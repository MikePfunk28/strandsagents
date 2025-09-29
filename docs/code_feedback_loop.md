# Code Feedback Loop

This document explains the GAN-inspired learning loop introduced in `swarm_system/learning`.

## Components
- **CodeCommentGeneratorAssistant** (`swarm_system/learning/generator_assistant.py`): creates JSON line annotations and scope summaries for a source file. Results are persisted to `coderl.db` and short summaries are cached in `memory.db`.
- **CodeCommentDiscriminatorAssistant** (`swarm_system/learning/discriminator_assistant.py`): scores generator output for coverage, accuracy, coherence, and formatting, yielding a reward signal and storing structured feedback.
- **CodeCommentAgitatorAssistant** (`swarm_system/learning/agitator_assistant.py`): produces targeted prompts and improvement plans that push the generator toward better behaviour while logging actionable guidance in `knowledge.db`.
- **CodeFeedbackLoop** (`swarm_system/learning/code_feedback_loop.py`): orchestrates generator → discriminator → agitator, tracks guidance history, and records full iteration payloads for later analysis.

All assistants run on local Ollama models. Default configuration uses `llama3.2` for generator/discriminator and `gemma2:27b` for the agitator (override `model_id` in `AssistantConfig` if needed).

## Running a Training Iteration
```python
from swarm_system.learning import CodeFeedbackLoop

loop = CodeFeedbackLoop()
with open('path/to/file.py', 'r', encoding='utf-8') as f:
    code = f.read()

record = loop.run_iteration('path/to/file.py', code)
print(record.discriminator_score.reward, record.agitator_feedback.guidance)
```

For multiple files or repeated passes:
```python
dataset = [("file_a.py", open("file_a.py").read()),
           ("file_b.py", open("file_b.py").read())]
loop.train(dataset, epochs=3)
```

Each iteration writes:
- `memory.db` (`feedback_iteration`, `generator_summary` entries)
- `coderl.db` (`code_explanations`, `file_scopes`)
- `knowledge.db` (`topic='code_gan_iteration'` with complete iteration payloads, agitator coaching entries)

## Benchmarking Progress
Use `FeedbackBenchmark` to analyse stored iterations.
```python
from swarm_system.learning.benchmark import FeedbackBenchmark

benchmark = FeedbackBenchmark()
report = benchmark.report('path/to/file.py', limit=30)
print(report['reward'])
print(report['top_issues'])
```

`report` includes per-metric means, reward delta between first and last iterations, and the five most common discriminator issues. You can also call `compare(baseline_records, candidate_records)` to measure gains between runs.

## Extending the Loop
- `IterationRecord` dataclasses (see `swarm_system/learning/schemas.py`) support round-trip serialization for future replay or RL fine-tuning workflows.
- Guidance history is maintained per file, so adding human-in-the-loop feedback is as simple as appending prompts to `loop.guidance_history[file_path]`.
- LoRA / RLHF integration can build on the captured reward traces (`reward` and `issues` fields) once the adaptation pipeline is ready.

## Suggested Next Steps
1. Wire the loop into existing workflows (e.g., `swarm_system/swarm_demo.py`) so agents can request iterative refinements automatically.
2. Add human feedback ingestion that pushes curated prompts into the guidance history.
3. Hook benchmark summaries into dashboards or CLI reports for quick regression checks.
## Swarm Integration
- **FeedbackGraph** (`swarm/orchestration/feedback_graph.py`): wraps `CodeFeedbackLoop` with explicit generator→discriminator→agitator edges so orchestration layers can visualise and reuse the pipeline topology. `describe()` returns node/edge metadata suitable for dashboards or workflow planners.
- **CodeFeedbackAssistant** (`swarm/agents/code_feedback/service.py`): lightweight 270M microservice that exposes the GAN loop over MCP. It listens for `task_request` messages containing `code` and `file_path`, runs the graph, and replies with rewards, guidance, and the iteration history. The assistant registers as agent type `code_feedback`, enabling agent-to-agent collaboration and orchestrator scheduling.
- **Swarm bootstrapping** (`swarm/main.py`): the default swarm now instantiates a code feedback assistant alongside the research/creative/critical/summariser services so orchestration plans can route code-review work automatically.

## Tools and MCP Access
- **code_feedback_tool** (`swarm/tools/code_feedback_tool.py`): Strands-compliant tool spec that triggers the graph synchronously. Assistants or meta-tooling flows can call it with `{ "file_path": "...", "code": "...", "iterations": 2 }` to obtain JSON-formatted iteration payloads.
- **Meta-tooling** (`swarm/meta_tooling/tool_builder.py`): because the tool lives in `swarm/tools`, the existing meta-tool builder can load it and chain new composite tools without extra wiring. Agents can therefore expose the full feedback loop as a callable tool inside MCP sessions.

## MCP & Agent-to-Agent Flow
- The `CodeFeedbackAssistant` registers through `SwarmMCPClient`, advertising capabilities `code_commentary`, `gan_feedback`, `reward_analysis`, and `guidance_generation`.
- `SwarmOrchestrator.submit_task` now ensures any task with `task_type="code_feedback"` (or a context payload containing `code`) allocates the `code_feedback` agent and plans for the GAN loop before distributing work.
- Results propagate back to the orchestrator as structured payloads; the coordinator captures the reward trace for synthesis with other agents (e.g., summariser) so downstream tools receive consolidated feedback and improvement guidance.

## Visualising the Graph
Call `FeedbackGraph.describe()` to obtain nodes, edges, and guidance keys for a given loop instance:
```python
from swarm.orchestration.feedback_graph import FeedbackGraph

graph = FeedbackGraph()
print(graph.describe())
```
This is the same representation returned in every `CodeFeedbackAssistant` response under the `graph` key, allowing dashboards or workflows to render the active feedback circuit.

## Workflow Graph Integration
- **FeedbackWorkflow** (`graph/feedback_workflow.py`): wraps `FeedbackGraph` inside a dependency-aware workflow (`WorkflowGraph`). Nodes cover input prep, history lookup, guidance application, loop execution, and logging with embeddings. Results persist to `workflow_runs/` and the graph storage layer for later retrieval.
- **WorkflowGraph Engine** (`graph/workflow_engine.py`): lightweight DAG executor used by the workflow. Register additional nodes or event handlers to extend the pipeline, e.g., auditing or notification steps.
- **CLI Support** (`run_graph.py`): run `python run_graph.py --code-feedback path/to/file.py --iterations 2 --guidance "Review function headers"` to execute the full loop, view rewards, and locate persisted logs. Interactive mode (`python run_graph.py --interactive`) now includes a `feedback` command.

### Human-in-the-Loop callbacks
Call `FeedbackWorkflow.add_human_guidance(path, guidance)` before running to enqueue manual hints. Subscribing to events is as simple as:
```python
workflow = FeedbackWorkflow()

async def on_iteration(state, payload):
    print("Iteration", payload.get("iteration_index"), payload.get("discriminator_score", {}).get("reward"))

workflow.register_event_handler("iteration_completed", on_iteration)
workflow.add_human_guidance("agent.py", "Focus on type hints")
await workflow.run(file_path="agent.py", iterations=2)
```
Each run writes a JSON report under `workflow_runs/` containing the raw generator/discriminator/agitator payloads, making it easy to diff guidance impact across iterations.

## Agent-to-Agent & MCP Integration
- **In-memory MCP broker** (`swarm/communication/inmemory_mcp.py`) exposes a lightweight message bus mirroring the MCP patterns used in the swarm. It lets orchestration logic exercise agent-to-agent requests without standing up the full socket server.
- **FeedbackAgentChannel** (`swarm/communication/feedback_channel.py`) provides a thin request/response helper. It tracks correlation IDs and resolves futures when the code-feedback agent responds, making MCP calls feel synchronous from the caller.
- **FeedbackAgentService** (`swarm/communication/feedback_service.py`) wraps `FeedbackWorkflow` so any MCP client can plug in and service `feedback_request` messages with full workflow execution.

## Adaptive Benchmarking
- **AdaptiveFeedbackBenchmark** (`swarm_system/learning/adaptive_benchmark.py`) captures run metadata (reward deltas, guidance count, log locations) and generates evolving challenge sets. Every workflow run updates the benchmark automatically via `FeedbackWorkflow`.
- `summary()` returns aggregate stats, while `build_challenge_set()` surfaces the toughest files to rerun so the loop continues to improve and the model is regularly challenged as memory/guidance changes.

