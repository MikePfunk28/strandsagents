# Deterministic Agent Scaffolder

The scaffolder (`agent/scaffolder.py`) turns a structured questionnaire into a fully configured agent module that:

- wraps the Strands `@agent` decorator,
- wires in tools (including the new `sandbox_test_code` sandbox hook),
- persists the system prompt and metadata, and
- optionally chunks and embeds supporting documents for retrieval.

Run it after the AI has gathered the required answers:

```bash
python -m agent.scaffolder
```

The CLI mirrors the question set below and writes artifacts into `assistants/generated/` by default.

## Question Flow

| Question ID | Purpose | Notes |
|-------------|---------|-------|
| `agent_name` | Python function + filename | Snake_case, becomes module name |
| `display_name` | Human-readable label | Defaults to title-cased name |
| `description` | One-line summary | Stored in metadata + docstring |
| `prompt_source` | `inline` or `file` | Select how the system prompt is supplied |
| `system_prompt` | Core instructions (inline) | Only required when `prompt_source=inline` |
| `prompt_path` | Prompt file location | Required when `prompt_source=file`; the file is copied into the generated prompts folder |
| `model_id` | Ollama model choice | Defaults to `qwen3:8b` |
| `enable_code_execution` | Toggle sandbox integration | Adds sandbox support when true |
| `sandbox_timeout` | Execution timeout | Seconds per sandbox call |
| `tool_selection` | Tools to attach | Uses keys from `TOOL_LIBRARY` |
| `memory_profile` | Chunk/embedding depth | `none`, `light`, or `full` |
| `embedding_namespace` | Vector namespace | Auto-generated if blank |
| `context_documents` | Extra knowledge sources | Comma separated paths |
| `output_dir` | Destination | Defaults to `assistants/generated` |

The answers are validated and normalised before generation to keep results deterministic.

## Generated Files

| File | Contents |
|------|----------|
| `<output_dir>/<agent_name>.py` | Agent module with `@agent` decorator |
| `assistants/generated/prompts/<agent_name>.prompt` | System prompt |
| `assistants/generated/metadata/<agent_name>.json` | Configuration metadata |
| `assistants/generated/metadata/<agent_name>_embeddings.json` | Optional chunk + embedding payload |

The metadata file references all related artifacts so downstream systems can ingest them.
The generated agent module always loads its system prompt from the colocated `.prompt` file so multi-line content can be edited freely outside Python source.

## UI / API Integration

Use `agent.scaffolder.generate_agent(answers_dict)` to drive the scaffolder from the app.
It returns a `ScaffoldingResult` containing a success flag, human-readable message, artifact paths, and the resolved metadata payload—ideal for rendering in the Agent Builder interface instead of reading CLI output.

## Memory Profiles & Embeddings

- `none`: Skips chunking/embedding entirely.
- `light`: Uses 900/180 chunk/overlap and embeds sections + chunks.
- `full`: Uses 700/120 chunk/overlap and embeds the full hierarchy (document, summary, sections, chunks, sentences).

Context documents are read, chunked with `ChunkingAssistant`, embedded via `EmbeddingAssistant`, and stored alongside the metadata. The namespace is derived from the agent name unless explicitly provided.

## Sandbox Tool

`sandbox_test_code` (from `agent/sandbox_tool.py`) is exposed as an agent tool so assistants can execute snippets without spawning other agents. Include `"sandbox_test_code"` in the tool selection to make it available in generated agents.

Example use inside a generated agent:

```python
from agent import agent, sandbox_test_code

TOOLS = [sandbox_test_code]
```

The scaffolder automatically wires this in when `tool_selection` includes the key, and the metadata records that sandbox execution is enabled.
