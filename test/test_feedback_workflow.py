import asyncio
from collections import defaultdict, deque
from pathlib import Path

from graph.feedback_workflow import FeedbackWorkflow


class DummyFeedbackGraph:
    def __init__(self):
        self.loop = type("Loop", (), {"guidance_history": defaultdict(lambda: deque(maxlen=10))})()

    async def run_iterations(self, *, file_path: str, code: str, count: int, start_index: int = 0,
                             metadata=None):
        return [
            {
                "iteration_index": start_index,
                "timestamp": 0.0,
                "generator_output": {
                    "overall_summary": "demo",
                    "line_annotations": [],
                    "scope_annotations": [],
                    "raw_response": "{}",
                },
                "discriminator_score": {
                    "reward": 0.8,
                    "coverage": 0.7,
                    "accuracy": 0.6,
                    "coherence": 0.5,
                    "formatting": 0.9,
                    "issues": [],
                },
                "agitator_feedback": {
                    "guidance": "keep improving",
                    "improvement_plan": [],
                    "targeted_prompts": [],
                    "raw_response": "{}",
                },
                "graph": {"nodes": [], "edges": [], "loop_guidance": []},
                "metadata": metadata or {},
            }
        ]

    def describe(self):
        return {
            "nodes": [],
            "edges": [],
            "loop_guidance": list(self.loop.guidance_history.keys()),
        }


class DummyEmbeddingManager:
    async def initialize(self):
        return None

    async def create_node_with_embedding(self, **kwargs):
        return f"node_{kwargs.get('node_type')}"

    async def create_relationship(self, **kwargs):
        return "edge_feedback"


def test_feedback_workflow_runs(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_text("print('hello')\n", encoding="utf-8")

    workflow = FeedbackWorkflow(
        feedback_graph=DummyFeedbackGraph(),
        embedding_manager=DummyEmbeddingManager(),
        run_log_dir=str(tmp_path / "runs"),
    )

    workflow.benchmark.report = lambda *_args, **_kwargs: {"reward": {"delta": 0.1}}
    workflow.add_human_guidance(str(file_path), "Check formatting")

    state = asyncio.run(workflow.run(file_path=str(file_path), iterations=1))

    assert state.results["graph_nodes"]["feedback"] == 1
    guidance_history = workflow.feedback_graph.loop.guidance_history[str(file_path)]
    assert "Check formatting" in guidance_history

    log_path = Path(state.results["run"]["log_path"])
    assert log_path.exists()