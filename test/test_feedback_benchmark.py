import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_system.learning.benchmark import FeedbackBenchmark


class FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def search_knowledge(self, query: str, limit: int = 50):
        return self._rows[:limit]


def _iteration_payload(iteration_index: int, reward: float, issue: str) -> dict:
    return {
        "iteration_index": iteration_index,
        "timestamp": float(iteration_index),
        "metadata": {},
        "generator_output": {
            "file_path": "sample.py",
            "raw_response": "{}",
            "line_annotations": [
                {
                    "line_number": 1,
                    "content": "print('hello')",
                    "explanation": "Print greeting",
                    "scope": "module",
                    "critique": None,
                }
            ],
            "scope_annotations": [],
            "overall_summary": "Demo",
            "confidence": 0.9,
            "metadata": {},
        },
        "discriminator_score": {
            "coverage": 0.8 + iteration_index * 0.05,
            "accuracy": 0.7 + iteration_index * 0.05,
            "coherence": 0.6 + iteration_index * 0.05,
            "formatting": 0.9,
            "reward": reward,
            "issues": [issue],
            "raw_response": "{}",
            "metadata": {},
        },
        "agitator_feedback": {
            "guidance": "Improve variable naming",
            "improvement_plan": ["Rename variables"],
            "targeted_prompts": ["Focus on descriptive names"],
            "raw_response": "{}",
            "metadata": {},
        },
    }


def test_feedback_benchmark_reports_positive_delta():
    rows = [
        {
            "topic": "code_gan_iteration",
            "metadata": json.dumps(_iteration_payload(0, 0.2, "missing detail")),
        },
        {
            "topic": "code_gan_iteration",
            "metadata": json.dumps(_iteration_payload(1, 0.6, "typo")),
        },
    ]
    benchmark = FeedbackBenchmark(db_manager_instance=FakeDB(rows))

    records = benchmark.load_iterations("sample.py")
    assert len(records) == 2

    report = benchmark.report("sample.py")
    assert report["iterations"] == 2
    assert report["reward"]["delta"] > 0
    assert report["top_issues"][0][0] in {"missing detail", "typo"}