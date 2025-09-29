"""Adaptive benchmarking for feedback loop evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .benchmark import FeedbackBenchmark
from .schemas import IterationRecord, GeneratorOutput, DiscriminatorScore, AgitatorFeedback


@dataclass
class AdaptiveRecord:
    """Stored snapshot summarising a workflow run."""

    run_id: str
    file_path: str
    reward_delta: float
    average_reward: float
    iteration_count: int
    guidance_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "file_path": self.file_path,
            "reward_delta": self.reward_delta,
            "average_reward": self.average_reward,
            "iteration_count": self.iteration_count,
            "guidance_count": self.guidance_count,
            "metadata": self.metadata,
        }


class AdaptiveFeedbackBenchmark(FeedbackBenchmark):
    """Extends `FeedbackBenchmark` with adaptive challenge generation."""

    def __init__(self, storage_path: str = "workflow_runs/adaptive_benchmark.json") -> None:
        super().__init__()
        self.storage_path = Path(storage_path)
        self.records: List[AdaptiveRecord] = []
        if self.storage_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.records = [AdaptiveRecord(**item) for item in data.get("records", [])]

    def _persist(self) -> None:
        payload = {"records": [item.to_dict() for item in self.records]}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update_from_run(self, run_payload: Dict[str, Any]) -> None:
        """Update adaptive benchmark statistics from a workflow run payload."""
        run_info = run_payload.get("run", {})
        history = run_payload.get("history", {})
        iterations = run_payload.get("iterations", [])
        reward_summary = history.get("reward", {}) if isinstance(history, dict) else {}

        rewards = [
            item.get("discriminator_score", {}).get("reward")
            for item in iterations
            if item.get("discriminator_score")
        ]
        rewards = [value for value in rewards if isinstance(value, (int, float))]
        average_reward = sum(rewards) / len(rewards) if rewards else 0.0

        record = AdaptiveRecord(
            run_id=str(run_info.get("id", "unknown")),
            file_path=str(run_info.get("file_path", "unknown")),
            reward_delta=float(reward_summary.get("delta", 0.0) or 0.0),
            average_reward=average_reward,
            iteration_count=len(iterations),
            guidance_count=len(run_info.get("guidance_history", [])),
            metadata={
                "log_path": run_info.get("log_path"),
                "timestamp": run_payload.get("timestamp"),
            },
        )

        # Replace existing record with same run_id if present
        self.records = [r for r in self.records if r.run_id != record.run_id]
        self.records.append(record)
        self.records.sort(key=lambda item: (item.reward_delta, item.average_reward), reverse=True)
        self._persist()

    def build_challenge_set(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top N challenge descriptors focusing on difficult files."""
        challenges = []
        for record in self.records[:top_n]:
            challenges.append(
                {
                    "file_path": record.file_path,
                    "target_reward": max(record.average_reward, 0.8),
                    "guidance_hint": record.metadata.get("log_path"),
                    "expected_iterations": record.iteration_count,
                    "run_id": record.run_id,
                }
            )
        return challenges

    def summarise(self) -> Dict[str, Any]:
        """Return summary statistics across stored records."""
        if not self.records:
            return {"records": 0, "avg_reward": 0.0, "avg_delta": 0.0}
        avg_reward = sum(record.average_reward for record in self.records) / len(self.records)
        avg_delta = sum(record.reward_delta for record in self.records) / len(self.records)
        return {
            "records": len(self.records),
            "avg_reward": avg_reward,
            "avg_delta": avg_delta,
        }

    @staticmethod
    def convert_iteration(payload: Dict[str, Any]) -> IterationRecord:
        """Utility to convert dict payload into `IterationRecord`."""
        return IterationRecord(
            generator_output=GeneratorOutput.from_dict(payload.get("generator_output", {})),
            discriminator_score=DiscriminatorScore.from_dict(payload.get("discriminator_score", {})),
            agitator_feedback=AgitatorFeedback.from_dict(payload.get("agitator_feedback", {})),
            iteration_index=int(payload.get("iteration_index", 0)),
            timestamp=float(payload.get("timestamp", 0.0)),
            metadata=payload.get("metadata", {}),
        )
