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
    memory_impact: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "file_path": self.file_path,
            "reward_delta": self.reward_delta,
            "average_reward": self.average_reward,
            "iteration_count": self.iteration_count,
            "guidance_count": self.guidance_count,
            "memory_impact": self.memory_impact,
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
        self.records = [self._coerce_record(item) for item in data.get("records", [])]

    def _persist(self) -> None:
        payload = {"records": [item.to_dict() for item in self.records]}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _coerce_record(payload: Dict[str, Any]) -> AdaptiveRecord:
        """Convert stored dictionary payloads into `AdaptiveRecord` instances."""
        memory_impact_raw = payload.get("memory_impact") or {}
        if not isinstance(memory_impact_raw, dict):
            memory_impact_raw = {}
        memory_impact = {
            str(key): float(value)
            for key, value in memory_impact_raw.items()
            if isinstance(value, (int, float))
        }
        return AdaptiveRecord(
            run_id=str(payload.get("run_id", "unknown")),
            file_path=str(payload.get("file_path", "unknown")),
            reward_delta=float(payload.get("reward_delta", 0.0) or 0.0),
            average_reward=float(payload.get("average_reward", 0.0) or 0.0),
            iteration_count=int(payload.get("iteration_count", 0) or 0),
            guidance_count=int(payload.get("guidance_count", 0) or 0),
            memory_impact=memory_impact,
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _normalise_memory_counts(section: Any) -> Dict[str, float]:
        """Extract numeric memory counters from payload sections."""
        if not isinstance(section, dict):
            return {}
        counts: Dict[str, float] = {}
        for key, value in section.items():
            if isinstance(value, (int, float)):
                slug = str(key)
                lowered = slug.lower()
                if lowered in {"total", "__total__"}:
                    # Totals are derived; avoid double-counting here
                    continue
                counts[slug] = float(value)
        return counts

    def _extract_memory_impact(self, run_payload: Dict[str, Any]) -> Dict[str, float]:
        """Compute before/after deltas for memory usage in a run payload."""
        snapshot = run_payload.get("memory_snapshot") or {}
        before = self._normalise_memory_counts(snapshot.get("before"))
        after = self._normalise_memory_counts(snapshot.get("after"))
        if not before and not after:
            return {}
        keys = set(before) | set(after)
        impact = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in keys
        }
        impact["__total__"] = sum(after.values()) - sum(before.values())
        return impact

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

        memory_impact = self._extract_memory_impact(run_payload)

        record = AdaptiveRecord(
            run_id=str(run_info.get("id", "unknown")),
            file_path=str(run_info.get("file_path", "unknown")),
            reward_delta=float(reward_summary.get("delta", 0.0) or 0.0),
            average_reward=average_reward,
            iteration_count=len(iterations),
            guidance_count=len(run_info.get("guidance_history", [])),
            memory_impact=memory_impact,
            metadata={
                "log_path": run_info.get("log_path"),
                "timestamp": run_payload.get("timestamp"),
                "memory_snapshot": run_payload.get("memory_snapshot"),
            },
        )

        # Replace existing record with same run_id if present
        self.records = [r for r in self.records if r.run_id != record.run_id]
        self.records.append(record)
        self.records.sort(
            key=lambda item: (
                item.reward_delta,
                item.average_reward,
                abs(item.memory_impact.get("__total__", 0.0)),
            ),
            reverse=True,
        )
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
                    "guidance_count": record.guidance_count,
                    "memory_pressure": record.memory_impact.get("__total__", 0.0),
                    "memory_breakdown": record.memory_impact,
                    "run_id": record.run_id,
                }
            )
        return challenges

    def summarise(self) -> Dict[str, Any]:
        """Return summary statistics across stored records."""
        if not self.records:
            return {
                "records": 0,
                "avg_reward": 0.0,
                "avg_delta": 0.0,
                "avg_guidance": 0.0,
                "memory": {
                    "avg_total_delta": 0.0,
                    "positive_runs": 0,
                    "negative_runs": 0,
                    "top_files": [],
                },
            }
        record_count = len(self.records)
        avg_reward = sum(record.average_reward for record in self.records) / record_count
        avg_delta = sum(record.reward_delta for record in self.records) / record_count
        avg_guidance = sum(record.guidance_count for record in self.records) / record_count
        memory_totals = [record.memory_impact.get("__total__", 0.0) for record in self.records]
        avg_memory = sum(memory_totals) / record_count
        positive_runs = sum(1 for value in memory_totals if value > 0)
        negative_runs = sum(1 for value in memory_totals if value < 0)
        top_memory = sorted(
            (
                {
                    "file_path": record.file_path,
                    "delta": record.memory_impact.get("__total__", 0.0),
                    "breakdown": record.memory_impact,
                }
                for record in self.records
            ),
            key=lambda item: abs(item["delta"]),
            reverse=True,
        )[:3]
        return {
            "records": record_count,
            "avg_reward": avg_reward,
            "avg_delta": avg_delta,
            "avg_guidance": avg_guidance,
            "memory": {
                "avg_total_delta": avg_memory,
                "positive_runs": positive_runs,
                "negative_runs": negative_runs,
                "top_files": top_memory,
            },
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
