"""Benchmark helpers for monitoring GAN-style feedback training."""

import json
from statistics import mean
from typing import Any, Dict, List, Optional

from swarm_system.learning.schemas import IterationRecord
from swarm_system.utils.database_manager import db_manager


class FeedbackBenchmark:
    """Loads iteration history and computes learning trends."""

    def __init__(self, db_manager_instance=db_manager) -> None:
        self.db = db_manager_instance

    def load_iterations(self, file_path: str, limit: int = 50) -> List[IterationRecord]:
        """Retrieve recent iteration records from the knowledge store."""
        rows = self.db.search_knowledge(file_path, limit=limit)
        records: List[IterationRecord] = []
        for row in rows:
            if row.get("topic") != "code_gan_iteration":
                continue
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if not metadata:
                continue
            try:
                record = IterationRecord.from_dict(metadata)
                records.append(record)
            except Exception:
                continue
        records.sort(key=lambda item: item.iteration_index)
        return records

    def summarize(self, records: List[IterationRecord]) -> Dict[str, Any]:
        """Compute descriptive statistics across iterations."""
        if not records:
            return {
                "iterations": 0,
                "reward": {"mean": 0.0, "first": 0.0, "last": 0.0, "delta": 0.0},
                "coverage": {"mean": 0.0},
                "accuracy": {"mean": 0.0},
                "coherence": {"mean": 0.0},
                "formatting": {"mean": 0.0},
                "top_issues": [],
            }

        rewards = [rec.discriminator_score.reward for rec in records]
        coverage = [rec.discriminator_score.coverage for rec in records]
        accuracy = [rec.discriminator_score.accuracy for rec in records]
        coherence = [rec.discriminator_score.coherence for rec in records]
        formatting = [rec.discriminator_score.formatting for rec in records]
        issues: Dict[str, int] = {}
        for rec in records:
            for issue in rec.discriminator_score.issues:
                slug = issue.strip().lower()
                if not slug:
                    continue
                issues[slug] = issues.get(slug, 0) + 1

        first_reward = rewards[0]
        last_reward = rewards[-1]
        delta = last_reward - first_reward

        return {
            "iterations": len(records),
            "reward": {
                "mean": mean(rewards),
                "first": first_reward,
                "last": last_reward,
                "delta": delta,
            },
            "coverage": {"mean": mean(coverage)},
            "accuracy": {"mean": mean(accuracy)},
            "coherence": {"mean": mean(coherence)},
            "formatting": {"mean": mean(formatting)},
            "top_issues": sorted(issues.items(), key=lambda item: item[1], reverse=True)[:5],
        }

    def report(self, file_path: str, limit: int = 50) -> Dict[str, Any]:
        """Generate a benchmark report for a file."""
        records = self.load_iterations(file_path, limit)
        summary = self.summarize(records)
        summary["file_path"] = file_path
        return summary

    def compare(self, baseline: List[IterationRecord], candidate: List[IterationRecord]) -> Dict[str, Any]:
        """Compare two training runs to quantify improvement."""
        baseline_summary = self.summarize(baseline)
        candidate_summary = self.summarize(candidate)
        return {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "reward_gain": candidate_summary["reward"]["mean"] - baseline_summary["reward"]["mean"],
        }