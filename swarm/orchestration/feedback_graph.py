"""Graph orchestration primitives for the code feedback learning loop."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from swarm_system.learning import CodeFeedbackLoop
from swarm_system.learning.schemas import IterationRecord


@dataclass
class GraphNode:
    """Lightweight description of a node participating in the loop."""

    name: str
    role: str
    description: str
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "model": self.model,
        }


@dataclass
class GraphEdge:
    """Directed connection between two graph nodes."""

    source: str
    target: str
    channel: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "channel": self.channel,
            "description": self.description,
        }


class FeedbackGraph:
    """Graph wrapper around :class:`CodeFeedbackLoop` for orchestration layers."""

    def __init__(self, loop: Optional[CodeFeedbackLoop] = None) -> None:
        self.loop = loop or CodeFeedbackLoop()
        self.nodes: List[GraphNode] = [
            GraphNode(
                name="generator",
                role="annotation",
                description="Produces structured line and scope commentary for source files.",
                model="llama3.2",
            ),
            GraphNode(
                name="discriminator",
                role="evaluation",
                description="Scores generator output for coverage, accuracy, coherence, and formatting.",
                model="llama3.2",
            ),
            GraphNode(
                name="agitator",
                role="coaching",
                description="Delivers corrective guidance and challenge prompts for the next iteration.",
                model="gemma2:27b",
            ),
        ]
        self.edges: List[GraphEdge] = [
            GraphEdge(
                source="generator",
                target="discriminator",
                channel="evaluation",
                description="Pass structured annotations for scoring and reward shaping.",
            ),
            GraphEdge(
                source="discriminator",
                target="agitator",
                channel="critique",
                description="Deliver reward signal and issues for targeted coaching.",
            ),
            GraphEdge(
                source="agitator",
                target="generator",
                channel="guidance",
                description="Feed improvement guidance back into the generator prompt.",
            ),
        ]

    def describe(self) -> Dict[str, Any]:
        """Return metadata describing the graph topology and models."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "loop_guidance": list(self.loop.guidance_history.keys()),
        }

    async def run_iteration(
        self,
        *,
        file_path: str,
        code: str,
        iteration_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single iteration and return a serialisable payload."""
        record = await self.loop.run_iteration_async(
            file_path=file_path,
            code=code,
            iteration_index=iteration_index,
        )

        payload = self._record_to_dict(record)
        payload["graph"] = self.describe()
        payload["metadata"] = {**(metadata or {}), **payload.get("metadata", {})}
        return payload

    async def run_iterations(
        self,
        *,
        file_path: str,
        code: str,
        count: int,
        start_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple iterations sequentially."""
        results: List[Dict[str, Any]] = []
        for offset in range(count):
            result = await self.run_iteration(
                file_path=file_path,
                code=code,
                iteration_index=start_index + offset,
                metadata=metadata,
            )
            results.append(result)
        return results

    @staticmethod
    def _record_to_dict(record: IterationRecord) -> Dict[str, Any]:
        """Convert :class:`IterationRecord` into a pure dictionary."""
        return {
            "iteration_index": record.iteration_index,
            "timestamp": record.timestamp,
            "generator_output": asdict(record.generator_output),
            "discriminator_score": asdict(record.discriminator_score),
            "agitator_feedback": asdict(record.agitator_feedback),
            "metadata": record.metadata,
        }


def serialize_iterations(records: Sequence[IterationRecord]) -> List[Dict[str, Any]]:
    """Utility helper mirroring :meth:`FeedbackGraph._record_to_dict` for batches."""
    return [FeedbackGraph._record_to_dict(record) for record in records]