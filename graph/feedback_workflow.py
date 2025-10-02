"""Workflow graph that orchestrates the CodeFeedbackLoop with storage and callbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from swarm.orchestration.feedback_graph import FeedbackGraph
from swarm_system.learning.benchmark import FeedbackBenchmark
from swarm_system.learning.adaptive_benchmark import AdaptiveFeedbackBenchmark
from swarm_system.learning.schemas import IterationRecord

from .embedding_integration import GraphEmbeddingManager
from .workflow_engine import WorkflowGraph, WorkflowState

logger = logging.getLogger(__name__)

HumanCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


def _sanitize_node_id(value: str) -> str:
    return value.replace("\\", "/").replace("/", "_")


@dataclass
class WorkflowRunConfig:
    file_path: str
    code: str
    iterations: int = 1
    iteration_offset: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackWorkflow:
    """High-level workflow that consumes :class:`FeedbackGraph` results."""

    def __init__(
        self,
        feedback_graph: Optional[FeedbackGraph] = None,
        embedding_manager: Optional[GraphEmbeddingManager] = None,
        run_log_dir: str = "workflow_runs",
    ) -> None:
        self.feedback_graph = feedback_graph or FeedbackGraph()
        self.embedding_manager = embedding_manager or GraphEmbeddingManager()
        self.workflow = WorkflowGraph(name="code_feedback_workflow")
        self.run_log_dir = Path(run_log_dir)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = FeedbackBenchmark()
        self.adaptive_benchmark = AdaptiveFeedbackBenchmark()
        self._guidance_inbox: Dict[str, Deque[str]] = defaultdict(deque)

        # Human-in-the-loop callbacks
        self._human_callbacks: Dict[str, List[HumanCallback]] = defaultdict(list)

        self._build_workflow()

    # ------------------------------------------------------------------
    # Public registration helpers
    # ------------------------------------------------------------------
    def register_event_handler(self, event: str, callback: Callable[[WorkflowState, Dict[str, Any]], Awaitable[None]]) -> None:
        self.workflow.on(event, callback)

    def register_human_callback(self, event: str, callback: HumanCallback) -> None:
        self._human_callbacks[event].append(callback)

    def add_human_guidance(self, file_path: str, guidance: str) -> None:
        normalized = os.path.abspath(file_path)
        self._guidance_inbox[normalized].append(guidance)

    # ------------------------------------------------------------------
    async def run(self, *, file_path: str, code: Optional[str] = None, iterations: int = 1,
                  iteration_offset: int = 0, metadata: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Execute the workflow and return the final state."""
        config = await self._prepare_config(file_path=file_path, code=code, iterations=iterations,
                                            iteration_offset=iteration_offset, metadata=metadata)
        state = await self.workflow.run(initial_payload={"config": config})
        return state

    def describe(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow.describe(),
            "feedback_graph": self.feedback_graph.describe(),
        }

    async def _prepare_config(self, *, file_path: str, code: Optional[str], iterations: int,
                               iteration_offset: int, metadata: Optional[Dict[str, Any]]) -> WorkflowRunConfig:
        path = Path(file_path)
        if code is None:
            code = path.read_text(encoding="utf-8")
        return WorkflowRunConfig(
            file_path=str(path.resolve()),
            code=code,
            iterations=max(1, iterations),
            iteration_offset=max(0, iteration_offset),
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    def _build_workflow(self) -> None:
        self.workflow.add_node("prepare_input", self._node_prepare_input)
        self.workflow.add_node("load_history", self._node_load_history, requires=["prepare_input"])
        self.workflow.add_node("apply_guidance", self._node_apply_guidance, requires=["load_history"])
        self.workflow.add_node("run_feedback", self._node_run_feedback, requires=["apply_guidance"])
        self.workflow.add_node("log_results", self._node_log_results, requires=["run_feedback"])
        self.workflow.add_node("finalize", self._node_finalize, requires=["log_results"])

    # ------------------------------------------------------------------
    async def _node_prepare_input(self, state: WorkflowState) -> None:
        config: WorkflowRunConfig = state.payload["config"]
        run_id = f"feedback_{int(time.time())}"
        state.results["run"] = {
            "id": run_id,
            "file_path": config.file_path,
            "iterations": config.iterations,
            "iteration_offset": config.iteration_offset,
            "metadata": config.metadata,
            "code_length": len(config.code),
        }
        state.log("Prepared workflow input", run_id=run_id, file_path=config.file_path)
        await self.workflow.emit("input_prepared", state, state.results["run"])

    async def _node_load_history(self, state: WorkflowState) -> None:
        config: WorkflowRunConfig = state.payload["config"]
        try:
            report = self.benchmark.report(config.file_path, limit=20)
        except Exception as exc:
            state.log("Failed to load benchmark history", error=str(exc))
            report = {"iterations": 0, "reward": {"delta": 0, "mean": 0}, "top_issues": []}
        state.results["history"] = report
        await self.workflow.emit("history_loaded", state, report)

    async def _node_apply_guidance(self, state: WorkflowState) -> None:
        config: WorkflowRunConfig = state.payload["config"]
        normalized = os.path.abspath(config.file_path)
        guidance_updates: List[str] = list(self._guidance_inbox.pop(normalized, []))
        additional = config.metadata.get("guidance") or []
        if isinstance(additional, str):
            additional = [additional]
        guidance_updates.extend(additional)

        if guidance_updates:
            history = self.feedback_graph.loop.guidance_history.setdefault(normalized, deque(maxlen=10))
            for item in guidance_updates:
                history.append(item)
            recorded_guidance = list(history)
            await self.workflow.emit("guidance_applied", state, {"guidance": recorded_guidance})
            state.log("Applied guidance", guidance=recorded_guidance)
        else:
            recorded_guidance = []
            await self.workflow.emit("guidance_applied", state, {"guidance": recorded_guidance})

        state.results["guidance_history"] = recorded_guidance

    async def _node_run_feedback(self, state: WorkflowState) -> None:
        config: WorkflowRunConfig = state.payload["config"]
        results = await self.feedback_graph.run_iterations(
            file_path=config.file_path,
            code=config.code,
            count=config.iterations,
            start_index=config.iteration_offset,
            metadata=config.metadata,
        )
        state.results["iterations"] = results
        for record in results:
            await self.workflow.emit("iteration_completed", state, record)
        state.log("Completed feedback iterations", iteration_count=len(results))

    async def _node_log_results(self, state: WorkflowState) -> None:
        config: WorkflowRunConfig = state.payload["config"]
        await self.embedding_manager.initialize()
        run_meta = state.results.get("run", {})
        iterations: List[Dict[str, Any]] = state.results.get("iterations", [])

        code_node_id = await self.embedding_manager.create_node_with_embedding(
            content=config.code,
            node_type="code",
            node_id=f"code::{_sanitize_node_id(config.file_path)}",
            metadata={
                "file_path": config.file_path,
                "length": len(config.code),
                "run_id": run_meta.get("id"),
            },
        )

        for record in iterations:
            summary = record.get("generator_output", {}).get("overall_summary") or "No summary provided"
            feedback_node_id = await self.embedding_manager.create_node_with_embedding(
                content=summary,
                node_type="feedback_iteration",
                metadata={
                    "reward": record.get("discriminator_score", {}).get("reward"),
                    "iteration_index": record.get("iteration_index"),
                    "timestamp": record.get("timestamp"),
                },
            )
            await self.embedding_manager.create_relationship(
                source_id=code_node_id,
                target_id=feedback_node_id,
                relationship_type="feedback",
                weight=record.get("discriminator_score", {}).get("reward", 0.0) or 0.0,
                metadata={"graph": record.get("graph")},
            )

        state.results["graph_nodes"] = {
            "code": code_node_id,
            "feedback": len(iterations),
        }
        state.results.setdefault("run", {})["guidance_history"] = state.results.get("guidance_history", [])

        run_payload = await self._persist_run(state)
        self.adaptive_benchmark.update_from_run(run_payload)
        state.results["run_payload"] = run_payload
        await self.workflow.emit("results_logged", state, state.results["graph_nodes"])

    async def _node_finalize(self, state: WorkflowState) -> None:
        run_summary = {
            "file_path": state.results.get("run", {}).get("file_path"),
            "iterations": state.results.get("iterations", []),
            "history": state.results.get("history"),
            "guidance_history": state.results.get("run", {}).get("guidance_history", []),
            "adaptive_summary": self.adaptive_benchmark.summarise(),
            "adaptive_challenges": self.adaptive_benchmark.build_challenge_set(3),
        }
        state.results["adaptive_summary"] = run_summary["adaptive_summary"]
        state.results["adaptive_challenges"] = run_summary["adaptive_challenges"]
        await self.workflow.emit("workflow_summary", state, run_summary)

        for event, callbacks in self._human_callbacks.items():
            payload = {
                "file_path": run_summary["file_path"],
                "event": event,
                "state": state,
            }
            for callback in callbacks:
                try:
                    await callback(event, payload)
                except Exception as exc:
                    logger.error("Human callback failed: %s", exc)

    # ------------------------------------------------------------------
    async def _persist_run(self, state: WorkflowState) -> Dict[str, Any]:
        run_meta = dict(state.results.get("run", {}))
        run_id = run_meta.get("id", f"run_{int(time.time())}")
        run_meta["id"] = run_id
        file_token = _sanitize_node_id(run_meta.get("file_path", "unknown"))
        out_path = self.run_log_dir / f"{file_token}_{run_id}.json"

        payload = {
            "run": run_meta,
            "history": state.results.get("history"),
            "iterations": state.results.get("iterations"),
            "graph_nodes": state.results.get("graph_nodes"),
            "logs": state.logs,
            "timestamp": time.time(),
        }
        payload["run"]["log_path"] = str(out_path)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        state.results["run"] = payload["run"]
        state.log("Persisted workflow run", log_path=str(out_path))
        return payload
