"""Lightweight workflow graph engine for coordinating async nodes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

WorkflowFunc = Callable[["WorkflowState"], Awaitable[None]]
EventCallback = Callable[["WorkflowState", Dict[str, Any]], Awaitable[None]]


@dataclass
class WorkflowState:
    """Shared state passed between workflow nodes."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def log(self, message: str, **metadata: Any) -> None:
        entry = {
            "timestamp": time.time(),
            "message": message,
            **metadata,
        }
        self.logs.append(entry)
        logger.info("[%s] %s", self.name, message, extra={"workflow": metadata})


@dataclass
class WorkflowNode:
    """Single unit of work in the workflow graph."""

    name: str
    func: WorkflowFunc
    requires: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def run(self, state: WorkflowState) -> None:
        await self.func(state)


class WorkflowGraph:
    """Directed acyclic graph executor for orchestrating workflow nodes."""

    def __init__(self, name: str):
        self.name = name
        self._nodes: Dict[str, WorkflowNode] = {}
        self._successors: Dict[str, Set[str]] = defaultdict(set)
        self._event_handlers: Dict[str, List[EventCallback]] = defaultdict(list)

    def add_node(
        self,
        name: str,
        func: WorkflowFunc,
        requires: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name in self._nodes:
            raise ValueError(f"Workflow node '{name}' already exists")

        dependencies = set(requires or [])
        node = WorkflowNode(name=name, func=func, requires=dependencies, metadata=metadata or {})
        self._nodes[name] = node

        for dependency in dependencies:
            if dependency not in self._nodes:
                logger.warning("Dependency '%s' added before definition", dependency)
            self._successors[dependency].add(name)

    def on(self, event: str, callback: EventCallback) -> None:
        """Register an async callback for a workflow event."""
        self._event_handlers[event].append(callback)

    async def emit(self, event: str, state: WorkflowState, payload: Optional[Dict[str, Any]] = None) -> None:
        handlers = self._event_handlers.get(event)
        if not handlers:
            return
        for handler in handlers:
            try:
                await handler(state, payload or {})
            except Exception as exc:  # pragma: no cover - logging only
                logger.error("Workflow event handler failed: %s", exc)

    async def run(self, initial_payload: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Execute the workflow graph topologically."""
        state = WorkflowState(name=self.name, payload=initial_payload or {})
        await self.emit("workflow_started", state, {"node_count": len(self._nodes)})

        in_degree: Dict[str, int] = {
            name: len(node.requires) for name, node in self._nodes.items()
        }
        ready = deque([name for name, degree in in_degree.items() if degree == 0])
        executed: Set[str] = set()

        while ready:
            node_name = ready.popleft()
            node = self._nodes[node_name]
            await self.emit("node_started", state, {"node": node_name})

            try:
                await node.run(state)
                executed.add(node_name)
                await self.emit("node_completed", state, {"node": node_name})
            except Exception as exc:
                state.log(f"Node '{node_name}' failed", error=str(exc))
                await self.emit("node_failed", state, {"node": node_name, "error": str(exc)})
                raise

            for successor in self._successors.get(node_name, set()):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.append(successor)

        if len(executed) != len(self._nodes):
            missing = set(self._nodes) - executed
            raise RuntimeError(f"Workflow '{self.name}' did not execute nodes: {missing}")

        await self.emit("workflow_completed", state, {"executed_nodes": list(executed)})
        return state

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": {
                name: {
                    "requires": sorted(node.requires),
                    "metadata": node.metadata,
                }
                for name, node in self._nodes.items()
            },
        }