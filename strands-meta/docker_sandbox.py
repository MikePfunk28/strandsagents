"""
Docker sandbox wrapper for strands-meta utilities.

Delegates to the core `agent.docker_sandbox` implementation so meta tooling can
reuse the same container execution engine without duplicating code. The import
logic mirrors other meta modules, attempting relative imports first and falling
back to top-level package imports when this file is executed standalone.
"""

from __future__ import annotations

try:
    from ..agent.docker_sandbox import DockerSandboxExecutor, ExecutionResult
except ImportError:  # pragma: no cover - allow running as script
    from agent.docker_sandbox import DockerSandboxExecutor, ExecutionResult  # type: ignore

__all__ = ["DockerSandboxExecutor", "ExecutionResult"]
