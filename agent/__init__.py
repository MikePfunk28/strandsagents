"""
StrandsAgents agent package.

Exports the core @agent decorator utilities, sandbox helpers, deterministic
scaffolder, and (when available) the high-level agent builder API.
"""

from __future__ import annotations

from typing import Any, Dict

__version__ = "1.0.0"
__author__ = "StrandsAgents Community"

from .agent_decorator import (
    agent,
    list_agents,
    get_agent_info,
    call_agent,
    get_agents_with_code_execution,
    get_agents_by_model,
    print_agent_summary,
    AGENT_REGISTRY,
)

from .sandbox_executor import SandboxExecutor, ExecutionResult
from .sandbox_tool import sandbox_test_code
from .scaffolder import (
    AgentConfig,
    ScaffoldingResult,
    QUESTIONNAIRE,
    materialise_agent_module,
    render_agent_module,
    validate_answers,
    generate_agent,
)

try:  # Optional: builder utilities may depend on heavy tool stacks.
    from .agent_builder import (
        AgentBuilder,
        AgentSpecification,
        agent_builder,
        create_agent_interactive,
        create_agent_from_type,
        list_agent_types,
        list_available_models,
        validate_generated_agent,
    )
    _AGENT_BUILDER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - avoid hard dependency at import time
    AgentBuilder = None  # type: ignore[assignment]
    AgentSpecification = None  # type: ignore[assignment]
    agent_builder = None  # type: ignore[assignment]
    _AGENT_BUILDER_IMPORT_ERROR = exc

    def _builder_stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError(
            "agent_builder module is unavailable. Restore agent/agent_builder.py or "
            "install its dependencies."
        ) from _AGENT_BUILDER_IMPORT_ERROR

    create_agent_interactive = _builder_stub  # type: ignore[assignment]
    create_agent_from_type = _builder_stub  # type: ignore[assignment]
    list_agent_types = _builder_stub  # type: ignore[assignment]
    list_available_models = _builder_stub  # type: ignore[assignment]
    validate_generated_agent = _builder_stub  # type: ignore[assignment]


__all__ = [
    "agent",
    "list_agents",
    "get_agent_info",
    "call_agent",
    "get_agents_with_code_execution",
    "get_agents_by_model",
    "print_agent_summary",
    "AGENT_REGISTRY",
    "SandboxExecutor",
    "ExecutionResult",
    "sandbox_test_code",
    "AgentConfig",
    "ScaffoldingResult",
    "QUESTIONNAIRE",
    "materialise_agent_module",
    "render_agent_module",
    "validate_answers",
    "generate_agent",
]

if _AGENT_BUILDER_IMPORT_ERROR is None:
    __all__.extend(
        [
            "AgentBuilder",
            "AgentSpecification",
            "agent_builder",
            "create_agent_interactive",
            "create_agent_from_type",
            "list_agent_types",
            "list_available_models",
            "validate_generated_agent",
        ]
    )
