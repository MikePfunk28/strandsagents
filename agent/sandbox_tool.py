# sandbox_tool.py
"""
Sandbox-backed tool for executing code snippets deterministically.

Exposes the enhanced SandboxExecutor through a `@tool` wrapper so that agents
can validate code without spawning additional helper agents.
"""

from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - depends on external package
    from strands import tool
    STRANDS_AVAILABLE = True
except ImportError:  # pragma: no cover
    STRANDS_AVAILABLE = False

    def tool(func):  # type: ignore
        """Fallback decorator that leaves the function unchanged."""
        logger.warning(
            "Strands SDK not available; sandbox tool will behave as a plain function."
        )
        return func

from .sandbox_executor import SandboxExecutor, ExecutionResult

logger = logging.getLogger("sandbox_tool")


def _format_execution(result: ExecutionResult) -> str:
    """Convert execution result into a compact textual summary."""
    status = "succeeded" if result.success else "failed"
    lines = [
        f"Sandbox execution {status} (language={result.language}, "
        f"returncode={result.returncode}, session={result.session_id or 'new'})"
    ]

    if result.stdout:
        lines.append("stdout:")
        lines.append(result.stdout.strip())

    if result.stderr:
        lines.append("stderr:")
        lines.append(result.stderr.strip())

    if result.error:
        lines.append(f"error: {result.error}")

    if not result.stdout and not result.stderr and not result.error:
        lines.append("No output produced.")

    return "\n".join(lines)


@tool
def sandbox_test_code(
    code: str,
    language: str = "python",
    session_id: Optional[str] = None,
    timeout: int = 30
) -> str:
    """
    Execute code inside the enhanced sandbox environment.

    Args:
        code: Source code to execute (multi-line supported).
        language: One of the supported languages (python, javascript, bash, powershell).
        session_id: Optional persistent session identifier to reuse state.
        timeout: Execution timeout in seconds for this request.

    Returns:
        Textual summary of the execution outcome, including stdout/stderr.
    """
    try:
        executor = SandboxExecutor(timeout=timeout)
        result = executor.execute_code(code, language=language, session_id=session_id)
        return _format_execution(result)
    except Exception as exc:
        logger.error("Sandbox execution failed: %s", exc)
        return f"Sandbox execution error: {exc}"
