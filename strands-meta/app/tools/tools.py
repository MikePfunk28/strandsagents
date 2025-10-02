from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from strands import tool

WORKDIR = Path(os.getenv("WORKDIR", os.getcwd())).resolve()


class ToolError(Exception):
    """Raised when a tool is misused or cannot complete safely."""


def _resolve_path(path: str) -> Path:
    candidate = (WORKDIR / path).resolve()
    if WORKDIR not in candidate.parents and candidate != WORKDIR:
        raise ToolError("Path escape blocked")
    return candidate


@tool(name="fs.read")
def fs_read(path: str) -> str:
    """Read a UTF-8 file relative to the workspace root."""
    full_path = _resolve_path(path)
    if not full_path.exists():
        raise ToolError(f"Path not found: {path}")
    return full_path.read_text(encoding="utf-8", errors="ignore")


@tool(name="fs.write")
def fs_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file relative to the workspace root."""
    full_path = _resolve_path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return "ok"


@tool(name="fs.diff")
def fs_diff(path: str, new: str) -> str:
    """Produce a JSON payload describing before/after text for a file."""
    full_path = _resolve_path(path)
    before = ""
    if full_path.exists():
        before = full_path.read_text(encoding="utf-8", errors="ignore")
    diff_payload = {"path": path, "before": before, "after": new}
    return json.dumps(diff_payload)



@tool(name="think")
def think(prompt: str) -> str:
    """Return a short internal reflection string."""
    return prompt


@tool(name="sh.run")
def sh(command: str, consent: bool = False) -> str:
    """Execute a shell command inside the workspace after explicit consent."""
    if not consent:
        raise ToolError("Consent required for shell execution")
    process = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        cwd=str(WORKDIR),
        check=False,
    )
    if process.returncode:
        raise ToolError(process.stderr.strip() or f"Command failed: {command}")
    return process.stdout


_TOOL_REGISTRY: Dict[str, Any] = {
    "fs.read": fs_read,
    "fs.write": fs_write,
    "fs.diff": fs_diff,
    "think": think,
    "sh.run": sh,
}


def get_toolset(capabilities: List[str] | None) -> List[Any]:
    """Return the Strands tool objects for the requested capabilities."""
    if not capabilities:
        return []
    missing = [name for name in capabilities if name not in _TOOL_REGISTRY]
    if missing:
        raise KeyError(f"Unknown tool(s): {', '.join(missing)}")
    return [_TOOL_REGISTRY[name] for name in capabilities]


ALL_TOOLS: List[Any] = list(_TOOL_REGISTRY.values())

