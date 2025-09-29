"""Strands tool for running the code feedback GAN pipeline."""

import asyncio
import json
from typing import Any, Dict

from strands.types.tool_types import ToolResult, ToolUse

from swarm.orchestration.feedback_graph import FeedbackGraph

TOOL_SPEC = {
    "name": "code_feedback_tool",
    "description": "Run the generator/discriminator/agitator feedback loop on a code snippet.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Logical path for the code sample (used for memory indexing)."
                },
                "code": {
                    "type": "string",
                    "description": "Source code to analyse."
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of feedback iterations to execute.",
                    "default": 1
                }
            },
            "required": ["file_path", "code"]
        }
    }
}

_GRAPH = FeedbackGraph()


def _run_async(coro):
    """Execute coroutine regardless of current loop state."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def code_feedback_tool(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Execute the feedback loop and return structured JSON in text form."""
    tool_use_id = tool_use["toolUseId"]
    params: Dict[str, Any] = tool_use.get("input", {})
    file_path = params["file_path"]
    code = params["code"]
    iterations = int(params.get("iterations", 1) or 1)

    if iterations <= 0:
        iterations = 1

    if iterations == 1:
        result = _run_async(
            _GRAPH.run_iteration(
                file_path=file_path,
                code=code,
                iteration_index=0,
                metadata={"invocation": "tool"},
            )
        )
        results = [result]
    else:
        results = _run_async(
            _GRAPH.run_iterations(
                file_path=file_path,
                code=code,
                count=iterations,
                start_index=0,
                metadata={"invocation": "tool"},
            )
        )

    summary = {
        "file_path": file_path,
        "iterations": len(results),
        "last_reward": results[-1]["discriminator_score"]["reward"] if results else None,
        "graph": results[-1]["graph"] if results else _GRAPH.describe(),
        "records": results,
    }

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(summary, indent=2)}]
    }