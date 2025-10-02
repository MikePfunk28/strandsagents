
from typing import Any
from strands.types.tools import ToolUse, ToolResult

TOOL_SPEC = {
    "name": "demo_greeter",
    "description": "A simple greeting tool",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"}
            },
            "required": ["name"]
        }
    }
}

def demo_greeter(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """A simple greeting tool"""
    tool_use_id = tool_use["toolUseId"]
    name = tool_use["input"]["name"]

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": f"Hello, {name}!"}]
    }
