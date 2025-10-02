
from typing import Any
from strands.types.tools import ToolUse, ToolResult

TOOL_SPEC = {
    "name": "demo_greeter",
    "description": "A simple greeting tool",
    "inputSchema": {'type': 'object', 'properties': {'name': {'type': 'string', 'description': 'Name to greet'}}, 'required': ['name']}
}

def demo_greeter(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """A simple greeting tool"""
    tool_use_id = tool_use["toolUseId"]

    try:
        # Execute the provided code (simplified for demo)
        result = f"Executed: result = f'Hello, {name}!'..."

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Tool executed successfully: {result}"}]
        }
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Tool execution failed: {str(e)}"}]
        }
