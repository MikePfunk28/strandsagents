"""Tool helpers for the code feedback assistant."""

from typing import List, Any

from swarm.tools.code_feedback_tool import code_feedback_tool


def get_code_feedback_tools() -> List[Any]:
    """Return tool list for the code feedback assistant."""
    return [code_feedback_tool]