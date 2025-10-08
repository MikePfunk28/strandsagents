# tools.py
"""
Strands community tools – imported as *module tools*.
Each module provides a TOOL_SPEC and a callable with the same name.
You can pass these modules directly to `Agent(tools=[...])`.

References:
- Python tools: function decorators vs. module tools, and how to load them
  https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/python-tools/
- Community tools list & extras (e.g., a2a_client, diagram, use_aws, use_agent, use_llm)
  https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/community-tools-package/
"""

# Install once in your app image/venv:
#   pip install strands-agents-tools
# Optional extras (only if you use them):
#   pip install 'strands-agents-tools[a2a_client,diagram]'

# Import the community tools as Python modules
from strands_tools import (
    use_aws,         # Interact with AWS services (STS/IAM/S3/etc.)
    # Start a nested LLM loop with custom prompts (meta-tooling)
    use_llm,
    # Start a nested Agent loop (agent creates/uses another agent)
    use_agent,
    # search_video,    # Search videos (YouTube etc.) with filtering
    mcp_client,      # Generic MCP client tool (call MCP servers by address)
    # other handy ones if you enable extras:
    diagram,       # Render architecture diagrams (Mermaid) if extras installed
    a2a_client,    # Agent-to-Agent calls if extras installed
)

# ---- Tool cheat sheet (summaries for the model to imitate) -------------------
# The exact JSON schemas live in each module’s TOOL_SPEC, but brief hints help:
#
# use_aws.TOOL_SPEC["inputSchema"]["json"] —
#   {
#     "service": "s3|sts|iam|... (AWS service id)",
#     "action":  "operation to perform (e.g., 'ListBuckets')",
#     "params":  { <operation parameters> },
#     "region":  "optional AWS region override",
#     "profile": "optional credential profile (if your app supports profiles)"
#   }
#
# use_llm.TOOL_SPEC —
#   {
#     "system_prompt": "str (system)",
#     "user_message":  "str (first user message to the nested loop)",
#     "model":         "optional provider/model override",
#     "max_turns":     "optional int, guardrail for nested loop"
#   }
#
# use_agent.TOOL_SPEC —
#   {
#     "system_prompt": "str (child agent system)",
#     "tools":         "optional list of tool names the child may load",
#     "model":         "optional provider/model override",
#     "input":         "initial user task for the child agent"
#   }
#
# search_video.TOOL_SPEC —
#   {
#     "query":   "str",
#     "site":    "optional str (e.g., 'youtube')",
#     "filters": "optional dict (duration, uploaded_after, etc.)",
#     "max":     "optional int"
#   }
#
# mcp_client.TOOL_SPEC —
#   {
#     "server":  "MCP server id/URL (ws/stdio depending on runtime)",
#     "tool":    "remote tool name exposed by that server",
#     "args":    "dict of arguments per the remote tool schema",
#     "timeout": "optional seconds"
#   }
# ------------------------------------------------------------------------------

# Curated list for your Solutions Architect agent
COMMUNITY_TOOLS_FOR_ARCHITECT = [
    use_aws,       # AWS actions (list/query/config helpers)
    use_llm,       # Spawn a nested short LLM loop for “deep think” or formatting
    use_agent,     # Spawn a nested agent with its own system prompt/tools
    # search_video,  # Optional: pull AWS re:Invent talks, service demos
    mcp_client,    # Optional: call MCP tools by address at runtime
    diagram,     # enable if you install '[diagram]' extra
    a2a_client,  # enable if you install '[a2a_client]' extra
]
