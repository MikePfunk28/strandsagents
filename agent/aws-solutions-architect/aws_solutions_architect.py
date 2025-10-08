# aws_solutions_architect.py

from typing import List, Dict, Any
from prompts import solutions_architect_agent as AWS_SOLUTIONS_ARCHITECT_SYSTEM_PROMPT
from tools import COMMUNITY_TOOLS_FOR_ARCHITECT

# If you're using Strands' Agent class:
try:
    from strands import Agent, Model
except ImportError:
    Agent = object

    class Model:
        def __init__(self, provider: str, name: str):
            self.provider = provider
            self.name = name
            ...


class AwsSolutionsArchitectAgent(Agent):
    """
    Strands agent definition for AWS Solutions Architect.
    Assumes your runtime discovers the tools by `uses_tools` names.
    """

    name: str = "aws_solutions_architect"
    # Pick your deployed model here (update as needed)
    model = Model(provider="bedrock", name="anthropic.claude-3-5-sonnet")

    # System prompt
    system_prompt: str = AWS_SOLUTIONS_ARCHITECT_SYSTEM_PROMPT

    # Allow only the curated tools for this agent (principle of least privilege)
    uses_tools: List[str] = [t.__name__.split('.')[-1]
                             for t in COMMUNITY_TOOLS_FOR_ARCHITECT]

    # Optional: guard-rails (simple example)
    max_context_tokens: int = 24_000
    max_output_tokens: int = 2_000

    # Optional: per-message policy hook (pseudo-API shown for clarity)
    def before_tool_call(self, tool_name: str, args: dict):
        if tool_name not in self.uses_tools:
            raise PermissionError(
                f"Tool '{tool_name}' not permitted for {self.name}")
        # Example: enforce minimal args for pricing calls
        if tool_name == "aws_pricing.estimate":
            if "services" not in args or not args["services"]:
                raise ValueError(
                    "aws_pricing.estimate requires a non-empty 'services' list.")

    # Optional: normalize citations in final answers
    def postprocess_answer(self, text: str) -> str:
        # Keep answers concise; ensure sections from the system prompt order
        return text.strip()
