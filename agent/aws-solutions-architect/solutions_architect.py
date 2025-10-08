# aws_solutions_architect.py
from __future__ import annotations
from typing import List

from strands import Agent, Model  # adjust to your SDK
from prompts import AWS_SOLUTIONS_ARCHITECT_SYSTEM_PROMPT

# Importing tools ensures decorators register with the runtime (if applicable).
# Keep even if unused directly here.
import tools as _registered_tools  # noqa


class AwsSolutionsArchitectAgent(Agent):
    name = "aws_solutions_architect"
    # pick your deployed model
    model = Model(provider="bedrock", name="anthropic.claude-3-5-sonnet")
    system_prompt = AWS_SOLUTIONS_ARCHITECT_SYSTEM_PROMPT

    # Principle of least privilege: only the curated tool names.
    uses_tools: List[str] = [
        "aws_docs.search",
        "aws_solutions_library.find",
        "aws_well_architected.check",
        "aws_pricing.estimate",
        "kb_vector.search",
        "diagram.mermaid_export",
        # built-ins if your runtime exposes them:
        "graph",
        "a2a",
    ]

    max_context_tokens: int = 24000
    max_output_tokens: int = 2000

    def before_tool_call(self, tool_name: str, args: dict):
        if tool_name not in self.uses_tools:
            raise PermissionError(
                f"Tool '{tool_name}' not permitted for {self.name}")
        if tool_name == "aws_pricing.estimate":
            services = args.get("services", [])
            if not services:
                raise ValueError(
                    "aws_pricing.estimate requires non-empty 'services'.")
        if tool_name == "aws_docs.search":
            if len(args.get("query", "")) < 2:
                raise ValueError(
                    "aws_docs.search requires a 'query' with length >= 2.")

    def postprocess_answer(self, text: str) -> str:
        return text.strip()
