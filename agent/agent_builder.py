#!/usr/bin/env python3
"""
Interactive agent builder for the StrandsAgents package.

This script collects the information needed to scaffold a Strands agent
(using agent.scaffolder) and optionally generates simple tool stubs so the
agent has concrete capabilities out of the box.

Usage:
    python -m agent.agent_builder
        or
    python agent/agent_builder.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import resolution so the script works when executed directly
# ---------------------------------------------------------------------------

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    sys.path.insert(0, str(PARENT_DIR))
    from agent.model_selector import (
        get_best_model_for_task,
        list_available_models as selector_list_available_models,
        model_selector,
    )
    from agent.scaffolder import generate_agent
    from agent.sandbox_tool import sandbox_test_code  # noqa: F401  (import side effect)
else:  # pragma: no cover - handled above when run as module
    from .model_selector import (
        get_best_model_for_task,
        list_available_models as selector_list_available_models,
        model_selector,
    )
    from .scaffolder import generate_agent
    from .sandbox_tool import sandbox_test_code  # noqa: F401

# Optional: use Strands SDK for validation when available
try:  # pragma: no cover - depends on external package
    from strands import Agent as StrandsAgent
    from strands import tool as strands_tool
    STRANDS_AVAILABLE = True
except ImportError:  # pragma: no cover
    StrandsAgent = None  # type: ignore
    strands_tool = None  # type: ignore
    STRANDS_AVAILABLE = False

# Simple utility ----------------------------------------------------------------

def prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def confirm(text: str, default: bool = False) -> bool:
    default_token = "Y/n" if default else "y/N"
    value = input(f"{text} ({default_token}): ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "true"}


@dataclass
class AgentSpecification:
    name: str
    display_name: str
    description: str
    model_id: str
    system_prompt: str
    enable_code_execution: bool
    tools: List[str]
    sandbox_timeout: int
    memory_profile: str
    context_documents: List[str]


class AgentBuilderCLI:
    """Interactive command line agent builder."""

    OUTPUT_DIR = Path("assistants/generated")

    def run(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("\n=== Strands Agent Builder ===")
        print("This wizard will gather the details required to scaffold a new agent.\n")

        spec = self._collect_agent_spec()
        answers = self._spec_to_answers(spec)

        result = generate_agent(answers)
        if not result.ok:
            print(f"\n[ERROR] Agent generation failed: {result.message}")
            return

        print("\n[OK] Agent generated.")
        for label, path in result.artifacts.items():
            print(f"  - {label}: {path}")

        if confirm("Add an auxiliary tool for this agent?", default=False):
            self._scaffold_tool(spec)

        if STRANDS_AVAILABLE and confirm("Validate by importing the generated agent?", default=True):
            self._validate_agent(spec)

        print("\nDone!")

    # ------------------------------------------------------------------
    # Input gathering
    # ------------------------------------------------------------------
    def _collect_agent_spec(self) -> AgentSpecification:
        name = prompt("Agent function name", "my_agent").strip()
        while not name.isidentifier():
            print("Name must be a valid Python identifier (snake_case).")
            name = prompt("Agent function name", "my_agent").strip()

        display_name = prompt("Display name", name.replace("_", " ").title())
        description = prompt("Short description", "Custom Strands agent")

        providers = self._list_providers()
        print("\nAvailable model providers:")
        for provider in providers:
            print(f"  - {provider}")
        provider_choice = prompt("Preferred provider (press Enter to auto)") or None
        if provider_choice:
            model_selector.set_provider(provider_choice)

        available_models = selector_list_available_models()
        print("\nAvailable models:")
        for model in available_models:
            row = f"  - {model['name']} ({model['family']}) capabilities: {', '.join(model['capabilities'])}"
            print(row)

        default_model = get_best_model_for_task("general")
        model_id = prompt("Model ID", default_model)

        system_prompt = self._default_prompt(display_name, description)
        if confirm("Would you like to edit the system prompt?", default=False):
            print("Enter prompt (finish with an empty line):")
            lines: List[str] = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            if lines:
                system_prompt = "\n" + "\n".join(lines)

        enable_code_execution = confirm("Enable sandboxed code execution?", default=True)
        sandbox_timeout = int(prompt("Sandbox timeout (seconds)", "45"))

        default_tools = ["sandbox_test_code"] if enable_code_execution else []
        tools_raw = prompt("Comma-separated tool names", ",".join(default_tools))
        tools = sorted({tool.strip() for tool in tools_raw.split(',') if tool.strip()})

        memory_profile = prompt("Memory profile (none/light/full)", "light")
        docs_raw = prompt("Context documents (comma-separated paths)", "")
        context_documents = [doc.strip() for doc in docs_raw.split(',') if doc.strip()]

        return AgentSpecification(
            name=name,
            display_name=display_name,
            description=description,
            model_id=model_id,
            system_prompt=system_prompt,
            enable_code_execution=enable_code_execution,
            tools=tools,
            sandbox_timeout=sandbox_timeout,
            memory_profile=memory_profile,
            context_documents=context_documents,
        )

    def _default_prompt(self, display_name: str, description: str) -> str:
        return dedent(
            f"""
            You are {display_name}.
            {description}

            Behaviours:
            - Provide step-by-step reasoning before final answers.
            - Use tools when appropriate.
            - Ask clarifying questions when requirements are ambiguous.
            """
        ).strip()

    def _list_providers(self) -> List[str]:
        providers = {info.provider for info in model_selector.MODEL_CAPABILITIES.values()}
        return sorted(providers)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _spec_to_answers(self, spec: AgentSpecification) -> Dict[str, Any]:
        return {
            "agent_name": spec.name,
            "display_name": spec.display_name,
            "description": spec.description,
            "prompt_source": "inline",
            "system_prompt": spec.system_prompt,
            "prompt_path": "",
            "model_id": spec.model_id,
            "enable_code_execution": spec.enable_code_execution,
            "sandbox_timeout": spec.sandbox_timeout,
            "tool_selection": ", ".join(spec.tools),
            "memory_profile": spec.memory_profile,
            "embedding_namespace": f"{spec.name}_namespace",
            "context_documents": ", ".join(spec.context_documents),
            "output_dir": str(self.OUTPUT_DIR),
        }

    # ------------------------------------------------------------------
    # Tool scaffolding & validation
    # ------------------------------------------------------------------
    def _scaffold_tool(self, spec: AgentSpecification) -> None:
        tool_name = prompt("Tool function name", f"{spec.name}_helper")
        while not tool_name.isidentifier():
            print("Tool name must be a valid Python identifier.")
            tool_name = prompt("Tool function name", f"{spec.name}_helper")

        description = prompt("Tool description", "Helper tool built by agent builder")
        tool_dir = self.OUTPUT_DIR / "tools"
        tool_dir.mkdir(parents=True, exist_ok=True)
        tool_path = tool_dir / f"{tool_name}.py"

        tool_code = dedent(
            f"""
            from typing import Any
            from strands.types.tools import ToolUse, ToolResult

            TOOL_SPEC = {{
                "name": "{tool_name}",
                "description": "{description}",
                "inputSchema": {{
                    "json": {{
                        "type": "object",
                        "properties": {{
                            "message": {{
                                "type": "string",
                                "description": "Message for the helper tool"
                            }}
                        }},
                        "required": ["message"]
                    }}
                }}
            }}

            def {tool_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
                message = tool_use.get("input", {{}}).get("message", "")
                response = f"Helper tool received: {{message}}"
                return {{
                    "toolUseId": tool_use["toolUseId"],
                    "status": "success",
                    "content": [{{"text": response}}]
                }}
            """
        ).strip()

        tool_path.write_text(tool_code, encoding="utf-8")
        print(f"  - Tool scaffolded at {tool_path}")

    def _validate_agent(self, spec: AgentSpecification) -> None:
        try:
            module_path = self.OUTPUT_DIR / "metadata" / f"{spec.name}.json"
            metadata = json.loads(module_path.read_text(encoding="utf-8"))
            module_file = metadata["module_file"]
            import_path = Path(module_file).with_suffix("")

            sys.path.insert(0, str(Path(module_file).resolve().parent.parent))
            module = __import__(Path(module_file).stem, fromlist=[spec.name])
            agent_callable = getattr(module, spec.name)

            if STRANDS_AVAILABLE and callable(agent_callable):
                response = agent_callable("Health check")
                print(f"  - Validation call response: {response}")
            else:
                print("  - Agent imported (Strands validation skipped)")

        except Exception as exc:  # pragma: no cover - best effort validation
            print(f"  - Validation failed: {exc}")


def main() -> None:
    cli = AgentBuilderCLI()
    cli.run()


if __name__ == "__main__":
    main()
