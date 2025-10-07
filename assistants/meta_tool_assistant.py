from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands_tools import use_llm


logger = logging.getLogger(__name__)


TOOL_SPEC: Dict[str, Any] = {
    "name": "meta_tool_orchestrator",
    "description": "Selects the next agent or decides if a new tool must be created based on workflow context.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "user_query": {"type": "string", "description": "The original user objective."},
                "context": {"type": "string", "description": "Additional context supplied by the workflow."},
                "history": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recent workflow messages or observations."
                },
                "thinking_results": {
                    "type": "object",
                    "description": "Structured thinking outputs gathered so far."
                },
                "current_results": {
                    "type": "object",
                    "description": "Intermediate agent results available to the orchestrator."
                },
                "available_tools": {
                    "type": "object",
                    "description": "Mapping of tool names to their metadata, including descriptions and schemas."
                }
            },
            "required": ["user_query", "available_tools"]
        }
    }
}


_ALLOWED_INTENTS = {"use_tool", "create_tool"}


ToolUseDict = Dict[str, Any]
ToolResultDict = Dict[str, Any]


class MetaToolAssistant:
    """Reasons about available agent tools and recommends the next action."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a meta-tool orchestrator for Strands Agents.\n"
        "Choose the next agent/tool to run based on the user's goal, thinking outputs,"
        " current results, and the provided tool specs.\n\n"
        "You must respond with a strict JSON object containing:\n"
        "  next_agent (string) - name of the selected tool or proposed new tool\n"
        "  intent (string) - `use_tool` to call an existing tool, `create_tool` to request new tool creation\n"
        "  reasoning (string) - brief justification (<=60 words)\n"
        "  confidence (string) - one of {low, medium, high}\n"
        "  follow_up (string, optional) - concise note about future steps\n"
        "  parameters (object, optional) - structured arguments for the chosen tool\n"
        "  tool_spec (object, required when intent is `create_tool`) - compliant tool specification\n"
        "If no tool fits, choose `planner_agent` with intent `use_tool`."
        " Do not include markdown or prose outside of the JSON object."
    )

    def __init__(
        self,
        model: Any,
        available_tools: Dict[str, Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> None:
        self.available_tools = available_tools
        self.agent = Agent(
            model=model,
            tools=[use_llm],
            conversation_manager=SlidingWindowConversationManager(
                window_size=15),
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )

    def select_next_action(
        self,
        user_query: str,
        context: str = "",
        history: Optional[List[str]] = None,
        thinking_results: Optional[Dict[str, Any]] = None,
        current_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a structured recommendation for the next agent to execute."""
        prompt = self._build_prompt(
            user_query=user_query,
            context=context,
            history=history or [],
            thinking_results=thinking_results or {},
            current_results=current_results or {},
        )
        try:
            raw = self.agent.tool.use_llm(prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - use safe default on failure
            logger.error("MetaToolAssistant failed to call LLM: %s", exc)
            return self._default_decision(reason="LLM call failed")

        decision = self._parse_decision(str(raw))
        if decision is None:
            logger.warning(
                "MetaToolAssistant returned unparseable response: %s", raw)
            return self._default_decision(reason="Unparseable response")
        return decision

    def _build_prompt(
        self,
        user_query: str,
        context: str,
        history: List[str],
        thinking_results: Dict[str, Any],
        current_results: Dict[str, Any],
    ) -> str:
        tool_lines: List[str] = []
        for name, meta in self.available_tools.items():
            description = meta.get("description", "No description provided.")
            schema = meta.get("input_schema")
            schema_text = ""
            if schema:
                try:
                    schema_text = f" | schema: {json.dumps(schema, ensure_ascii=False)}"
                except (TypeError, ValueError):
                    schema_text = " | schema: <unserializable>"
            tool_lines.append(f"- {name}: {description}{schema_text}")
        tools_section = "\n".join(tool_lines)
        history_section = "\n".join(
            f"- {item}" for item in history[-5:]) if history else "None"
        thinking_section = json.dumps(
            thinking_results, indent=2, ensure_ascii=False) if thinking_results else "{}"
        current_section = json.dumps(
            current_results, indent=2, ensure_ascii=False) if current_results else "{}"

        return (
            f"Available tools:\n{tools_section}\n\n"
            f"User query: {user_query}\n"
            f"Supplementary context: {context or 'None'}\n"
            f"Recent conversation/history:\n{history_section}\n\n"
            "Recent thinking results (JSON):\n" + thinking_section + "\n\n"
            "Current intermediate results (JSON):\n" + current_section + "\n\n"
            "Respond with the JSON object only."
        )

    def _parse_decision(self, response: str) -> Optional[Dict[str, Any]]:
        candidate = self._extract_json_object(response)
        if candidate is None:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None

        intent_raw = parsed.get("intent", "use_tool")
        intent = intent_raw.lower().strip() if isinstance(intent_raw, str) else "use_tool"
        if intent not in _ALLOWED_INTENTS:
            intent = "use_tool"

        next_agent_raw = parsed.get("next_agent", "")
        next_agent = next_agent_raw.strip() if isinstance(next_agent_raw, str) else ""
        if intent == "use_tool" and next_agent not in self.available_tools:
            logger.warning(
                "Unknown agent '%s' requested by meta tool", next_agent)
            next_agent = ""

        reasoning_raw = parsed.get("reasoning")
        reasoning = reasoning_raw.strip() if isinstance(reasoning_raw,
                                                        str) and reasoning_raw.strip() else "Decision based on meta-tool analysis."

        confidence_raw = parsed.get("confidence", "medium")
        confidence = confidence_raw.strip() if isinstance(
            confidence_raw, str) and confidence_raw.strip() else "medium"

        decision: Dict[str, Any] = {
            "next_agent": next_agent or "planner_agent",
            "intent": intent,
            "reasoning": reasoning,
            "confidence": confidence,
        }

        follow_up = parsed.get("follow_up")
        if isinstance(follow_up, str) and follow_up.strip():
            decision["follow_up"] = follow_up.strip()

        parameters = parsed.get("parameters")
        if isinstance(parameters, dict):
            decision["parameters"] = parameters

        if intent == "create_tool":
            tool_spec = parsed.get("tool_spec")
            if isinstance(tool_spec, dict):
                decision["tool_spec"] = tool_spec
            elif tool_spec is not None:
                decision["tool_spec"] = str(tool_spec)
            if not next_agent:
                proposed_name = parsed.get("proposed_name")
                if isinstance(proposed_name, str) and proposed_name.strip():
                    decision["next_agent"] = proposed_name.strip()

        return decision

    @staticmethod
    def _extract_json_object(response: str) -> Optional[str]:
        start = response.find("{")
        if start == -1:
            return None
        return response[start:]

    @staticmethod
    def _default_decision(reason: str) -> Dict[str, Any]:
        return {
            "next_agent": "planner_agent",
            "intent": "use_tool",
            "reasoning": reason,
            "confidence": "medium",
        }


def meta_tool_orchestrator(tool_use: ToolUseDict, **kwargs: Any) -> ToolResultDict:
    """Selects the next agent or decides if a new tool must be created based on workflow context."""
    tool_use_id = tool_use["toolUseId"]
    payload = tool_use.get("input", {}) or {}

    available_tools = payload.get(
        "available_tools") or kwargs.get("available_tools")
    if not isinstance(available_tools, dict) or not available_tools:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "meta_tool_orchestrator missing available_tools"}],
        }

    model = kwargs.get("model")
    if model is None:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "meta_tool_orchestrator requires model"}],
        }

    assistant = MetaToolAssistant(model=model, available_tools=available_tools)
    decision = assistant.select_next_action(
        user_query=str(payload.get("user_query", "")),
        context=str(payload.get("context", "")),
        history=payload.get("history") if isinstance(
            payload.get("history"), list) else None,
        thinking_results=payload.get("thinking_results") if isinstance(
            payload.get("thinking_results"), dict) else None,
        current_results=payload.get("current_results") if isinstance(
            payload.get("current_results"), dict) else None,
    )

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": decision}],
    }
