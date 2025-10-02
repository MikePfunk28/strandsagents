from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from strands import Agent

from app.orchestrator.router import get_model_for
from app.sa import get_sa_model
from app.tools.tools import get_toolset


@dataclass
class Task:
    goal: str
    context: Dict[str, Any] | None = None
    risk: str = "low"
    role: str = "worker"


@dataclass
class StepResult:
    ok: bool
    artifacts: Dict[str, Any]
    next_hint: str = ""


class BaseAssistant:
    SYSTEM: str = ""
    NAME: str = "base"
    PURPOSE: str = ""
    CAPABILITIES: list[str] = []
    ROLE: str = "worker"

    async def run(self, task: Task) -> StepResult:
        try:
            agent = self._build_agent()
            prompt = self._format_prompt(task)
            result = await agent.invoke_async(prompt)
        except Exception as exc:  # noqa: BLE001 - surface the error to the orchestrator
            return StepResult(ok=False, artifacts={"error": str(exc)})

        usage = result.metrics.accumulated_usage
        metrics = {
            "cycles": result.metrics.cycle_count,
            "latency_ms": result.metrics.accumulated_metrics.get("latencyMs"),
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        }

        return StepResult(
            ok=True,
            artifacts={
                "text": str(result).strip(),
                "message": result.message,
                "stop_reason": result.stop_reason,
                "metrics": metrics,
            },
        )

    def _build_agent(self) -> Agent:
        model_spec = get_model_for(role=self.ROLE, assistant_name=self.NAME)
        model = get_sa_model(model_spec)
        tools = get_toolset(self.CAPABILITIES)
        return Agent(
            name=self.NAME,
            model=model,
            system_prompt=self.SYSTEM,
            tools=tools,
        )

    def _format_prompt(self, task: Task) -> str:
        payload = {
            "goal": task.goal,
            "context": task.context or {},
            "risk": task.risk,
            "caller_role": task.role,
            "assistant": self.NAME,
        }
        return json.dumps(payload)

