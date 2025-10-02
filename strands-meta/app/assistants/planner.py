from __future__ import annotations

import json

from app.prompts.prompts import PLANNER_SYSTEM

from .base import BaseAssistant, StepResult, Task


class Planner(BaseAssistant):
    ROLE = "coordinator"
    NAME = "planner"
    PURPOSE = "Make tiny-step plans"
    SYSTEM = PLANNER_SYSTEM

    async def run(self, task: Task) -> StepResult:
        base_result = await super().run(task)
        if not base_result.ok:
            return base_result

        raw_text = base_result.artifacts.get("text", "{}")
        try:
            plan = json.loads(raw_text or "{}")
        except json.JSONDecodeError:
            plan = {"steps": []}

        base_result.artifacts["plan"] = plan
        return base_result

