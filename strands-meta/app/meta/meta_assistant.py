from __future__ import annotations

from typing import Awaitable, Callable

from .registry import get, register
from app.assistants.base import StepResult, Task
from app.assistants.explainer import Explainer
from app.assistants.planner import Planner
from app.assistants.scaffolder import Scaffolder

AssistantFn = Callable[[Task], Awaitable[StepResult]]


class MetaAssistant:
    def __init__(self) -> None:
        register("planner", Planner().run)
        register("scaffolder", Scaffolder().run)
        register("explainer", Explainer().run)

    async def create_assistant(self, name: str, system_prompt: str, capabilities: list[str]) -> StepResult:
        # NOTE: Keep simple for demo; policy checks omitted here.
        code = f'''from .base import BaseAssistant
class {name.capitalize()}(BaseAssistant):
    SYSTEM={system_prompt!r}
    NAME="{name}"; PURPOSE="{name} purpose"
    CAPABILITIES={capabilities!r}
'''
        from app.tools import tools  # local import to avoid import cycle

        tools.fs_write(f"app/assistants/{name}.py", code)
        return StepResult(ok=True, artifacts={"created": name})

    async def run_goal(self, goal: str, *, context: dict | None = None) -> StepResult:
        planner = self._ensure_callable("planner")
        planning_task = Task(goal=goal, context=context or {}, role="coordinator")
        plan_res = await planner(planning_task)
        steps = plan_res.artifacts.get("plan", {}).get("steps", [])
        if not steps:
            fallback_context = {
                "file": "app/main_generated.py",
                "content": "print('hello')\n",
            }
            if context:
                fallback_context["meta_context"] = context
            return await self._ensure_callable("scaffolder")(
                Task(goal=goal, context=fallback_context, role="builder")
            )

        first = steps[0]
        assistant_name = first.get("assistant", "scaffolder")
        assistant_callable = self._resolve_assistant(assistant_name)
        raw_inputs = first.get(
            "inputs",
            {"file": "app/main_generated.py", "content": "print('hello')\n"},
        )
        task_context = dict(raw_inputs)
        if context:
            task_context["meta_context"] = context
        task = Task(goal=first.get("name", "step-1"), context=task_context, role="builder")
        return await assistant_callable(task)

    def _resolve_assistant(self, descriptor: str) -> AssistantFn:
        candidates = [candidate.strip() for candidate in descriptor.split("|") if candidate.strip()]
        if not candidates:
            candidates = ["scaffolder"]
        for candidate in candidates:
            maybe = get(candidate)
            if maybe:
                return maybe
        return self._ensure_callable("scaffolder")

    def _ensure_callable(self, name: str) -> AssistantFn:
        fn = get(name)
        if fn is None:
            raise RuntimeError(f"Assistant '{name}' is not registered")
        return fn
