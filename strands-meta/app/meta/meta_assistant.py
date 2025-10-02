from .registry import register, get
from app.assistants.planner import Planner
from app.assistants.scaffolder import Scaffolder
from app.assistants.explainer import Explainer
from app.assistants.base import Task, StepResult

class MetaAssistant:
    def __init__(self):
        register("planner", Planner().run)
        register("scaffolder", Scaffolder().run)
        register("explainer", Explainer().run)

    async def create_assistant(self, name:str, system_prompt:str, capabilities:list[str]):
        # NOTE: Keep simple for demo; policy checks omitted here.
        code = f'''from .base import BaseAssistant
class {name.capitalize()}(BaseAssistant):
    SYSTEM={system_prompt!r}
    NAME="{name}"; PURPOSE="{name} purpose"
    CAPABILITIES={capabilities!r}
'''
        from app.tools import tools
        tools.fs_write(f"app/assistants/{name}.py", code)
        return StepResult(ok=True, artifacts={"created": name})

    async def run_goal(self, goal:str):
        # 1) plan (gets JSON plan)
        plan_res = await get("planner")(Task(goal=goal, role="coordinator"))
        steps = plan_res.artifacts.get("plan",{}).get("steps",[])
        if not steps:
            # fallback: a single scaffolder step
            return await get("scaffolder")(Task(goal=goal, context={"file":"app/main_generated.py","content":"print('hello')\n"}, role="builder"))
        # 2) execute first step only (tiny steps policy)
        first = steps[0]
        assistant = first.get("assistant","scaffolder")
        ctx = first.get("inputs",{"file":"app/main_generated.py","content":"print('hello')\n"})
        return await get(assistant)(Task(goal=first.get("name","step-1"), context=ctx, role="builder"))
