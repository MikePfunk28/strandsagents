from .base import BaseAssistant, Task, StepResult
from prompts.prompts import PLANNER_SYSTEM
import json

class Planner(BaseAssistant):
    ROLE="coordinator"; NAME="planner"; PURPOSE="Make tiny-step Plans"
    SYSTEM=PLANNER_SYSTEM

    async def run(self, task:Task)->StepResult:
        res = await super().run(task)
        # naive extraction expecting JSON in first content block
        msg = res.artifacts["message"]
        content = msg.get("message",{}).get("content",[{"text":"{"steps":[]}"}])[0].get("text","{"steps":[]}")
        plan = json.loads(content)
        return StepResult(ok=True, artifacts={"plan":plan})
