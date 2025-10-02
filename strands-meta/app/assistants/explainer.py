from .base import BaseAssistant, Task, StepResult
from prompts.prompts import EXPLAINER_SYSTEM
from tools import tools
import json, os

class Explainer(BaseAssistant):
    ROLE="worker"; NAME="explainer"; PURPOSE="Explain diffs simply"
    SYSTEM=EXPLAINER_SYSTEM

    async def run(self, task:Task)->StepResult:
        diff_json = task.context.get("diff_json")
        if not diff_json:
            path = task.context.get("file","app/hello.py")
            before = tools.fs_read(path) if os.path.exists(path) else ""
            after = before
            diff_json = json.dumps({"path":path,"before":before,"after":after})
        # In a real version, we'd LLM-summarize diff_json here.
        return StepResult(ok=True, artifacts={"explanation":"Tiny change applied.", "diff": diff_json})
