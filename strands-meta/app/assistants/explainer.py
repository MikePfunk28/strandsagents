from __future__ import annotations

import json

from app.prompts.prompts import EXPLAINER_SYSTEM
from app.tools import tools
from app.tools.tools import ToolError

from .base import BaseAssistant, StepResult, Task


class Explainer(BaseAssistant):
    ROLE = "worker"
    NAME = "explainer"
    PURPOSE = "Explain diffs in plain language"
    SYSTEM = EXPLAINER_SYSTEM

    async def run(self, task: Task) -> StepResult:
        diff_json = (task.context or {}).get("diff_json")
        if not diff_json:
            path = (task.context or {}).get("file", "app/hello.py")
            try:
                before = tools.fs_read(path)
            except ToolError:
                before = ""
            diff_json = json.dumps({"path": path, "before": before, "after": before})
        return StepResult(ok=True, artifacts={"explanation": "Tiny change applied.", "diff": diff_json})

