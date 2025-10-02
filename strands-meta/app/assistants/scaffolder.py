from __future__ import annotations

from app.prompts.prompts import SCAFFOLDER_SYSTEM
from app.tools import tools

from .base import BaseAssistant, StepResult, Task


class Scaffolder(BaseAssistant):
    ROLE = "builder"
    NAME = "scaffolder"
    PURPOSE = "Create and modify files"
    SYSTEM = SCAFFOLDER_SYSTEM
    CAPABILITIES = ["fs.read", "fs.write", "fs.diff"]

    async def run(self, task: Task) -> StepResult:
        context = task.context or {}
        file_path = context.get("file", "app/hello.py")
        content = context.get("content", 'print("hello world")\n')
        diff = tools.fs_diff(file_path, content)
        tools.fs_write(file_path, content)
        return StepResult(ok=True, artifacts={"diff": diff})

