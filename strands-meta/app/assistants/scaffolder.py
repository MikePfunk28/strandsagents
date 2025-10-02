from .base import BaseAssistant, Task, StepResult
from prompts.prompts import SCAFFOLDER_SYSTEM
from tools import tools

class Scaffolder(BaseAssistant):
    ROLE="builder"; NAME="scaffolder"; PURPOSE="Create/modify files"
    SYSTEM=SCAFFOLDER_SYSTEM
    CAPABILITIES=["fs.read","fs.write","fs.diff"]

    async def run(self, task:Task)->StepResult:
        file = task.context.get("file","app/hello.py")
        content = task.context.get("content",'print("hello world")\n')
        diff = tools.fs_diff(file, content)
        tools.fs_write(file, content)
        return StepResult(ok=True, artifacts={"diff": diff})
