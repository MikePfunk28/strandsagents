from .base import BaseAssistant, Task, StepResult

class Tester(BaseAssistant):
    ROLE="reviewer"; NAME="tester"; PURPOSE="Run tests and summarize"
