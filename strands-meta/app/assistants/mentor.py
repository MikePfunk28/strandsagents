from .base import BaseAssistant, Task, StepResult
from prompts.prompts import MENTOR_SYSTEM

class Mentor(BaseAssistant):
    ROLE="coordinator"; NAME="mentor"; PURPOSE="Teach+supervise with tiny steps"
    SYSTEM = MENTOR_SYSTEM
