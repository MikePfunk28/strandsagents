from dataclasses import dataclass
from typing import Dict, Any
from tools.adapters.ollama_client import chat
from app.orchestrator.router import get_model_for

@dataclass
class Task:
    goal:str
    context:Dict[str,Any]|None=None
    risk:str="low"
    role:str="worker"

@dataclass
class StepResult:
    ok:bool
    artifacts:Dict[str,Any]
    next_hint:str=""

class BaseAssistant:
    SYSTEM:str=""; NAME:str="base"; PURPOSE:str=""; CAPABILITIES:list[str]=[]
    ROLE:str="worker"

    async def run(self, task:Task)->StepResult:
        model_spec = get_model_for(role=self.ROLE, assistant_name=self.NAME)
        msgs=[{"role":"system","content":self.SYSTEM},
              {"role":"user","content":{"goal":task.goal,"context":task.context or {}}}]
        out = await chat(msgs, stream=False, model_id=model_spec["id"], **model_spec["default_params"])
        return StepResult(ok=True, artifacts={"message": out})
