from fastapi import FastAPI, HTTPException
from app.orchestrator.router import list_available, set_runtime_model

app = FastAPI()

@app.get("/models")
def models():
    return list_available()

@app.post("/models/switch")
def switch(target: str, registry_key: str):
    try:
        set_runtime_model(target, registry_key)
        return {"ok": True, "target": target, "model_key": registry_key}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
