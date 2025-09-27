from typing import Union

from fastapi import FastAPI
import thought_agent

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.start("/thought")
async def run_thought_agent(goal: str):
    result = thought_agent.run_thought_agent(goal)
    return {"result": result}