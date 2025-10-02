import httpx, os

BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

async def chat(messages, stream=False, model_id:str=None, **params):
    body = {"model": model_id, "messages": messages, "stream": stream}
    body.update(params)
    async with httpx.AsyncClient(timeout=params.get("timeout", 60)) as client:
        r = await client.post(f"{BASE}/api/chat", json=body)
        r.raise_for_status()
        return r.json()
