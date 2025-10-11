# builder/providers/ollama.py
import requests
from typing import Dict, Any


def chat(model: str, messages: list[dict], host: str = "http://localhost:11434"):
    r = requests.post(f"{host}/api/chat", json={"model": model,
                      "messages": messages, "stream": False}, timeout=600)
    r.raise_for_status()
    return r.json()
