import sys
import types
from pathlib import Path
import importlib.util

if "strands" not in sys.modules:
    models_pkg = types.ModuleType("strands.models")
    ollama_pkg = types.ModuleType("strands.models.ollama")
    class _StubOllamaModel:
        def __init__(self, *args, **kwargs):
            pass
    ollama_pkg.OllamaModel = _StubOllamaModel
    models_pkg.ollama = ollama_pkg
    sys.modules["strands.models"] = models_pkg
    sys.modules["strands.models.ollama"] = ollama_pkg
    strands_pkg = types.ModuleType("strands")
    strands_pkg.Agent = object
    sys.modules["strands"] = strands_pkg

root = Path(__file__).resolve().parent

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

mcp_module = load("swarm.communication.inmemory_mcp", "swarm/communication/inmemory_mcp.py")
channel_module = load("swarm.communication.feedback_channel", "swarm/communication/feedback_channel.py")

async def main():
    server = mcp_module.InMemoryMCPServer()
    requester = mcp_module.InMemoryMCPClient("req", server)
    responder = mcp_module.InMemoryMCPClient("res", server)
    channel = channel_module.FeedbackAgentChannel(requester, "res")

    async def handler(payload):
        print("handler called with", payload)
        return {"ack": True}

    await channel_module.register_feedback_responder(responder, handler)
    await channel.connect()

    async def intercept(message):
        print("response message", message)
        await channel._handle_response(message)

    channel.client.register_message_handler("feedback_response", intercept)

    response = await channel.request_feedback(file_path="demo.py", code="print('hi')\n", iterations=1, timeout=5)
    print("response", response)

import asyncio
asyncio.run(main())
