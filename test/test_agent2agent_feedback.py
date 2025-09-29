import sys
import types
from pathlib import Path
import importlib.util

if "strands" not in sys.modules:
    models_pkg = types.ModuleType("strands.models")
    ollama_pkg = types.ModuleType("strands.models.ollama")
    class _StubOllamaModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    ollama_pkg.OllamaModel = _StubOllamaModel
    models_pkg.ollama = ollama_pkg
    sys.modules["strands.models"] = models_pkg
    sys.modules["strands.models.ollama"] = ollama_pkg
    strands_pkg = types.ModuleType("strands")
    strands_pkg.__path__ = []
    class _StubAgent:
        async def run_async(self, prompt):
            return f"stub-response:{prompt}"
    strands_pkg.Agent = _StubAgent
    sys.modules["strands"] = strands_pkg


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

mcp_module = _load_module("swarm.communication.inmemory_mcp", "swarm/communication/inmemory_mcp.py")
channel_module = _load_module("swarm.communication.feedback_channel", "swarm/communication/feedback_channel.py")
FeedbackAgentChannel = channel_module.FeedbackAgentChannel
register_feedback_responder = channel_module.register_feedback_responder
InMemoryMCPClient = mcp_module.InMemoryMCPClient
InMemoryMCPServer = mcp_module.InMemoryMCPServer

"""Agent-to-agent feedback channel tests with detailed commentary."""

import asyncio


def test_agent2agent_feedback_flow():
    async def run_scenario():
        server = InMemoryMCPServer()
        requester = InMemoryMCPClient(agent_id="orchestrator", server=server)
        responder = InMemoryMCPClient(agent_id="code_feedback", server=server)
        channel = FeedbackAgentChannel(requester, target_agent_id="code_feedback")

        async def fake_handler(payload):
            return {
                "file_path": payload.get("file_path"),
                "reward": 0.9,
                "summary": "Stub summary",
            }

        await register_feedback_responder(responder, fake_handler)
        await channel.connect()

        result = await channel.request_feedback(
            file_path="demo.py",
            code="print('hi')\n",
            iterations=1,
        )

        assert result["file_path"] == "demo.py"
        assert result["reward"] == 0.9
        assert result["summary"] == "Stub summary"

    asyncio.run(run_scenario())
if "strandsagents" not in sys.modules:
    sys.modules["strandsagents"] = types.ModuleType("strandsagents")
