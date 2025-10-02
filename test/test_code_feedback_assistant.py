import asyncio
import importlib.util
import sys
import types
from pathlib import Path

_swarm_root = Path(__file__).resolve().parents[1] / "swarm"
if "swarm" not in sys.modules:
    swarm_pkg = types.ModuleType("swarm")
    swarm_pkg.__path__ = [str(_swarm_root)]
    sys.modules["swarm"] = swarm_pkg
if "swarm.agents" not in sys.modules:
    agents_pkg = types.ModuleType("swarm.agents")
    agents_pkg.__path__ = [str(_swarm_root / "agents")]
    sys.modules["swarm.agents"] = agents_pkg
if "swarm.agents.code_feedback" not in sys.modules:
    cf_pkg = types.ModuleType("swarm.agents.code_feedback")
    cf_pkg.__path__ = [str(_swarm_root / "agents" / "code_feedback")]
    sys.modules["swarm.agents.code_feedback"] = cf_pkg
if "swarm.orchestration" not in sys.modules:
    orch_pkg = types.ModuleType("swarm.orchestration")
    orch_pkg.__path__ = [str(_swarm_root / "orchestration")]
    sys.modules["swarm.orchestration"] = orch_pkg

if "swarm.communication" not in sys.modules:
    comm_pkg = types.ModuleType("swarm.communication")
    comm_pkg.__path__ = [str(_swarm_root / "communication")]
    sys.modules["swarm.communication"] = comm_pkg

mcp_client_module = types.ModuleType("swarm.communication.mcp_client")

class _StubMCPClient:
    def __init__(self, *args, **kwargs):
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    def register_message_handler(self, *args, **kwargs):
        return None

    async def send_message(self, *args, **kwargs):
        return None

    async def get_next_message(self, timeout=None):
        return None

mcp_client_module.SwarmMCPClient = _StubMCPClient
sys.modules["swarm.communication.mcp_client"] = mcp_client_module

mcp_server_module = types.ModuleType("swarm.communication.mcp_server")

class _StubMessage(dict):
    pass

class _StubAgentInfo(dict):
    pass

mcp_server_module.SwarmMessage = _StubMessage
mcp_server_module.AgentInfo = _StubAgentInfo
sys.modules["swarm.communication.mcp_server"] = mcp_server_module

if "strands" not in sys.modules:
    strands_pkg = types.ModuleType("strands")
    strands_pkg.__path__ = []
    sys.modules["strands"] = strands_pkg
if "strands.types" not in sys.modules:
    strands_types_pkg = types.ModuleType("strands.types")
    sys.modules["strands.types"] = strands_types_pkg
if "strands.types.tool_types" not in sys.modules:
    tool_types_module = types.ModuleType("strands.types.tool_types")

    class ToolUse(dict):
        pass

    class ToolResult(dict):
        pass

    tool_types_module.ToolUse = ToolUse
    tool_types_module.ToolResult = ToolResult
    sys.modules["strands.types.tool_types"] = tool_types_module

_service_path = _swarm_root / "agents" / "code_feedback" / "service.py"
service_spec = importlib.util.spec_from_file_location(
    "swarm.agents.code_feedback.service", _service_path
)
service_module = importlib.util.module_from_spec(service_spec)
sys.modules[service_spec.name] = service_module
service_spec.loader.exec_module(service_module)
CodeFeedbackAssistant = service_module.CodeFeedbackAssistant


class DummyGraph:
    def __init__(self):
        self.calls = []
        self._description = {"nodes": [], "edges": [], "loop_guidance": []}

    def describe(self):
        return self._description

    async def run_iteration(self, *, file_path: str, code: str, iteration_index: int = 0, metadata=None):
        self.calls.append((file_path, code, iteration_index))
        return {
            "iteration_index": iteration_index,
            "timestamp": 0.0,
            "generator_output": {
                "overall_summary": "demo-summary",
                "line_annotations": [],
                "scope_annotations": [],
                "raw_response": "{}",
                "file_path": file_path,
                "confidence": 0.9,
                "metadata": {},
            },
            "discriminator_score": {
                "reward": 0.75,
                "coverage": 0.8,
                "accuracy": 0.7,
                "coherence": 0.65,
                "formatting": 0.9,
                "issues": ["demo"],
                "raw_response": "{}",
                "metadata": {},
            },
            "agitator_feedback": {
                "guidance": "demo guidance",
                "improvement_plan": ["plan"],
                "targeted_prompts": ["prompt"],
                "raw_response": "{}",
                "metadata": {},
            },
            "graph": self._description,
            "metadata": metadata or {},
        }

    async def run_iterations(self, *, file_path: str, code: str, count: int, start_index: int = 0, metadata=None):
        results = []
        for offset in range(count):
            result = await self.run_iteration(
                file_path=file_path,
                code=code,
                iteration_index=start_index + offset,
                metadata=metadata,
            )
            results.append(result)
        return results


def test_code_feedback_assistant_processes_code_task():
    graph = DummyGraph()
    assistant = CodeFeedbackAssistant(
        assistant_id="code_feedback_test",
        feedback_graph=graph,
    )

    task = {
        "task_id": "task-1",
        "description": "Provide detailed commentary",
        "context": {
            "file_path": "demo.py",
            "code": "print('hello')\n",
            "iterations": 1,
        },
    }

    result = asyncio.run(assistant.process_task(task))

    assert result["status"] == "completed"
    assert result["reward"] == 0.75
    assert result["iterations"][0]["generator_output"]["overall_summary"] == "demo-summary"
    assert graph.calls == [("demo.py", "print('hello')\n", 0)]

    cached = asyncio.run(assistant.process_task(task))
    assert graph.calls == [("demo.py", "print('hello')\n", 0)]
    assert cached["task_id"] == "task-1"