import sys  # Access Python's module registry for stubbing
import types  # Create lightweight stand-in modules during the test
from pathlib import Path  # Locate project files relative to the test file
import importlib.util  # Dynamically load project modules without full package imports

if "strands" not in sys.modules:  # Provide a minimal strands package for imports that expect it
    models_pkg = types.ModuleType("strands.models")  # Create the container for strands.models
    ollama_pkg = types.ModuleType("strands.models.ollama")  # Create the submodule used by swarm code
    class _StubOllamaModel:  # Define a placeholder OllamaModel
        def __init__(self, *args, **kwargs):  # Accept arbitrary constructor arguments
            pass  # No behaviour required for the test
    ollama_pkg.OllamaModel = _StubOllamaModel  # Expose the stub model class
    models_pkg.ollama = ollama_pkg  # Attach the ollama module to the models package
    sys.modules["strands.models"] = models_pkg  # Register the models package
    sys.modules["strands.models.ollama"] = ollama_pkg  # Register the ollama submodule
    strands_pkg = types.ModuleType("strands")  # Create the strands package itself
    strands_pkg.__path__ = []  # Mark it as a package
    class _StubAgent:  # Minimal Agent replacement with required API
        async def run_async(self, prompt):  # Matching signature used by code elsewhere
            return f"stub-response:{prompt}"  # Return deterministic text for debugging
    strands_pkg.Agent = _StubAgent  # Expose the stub Agent
    sys.modules["strands"] = strands_pkg  # Register the strands package in sys.modules

if "strandsagents" not in sys.modules:  # Stub the project package to avoid heavy imports during collection
    sys.modules["strandsagents"] = types.ModuleType("strandsagents")  # Register an empty module placeholder


def _load_module(name: str, relative_path: str):  # Helper to load modules by file path
    module_path = Path(__file__).resolve().parents[1] / relative_path  # Resolve the absolute path to the module
    spec = importlib.util.spec_from_file_location(name, module_path)  # Build an import spec referencing the file
    module = importlib.util.module_from_spec(spec)  # Create the module object from the spec
    sys.modules[name] = module  # Register the module so nested imports work
    spec.loader.exec_module(module)  # Execute the module source code
    return module  # Return the ready-to-use module

mcp_module = _load_module("swarm.communication.inmemory_mcp", "swarm/communication/inmemory_mcp.py")  # Load the in-memory MCP broker implementation
channel_module = _load_module("swarm.communication.feedback_channel", "swarm/communication/feedback_channel.py")  # Load the feedback channel utilities
FeedbackAgentChannel = channel_module.FeedbackAgentChannel  # Alias for readability in the test body
register_feedback_responder = channel_module.register_feedback_responder  # Alias the responder registration helper
InMemoryMCPClient = mcp_module.InMemoryMCPClient  # Alias the in-memory MCP client
InMemoryMCPServer = mcp_module.InMemoryMCPServer  # Alias the in-memory MCP server

import asyncio  # Used to run asynchronous code within the test


def test_agent2agent_feedback_flow():  # Test ensuring an agent can request feedback from another agent
    async def run_scenario():  # Encapsulate async logic for use with asyncio.run
        server = InMemoryMCPServer()  # Create the in-memory messaging hub
        requester = InMemoryMCPClient(agent_id="orchestrator", server=server)  # Instantiate the requesting agent client
        responder = InMemoryMCPClient(agent_id="code_feedback", server=server)  # Instantiate the responding agent client
        channel = FeedbackAgentChannel(requester, target_agent_id="code_feedback")  # Assemble the helper channel for requests

        async def fake_handler(payload):  # Define how the responder reacts to requests
            return {  # Return a deterministic payload mimicking a real feedback result
                "file_path": payload.get("file_path"),  # Echo back the file path
                "reward": 0.9,  # Provide a synthetic reward score
                "summary": "Stub summary",  # Provide a canned summary for verification
            }

        await register_feedback_responder(responder, fake_handler)  # Register the responder with the in-memory bus
        await channel.connect()  # Connect the requester and set up response handling

        result = await channel.request_feedback(  # Send the feedback request and await the response
            file_path="demo.py",  # Target file path for the evaluation
            code="print('hi')\n",  # Sample source code passed through the channel
            iterations=1,  # Request a single iteration to keep the test quick
        )

        assert result["file_path"] == "demo.py"  # Check that the file path travelled through the channel
        assert result["reward"] == 0.9  # Verify the reward score survives the round trip
        assert result["summary"] == "Stub summary"  # Confirm the summary text matches expectations

    asyncio.run(run_scenario())  # Execute the asynchronous scenario inside the synchronous test