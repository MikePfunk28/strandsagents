import asyncio  # Run asynchronous helper coroutines inside the tests
import sys  # Access Python's module registry for stubbing
import types  # Create lightweight stand-in modules during the test
from pathlib import Path  # Locate project files relative to the test file
import importlib.util  # Dynamically load project modules without full package imports

import pytest  # Provide assertion helpers like approx for floating point comparisons

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

# Ensure the swarm_system namespace exists so dynamically loaded modules can resolve relatives
if "swarm_system" not in sys.modules:  # Provide a synthetic package for swarm_system
    swarm_system_pkg = types.ModuleType("swarm_system")  # Create the root package shell
    swarm_system_pkg.__path__ = []  # Mark as namespace package to satisfy imports
    sys.modules["swarm_system"] = swarm_system_pkg  # Register the root package placeholder

if "swarm_system.learning" not in sys.modules:  # Provide the learning subpackage placeholder
    learning_pkg = types.ModuleType("swarm_system.learning")  # Create the subpackage shell
    learning_pkg.__path__ = []  # Mark it as a package container
    sys.modules["swarm_system.learning"] = learning_pkg  # Register for module resolution


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

benchmark_module = _load_module("swarm_system.learning.adaptive_benchmark", "swarm_system/learning/adaptive_benchmark.py")  # Load the adaptive benchmark implementation
manager_module = _load_module("swarm_system.learning.adaptive_challenge_manager", "swarm_system/learning/adaptive_challenge_manager.py")  # Load the adaptive challenge manager
AdaptiveFeedbackBenchmark = benchmark_module.AdaptiveFeedbackBenchmark  # Alias the adaptive benchmark class for tests
AdaptiveChallengeManager = manager_module.AdaptiveChallengeManager  # Alias the challenge manager helper


def test_adaptive_benchmark_tracks_memory(tmp_path):  # Validate memory-aware summary calculations on the adaptive benchmark
    storage = tmp_path / "adaptive.json"  # Choose a temporary file for benchmark persistence
    bench = AdaptiveFeedbackBenchmark(storage_path=str(storage))  # Instantiate the benchmark pointing at the temp storage

    first_run = {  # Build a payload representing an improving workflow run
        "run": {  # Describe core run metadata
            "id": "run-1",  # Unique run identifier
            "file_path": "alpha.py",  # File evaluated during the run
            "guidance_history": ["hint-a", "hint-b"],  # Guidance applied before the run
        },
        "history": {"reward": {"delta": 0.3}},  # Reward delta showing improvement over baseline
        "iterations": [  # Details for each iteration of the workflow
            {
                "discriminator_score": {"reward": 0.9},  # Reward score contributed to averages
                "generator_output": {"overall_summary": "alpha summary"},  # Minimal generator output context
            }
        ],
        "memory_snapshot": {  # Memory counts before and after the run
            "before": {"short_term": 3, "long_term": 2},  # Baseline memory counts
            "after": {"short_term": 5, "long_term": 3},  # Post-run memory counts used for delta
        },
        "timestamp": 100.0,  # Stable timestamp for deterministic ordering
    }
    bench.update_from_run(first_run)  # Store the first run inside the benchmark

    second_run = {  # Build a payload representing a regressing workflow run
        "run": {
            "id": "run-2",  # Unique identifier for the second run
            "file_path": "beta.py",  # Second file analysed by the workflow
            "guidance_history": [],  # No guidance supplied this time
        },
        "history": {"reward": {"delta": 0.2}},  # Slight improvement captured for ordering
        "iterations": [
            {
                "discriminator_score": {"reward": 0.75},  # Lower reward for contrast
                "generator_output": {"overall_summary": "beta summary"},  # Provide accompanying generator output
            }
        ],
        "memory_snapshot": {  # Capture a contraction in memory usage
            "before": {"short_term": 4, "long_term": 3},  # Starting memory counts
            "after": {"short_term": 3, "long_term": 2},  # Ending memory counts after cleanup
        },
        "timestamp": 101.0,  # Distinct timestamp for ordering
    }
    bench.update_from_run(second_run)  # Store the second run for comparison

    summary = bench.summarise()  # Generate summary statistics including memory insights
    assert summary["memory"]["avg_total_delta"] == pytest.approx(0.5)  # Average delta should reflect both runs (3 and -2)
    assert summary["memory"]["positive_runs"] == 1  # Only the first run expanded memory
    assert summary["memory"]["negative_runs"] == 1  # Only the second run contracted memory

    challenges = bench.build_challenge_set(top_n=2)  # Request challenge descriptors for both runs
    assert challenges[0]["file_path"] == "alpha.py"  # First challenge should reference the higher-reward run
    assert challenges[0]["memory_pressure"] == pytest.approx(3.0)  # Memory pressure equals the total delta from the first run
    assert challenges[0]["memory_breakdown"]["short_term"] == pytest.approx(2.0)  # Short-term memory delta captured explicitly
    assert challenges[1]["memory_pressure"] == pytest.approx(-2.0)  # Second challenge reflects the contraction in memory usage


def test_adaptive_challenge_manager_agent_flow(tmp_path):  # Ensure the challenge manager answers challenge requests over MCP
    async def run_scenario():  # Wrap the asynchronous scenario for execution via asyncio.run
        server = InMemoryMCPServer()  # Create the in-memory MCP broker
        manager_client = InMemoryMCPClient(agent_id="benchmark_manager", server=server)  # Instantiate the manager's MCP client
        requester = InMemoryMCPClient(agent_id="research_agent", server=server)  # Instantiate the requesting agent client

        bench = AdaptiveFeedbackBenchmark(storage_path=str(tmp_path / "manager.json"))  # Prepare the adaptive benchmark for the manager
        bench.update_from_run({  # Seed the benchmark with a single run so challenges are available
            "run": {
                "id": "run-challenge",  # Unique run identifier for tracking
                "file_path": "gamma.py",  # File evaluated during the seeded run
                "guidance_history": ["prep"],  # Record that guidance was supplied
            },
            "history": {"reward": {"delta": 0.4}},  # Strong improvement to rank this run highly
            "iterations": [
                {
                    "discriminator_score": {"reward": 0.88},  # Reward value for averages
                    "generator_output": {"overall_summary": "gamma summary"},  # Provide generator output metadata
                }
            ],
            "memory_snapshot": {  # Provide memory counts to enrich challenge output
                "before": {"short_term": 2, "long_term": 1},  # Baseline memory counts
                "after": {"short_term": 4, "long_term": 2},  # Expanded memory counts after the run
            },
            "timestamp": 200.0,  # Timestamp used in persisted record
        })

        manager = AdaptiveChallengeManager(  # Assemble the adaptive challenge manager bridging benchmark and MCP
            bench,
            manager_client,
            manager_id="benchmark_manager",
        )

        await manager.start()  # Connect the manager and register message handlers
        await requester.connect()  # Connect the requesting agent client

        loop = asyncio.get_running_loop()  # Access the running event loop to create futures
        response_future = loop.create_future()  # Future used to capture the manager's response

        async def capture_response(message):  # Handler called when the response message arrives
            response_future.set_result(message.get("payload"))  # Store the payload inside the future for later assertions

        requester.register_message_handler("challenge_response", capture_response)  # Register to receive challenge responses

        await requester.send_message(  # Dispatch a challenge request to the manager
            recipient_id="benchmark_manager",  # Direct the message to the manager's agent id
            message_type="challenge_request",  # Use the request message type expected by the manager
            payload={"top_n": 1, "include_summary": True},  # Ask for the top challenge and include summary statistics
        )

        response = await asyncio.wait_for(response_future, timeout=1.0)  # Await the manager response with a timeout for safety
        assert response["challenges"][0]["file_path"] == "gamma.py"  # Ensure the challenge references the seeded run
        assert response["challenges"][0]["memory_pressure"] == pytest.approx(3.0)  # Memory pressure should reflect the seeded delta
        assert response["summary"]["memory"]["avg_total_delta"] == pytest.approx(3.0)  # Summary should report the same total delta

        await manager.stop()  # Disconnect the manager's MCP client
        await requester.disconnect()  # Disconnect the requesting agent client

    asyncio.run(run_scenario())  # Execute the asynchronous scenario for the challenge manager test


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
