"""Base assistant class for lightweight 270M microservices."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from strands import Agent
from strands.tools.executors import SequentialToolExecutor, ConcurrentToolExecutor
from swarm.communication.mcp_client import SwarmMCPClient

logger = logging.getLogger(__name__)

class BaseAssistant(ABC):
    """Base class for lightweight assistant microservices.

    Each assistant:
    - Does ONE specific thing well
    - Uses 270M model for speed
    - Has specialized prompts and tools
    - Communicates via MCP
    - Operates independently as microservice
    """

    def __init__(self, assistant_id: str, assistant_type: str,
                 capabilities: List[str], model_name: str = "gemma:270m",
                 host: str = "localhost:11434"):
        self.assistant_id = assistant_id
        self.assistant_type = assistant_type
        self.capabilities = capabilities
        self.model_name = model_name
        self.host = host

        # Core components
        self.agent: Optional[Agent] = None
        self.mcp_client: Optional[SwarmMCPClient] = None
        self.tools = []

        # State
        self.running = False
        self.task_queue = asyncio.Queue()
        self.results_cache: Dict[str, Any] = {}

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the specialized system prompt for this assistant."""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Get the specialized tools for this assistant."""
        pass

    async def initialize(self):
        """Initialize the assistant microservice."""
        try:
            # Load tools
            self.tools = self.get_tools()

            # Create StrandsAgent with model string (Strands will handle the model provider)
            # Note: Since Strands doesn't have built-in Ollama support, we use string model names
            # In production, you'd configure a custom model provider or use supported providers
            self.agent = Agent(
                model=self.model_name,  # Use string model name directly
                system_prompt=self.get_system_prompt(),
                tools=self.tools
                # tool_executor=SequentialToolExecutor()  # Not needed for basic usage
            )

            # Create MCP client for communication
            self.mcp_client = SwarmMCPClient(
                agent_id=self.assistant_id,
                agent_type=self.assistant_type,
                capabilities=self.capabilities,
                model_name=self.model_name
            )

            # Connect to swarm
            await self.mcp_client.connect()

            # Register message handlers
            self.mcp_client.register_message_handler("task_request", self.handle_task_request)
            self.mcp_client.register_message_handler("collaboration_request", self.handle_collaboration)

            self.running = True
            logger.info(f"Assistant {self.assistant_id} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize assistant {self.assistant_id}: {e}")
            raise

    async def start_service(self):
        """Start the assistant microservice."""
        if not self.running:
            await self.initialize()

        # Start background task processing
        asyncio.create_task(self.process_tasks())

        logger.info(f"Assistant microservice {self.assistant_id} started")

    async def stop_service(self):
        """Stop the assistant microservice."""
        self.running = False

        if self.mcp_client:
            await self.mcp_client.disconnect()

        logger.info(f"Assistant microservice {self.assistant_id} stopped")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task using the specialized agent."""
        try:
            task_id = task.get("task_id", "unknown")
            task_description = task.get("description", "")
            context = task.get("context", {})

            # Check cache first
            cache_key = f"{task_description}:{hash(str(context))}"
            if cache_key in self.results_cache:
                logger.info(f"Returning cached result for task {task_id}")
                return self.results_cache[cache_key]

            # Build prompt with context
            if context:
                prompt = f"Context: {context}\n\nTask: {task_description}"
            else:
                prompt = task_description

            # Execute with agent (using correct Strands API)
            result = await self.agent.invoke_async(prompt)

            # Prepare response
            response = {
                "task_id": task_id,
                "assistant_id": self.assistant_id,
                "assistant_type": self.assistant_type,
                "result": result,
                "status": "completed",
                "model_used": self.model_name
            }

            # Cache result
            self.results_cache[cache_key] = response

            # Limit cache size
            if len(self.results_cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self.results_cache))
                del self.results_cache[oldest_key]

            return response

        except Exception as e:
            logger.error(f"Error processing task in {self.assistant_id}: {e}")
            return {
                "task_id": task.get("task_id", "unknown"),
                "assistant_id": self.assistant_id,
                "error": str(e),
                "status": "error"
            }

    async def handle_task_request(self, message_data: Dict[str, Any]):
        """Handle incoming task request via MCP."""
        payload = message_data.get("payload", {})
        task = payload.get("task", {})

        # Process task
        result = await self.process_task(task)

        # Send result back
        if self.mcp_client:
            await self.mcp_client.send_message(
                payload.get("sender_id", "unknown"),
                "task_result",
                result
            )

    async def handle_collaboration(self, message_data: Dict[str, Any]):
        """Handle collaboration request from other agents."""
        payload = message_data.get("payload", {})
        collaboration_id = payload.get("collaboration_id")
        task_description = payload.get("task_description")

        # Process the collaborative task
        task = {
            "task_id": collaboration_id,
            "description": task_description,
            "collaboration": True
        }

        result = await self.process_task(task)

        # Send collaboration response
        if self.mcp_client:
            await self.mcp_client.respond_to_collaboration(collaboration_id, result)

    async def process_tasks(self):
        """Background task processing loop."""
        while self.running:
            try:
                # Check for MCP messages
                if self.mcp_client:
                    message = await self.mcp_client.get_next_message(timeout=0.1)
                    if message:
                        await self.handle_incoming_message(message)

                # Process local task queue
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                    result = await self.process_task(task)
                    logger.info(f"Processed local task: {result.get('task_id')}")
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(0.1)

    async def handle_incoming_message(self, message: Dict[str, Any]):
        """Handle incoming MCP message."""
        message_type = message.get("type")

        if message_type == "task_request":
            await self.handle_task_request(message)
        elif message_type == "collaboration_request":
            await self.handle_collaboration(message)
        else:
            logger.debug(f"Unhandled message type: {message_type}")

    async def add_task(self, task: Dict[str, Any]):
        """Add task to local processing queue."""
        await self.task_queue.put(task)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the assistant."""
        return {
            "assistant_id": self.assistant_id,
            "assistant_type": self.assistant_type,
            "model_name": self.model_name,
            "capabilities": self.capabilities,
            "running": self.running,
            "tasks_queued": self.task_queue.qsize(),
            "cache_size": len(self.results_cache),
            "connected_to_swarm": self.mcp_client.connected if self.mcp_client else False
        }

# Factory function for creating lightweight assistants
def create_lightweight_assistant(assistant_type: str, assistant_id: Optional[str] = None) -> BaseAssistant:
    """Factory function to create lightweight assistant instances."""

    if assistant_id is None:
        import uuid
        assistant_id = f"{assistant_type}_{str(uuid.uuid4())[:8]}"

    # Import specific assistant classes
    if assistant_type == "research":
        from .research_assistant.service import ResearchAssistant
        return ResearchAssistant(assistant_id)
    elif assistant_type == "creative":
        from .creative_assistant.service import CreativeAssistant
        return CreativeAssistant(assistant_id)
    elif assistant_type == "critical":
        from .critical_assistant.service import CriticalAssistant
        return CriticalAssistant(assistant_id)
    elif assistant_type == "summarizer":
        from .summarizer_assistant.service import SummarizerAssistant
        return SummarizerAssistant(assistant_id)
    else:
        raise ValueError(f"Unknown assistant type: {assistant_type}")

# Example usage for testing
async def demo_base_assistant():
    """Demonstrate base assistant functionality."""

    # Create a simple test assistant
    class TestAssistant(BaseAssistant):
        def get_system_prompt(self) -> str:
            return "You are a test assistant. Respond briefly and helpfully."

        def get_tools(self) -> List[Any]:
            return []  # No tools for test

    # Create and start assistant
    assistant = TestAssistant(
        assistant_id="test_assistant_001",
        assistant_type="test",
        capabilities=["testing"]
    )

    try:
        await assistant.start_service()

        # Test task processing
        test_task = {
            "task_id": "test_001",
            "description": "Say hello and explain what you do",
            "context": {"user": "developer"}
        }

        result = await assistant.process_task(test_task)
        print(f"Test result: {result}")

        # Check status
        status = assistant.get_status()
        print(f"Assistant status: {status}")

    finally:
        await assistant.stop_service()

if __name__ == "__main__":
    asyncio.run(demo_base_assistant())