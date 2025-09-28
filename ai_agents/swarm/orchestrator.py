"""Swarm Orchestrator using Strands Agents patterns for multi-agent coordination."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from strands import Agent
from strands.tools.executors import ConcurrentToolExecutor, SequentialToolExecutor

from swarm.agents.research_assistant.service import create_research_assistant
from swarm.agents.creative_assistant.service import create_creative_assistant
from swarm.agents.critical_assistant.service import create_critical_assistant
from swarm.agents.summarizer_assistant.service import create_summarizer_assistant
from swarm.communication.mcp_client import create_orchestrator_client

logger = logging.getLogger(__name__)

@dataclass
class SwarmTask:
    """Represents a task for the swarm to process."""
    task_id: str
    description: str
    required_agents: List[str]
    priority: int = 1
    context: Dict[str, Any] = None

class SwarmOrchestrator:
    """Swarm orchestrator implementing Strands multi-agent patterns."""

    def __init__(self, orchestrator_id: str = "swarm_orchestrator",
                 orchestrator_model: str = "llama3.2:3b"):
        self.orchestrator_id = orchestrator_id
        self.orchestrator_model = orchestrator_model

        # Core orchestrator agent with larger model for complex reasoning
        self.orchestrator_agent: Optional[Agent] = None
        self.mcp_client = None

        # Lightweight swarm agents (270M models)
        self.swarm_agents: Dict[str, Any] = {}
        self.agent_tools = {}

        # Task management
        self.active_tasks: Dict[str, SwarmTask] = {}
        self.task_results: Dict[str, Any] = {}

        self.running = False

    async def initialize_orchestrator(self):
        """Initialize the main orchestrator agent."""
        try:
            # Create specialized agent tools for delegation
            self.agent_tools = self._create_agent_tools()

            # Main orchestrator agent with delegation tools
            # Using string model name - in production you'd configure proper model provider
            self.orchestrator_agent = Agent(
                model=self.orchestrator_model,  # String model name
                system_prompt=self._get_orchestrator_prompt(),
                tools=list(self.agent_tools.values())
                # tool_executor=ConcurrentToolExecutor()  # Not needed for basic usage
            )

            # MCP client for swarm communication
            self.mcp_client = create_orchestrator_client(self.orchestrator_id)
            await self.mcp_client.connect()

            logger.info(f"Orchestrator {self.orchestrator_id} initialized with {self.orchestrator_model}")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def _create_agent_tools(self) -> Dict[str, Any]:
        """Create tools that represent each specialized agent."""
        from strands.types.tool_types import ToolUse, ToolResult

        def research_agent_tool(tool_use: ToolUse, **kwargs) -> ToolResult:
            """Delegate research tasks to research assistant."""
            tool_use_id = tool_use["toolUseId"]
            query = tool_use["input"]["query"]

            # This would be handled by the research agent
            result = f"Research task '{query}' delegated to research assistant"

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": result}]
            }

        def creative_agent_tool(tool_use: ToolUse, **kwargs) -> ToolResult:
            """Delegate creative tasks to creative assistant."""
            tool_use_id = tool_use["toolUseId"]
            task = tool_use["input"]["task"]

            result = f"Creative task '{task}' delegated to creative assistant"

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": result}]
            }

        def critical_agent_tool(tool_use: ToolUse, **kwargs) -> ToolResult:
            """Delegate critical analysis to critical assistant."""
            tool_use_id = tool_use["toolUseId"]
            analysis_target = tool_use["input"]["analysis_target"]

            result = f"Critical analysis of '{analysis_target}' delegated to critical assistant"

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": result}]
            }

        def summarizer_agent_tool(tool_use: ToolUse, **kwargs) -> ToolResult:
            """Delegate summarization to summarizer assistant."""
            tool_use_id = tool_use["toolUseId"]
            content = tool_use["input"]["content"]

            result = f"Summarization of content delegated to summarizer assistant"

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": result}]
            }

        # Tool specifications following Strands patterns
        tools = {
            "research_assistant": {
                "spec": {
                    "name": "research_assistant",
                    "description": "Delegate research and fact-finding tasks to specialized research agent",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Research query or question"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                "function": research_agent_tool
            },
            "creative_assistant": {
                "spec": {
                    "name": "creative_assistant",
                    "description": "Delegate creative and ideation tasks to specialized creative agent",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Creative task or brainstorming request"
                                }
                            },
                            "required": ["task"]
                        }
                    }
                },
                "function": creative_agent_tool
            },
            "critical_assistant": {
                "spec": {
                    "name": "critical_assistant",
                    "description": "Delegate critical analysis and evaluation tasks to specialized critical agent",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "analysis_target": {
                                    "type": "string",
                                    "description": "Content or topic to analyze critically"
                                }
                            },
                            "required": ["analysis_target"]
                        }
                    }
                },
                "function": critical_agent_tool
            },
            "summarizer_assistant": {
                "spec": {
                    "name": "summarizer_assistant",
                    "description": "Delegate summarization tasks to specialized summarizer agent",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Content to summarize"
                                }
                            },
                            "required": ["content"]
                        }
                    }
                },
                "function": summarizer_agent_tool
            }
        }

        return tools

    def _get_orchestrator_prompt(self) -> str:
        """Get the orchestrator system prompt with tool selection guidance."""
        return """
You are a swarm orchestrator that coordinates specialized AI agents:

- For research questions and factual information → Use the research_assistant tool
- For creative tasks, brainstorming, and ideation → Use the creative_assistant tool
- For critical analysis, evaluation, and review → Use the critical_assistant tool
- For summarization and synthesis → Use the summarizer_assistant tool

You can use multiple agents in parallel for complex tasks that benefit from different perspectives.

Always select the most appropriate agent(s) based on the user's query and coordinate their outputs effectively.
"""

    async def start_swarm_agents(self):
        """Start all lightweight swarm agents (270M models)."""
        try:
            # Create lightweight agents
            agent_configs = [
                ("research_001", "research", create_research_assistant),
                ("creative_001", "creative", create_creative_assistant),
                ("critical_001", "critical", create_critical_assistant),
                ("summarizer_001", "summarizer", create_summarizer_assistant)
            ]

            for agent_id, agent_type, creator_func in agent_configs:
                agent = creator_func(agent_id)
                await agent.start_service()
                self.swarm_agents[agent_type] = agent
                logger.info(f"Started {agent_type} agent: {agent_id}")

            self.running = True
            logger.info("All swarm agents started successfully")

        except Exception as e:
            logger.error(f"Failed to start swarm agents: {e}")
            raise

    async def stop_swarm_agents(self):
        """Stop all swarm agents."""
        for agent_type, agent in self.swarm_agents.items():
            try:
                await agent.stop_service()
                logger.info(f"Stopped {agent_type} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_type} agent: {e}")

        self.running = False

        if self.mcp_client:
            await self.mcp_client.disconnect()

    async def process_user_request(self, user_input: str) -> str:
        """Process user request through the orchestrator."""
        if not self.orchestrator_agent:
            raise RuntimeError("Orchestrator not initialized")

        try:
            # Use the orchestrator agent to coordinate swarm response
            logger.info(f"Processing user request: {user_input}")
            result = await self.orchestrator_agent.invoke_async(user_input)

            return result

        except Exception as e:
            logger.error(f"Error processing user request: {e}")
            return f"Error processing request: {str(e)}"

    async def process_collaborative_task(self, task: SwarmTask) -> Dict[str, Any]:
        """Process a task requiring multiple agents."""
        task_results = {}

        try:
            # Delegate to required agents based on task requirements
            for agent_type in task.required_agents:
                if agent_type in self.swarm_agents:
                    agent = self.swarm_agents[agent_type]
                    result = await agent.process_task({
                        "task_id": f"{task.task_id}_{agent_type}",
                        "description": task.description,
                        "context": task.context
                    })
                    task_results[agent_type] = result

            # Store results
            self.task_results[task.task_id] = task_results

            return task_results

        except Exception as e:
            logger.error(f"Error in collaborative task {task.task_id}: {e}")
            return {"error": str(e)}

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get status of the entire swarm."""
        return {
            "orchestrator_id": self.orchestrator_id,
            "orchestrator_model": self.orchestrator_model,
            "running": self.running,
            "active_agents": len(self.swarm_agents),
            "agent_status": {
                agent_type: agent.get_status()
                for agent_type, agent in self.swarm_agents.items()
            },
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results)
        }

# Factory function for easy swarm creation
async def create_swarm_orchestrator(orchestrator_id: str = "swarm_001") -> SwarmOrchestrator:
    """Create and initialize a swarm orchestrator."""
    orchestrator = SwarmOrchestrator(orchestrator_id)

    await orchestrator.initialize_orchestrator()
    await orchestrator.start_swarm_agents()

    return orchestrator

# Demo usage following Strands patterns
async def demo_swarm_orchestrator():
    """Demonstrate swarm orchestrator functionality."""
    print("Starting Swarm Orchestrator Demo")
    print("=" * 40)

    orchestrator = None
    try:
        # Create and start swarm
        orchestrator = await create_swarm_orchestrator("demo_swarm")

        # Test user requests
        test_requests = [
            "Research the latest developments in renewable energy and provide creative solutions for urban implementation",
            "Analyze the pros and cons of remote work and summarize the key findings",
            "Brainstorm innovative solutions for reducing plastic waste in oceans"
        ]

        for request in test_requests:
            print(f"\nProcessing: {request}")
            result = await orchestrator.process_user_request(request)
            print(f"Result: {result}")

        # Show swarm status
        status = orchestrator.get_swarm_status()
        print(f"\nSwarm Status: {status}")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        if orchestrator:
            await orchestrator.stop_swarm_agents()

if __name__ == "__main__":
    asyncio.run(demo_swarm_orchestrator())