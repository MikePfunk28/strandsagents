#!/usr/bin/env python3
"""
Main entry point for the AI Agent Swarm System.

This is the proper way to import and use the swarm system:
- Uses Strands Agents SDK for all agent creation
- Implements proper multi-agent orchestration
- Supports both individual agents and swarm coordination
- Uses only local Ollama models (no cloud services)
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import swarm components with proper Strands patterns
from swarm.orchestrator import create_swarm_orchestrator, SwarmOrchestrator
from swarm.agents.research_assistant.service import create_research_assistant
from swarm.agents.creative_assistant.service import create_creative_assistant
from swarm.agents.critical_assistant.service import create_critical_assistant
from swarm.agents.summarizer_assistant.service import create_summarizer_assistant

class AIAgentSwarmSystem:
    """Main system class for AI Agent Swarm coordination."""

    def __init__(self):
        self.orchestrator: SwarmOrchestrator = None
        self.individual_agents: Dict[str, Any] = {}
        self.running = False

    async def initialize_system(self, mode: str = "swarm"):
        """Initialize the AI agent system."""
        try:
            if mode == "swarm":
                logger.info("Initializing AI Agent Swarm System...")
                self.orchestrator = await create_swarm_orchestrator("main_swarm")
                logger.info("✅ Swarm orchestrator initialized with 4 specialized agents")

            elif mode == "individual":
                logger.info("Initializing individual agents...")
                # Create individual agents for direct use
                self.individual_agents = {
                    "research": create_research_assistant("research_main"),
                    "creative": create_creative_assistant("creative_main"),
                    "critical": create_critical_assistant("critical_main"),
                    "summarizer": create_summarizer_assistant("summarizer_main")
                }

                # Start individual agents
                for agent_type, agent in self.individual_agents.items():
                    await agent.start_service()
                    logger.info(f"✅ {agent_type} agent started")

            self.running = True
            logger.info(f"🚀 AI Agent System ready in {mode} mode")

        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise

    async def shutdown_system(self):
        """Shutdown the AI agent system."""
        try:
            logger.info("Shutting down AI Agent System...")

            if self.orchestrator:
                await self.orchestrator.stop_swarm_agents()
                logger.info("✅ Swarm orchestrator stopped")

            for agent_type, agent in self.individual_agents.items():
                await agent.stop_service()
                logger.info(f"✅ {agent_type} agent stopped")

            self.running = False
            logger.info("🛑 AI Agent System shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def process_user_request(self, user_input: str) -> str:
        """Process user request through the system."""
        if not self.running:
            return "❌ System not initialized. Please start the system first."

        try:
            if self.orchestrator:
                # Use swarm orchestration
                logger.info(f"Processing request through swarm: {user_input[:50]}...")
                result = await self.orchestrator.process_user_request(user_input)
                return result
            else:
                # Use individual agents (simplified routing)
                logger.info(f"Processing request through individual agents: {user_input[:50]}...")

                # Simple routing logic (in production, use more sophisticated routing)
                if any(keyword in user_input.lower() for keyword in ["research", "find", "search", "analyze data"]):
                    agent = self.individual_agents["research"]
                elif any(keyword in user_input.lower() for keyword in ["creative", "brainstorm", "ideas", "innovative"]):
                    agent = self.individual_agents["creative"]
                elif any(keyword in user_input.lower() for keyword in ["critical", "evaluate", "assess", "risks"]):
                    agent = self.individual_agents["critical"]
                elif any(keyword in user_input.lower() for keyword in ["summarize", "summary", "synthesis"]):
                    agent = self.individual_agents["summarizer"]
                else:
                    # Default to research agent
                    agent = self.individual_agents["research"]

                result = await agent.process_task({
                    "task_id": "user_request",
                    "description": user_input
                })

                return result.get("result", "No result generated")

        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            return f"Error processing request: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if self.orchestrator:
            return {
                "mode": "swarm",
                "status": "running" if self.running else "stopped",
                "swarm_status": self.orchestrator.get_swarm_status()
            }
        else:
            return {
                "mode": "individual",
                "status": "running" if self.running else "stopped",
                "agents": {
                    agent_type: agent.get_status()
                    for agent_type, agent in self.individual_agents.items()
                }
            }

async def interactive_mode():
    """Run the system in interactive mode."""
    print("🤖 AI Agent Swarm System - Interactive Mode")
    print("=" * 50)

    # Ask user for mode
    print("\nSelect mode:")
    print("1. Swarm mode (orchestrated multi-agent)")
    print("2. Individual agents mode")

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")

    mode = "swarm" if choice == "1" else "individual"

    # Initialize system
    system = AIAgentSwarmSystem()

    try:
        await system.initialize_system(mode)

        print(f"\n✅ System initialized in {mode} mode")
        print("\nAvailable commands:")
        print("- Type your request naturally")
        print("- Type 'status' to see system status")
        print("- Type 'exit' to quit")
        print("\nExamples:")
        print("- Research the latest trends in AI")
        print("- Generate creative ideas for reducing plastic waste")
        print("- Critically evaluate remote work policies")
        print("- Summarize the benefits of renewable energy")

        while True:
            try:
                user_input = input("\n🎯 Your request: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                elif user_input.lower() == 'status':
                    status = system.get_system_status()
                    print(f"\n📊 System Status: {status}")
                elif user_input:
                    print("\n🤔 Processing...")
                    result = await system.process_user_request(user_input)
                    print(f"\n🎉 Result:\n{result}")
                else:
                    print("Please enter a request or command")

            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    finally:
        await system.shutdown_system()

async def demo_mode():
    """Run demo scenarios."""
    print("🤖 AI Agent Swarm System - Demo Mode")
    print("=" * 50)

    system = AIAgentSwarmSystem()

    try:
        # Demo swarm mode
        await system.initialize_system("swarm")

        demo_requests = [
            "Research the latest developments in renewable energy and provide creative solutions for urban implementation",
            "Analyze the pros and cons of remote work policies and summarize key findings",
            "Generate innovative ideas for reducing ocean plastic pollution and evaluate their feasibility"
        ]

        for i, request in enumerate(demo_requests, 1):
            print(f"\n🎯 Demo Request {i}: {request}")
            print("🤔 Processing...")

            result = await system.process_user_request(request)
            print(f"🎉 Result:\n{result}\n")
            print("-" * 80)

        # Show final status
        status = system.get_system_status()
        print(f"\n📊 Final System Status:")
        print(f"Mode: {status['mode']}")
        print(f"Status: {status['status']}")
        if 'swarm_status' in status:
            swarm_status = status['swarm_status']
            print(f"Active Agents: {swarm_status['active_agents']}")
            print(f"Completed Tasks: {swarm_status['completed_tasks']}")

    finally:
        await system.shutdown_system()

async def test_individual_agents():
    """Test individual agents separately."""
    print("🧪 Testing Individual Agents")
    print("=" * 30)

    system = AIAgentSwarmSystem()

    try:
        await system.initialize_system("individual")

        test_cases = [
            ("research", "Find information about sustainable energy trends"),
            ("creative", "Brainstorm ideas for improving online education"),
            ("critical", "Evaluate the risks of AI in healthcare"),
            ("summarizer", "Create a summary of renewable energy benefits")
        ]

        for agent_type, request in test_cases:
            print(f"\n🔬 Testing {agent_type} agent: {request}")

            # Direct agent access for testing
            agent = system.individual_agents[agent_type]
            result = await agent.process_task({
                "task_id": f"test_{agent_type}",
                "description": request
            })

            print(f"✅ {agent_type} result: {result.get('result', 'No result')[:200]}...")

    finally:
        await system.shutdown_system()

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "demo":
            asyncio.run(demo_mode())
        elif mode == "test":
            asyncio.run(test_individual_agents())
        elif mode == "interactive":
            asyncio.run(interactive_mode())
        else:
            print("Usage: python main.py [demo|test|interactive]")
            sys.exit(1)
    else:
        # Default to interactive mode
        asyncio.run(interactive_mode())

if __name__ == "__main__":
    main()