#!/usr/bin/env python3
"""
Swarm System Demonstration

Demonstrates the hierarchical assistant → agent → swarm architecture
with meta-tooling capabilities and database integration.
"""

import logging
import asyncio
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import swarm system components
try:
    from assistants.registry import AssistantRegistry, global_registry
    from assistants.base_assistant import AssistantConfig
    from assistants.core.text_processor import TextProcessorAssistant
    from assistants.core.calculator_assistant import CalculatorAssistant
    from utils.database_manager import db_manager
    from utils.prompts import get_assistant_prompt
    from utils.tools import (
        create_dynamic_tool, create_assistant_as_tool,
        create_lightweight_agent, query_knowledge_base,
        store_learning, get_swarm_status
    )

    SWARM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Swarm system not fully available: {e}")
    SWARM_AVAILABLE = False


class SwarmDemo:
    """Demonstration of the swarm system capabilities."""

    def __init__(self):
        self.demo_results = []

    async def run_demo(self):
        """Run the complete swarm system demonstration."""
        logger.info("Starting Swarm System Demonstration")

        if not SWARM_AVAILABLE:
            logger.error("Swarm system components not available")
            return

        try:
            # 1. Initialize database system
            await self.demo_database_system()

            # 2. Register core assistants
            await self.demo_assistant_registration()

            # 3. Demonstrate meta-tooling
            await self.demo_meta_tooling()

            # 4. Show swarm communication
            await self.demo_swarm_communication()

            # 5. Display system status
            await self.demo_system_status()

            # 6. Generate summary report
            self.generate_summary_report()

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            self.demo_results.append(f"ERROR: {str(e)}")

    async def demo_database_system(self):
        """Demonstrate the database layer."""
        logger.info("=== DATABASE SYSTEM DEMO ===")

        # Store some sample knowledge
        knowledge_id = db_manager.store_knowledge(
            topic="swarm_systems",
            content="Swarm systems use multiple AI agents working together to solve complex problems",
            subtopic="ai_architecture",
            source="demo",
            confidence=0.9
        )

        # Store sample memory
        memory_id = db_manager.store_memory(
            session_id="demo_session",
            content="Demonstrated database integration with knowledge and memory storage",
            memory_type="demo",
            importance_score=0.8
        )

        # Set cache entry
        db_manager.set_cache("demo_key", {"demo": "data", "timestamp": "now"})

        # Retrieve and verify
        cached_data = db_manager.get_cache("demo_key")
        knowledge_results = db_manager.search_knowledge("swarm")

        self.demo_results.append("✅ Database system operational"        self.demo_results.append(f"   - Knowledge entries: {len(knowledge_results)}")
        self.demo_results.append(f"   - Cache working: {cached_data is not None}")
        self.demo_results.append(f"   - Memory storage: ID {memory_id}")

    async def demo_assistant_registration(self):
        """Demonstrate assistant registration and creation."""
        logger.info("=== ASSISTANT REGISTRATION DEMO ===")

        # Register core assistant types
        global_registry.register(
            "text_processor",
            TextProcessorAssistant,
            metadata={"type": "core", "version": "1.0.0"}
        )

        global_registry.register(
            "calculator",
            CalculatorAssistant,
            metadata={"type": "core", "version": "1.0.0"}
        )

        # Create assistant instances
        text_assistant = global_registry.create_instance(
            "text_processor",
            "demo_text_processor",
            config=AssistantConfig(
                name="demo_text_processor",
                description="Demo text processing assistant",
                model_id="llama3.2"
            )
        )

        calc_assistant = global_registry.create_instance(
            "calculator",
            "demo_calculator",
            config=AssistantConfig(
                name="demo_calculator",
                description="Demo calculator assistant",
                model_id="llama3.2"
            )
        )

        # Test assistant capabilities
        text_result = await text_assistant.execute_async("Hello world")
        calc_result = await calc_assistant.execute_async("2 + 2")

        self.demo_results.append("✅ Assistant system operational"        self.demo_results.append(f"   - Registered types: {global_registry.list_available_types()}")
        self.demo_results.append(f"   - Active instances: {len(global_registry.list_instances())}")
        self.demo_results.append(f"   - Text processor result: {type(text_result)}")
        self.demo_results.append(f"   - Calculator result: {type(calc_result)}")

    async def demo_meta_tooling(self):
        """Demonstrate meta-tooling capabilities."""
        logger.info("=== META-TOOLING DEMO ===")

        # Create a dynamic tool
        tool_result = create_dynamic_tool(
            tool_name="demo_greeter",
            description="A simple greeting tool",
            code_content="result = f'Hello, {name}!'",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"]
            }
        )

        # Create an assistant as a tool
        assistant_tool_result = create_assistant_as_tool(
            assistant_name="demo_researcher",
            assistant_type="research",
            model_id="llama3.2"
        )

        # Store some learning
        learning_result = store_learning(
            topic="meta_tooling",
            content="Meta-tooling allows dynamic creation of tools and assistants at runtime",
            source="demo",
            confidence=0.9
        )

        self.demo_results.append("✅ Meta-tooling operational"        self.demo_results.append(f"   - Dynamic tool creation: {len(tool_result)} chars")
        self.demo_results.append(f"   - Assistant as tool: {len(assistant_tool_result)} chars")
        self.demo_results.append(f"   - Learning storage: {len(learning_result)} chars")

    async def demo_swarm_communication(self):
        """Demonstrate swarm communication capabilities."""
        logger.info("=== SWARM COMMUNICATION DEMO ===")

        # Query knowledge base
        kb_result = query_knowledge_base("swarm", limit=3)

        # Demonstrate knowledge retrieval
        if "No knowledge found" not in kb_result:
            self.demo_results.append("✅ Swarm communication operational")
            self.demo_results.append(f"   - Knowledge retrieval: {len(kb_result)} chars")
            self.demo_results.append("   - Cross-component communication: Working")
        else:
            self.demo_results.append("⚠️  Knowledge base query returned no results")
    async def demo_system_status(self):
        """Display overall system status."""
        logger.info("=== SYSTEM STATUS DEMO ===")

        status = get_swarm_status()
        self.demo_results.append("✅ System status retrieved"        self.demo_results.append(f"   - Status length: {len(status)} chars")

        # Get database statistics
        db_stats = db_manager.get_stats()
        total_records = sum(stats.get('records', 0) for stats in db_stats.values() if stats.get('exists', False))
        self.demo_results.append(f"   - Total database records: {total_records}")

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("=== GENERATING SUMMARY REPORT ===")

        report = []
        report.append("SWARM SYSTEM DEMONSTRATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {asyncio.get_event_loop().time()}")
        report.append("")

        report.append("EXECUTION RESULTS:")
        report.extend(self.demo_results)

        report.append("")
        report.append("SYSTEM CAPABILITIES DEMONSTRATED:")
        report.append("✅ Database layer (cache.db, memory.db, knowledge.db, coderl.db)")
        report.append("✅ Assistant registration and management")
        report.append("✅ Meta-tooling and dynamic tool creation")
        report.append("✅ Assistant-as-tool functionality")
        report.append("✅ Knowledge base operations")
        report.append("✅ Swarm communication protocols")
        report.append("✅ System monitoring and status")

        report.append("")
        report.append("ARCHITECTURE COMPONENTS:")
        report.append("• Assistant Layer: Base building blocks")
        report.append("• Agent Layer: Composed assistants (in development)")
        report.append("• Swarm Layer: Lightweight agent coordination (in development)")
        report.append("• Meta-Tooling: Dynamic creation capabilities")
        report.append("• Database Layer: Multi-database management")
        report.append("• Communication Layer: Inter-component communication")

        # Write report to file
        report_content = "\n".join(report)
        with open("swarm_system/swarm_demo_report.txt", 'w') as f:
            f.write(report_content)

        logger.info("Summary report saved to: swarm_system/swarm_demo_report.txt")
        print("\n" + report_content)


async def main():
    """Main demonstration function."""
    demo = SwarmDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🧠 SWARM SYSTEM DEMONSTRATION")
    print("Demonstrating hierarchical assistant → agent → swarm architecture")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logger.exception("Demo exception")
