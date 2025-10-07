"""Example usage of the swarm system.

This script demonstrates how to use the SwarmSystem with local Ollama models
for real task processing.
"""

import asyncio
import logging
from swarm import SwarmSystem, create_research_swarm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def example_basic_usage():
    """Example: Basic swarm usage with custom configuration."""
    print("🚀 Starting basic swarm example...")

    # Create swarm with custom model configuration
    config = {
        "orchestrator_model": "llama3.2:3b",
        "assistant_model": "gemma:270m",
        "host": "localhost:11434",
        "max_assistants": 4
    }

    swarm = SwarmSystem(config)

    try:
        # Start the swarm system
        await swarm.start()
        print("✅ Swarm system started")

        # Check system status
        status = swarm.get_status()
        print(f"📊 System status: {status['running']}")
        print(f"🤖 Orchestrator model: {status['orchestrator_model']}")
        print(f"🔧 Assistant model: {status['assistant_model']}")
        print(f"👥 Active assistants: {len(status['assistants'])}")

        # Process a real task
        task_result = await swarm.process_task(
            "Research the advantages of using local AI models vs cloud-based solutions",
            context={
                "domain": "ai_infrastructure",
                "focus": "cost_privacy_performance"
            }
        )

        print(f"📋 Task status: {task_result.get('status', 'unknown')}")
        if task_result.get('result'):
            print(f"📝 Task result: {task_result['result'][:200]}...")

        # Switch models during runtime
        print("\n🔄 Switching to different models...")
        await swarm.switch_model("orchestrator", "qwen:3b")
        await swarm.switch_model("assistants", "phi:4b")

        # Process another task with new models
        task_result2 = await swarm.process_task(
            "Summarize the key benefits of model switching in AI systems"
        )

        print(f"📋 Second task status: {task_result2.get('status', 'unknown')}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean shutdown
        await swarm.stop()
        print("🛑 Swarm system stopped")

async def example_specialized_swarm():
    """Example: Using a specialized research swarm."""
    print("\n🔬 Starting specialized research swarm example...")

    # Use the factory function for research-optimized swarm
    research_swarm = create_research_swarm({
        "database_path": "research_data",
        "log_level": "DEBUG"
    })

    try:
        await research_swarm.start()
        print("✅ Research swarm started")

        # Process research-specific tasks
        tasks = [
            "What are the latest developments in local AI model optimization?",
            "Compare the performance characteristics of Gemma vs Qwen models",
            "Analyze the trade-offs between model size and inference speed"
        ]

        for i, task in enumerate(tasks, 1):
            print(f"\n📚 Processing research task {i}...")
            result = await research_swarm.process_task(task)
            print(f"✅ Task {i} completed: {result.get('status', 'unknown')}")

    except Exception as e:
        print(f"❌ Research swarm error: {e}")
    finally:
        await research_swarm.stop()
        print("🛑 Research swarm stopped")

async def main():
    """Run all examples."""
    print("🌟 Swarm System Examples")
    print("=" * 40)

    # Run basic usage example
    await example_basic_usage()

    # Run specialized swarm example
    await example_specialized_swarm()

    print("\n✨ All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())