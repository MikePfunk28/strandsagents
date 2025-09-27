"""Example usage of the coding assistant system."""

import asyncio
import logging
from pathlib import Path

from simple_coding_agent import create_simple_coding_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate the coding assistant capabilities."""

    # Create the coding agent
    agent = create_simple_coding_agent(
        db_dir="./assistant_data",
        model_name="llama3.2",
        ollama_host="http://localhost:11434"
    )

    print("🤖 Coding Assistant Demo")
    print("=" * 50)

    # Example 1: Simple chat interaction
    print("\n1. Simple Chat Interaction:")
    response = agent.chat("Hello! Can you help me write a Python function to calculate fibonacci numbers?")
    print(f"Assistant: {response}")

    # Example 2: Code analysis task
    print("\n2. Code Analysis Task:")
    analysis_result = await agent.execute_task(
        "Analyze the structure of a Python project and provide insights",
        task_type="analysis",
        project_path="."
    )
    print(f"Analysis Result: {analysis_result['success']}")
    if analysis_result['success']:
        print(f"Details: {str(analysis_result['result'])[:200]}...")

    # Example 3: Feature implementation task
    print("\n3. Feature Implementation Task:")
    feature_result = await agent.execute_task(
        "Create a simple calculator class with basic arithmetic operations",
        task_type="feature",
        file_path="calculator.py"
    )
    print(f"Feature Implementation: {feature_result['success']}")

    # Example 4: Debugging task
    print("\n4. Debugging Task:")
    debug_result = await agent.execute_task(
        "Debug a Python script that has a syntax error in line 10",
        task_type="debug",
        file_path="buggy_script.py"
    )
    print(f"Debug Result: {debug_result['success']}")

    # Example 5: Show assistant statistics
    print("\n5. Assistant Statistics:")
    stats = agent.get_stats()
    print(f"Session ID: {stats['session_id']}")
    print(f"Memory entries: {stats['memory_stats'].get('memory_entries', 0)}")
    print(f"Knowledge entries: {stats['memory_stats'].get('knowledge_entries', 0)}")
    print(f"Model: {stats['model_info']['name']}")
    print(f"Server healthy: {stats['model_info']['server_healthy']}")

    # Example 6: Streaming task execution
    print("\n6. Streaming Task Execution:")
    async for event in agent.stream_task(
        "Write unit tests for the calculator class",
        task_type="testing"
    ):
        print(f"Event: {event['type']}")
        if event['type'] == 'complete':
            print(f"Result summary: {str(event['result'])[:100]}...")

    # Cleanup
    agent.cleanup()
    print("\n✅ Demo completed!")


async def interactive_demo():
    """Interactive demo where user can chat with the assistant."""
    agent = create_simple_coding_agent()

    print("🤖 Interactive Coding Assistant")
    print("Type 'quit' to exit, 'stats' for statistics, 'help' for help")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = agent.get_stats()
                print(f"\n📊 Statistics:")
                print(f"Session: {stats['session_id']}")
                print(f"Memory entries: {stats['memory_stats'].get('memory_entries', 0)}")
                print(f"Tools available: {stats['tools_available']}")
                continue
            elif user_input.lower() == 'help':
                print(f"\n🆘 Available commands:")
                print("- Ask any coding question")
                print("- Request code analysis: 'analyze file.py'")
                print("- Request feature implementation: 'implement a sorting function'")
                print("- Request debugging help: 'debug this error: ...'")
                print("- 'stats' - show statistics")
                print("- 'quit' - exit")
                continue

            if not user_input:
                continue

            # Determine task type based on keywords
            task_type = "general"
            if any(word in user_input.lower() for word in ["analyze", "analysis"]):
                task_type = "analysis"
            elif any(word in user_input.lower() for word in ["implement", "create", "build"]):
                task_type = "feature"
            elif any(word in user_input.lower() for word in ["debug", "fix", "error", "bug"]):
                task_type = "debug"
            elif any(word in user_input.lower() for word in ["test", "testing"]):
                task_type = "testing"

            # For complex tasks, use workflow execution
            if task_type != "general":
                print("🔄 Executing workflow...")
                result = await agent.execute_task(user_input, task_type)
                if result['success']:
                    print(f"\n🤖 Assistant: Workflow completed successfully!")
                    print(f"Result: {str(result['result'])[:500]}...")
                else:
                    print(f"\n❌ Assistant: Workflow failed: {result['error']}")
            else:
                # Simple chat for general questions
                response = agent.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    agent.cleanup()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(main())