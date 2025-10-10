#!/usr/bin/env python3
"""Simple runner script for the swarm system.

Usage:
    python run_swarm.py                    # Start interactive swarm
    python run_swarm.py --task "research AI trends"  # Run single task
    python run_swarm.py --status           # Show system status
"""

from swarm import SwarmSystem, create_research_swarm, create_development_swarm
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def run_interactive_swarm(swarm_type: str = "basic", model_config: dict = None):
    """Run swarm in interactive mode."""
    print(f" Starting {swarm_type} swarm system...")

    # Create swarm based on type
    if swarm_type == "research":
        swarm = create_research_swarm(model_config)
    elif swarm_type == "development":
        swarm = create_development_swarm(model_config)
    else:
        swarm = SwarmSystem(model_config)

    try:
        await swarm.start()
        print(" Swarm system started successfully!")

        # Show initial status
        status = swarm.get_status()
        print(f" System running: {status['running']}")
        print(f" Orchestrator: {status['orchestrator_model']}")
        print(f"🔧 Assistants: {status['assistant_model']}")
        print(f"👥 Active assistants: {len(status['assistants'])}")

        print("\n" + "="*50)
        print("Interactive Swarm Console")
        print("Commands:")
        print("  task <description>  - Process a task")
        print("  status             - Show system status")
        print("  switch <component> <model> - Switch models")
        print("  quit               - Exit")
        print("="*50)

        while True:
            try:
                command = input("\nswarm> ").strip()

                if not command:
                    continue

                if command.lower() in ['quit', 'exit', 'q']:
                    break

                elif command.lower() == 'status':
                    status = swarm.get_status()
                    print(f" Running: {status['running']}")
                    print(f" Orchestrator: {status['orchestrator_model']}")
                    print(f"🔧 Assistants: {status['assistant_model']}")
                    print(f"👥 Assistants: {len(status['assistants'])}")

                elif command.lower().startswith('task '):
                    task_desc = command[5:].strip()
                    if task_desc:
                        print(f"🔄 Processing task: {task_desc}")
                        result = await swarm.process_task(task_desc)
                        print(f" Status: {result.get('status', 'unknown')}")
                        if result.get('result'):
                            print(f"📝 Result: {result['result']}")
                    else:
                        print("❌ Please provide a task description")

                elif command.lower().startswith('switch '):
                    parts = command[7:].strip().split(' ', 1)
                    if len(parts) == 2:
                        component, model = parts
                        try:
                            await swarm.switch_model(component, model)
                            print(f" Switched {component} to {model}")
                        except Exception as e:
                            print(f"❌ Error switching model: {e}")
                    else:
                        print(
                            "❌ Usage: switch <orchestrator|assistants> <model_name>")

                else:
                    print("❌ Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to start swarm: {e}")
    finally:
        print("\n🛑 Shutting down swarm...")
        await swarm.stop()
        print(" Swarm stopped")


async def run_single_task(task: str, swarm_type: str = "basic", model_config: dict = None):
    """Run a single task and exit."""
    print(f"🔄 Running single task: {task}")

    # Create swarm
    if swarm_type == "research":
        swarm = create_research_swarm(model_config)
    elif swarm_type == "development":
        swarm = create_development_swarm(model_config)
    else:
        swarm = SwarmSystem(model_config)

    try:
        await swarm.start()
        print(" Swarm started")

        result = await swarm.process_task(task)

        print(f" Task completed: {result.get('status', 'unknown')}")
        if result.get('result'):
            print(f"📝 Result:\n{result['result']}")

    except Exception as e:
        print(f"❌ Error processing task: {e}")
    finally:
        await swarm.stop()


async def show_status(swarm_type: str = "basic", model_config: dict = None):
    """Show swarm status without starting interactive mode."""
    print(f" Checking {swarm_type} swarm status...")

    if swarm_type == "research":
        swarm = create_research_swarm(model_config)
    elif swarm_type == "development":
        swarm = create_development_swarm(model_config)
    else:
        swarm = SwarmSystem(model_config)

    try:
        await swarm.start()
        status = swarm.get_status()

        print(" Swarm System Status:")
        print(f"   Running: {status['running']}")
        print(f"   Orchestrator Model: {status['orchestrator_model']}")
        print(f"   Assistant Model: {status['assistant_model']}")
        print(f"   Available Models: {status['available_models']}")
        print(f"   Active Assistants: {len(status['assistants'])}")
        print(f"   Database Connected: {status['database_connected']}")

        print("\n👥 Assistant Details:")
        for aid, assistant_status in status['assistants'].items():
            print(
                f"   {aid}: {assistant_status.get('assistant_type', 'unknown')}")

    except Exception as e:
        print(f"❌ Error checking status: {e}")
    finally:
        await swarm.stop()


def main():
    parser = argparse.ArgumentParser(description="Swarm System Runner")
    parser.add_argument("--task", help="Run single task and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show status and exit")
    parser.add_argument("--type", choices=["basic", "research", "development"],
                        default="basic", help="Swarm type")
    parser.add_argument("--orchestrator-model", default="llama3.2:3b",
                        help="Orchestrator model")
    parser.add_argument("--assistant-model", default="gemma:270m",
                        help="Assistant model")
    parser.add_argument("--host", default="localhost:11434",
                        help="Ollama host")

    args = parser.parse_args()

    # Build model configuration
    model_config = {
        "orchestrator_model": args.orchestrator_model,
        "assistant_model": args.assistant_model,
        "host": args.host
    }

    # Run based on arguments
    if args.task:
        asyncio.run(run_single_task(args.task, args.type, model_config))
    elif args.status:
        asyncio.run(show_status(args.type, model_config))
    else:
        asyncio.run(run_interactive_swarm(args.type, model_config))


if __name__ == "__main__":
    main()
