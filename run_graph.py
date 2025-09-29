#!/usr/bin/env python3
"""Simple runner script for the graph system.

Usage:
    python run_graph.py --analyze swarm/main.py     # Analyze code file
    python run_graph.py --debug "error info"        # Debug workflow
    python run_graph.py --data-structures "code"    # Analyze data structures
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from graph import create_code_analysis_graph, create_data_structure_graph, ProgrammingGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def analyze_code_file(file_path: str, model: str = "llama3.2:3b"):
    """Analyze a code file using graph workflow."""
    print(f"📊 Analyzing code file: {file_path}")

    graph = create_code_analysis_graph(model)

    try:
        result = await graph.analyze_code(file_path, {
            "focus": "comprehensive_analysis",
            "include_patterns": True,
            "include_optimizations": True
        })

        print(f"✅ Analysis completed for: {result['file_path']}")
        print(f"📋 Graph type: {result['graph_type']}")
        print(f"⏰ Timestamp: {result['timestamp']}")
        print("\n📝 Analysis Result:")
        print("="*60)
        print(result['result'])
        print("="*60)

        # Show graph status
        status = graph.get_graph_status()
        print(f"\n📊 Graph used {status['agents_count']} agents: {', '.join(status['agents'])}")

    except Exception as e:
        print(f"❌ Error analyzing code: {e}")

async def analyze_data_structures(code_input: str, model: str = "llama3.2:3b"):
    """Analyze data structures in code."""
    print(f"🏗️  Analyzing data structures...")

    graph = create_data_structure_graph(model)

    try:
        # If it's a file path, read the file
        if Path(code_input).exists():
            with open(code_input, 'r', encoding='utf-8') as f:
                code_content = f.read()
            print(f"📁 Reading from file: {code_input}")
        else:
            code_content = code_input
            print("📝 Analyzing provided code snippet")

        result = await graph.analyze_data_structures(code_content, {
            "focus": "comprehensive_mapping",
            "include_optimizations": True,
            "include_flow_diagram": True
        })

        print(f"✅ Data structure analysis completed")
        print(f"📋 Graph type: {result['graph_type']}")
        print(f"⏰ Timestamp: {result['timestamp']}")
        print("\n🏗️  Analysis Result:")
        print("="*60)
        print(result['result'])
        print("="*60)

    except Exception as e:
        print(f"❌ Error analyzing data structures: {e}")

async def debug_issue(error_info: str, code_context: str = "", model: str = "llama3.2:3b"):
    """Debug an issue using debugging workflow."""
    print(f"🐛 Starting debugging workflow...")

    graph = ProgrammingGraph("debugging", model)

    try:
        # If code_context is a file path, read it
        if code_context and Path(code_context).exists():
            with open(code_context, 'r', encoding='utf-8') as f:
                code_content = f.read()
            print(f"📁 Reading code context from: {code_context}")
        else:
            code_content = code_context or "No code context provided"

        result = await graph.debug_issue(error_info, code_content, {
            "severity": "high",
            "detailed_analysis": True,
            "include_fixes": True
        })

        print(f"✅ Debugging analysis completed")
        print(f"📋 Graph type: {result['graph_type']}")
        print(f"⏰ Timestamp: {result['timestamp']}")
        print("\n🐛 Debug Result:")
        print("="*60)
        print(result['result'])
        print("="*60)

    except Exception as e:
        print(f"❌ Error debugging issue: {e}")

async def interactive_graph():
    """Run graph system in interactive mode."""
    print("🌟 Interactive Graph System")
    print("="*50)
    print("Commands:")
    print("  analyze <file_path>        - Analyze code file")
    print("  debug <error_info>         - Debug an error")
    print("  data <code_or_file>        - Analyze data structures")
    print("  status                     - Show available graphs")
    print("  quit                       - Exit")
    print("="*50)

    while True:
        try:
            command = input("\ngraph> ").strip()

            if not command:
                continue

            if command.lower() in ['quit', 'exit', 'q']:
                break

            elif command.lower() == 'status':
                print("📊 Available Graph Types:")
                print("   • code_analysis - Parse, analyze, document code")
                print("   • data_structures - Identify, map, optimize data flows")
                print("   • debugging - Analyze errors, generate fixes, validate")
                print("   • Model: llama3.2:3b (default)")

            elif command.lower().startswith('analyze '):
                file_path = command[8:].strip()
                if file_path:
                    await analyze_code_file(file_path)
                else:
                    print("❌ Please provide a file path")

            elif command.lower().startswith('debug '):
                error_info = command[6:].strip()
                if error_info:
                    code_context = input("Code context (file path or code snippet, optional): ").strip()
                    await debug_issue(error_info, code_context)
                else:
                    print("❌ Please provide error information")

            elif command.lower().startswith('data '):
                data_input = command[5:].strip()
                if data_input:
                    await analyze_data_structures(data_input)
                else:
                    print("❌ Please provide code or file path")

            else:
                print("❌ Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Graph system session ended")

def main():
    parser = argparse.ArgumentParser(description="Graph System Runner")
    parser.add_argument("--analyze", help="Analyze code file")
    parser.add_argument("--debug", help="Debug error information")
    parser.add_argument("--data-structures", help="Analyze data structures in code/file")
    parser.add_argument("--code-context", help="Code context for debugging (file path)")
    parser.add_argument("--model", default="llama3.2:3b", help="Model to use")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Run based on arguments
    if args.analyze:
        asyncio.run(analyze_code_file(args.analyze, args.model))
    elif args.debug:
        code_context = args.code_context or ""
        asyncio.run(debug_issue(args.debug, code_context, args.model))
    elif args.data_structures:
        asyncio.run(analyze_data_structures(args.data_structures, args.model))
    else:
        # Default to interactive mode
        asyncio.run(interactive_graph())

if __name__ == "__main__":
    main()