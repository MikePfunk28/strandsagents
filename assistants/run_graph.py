#!/usr/bin/env python3
"""Graph system runner for analysis, debugging, and feedback workflows.

Usage examples:
    python run_graph.py --analyze swarm/main.py
    python run_graph.py --data-structures path/to/code.py
    python run_graph.py --code-feedback agent.py --iterations 2 --guidance "Focus on docstrings"
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import aiofiles
except ImportError:
    aiofiles = None

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from graph import (
    FeedbackWorkflow,
    ProgrammingGraph,
    create_code_analysis_graph,
    create_data_structure_graph,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def run_code_feedback(
    file_path: str,
    iterations: int = 1,
    guidance: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,) -> None:
    """Run the code-feedback workflow graph and display results."""
    print(f"[workflow] running code-feedback loop for: {file_path}")

    workflow = FeedbackWorkflow()
    for entry in guidance or []:
        workflow.add_human_guidance(file_path, entry)

    metadata = metadata or {}
    try:
        state = await workflow.run(
            file_path=file_path,
            iterations=iterations,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"[workflow] failed: {exc}")
        return

    run_info = state.results.get("run", {})
    iterations_payload = state.results.get("iterations", [])

    print("=== Workflow Summary ===")
    print(f"Run ID       : {run_info.get('id')}")
    print(f"Iterations   : {len(iterations_payload)}")
    history = state.results.get("history", {}).get("reward", {})
    print(f"History dReward: {history.get('delta')}")
    print(f"Log file     : {run_info.get('log_path')}")

    if iterations_payload:
        latest = iterations_payload[-1]
        ds = latest.get("discriminator_score", {})
        print("--- Latest Iteration ---")
        print(f"Reward     : {ds.get('reward')}")
        print(f"Coverage   : {ds.get('coverage')}")
        print(f"Accuracy   : {ds.get('accuracy')}")
        print(f"Coherence  : {ds.get('coherence')}")
        print(f"Formatting : {ds.get('formatting')}")
        print(f"Summary    : {latest.get('generator_output', {}).get('overall_summary')}")

    print("Guidance history:")
    graph_description = workflow.describe()["feedback_graph"]
    print(json.dumps(graph_description.get("loop_guidance", []), indent=2))


async def analyze_code_file(file_path: str, model: str = "llama3.2:3b") -> None:
    """Analyze a code file using graph workflow."""
    print(f"[graph] analyzing code file: {file_path}")

    graph = create_code_analysis_graph(model)

    try:
        result = await graph.analyze_code(
            file_path,
            {
                "focus": "comprehensive_analysis",
                "include_patterns": True,
                "include_optimizations": True,
            },
        )

        print(f"[graph] analysis completed for: {result['file_path']}")
        print(f"graph type : {result['graph_type']}")
        print(f"timestamp  : {result['timestamp']}")
        print("=== Analysis Result ===")
        print(result["result"])
        print("=======================")

        status = graph.get_graph_status()
        agents = ", ".join(status.get('agents', []))
        print(f"agents used: {status.get('agents_count')} -> {agents}")

    except Exception as exc:
        print(f"[graph] error analyzing code: {exc}")


async def analyze_data_structures(code_input: str, model: str = "llama3.2:3b") -> None:
    """Analyze data structures in code."""
    print("[graph] analyzing data structures")

    graph = create_data_structure_graph(model)

    try:
        if Path(code_input).exists():
            if aiofiles:
                async with aiofiles.open(code_input, "r", encoding="utf-8") as file:
                    code_content = await file.read()
            else:
                # Fallback to sync I/O in thread
                code_content = await asyncio.to_thread(Path(code_input).read_text, encoding="utf-8")
            print(f"[graph] reading from file: {code_input}")
        else:
            code_content = code_input
            print("[graph] analyzing provided snippet")

        result = await graph.analyze_data_structures(
            code_content,
            {
                "focus": "comprehensive_mapping",
                "include_optimizations": True,
                "include_flow_diagram": True,
            },
        )

        print("[graph] data structure analysis completed")
        print(f"graph type : {result['graph_type']}")
        print(f"timestamp  : {result['timestamp']}")
        print("=== Analysis Result ===")
        print(result["result"])
        print("=======================")

    except Exception as exc:
        print(f"[graph] error analyzing data structures: {exc}")


async def debug_issue(error_info: str, code_context: str = "", model: str = "llama3.2:3b") -> None:
    """Debug an issue using debugging workflow."""
    print("[graph] starting debugging workflow")

    graph = ProgrammingGraph("debugging", model)

    try:
        if code_context and Path(code_context).exists():
            if aiofiles:
                async with aiofiles.open(code_context, "r", encoding="utf-8") as file:
                    code_content = await file.read()
            else:
                # Fallback to sync I/O in thread
                code_content = await asyncio.to_thread(Path(code_context).read_text, encoding="utf-8")
            print(f"[graph] reading code context from: {code_context}")
        else:
            code_content = code_context or "No code context provided"

        result = await graph.debug_issue(
            error_info,
            code_content,
            {
                "severity": "high",
                "detailed_analysis": True,
                "include_fixes": True,
            },
        )

        print("[graph] debugging analysis completed")
        print(f"graph type : {result['graph_type']}")
        print(f"timestamp  : {result['timestamp']}")
        print("=== Debug Result ===")
        print(result["result"])
        print("====================")

    except Exception as exc:
        print(f"[graph] error debugging issue: {exc}")


def _print_help() -> None:
    """Print interactive help."""
    print("Interactive Graph System")
    print("=" * 50)
    print("Commands:")
    print("  analyze <file_path>        - Analyze code file")
    print("  debug <error_info>         - Debug an error")
    print("  data <code_or_file>        - Analyze data structures")
    print("  feedback <file_path>       - Run code feedback workflow")
    print("  status                     - Show available graphs")
    print("  quit                       - Exit")
    print("=" * 50)


async def _handle_analyze_command(command: str) -> None:
    """Handle analyze command."""
    file_path = command[8:].strip()
    if file_path:
        await analyze_code_file(file_path)
    else:
        print("provide a file path")


async def _handle_debug_command(command: str) -> None:
    """Handle debug command."""
    error_info = command[6:].strip()
    if error_info:
        code_context = (await asyncio.to_thread(input, "Code context (file path or snippet, optional): ")).strip()
        await debug_issue(error_info, code_context)
    else:
        print("provide error information")


async def _handle_data_command(command: str) -> None:
    """Handle data structures command."""
    data_input = command[5:].strip()
    if data_input:
        await analyze_data_structures(data_input)
    else:
        print("provide code or file path")


async def _handle_feedback_command(command: str) -> None:
    """Handle feedback command."""
    file_path = command[9:].strip()
    if not file_path:
        print("provide a file path")
        return
    
    guidance = (await asyncio.to_thread(input, "Guidance (comma separated, optional): ")).strip()
    guidance_items = [item.strip() for item in guidance.split(",") if item.strip()] if guidance else []
    iterations = (await asyncio.to_thread(input, "Iterations (default 1): ")).strip()
    try:
        iteration_count = int(iterations) if iterations else 1
    except ValueError:
        iteration_count = 1
    await run_code_feedback(file_path, iterations=iteration_count, guidance=guidance_items)


async def _process_command(command: str) -> bool:
    """Process a single command. Returns True to continue, False to exit."""
    if not command:
        return True
    
    cmd_lower = command.lower()
    if cmd_lower in {"quit", "exit", "q"}:
        return False
    
    if cmd_lower == "status":
        print("Available graph types: code_analysis, data_structures, debugging, code_feedback")
    elif cmd_lower.startswith("analyze "):
        await _handle_analyze_command(command)
    elif cmd_lower.startswith("debug "):
        await _handle_debug_command(command)
    elif cmd_lower.startswith("data "):
        await _handle_data_command(command)
    elif cmd_lower.startswith("feedback "):
        await _handle_feedback_command(command)
    else:
        print("unknown command. type 'status' for options")
    
    return True


async def interactive_graph() -> None:
    """Run graph system in interactive mode."""
    _print_help()
    
    while True:
        try:
            command = (await asyncio.to_thread(input, "graph> ")).strip()
            if not await _process_command(command):
                break
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"error: {exc}")
    
    print("graph system session ended")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph System Runner")
    parser.add_argument("--analyze", help="Analyze code file")
    parser.add_argument("--debug", help="Debug error information")
    parser.add_argument("--data-structures", help="Analyze data structures in code/file")
    parser.add_argument("--code-feedback", help="Run code feedback workflow on file")
    parser.add_argument("--code-context", help="Code context for debugging (file path)")
    parser.add_argument("--model", default="llama3.2:3b", help="Model to use")
    parser.add_argument("--iterations", type=int, default=1, help="Number of feedback iterations")
    parser.add_argument("--guidance", action="append", help="Guidance string for feedback (repeatable)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_graph())
        return

    if args.analyze:
        asyncio.run(analyze_code_file(args.analyze, args.model))
        return

    if args.debug:
        code_context = args.code_context or ""
        asyncio.run(debug_issue(args.debug, code_context, args.model))
        return

    if args.data_structures:
        asyncio.run(analyze_data_structures(args.data_structures, args.model))
        return

    if args.code_feedback:
        asyncio.run(
            run_code_feedback(
                args.code_feedback,
                iterations=args.iterations,
                guidance=args.guidance,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
