#!/usr/bin/env python3
"""
Interactive Swarm System with Individual Assistant Files

This creates a proper swarm system where each assistant is defined
in its own file with its own model, tools, and prompts.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List
from dataclasses import dataclass

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from strands.models.ollama import OllamaModel
from strands import Agent, tool

@dataclass
class AssistantSpec:
    name: str
    file_path: str
    model: str
    description: str
    tools: List[str]
    system_prompt: str

# Define all assistants
ASSISTANTS = {
    "researcher": AssistantSpec(
        name="researcher",
        file_path="assistants/researcher_assistant.py",
        model="llama3.2",
        description="Research and information gathering specialist",
        tools=["http_request", "file_read"],
        system_prompt="""You are a research specialist focused on gathering accurate information from reliable sources.
        You use web requests and file reading to collect data, then provide structured analysis with citations.
        Always cite your sources and distinguish between facts and assumptions."""
    ),

    "writer": AssistantSpec(
        name="writer",
        file_path="assistants/writer_assistant.py",
        model="llama3.2",
        description="Content creation and writing specialist",
        tools=["file_write", "editor"],
        system_prompt="""You are a professional writer who creates clear, engaging content.
        You organize information logically, use proper grammar and style, and adapt your writing
        to the intended audience and purpose."""
    ),

    "analyst": AssistantSpec(
        name="analyst",
        file_path="assistants/analyst_assistant.py",
        model="llama3.2",
        description="Data analysis and critical thinking specialist",
        tools=["calculator", "think"],
        system_prompt="""You are a critical analyst who examines information objectively.
        You identify patterns, evaluate evidence quality, assess risks and opportunities,
        and provide balanced, well-reasoned conclusions."""
    ),

    "coder": AssistantSpec(
        name="coder",
        file_path="assistants/coder_assistant.py",
        model="qwen:14b",
        description="Programming and development specialist",
        tools=["python_repl", "file_write", "shell"],
        system_prompt="""You are an expert programmer who writes clean, efficient, well-documented code.
        You understand multiple programming languages and frameworks, follow best practices,
        and create maintainable solutions."""
    ),

    "creative": AssistantSpec(
        name="creative",
        file_path="assistants/creative_assistant.py",
        model="llama3.2",
        description="Creative thinking and innovation specialist",
        tools=["think", "editor"],
        system_prompt="""You are a creative thinker who generates innovative ideas and solutions.
        You think outside the box, make unexpected connections, and propose novel approaches
        to problems and opportunities."""
    ),

    "critic": AssistantSpec(
        name="critic",
        file_path="assistants/critic_assistant.py",
        model="llama3.2",
        description="Critical evaluation and quality assessment specialist",
        tools=["think", "editor"],
        system_prompt="""You are a critical evaluator who assesses quality, identifies flaws,
        and suggests improvements. You provide constructive feedback, challenge assumptions,
        and ensure high standards are maintained."""
    ),

    "planner": AssistantSpec(
        name="planner",
        file_path="assistants/planner_assistant.py",
        model="llama3.2",
        description="Strategic planning and organization specialist",
        tools=["think", "editor"],
        system_prompt="""You are a strategic planner who creates detailed plans and organizes complex tasks.
        You break down large goals into manageable steps, identify dependencies, allocate resources,
        and create realistic timelines."""
    ),

    "summarizer": AssistantSpec(
        name="summarizer",
        file_path="assistants/summarizer_assistant.py",
        model="llama3.2",
        description="Information synthesis and summarization specialist",
        tools=["file_read", "editor"],
        system_prompt="""You are a summarization expert who distills complex information into clear,
        concise formats. You identify key points, eliminate redundancy, and present information
        in the most useful format for the intended audience."""
    )
}

class AssistantManager:
    def __init__(self):
        self.assistants = {}
        self.agents = {}

    def create_assistant_file(self, spec: AssistantSpec):
        """Create an individual assistant file"""

        # Windows compatibility for tools
        try:
            from strands_tools import http_request, file_read, file_write, editor, python_repl, shell, think
            tools_available = True
        except ImportError:
            print(f"Warning: Some tools not available on Windows")
            tools_available = False
            # Create fallback functions
            def http_request(*args, **kwargs): return "Web requests not available on Windows"
            def file_read(*args, **kwargs): return "File reading not available"
            def file_write(*args, **kwargs): return "File writing not available"
            def editor(*args, **kwargs): return "Editor not available on Windows"
            def python_repl(*args, **kwargs): return "Python REPL not available on Windows"
            def shell(*args, **kwargs): return "Shell not available on Windows"
            def think(*args, **kwargs): return "Deep thinking not available"

        # Map tool names to actual functions
        tool_map = {
            "http_request": http_request,
            "file_read": file_read,
            "file_write": file_write,
            "editor": editor,
            "python_repl": python_repl,
            "shell": shell,
            "think": think
        }

        # Get tools for this assistant
        tools = []
        for tool_name in spec.tools:
            if tools_available and tool_name in tool_map:
                tools.append(tool_map[tool_name])

        # Create the assistant file
        file_content = f'''"""
{spec.name.title()} Assistant

{spec.description}
Generated as part of the swarm system.
"""

from strands.models.ollama import OllamaModel
from strands import Agent

# Windows compatibility for tools
try:
    from strands_tools import {", ".join(spec.tools)}
    tools_available = True
except ImportError:
    print(f"Warning: Some tools not available on Windows")
    tools_available = False
    # Create fallback functions
    {chr(10).join([f"def {tool}(*args, **kwargs): return '{tool} not available on Windows'" for tool in spec.tools])}

# Create the assistant
model = OllamaModel(
    host="http://localhost:11434",
    model_id="{spec.model}"
)

agent = Agent(
    model=model,
    tools=[{", ".join([tool for tool in spec.tools if tools_available])}],
    system_prompt="""{spec.system_prompt}"""
)

def run_{spec.name}_assistant(query: str) -> str:
    """Run the {spec.name} assistant with a query"""
    try:
        return str(agent(query))
    except Exception as e:
        return f"Error running {spec.name} assistant: {{str(e)}}"

if __name__ == "__main__":
    print("🔧 {spec.name.title()} Assistant")
    print("=" * 50)
    print("{spec.description}")
    print(f"Model: {spec.model}")
    print(f"Available Tools: {', '.join(spec.tools)}")
    print()
    print("Enter queries for the {spec.name} assistant (type 'exit' to quit):")

    while True:
        try:
            user_input = input(f"{spec.name}> ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            response = run_{spec.name}_assistant(user_input)
            print(f"Assistant: {{response}}")
            print()

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {{e}}")
'''

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(spec.file_path), exist_ok=True)

        # Write the file
        with open(spec.file_path, 'w') as f:
            f.write(file_content)

        print(f"✅ Created {spec.name} assistant: {spec.file_path}")

    def create_all_assistants(self):
        """Create all assistant files"""
        print("🚀 Creating Individual Assistant Files")
        print("=" * 50)

        for name, spec in ASSISTANTS.items():
            self.create_assistant_file(spec)

        print(f"\\n✅ Created {len(ASSISTANTS)} assistant files")

    def run_interactive_swarm(self):
        """Run an interactive swarm system"""
        print("🧠 Interactive Swarm System")
        print("=" * 50)
        print("Available assistants:")
        for name, spec in ASSISTANTS.items():
            print(f"  • {name}: {spec.description}")

        print("\\nEnter queries and I'll route them to the appropriate assistant.")
        print("Format: [assistant_name] query")
        print("Example: researcher what is quantum computing?")
        print("Type 'list' to see assistants, 'exit' to quit")
        print()

        while True:
            try:
                user_input = input("swarm> ").strip()

                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break

                if user_input.lower() == 'list':
                    print("\\nAvailable assistants:")
                    for name, spec in ASSISTANTS.items():
                        print(f"  • {name}: {spec.description}")
                    print()
                    continue

                # Parse assistant name from input
                parts = user_input.split(' ', 1)
                if len(parts) < 2:
                    print("❌ Please specify an assistant. Example: researcher what is AI?")
                    continue

                assistant_name = parts[0]
                query = parts[1]

                if assistant_name not in ASSISTANTS:
                    print(f"❌ Unknown assistant: {assistant_name}")
                    print("Use 'list' to see available assistants")
                    continue

                # Import and run the assistant
                spec = ASSISTANTS[assistant_name]

                # Dynamic import of the assistant module
                try:
                    spec_module = importlib.util.spec_from_file_location(
                        spec.name, spec.file_path
                    )
                    assistant_module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(assistant_module)

                    # Get the run function
                    run_function = getattr(assistant_module, f"run_{spec.name}_assistant")

                    print(f"🔧 Routing to {assistant_name} assistant...")
                    response = run_function(query)
                    print(f"{assistant_name.title()}: {response}")
                    print()

                except Exception as e:
                    print(f"❌ Error running {assistant_name} assistant: {e}")

            except KeyboardInterrupt:
                print("\\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function"""
    manager = AssistantManager()

    print("🤖 StrandsAgents Swarm System")
    print("=" * 50)
    print("This system creates individual assistant files with their own:")
    print("  • Models (Ollama)")
    print("  • Tools (@tool decorated functions)")
    print("  • Prompts (specialized system prompts)")
    print("  • Separation of concerns")
    print()

    choice = input("Choose an option:\\n1. Create all assistant files\\n2. Run interactive swarm\\n3. Both\\nEnter choice (1-3): ")

    if choice in ['1', '3']:
        manager.create_all_assistants()
        print()

    if choice in ['2', '3']:
        manager.run_interactive_swarm()

if __name__ == "__main__":
    main()
