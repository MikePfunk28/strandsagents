#!/usr/bin/env python3
"""
Interactive Swarm System with Individual Assistant Files

This creates a proper swarm system where each assistant is defined
in its own file with its own model, tools, and prompts.
Following official Strands Agents documentation patterns.
"""

import os
import sys
import importlib.util
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables and MCP context7 integration
load_dotenv()

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import all necessary strandsagents components
from strands.models.ollama import OllamaModel
from strands import Agent, tool
from strands_tools import think, editor, http_request, file_read, file_write, calculator

# Setup logging following Strands Agents style guide
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("assistants_directory")

# Create multiple model instances for different assistant types
# Using qwen models for better performance (user has 4b and 8b versions)
qwen_8b_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:8b"
)

qwen_4b_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:4b"
)

llama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2"
)



@dataclass
class AssistantSpec:
    name: str
    file_path: str
    model_id: str
    description: str
    tools: List[str]
    system_prompt: str

# Define all 9 assistants that exist in the assistants directory
# Using qwen models for better performance (user has 4b and 8b versions)
ASSISTANTS = {
    "aws_cost": AssistantSpec(
        name="aws_cost",
        file_path="assistants/aws_cost_assistant.py",
        model_id="qwen3:8b",  # Using 8b for complex cost analysis
        description="AWS cost analysis and optimization specialist",
        tools=["http_request", "calculator"],
        system_prompt="""You are an AWS cost analysis specialist.
        Analyze AWS costs, identify optimization opportunities, and provide cost-saving recommendations.
        Use web requests to fetch current pricing data and calculator for cost analysis."""
    ),

    "browser": AssistantSpec(
        name="browser",
        file_path="assistants/browser_assistant.py",
        model_id="qwen3:4b",  # Using 4b for web browsing tasks
        description="Web browsing and automation specialist",
        tools=["http_request"],
        system_prompt="""You are a web browsing specialist.
        Navigate websites, extract information, and perform web-based tasks.
        Use web requests to fetch and analyze web content."""
    ),

    "chunking": AssistantSpec(
        name="chunking",
        file_path="assistants/chunking_assistant.py",
        model_id="qwen3:8b",  # Using 8b for complex text processing
        description="Text chunking and processing specialist",
        tools=["file_read", "file_write"],
        system_prompt="""You are a text processing specialist.
        Break down large texts into manageable chunks, process content, and reorganize information.
        Use file operations to read, process, and write text content."""
    ),

    "embedding": AssistantSpec(
        name="embedding",
        file_path="assistants/embedding_assistant.py",
        model_id="qwen3:8b",  # Using 8b for embedding computations
        description="Vector embeddings and similarity specialist",
        tools=["http_request", "calculator"],
        system_prompt="""You are a vector embeddings specialist.
        Create, compare, and analyze text embeddings for similarity and relevance.
        Use web requests for embedding APIs and calculator for similarity computations."""
    ),

    "file": AssistantSpec(
        name="file",
        file_path="assistants/file_assistant.py",
        model_id="qwen3:4b",  # Using 4b for file operations
        description="File operations and management specialist",
        tools=["file_read", "file_write"],
        system_prompt="""You are a file management specialist.
        Read, write, organize, and manipulate files of various types.
        Use file operations to manage documents and data."""
    ),

    "financial": AssistantSpec(
        name="financial",
        file_path="assistants/financial_assistant.py",
        model_id="qwen3:8b",  # Using 8b for financial analysis
        description="Financial analysis and planning specialist",
        tools=["calculator", "http_request"],
        system_prompt="""You are a financial analysis specialist.
        Analyze financial data, create projections, and provide investment insights.
        Use calculator for financial computations and web requests for market data."""
    ),

    "memory": AssistantSpec(
        name="memory",
        file_path="assistants/memory_assistant.py",
        model_id="qwen3:8b",  # Using 8b for memory management
        description="Memory management and retrieval specialist",
        tools=["file_read", "file_write"],
        system_prompt="""You are a memory management specialist.
        Store, retrieve, and organize information for long-term access.
        Use file operations to manage knowledge bases and memory systems."""
    ),

    "meta_tool": AssistantSpec(
        name="meta_tool",
        file_path="assistants/meta_tool_assistant.py",
        model_id="qwen3:8b",  # Using 8b for meta-tooling complexity
        description="Meta-tooling and assistant management specialist",
        tools=["file_write", "editor"],
        system_prompt="""You are a meta-tooling specialist.
        Create, modify, and manage other assistants and their tools.
        Use file operations and editing capabilities to build and modify assistant systems."""
    ),

    "research": AssistantSpec(
        name="research",
        file_path="assistants/research_assistant.py",
        model_id="qwen3:8b",  # Using 8b for research tasks
        description="Research and information gathering specialist",
        tools=["http_request", "file_read"],
        system_prompt="""You are a research specialist focused on gathering accurate information from reliable sources.
        You use web requests and file reading to collect data, then provide structured analysis with citations.
        Always cite your sources and distinguish between facts and assumptions."""
    )
}

class AssistantManager:
    def __init__(self):
        self.assistants = {}
        self.agents = {}
        self.orchestrator_agent = None
        self.meta_tools = {}

    def create_meta_tools(self):
        """Create meta-tooling capabilities for assistant management"""

        @tool
        def list_assistants() -> str:
            """List all available assistants in the swarm"""
            assistant_list = []
            for name, spec in ASSISTANTS.items():
                assistant_list.append(f"• {name}: {spec.description}")
            return "\\n".join(assistant_list)

        @tool
        def get_assistant_info(assistant_name: str) -> str:
            """Get detailed information about a specific assistant"""
            if assistant_name not in ASSISTANTS:
                return f"Assistant '{assistant_name}' not found. Use list_assistants() to see available assistants."

            spec = ASSISTANTS[assistant_name]
            return f"""
Assistant: {spec.name}
Description: {spec.description}
Model: {spec.model_id}
Tools: {', '.join(spec.tools)}
System Prompt: {spec.system_prompt[:200]}...
"""

        @tool
        def analyze_assistant_capabilities(query: str) -> str:
            """Analyze which assistant would be best for a given query"""
            # Simple keyword-based routing
            query_lower = query.lower()

            if any(word in query_lower for word in ['aws', 'cost', 'cloud', 'billing']):
                return "aws_cost: AWS cost analysis and optimization specialist"
            elif any(word in query_lower for word in ['web', 'browse', 'website', 'url']):
                return "browser: Web browsing and automation specialist"
            elif any(word in query_lower for word in ['text', 'chunk', 'document', 'content']):
                return "chunking: Text chunking and processing specialist"
            elif any(word in query_lower for word in ['embedding', 'vector', 'similarity']):
                return "embedding: Vector embeddings and similarity specialist"
            elif any(word in query_lower for word in ['file', 'read', 'write', 'document']):
                return "file: File operations and management specialist"
            elif any(word in query_lower for word in ['financial', 'money', 'investment', 'stock']):
                return "financial: Financial analysis and planning specialist"
            elif any(word in query_lower for word in ['memory', 'remember', 'recall', 'knowledge']):
                return "memory: Memory management and retrieval specialist"
            elif any(word in query_lower for word in ['create', 'build', 'modify', 'assistant', 'tool']):
                return "meta_tool: Meta-tooling and assistant management specialist"
            elif any(word in query_lower for word in ['research', 'information', 'fact', 'source']):
                return "research: Research and information gathering specialist"
            else:
                return "research: General research and information gathering specialist"

        @tool
        def orchestrate_task(query: str) -> str:
            """Orchestrate a complex task across multiple assistants"""
            best_assistant = analyze_assistant_capabilities(query)

            # Extract assistant name from the recommendation
            assistant_name = best_assistant.split(':')[0]

            if assistant_name in ASSISTANTS:
                # Route to the appropriate assistant
                spec = ASSISTANTS[assistant_name]
                try:
                    spec_module = importlib.util.spec_from_file_location(
                        spec.name, spec.file_path
                    )
                    assistant_module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(assistant_module)

                    run_function = getattr(assistant_module, f"run_{spec.name}_assistant")
                    return run_function(query)
                except Exception as e:
                    return f"Error orchestrating task: {str(e)}"
            else:
                return f"Could not determine appropriate assistant for: {query}"

        self.meta_tools = {
            "list_assistants": list_assistants,
            "get_assistant_info": get_assistant_info,
            "analyze_assistant_capabilities": analyze_assistant_capabilities,
            "orchestrate_task": orchestrate_task
        }

        # Create orchestrator agent with meta-tools
        self.orchestrator_agent = Agent(
            model=qwen_8b_model,
            system_prompt="""You are an intelligent orchestrator for a swarm of specialized assistants.
            Analyze user queries and route them to the most appropriate assistant.
            Use the meta-tools to understand assistant capabilities and make intelligent routing decisions.
            If unsure, default to the research assistant.""",
            tools=list(self.meta_tools.values())
        )

        logger.info("🔧 Meta-tools and orchestrator created successfully")

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

# Create the assistant file following proper @tool decorator pattern from documentation
        file_content = f'''"""
{spec.name.title()} Assistant

{spec.description}
Generated as part of the swarm system following Strands Agents documentation patterns.
"""

import logging
from strands.models.ollama import OllamaModel
from strands import Agent, tool

# Setup logging following Strands Agents style guide
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("{spec.name}_assistant")

# Create Ollama model following documentation pattern
model = OllamaModel(
    host="http://localhost:11434",
    model_id="{spec.model_id}"
)

# Create agent following documentation pattern
agent = Agent(
    model=model,
    system_prompt="""{spec.system_prompt}"""
)

@tool
def {spec.name}_assistant(query: str) -> str:
    """
    {spec.description}

    Args:
        query: A query string for the {spec.name} assistant

    Returns:
        Response from the {spec.name} assistant
    """
    try:
        logger.info(f"🔧 {spec.name.title()} Assistant - Processing query: {{query}}")
        response = str(agent(query))
        logger.info(f"🔧 {spec.name.title()} Assistant - Response generated successfully")
        return response
    except Exception as e:
        error_msg = f"Error in {spec.name} assistant: {{str(e)}}"
        logger.error(f"🔧 {spec.name.title()} Assistant - {{error_msg}}")
        return error_msg

def run_{spec.name}_assistant(query: str) -> str:
    """Run the {spec.name} assistant with a query"""
    return {spec.name}_assistant(query)

if __name__ == "__main__":
    print("🔧 {spec.name.title()} Assistant")
    print("=" * 50)
    print("{spec.description}")
    print(f"Model: {spec.model_id}")
    print(f"System Prompt: {{agent.system_prompt[:100]}}...")
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
        with open(spec.file_path, 'w', encoding='utf-8') as f:
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
                    assistant_response = run_function(query)
                    print(f"{assistant_name.title()}: {assistant_response}")
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

    # Initialize meta-tools and orchestrator
    manager.create_meta_tools()

    print("🤖 StrandsAgents Swarm System")
    print("=" * 50)
    print("This system creates individual assistant files with their own:")
    print("  • Models (Ollama)")
    print("  • Tools (@tool decorated functions)")
    print("  • Prompts (specialized system prompts)")
    print("  • Meta-tooling capabilities")
    print("  • Intelligent orchestration")
    print()

    print("Available modes:")
    print("1. Create all assistant files")
    print("2. Run interactive swarm (manual routing)")
    print("3. Run intelligent orchestration (AI-powered routing)")
    print("4. Both create files and run orchestration")
    print()

    choice = input("Enter choice (1-4): ")

    if choice in ['1', '4']:
        manager.create_all_assistants()
        print()

    if choice in ['2']:
        manager.run_interactive_swarm()
    elif choice in ['3', '4']:
        manager.run_intelligent_orchestration()

def run_intelligent_orchestration(self):
    """Run intelligent orchestration using AI-powered routing"""
    print("🧠 Intelligent Orchestration Mode")
    print("=" * 50)
    print("AI-powered assistant routing with meta-tooling capabilities")
    print()
    print("Commands:")
    print("  • 'list' - List all assistants")
    print("  • 'info [assistant]' - Get assistant details")
    print("  • 'analyze [query]' - See which assistant would handle a query")
    print("  • 'exit' - Exit the system")
    print()

    while True:
        try:
            user_input = input("orchestrator> ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            if user_input.lower() == 'list':
                # Use meta-tool to list assistants
                list_tool = self.meta_tools["list_assistants"]
                assistants_list = list_tool()
                print("\\nAvailable assistants:")
                print(assistants_list)
                print()
                continue

            if user_input.lower().startswith('info '):
                # Get assistant info
                assistant_name = user_input[5:].strip()
                info_tool = self.meta_tools["get_assistant_info"]
                info = info_tool(assistant_name)
                print(f"\\n{info}\\n")
                continue

            if user_input.lower().startswith('analyze '):
                # Analyze which assistant would be best
                query = user_input[8:].strip()
                analyze_tool = self.meta_tools["analyze_assistant_capabilities"]
                analysis = analyze_tool(query)
                print(f"\\nBest assistant for '{query}':")
                print(analysis)
                print()
                continue

            if not user_input.strip():
                continue

            # Use orchestrator agent for intelligent routing
            print(f"🔧 Analyzing query: '{user_input}'")
            try:
                # Use the orchestrate_task meta-tool
                orchestrate_tool = self.meta_tools["orchestrate_task"]
                response = orchestrate_tool(user_input)
                print(f"\\nResponse: {response}\\n")

            except Exception as e:
                print(f"❌ Error in orchestration: {e}")
                print("Falling back to research assistant...")

                # Fallback to research assistant
                try:
                    spec = ASSISTANTS["research"]
                    spec_module = importlib.util.spec_from_file_location(
                        spec.name, spec.file_path
                    )
                    assistant_module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(assistant_module)

                    run_function = getattr(assistant_module, f"run_{spec.name}_assistant")
                    response = run_function(user_input)
                    print(f"Research Assistant: {response}\\n")

                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# Add the method to the AssistantManager class
AssistantManager.run_intelligent_orchestration = run_intelligent_orchestration

if __name__ == "__main__":
    main()
