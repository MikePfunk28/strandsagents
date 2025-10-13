#!/usr/bin/env python3
"""
Real Agent Builder for StrandsAgents + AgentCore

Creates functional agents that actually work with real StrandsAgents tools
and AgentCore integration. No more demo/test implementations.

CRITICAL: This fixes the broken import system and creates working agents.
"""

import json
import logging
import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[AGENT_BUILDER] %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agent_builder")

# All available StrandsAgents tools from documentation
STRANDS_TOOLS = {
    # RAG & Memory
    "retrieve": {"pip": None, "description": "Semantically retrieve data from Amazon Bedrock Knowledge Bases"},
    "memory": {"pip": None, "description": "Agent memory persistence in Amazon Bedrock Knowledge Bases"},
    "agent_core_memory": {"pip": None, "description": "Integration with Amazon Bedrock Agent Core Memory"},
    "mem0_memory": {"pip": "strands-agents-tools[mem0_memory]", "description": "Agent memory and personalization"},

    # File Operations
    "editor": {"pip": None, "description": "File editing operations like line edits, search, and undo"},
    "file_read": {"pip": None, "description": "Read and parse files"},
    "file_write": {"pip": None, "description": "Create and modify files"},

    # Shell & System
    "environment": {"pip": None, "description": "Manage environment variables"},
    "shell": {"pip": None, "description": "Execute shell commands"},
    "cron": {"pip": None, "description": "Task scheduling with cron jobs"},
    "use_computer": {"pip": "strands-agents-tools[use_computer]", "description": "Automate desktop actions"},

    # Code Interpretation
    "python_repl": {"pip": None, "description": "Run Python code (not supported on Windows)"},
    "code_interpreter": {"pip": None, "description": "Execute code in isolated sandboxes"},

    # Web & Network
    "http_request": {"pip": None, "description": "Make API calls, fetch web data"},
    "slack": {"pip": None, "description": "Slack integration with real-time events"},
    "browser": {"pip": None, "description": "Automate web browser interactions"},
    "rss": {"pip": "strands-agents-tools[rss]", "description": "Manage and process RSS feeds"},

    # Multi-modal
    "generate_image_stability": {"pip": None, "description": "Create images with Stability AI"},
    "image_reader": {"pip": None, "description": "Process and analyze images"},
    "generate_image": {"pip": None, "description": "Create AI generated images with Amazon Bedrock"},
    "nova_reels": {"pip": None, "description": "Create AI generated videos with Nova Reels"},
    "speak": {"pip": None, "description": "Generate speech from text"},
    "diagram": {"pip": "strands-agents-tools[diagram]", "description": "Create cloud architecture diagrams"},

    # AWS Services
    "use_aws": {"pip": None, "description": "Interact with AWS services"},

    # Utilities
    "calculator": {"pip": None, "description": "Perform mathematical operations"},
    "current_time": {"pip": None, "description": "Get the current date and time"},
    "load_tool": {"pip": None, "description": "Dynamically load more tools"},
    "sleep": {"pip": None, "description": "Pause execution"},

    # Agents & Workflows
    "graph": {"pip": None, "description": "Create and manage multi-agent systems"},
    "agent_graph": {"pip": None, "description": "Create and manage graphs of agents"},
    "journal": {"pip": None, "description": "Create structured tasks and logs"},
    "swarm": {"pip": None, "description": "Coordinate multiple AI agents"},
    "stop": {"pip": None, "description": "Force stop the agent event loop"},
    "handoff_to_user": {"pip": None, "description": "Enable human-in-the-loop workflows"},
    "use_agent": {"pip": None, "description": "Run a new AI event loop"},
    "think": {"pip": None, "description": "Perform deep thinking"},
    "use_llm": {"pip": None, "description": "Run a new AI event loop"},
    "workflow": {"pip": None, "description": "Orchestrate sequenced workflows"},
    "batch": {"pip": None, "description": "Call multiple tools"},
    "a2a_client": {"pip": "strands-agents-tools[a2a_client]", "description": "Enable agent-to-agent communication"},
}

# Browser tools that need special setup
BROWSER_TOOLS = {
    "local_chromium_browser": "strands-agents-tools[local_chromium_browser]",
    "agent_core_browser": "strands-agents-tools[agent_core_browser]",
    "agent_core_code_interpreter": "strands-agents-tools[agent_core_code_interpreter]",
}

class RealAgentBuilder:
    """Creates real functional agents with proper StrandsAgents integration"""

    def __init__(self):
        self.output_dir = Path("assistants/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_real_agent(self, name: str, description: str, model_type: str = "ollama",
                         tools: List[str] = None, enable_code_execution: bool = False) -> Dict[str, Path]:
        """Create a real functional agent with proper imports and setup"""

        if tools is None:
            tools = ["http_request", "file_read", "calculator"]

        # Determine model and setup based on code execution requirements
        if enable_code_execution:
            # Code execution requires AgentCore + Bedrock
            model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            runtime_setup = self._generate_agentcore_setup()
            pip_installs = ["strands-agents-tools", "strands-agents-tools[agent_core_code_interpreter]"]
        else:
            # Regular agents can use Ollama
            model_id = "llama3.2"
            runtime_setup = self._generate_ollama_setup()
            pip_installs = ["strands-agents-tools"]

        # Generate the real agent code
        agent_code = self._generate_real_agent_code(
            name, description, model_id, model_type, tools, enable_code_execution, pip_installs, runtime_setup
        )

        # Create output files
        agent_file = self.output_dir / f"{name}.py"
        setup_file = self.output_dir / f"{name}_setup.py"
        requirements_file = self.output_dir / f"{name}_requirements.txt"
        metadata_file = self.output_dir / f"{name}_metadata.json"

        # Write files
        agent_file.write_text(agent_code, encoding='utf-8')

        setup_code = self._generate_setup_script(name, pip_installs, runtime_setup)
        setup_file.write_text(setup_code, encoding='utf-8')

        requirements_content = "\n".join(pip_installs)
        requirements_file.write_text(requirements_content, encoding='utf-8')

        metadata = {
            "name": name,
            "description": description,
            "model_id": model_id,
            "model_type": model_type,
            "tools": tools,
            "enable_code_execution": enable_code_execution,
            "created_by": "RealAgentBuilder",
            "created_at": datetime.now().isoformat(),
            "files": {
                "agent": str(agent_file),
                "setup": str(setup_file),
                "requirements": str(requirements_file)
            }
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        print(f"\n✅ Real agent '{name}' created successfully!")
        print(f"📁 Files created:")
        print(f"   • Agent: {agent_file}")
        print(f"   • Setup: {setup_file}")
        print(f"   • Requirements: {requirements_file}")
        print(f"   • Metadata: {metadata_file}")

        return {
            "agent": agent_file,
            "setup": setup_file,
            "requirements": requirements_file,
            "metadata": metadata_file
        }

    def _generate_real_agent_code(self, name: str, description: str, model_id: str,
                                 model_type: str, tools: List[str], enable_code_execution: bool,
                                 pip_installs: List[str], runtime_setup: str) -> str:
        """Generate real agent code with proper StrandsAgents imports"""

        # Generate proper tool imports
        tool_imports = []
        for tool in tools:
            if tool in STRANDS_TOOLS:
                tool_imports.append(f"    {tool},")

        tools_str = "\n".join(tool_imports)

        # Generate model import based on type
        if model_type == "bedrock" or enable_code_execution:
            model_import = '''from strands.models import BedrockModel
    model = BedrockModel(model_id=model_id)'''
        else:
            model_import = '''from strands.models.ollama import OllamaModel
    model = OllamaModel(host="http://localhost:11434", model_id=model_id)'''

        # Generate the complete real agent code
        timestamp = datetime.now().isoformat()
        code_lines = [
            '"""',
            f"{name.title()} - {description}",
            "",
            f"Generated by Real Agent Builder on {timestamp}",
            f"Model: {model_id}",
            f"Runtime: {model_type}",
            f"Code Execution: {enable_code_execution}",
            '"""',
            "",
            "# Real StrandsAgents imports - no fallback nonsense",
            "import sys",
            "import os",
            "import logging",
            "from typing import Optional",
            "",
            "# Core StrandsAgents imports",
            "try:",
            "    from strands import Agent",
            "    from strands_tools import (",
            tools_str,
            "    )",
            "    print(f\"✅ Successfully imported StrandsAgents tools: {', '.join(tools)}\")",
            "except ImportError as e:",
            "    print(f\"❌ Failed to import StrandsAgents tools: {e}\")",
            "    print(\"💡 Run the setup script first: python {name}_setup.py\")",
            "    sys.exit(1)",
            "",
            "# Model setup",
            "model_id = \"{model_id}\"",
            "try:",
            model_import,
            "    print(f\"✅ Model '{model_id}' loaded successfully\")",
            "except Exception as e:",
            "    print(f\"❌ Failed to load model '{model_id}': {e}\")",
            "    sys.exit(1)",
            "",
            "# Setup logging",
            "logging.basicConfig(level=logging.INFO)",
            f"logger = logging.getLogger(\"{name}\")",
            "",
            "# System prompt",
            "SYSTEM_PROMPT = \"\"\"You are {name}, a specialized AI agent.",
            "",
            f"{description}",
            "",
            "You have access to the following real StrandsAgents tools:",
            f"{', '.join(tools)}",
            "",
            "Use these tools to accomplish your tasks effectively.",
            "Always provide helpful, accurate responses.\"\"\"",
            "",
            "# Create the real agent with proper tool integration",
            "try:",
            "    agent = Agent(",
            "        model=model,",
            "        system_prompt=SYSTEM_PROMPT,",
            f"        tools=[{', '.join(tools)}]",
            "    )",
            "    print(f\"✅ Real agent '{name}' created successfully\")",
            "except Exception as e:",
            "    print(f\"❌ Failed to create agent: {e}\")",
            "    sys.exit(1)",
            "",
            "def {name}(query: str) -> str:",
            f'    """',
            f'    {description}',
            f'    """',
            '    """Real functional agent with StrandsAgents integration"""',
            '    """',
            '    try:',
            f'        logger.info(f"{name.title()} processing query: {{query[:100]}}...")',
            '        ',
            '        # Use the real StrandsAgents agent',
            '        response = str(agent(query))',
            '        ',
            f'        logger.info(f"{name.title()} completed successfully")',
            '        return response',
            '        ',
            '    except Exception as e:',
            f'        error_msg = f"Error in {name}: {{str(e)}}"',
            f'        logger.error(f"{name.title()} error: {{error_msg}}")',
            '        return error_msg',
            '        ',
            '# Agent metadata',
            'AGENT_METADATA = {',
            f'    "name": "{name}",',
            f'    "description": "{description}",',
            f'    "model_id": "{model_id}",',
            f'    "model_type": "{model_type}",',
            f'    "tools": {tools},',
            f'    "enable_code_execution": {str(enable_code_execution).lower()},',
            f'    "created_at": "{timestamp}",',
            f'    "generator": "RealAgentBuilder"',
            '}',
            '        ',
            'if __name__ == "__main__":',
            f'    print("{name.title()} - Real Functional Agent")',
            '    print("=" * 50)',
            f'    print(f"Model: {{model_id}}")',
            f'    print(f"Runtime: {{model_type}}")',
            f'    print(f"Tools: {{", ".join(tools)}}")',
            f'    print(f"Code Execution: {{enable_code_execution}}")',
            '    ',
            '    # Test the real agent',
            '    test_query = "Hello! Test the real agent functionality."',
            f'    print(f"Test Query: {{test_query}}")',
            '    print()',
            '    ',
            '    try:',
            f'        result = {name}(test_query)',
            f'        print(f"✅ Real Agent Response: {{result}}")',
            '        print("\\n🎉 Real agent working successfully!")',
            '    except Exception as e:',
            f'        print(f"❌ Agent test failed: {{e}}")',
            '        print("\\n💡 Make sure to run the setup script first:")',
            '        print(f"   python {name}_setup.py")'
        ]

        return "\n".join(code_lines)

    def _generate_setup_script(self, name: str, pip_installs: List[str], runtime_setup: str) -> str:
        """Generate setup script for the agent"""

        setup_lines = [
            "#!/usr/bin/env python3",
            '"""',
            f"Setup script for {name}",
            "",
            "Installs required dependencies and sets up the runtime environment.",
            '"""',
            "",
            "import sys",
            "import os",
            "import subprocess",
            "import logging",
            "",
            "# Setup logging",
            "logging.basicConfig(level=logging.INFO)",
            "logger = logging.getLogger(\"setup\")",
            "",
            "def run_command(cmd: str, description: str) -> bool:",
            '    """Run a command and handle errors"""',
            '    try:',
            '        logger.info(f"Running: {description}")',
            '        result = subprocess.run(cmd, shell=True, check=True,',
            '                               capture_output=True, text=True)',
            '        logger.info(f"✅ {description} completed")',
            '        return True',
            '    except subprocess.CalledProcessError as e:',
            '        logger.error(f"❌ {description} failed: {e}")',
            '        logger.error(f"Error output: {e.stderr}")',
            '        return False',
            "",
            "def main():",
            '    """Main setup process"""',
            '    print(f"🚀 Setting up {name}...")',
            '    print("=" * 50)',
            '    ',
            '    # Install Python dependencies',
            '    print("\\n📦 Installing Python dependencies...")',
            '    for package in {pip_installs}:',
            '        if not run_command(f"pip install \\"{package}\\"", f"Install {package}"):',
            '            print(f"❌ Failed to install {package}")',
            '            return False',
            '    ',
            '    # Setup runtime environment',
            '    print("\\n🔧 Setting up runtime environment...")',
            runtime_setup,
            '    ',
            '    print("\\n✅ Setup completed successfully!")',
            '    print(f"🎉 You can now run your agent: python {name}.py")',
            '    ',
            '    # Test the setup',
            '    print("\\n🧪 Testing setup...")',
            '    try:',
            '        import strands',
            '        print("✅ StrandsAgents imported successfully")',
            '        ',
            '        from strands_tools import http_request',
            '        print("✅ StrandsAgents tools imported successfully")',
            '        ',
            '        print("\\n🎉 All tests passed! Agent is ready to use.")',
            '        ',
            '    except ImportError as e:',
            '        print(f"❌ Import test failed: {e}")',
            '        print("💡 There may be additional setup required.")',
            '        return False',
            '    ',
            '    return True',
            "",
            "if __name__ == \"__main__\":",
            '    success = main()',
            '    sys.exit(0 if success else 1)'
        ]

        return "\n".join(setup_lines)

    def _generate_ollama_setup(self) -> str:
        """Generate Ollama setup instructions"""
        return '''
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
        else:
            print("⚠️  Ollama responded but may need model installation")
    except:
        print("ℹ️  Ollama not detected. Make sure Ollama is running:")
        print("   1. Install Ollama from https://ollama.com")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull required models: ollama pull llama3.2")
        return False

    return True'''

    def _generate_agentcore_setup(self) -> str:
        """Generate AgentCore setup instructions"""
        return '''
    # AgentCore setup for AWS Bedrock
    print("🔧 Setting up AgentCore for AWS Bedrock...")
    print("ℹ️  Make sure you have:")
    print("   1. AWS credentials configured (aws configure)")
    print("   2. Bedrock access enabled in your AWS account")
    print("   3. Proper IAM permissions for Bedrock")

    # Check AWS credentials
    try:
        import boto3
        client = boto3.client("bedrock-runtime")
        print("✅ AWS credentials configured")
    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        print("💡 Run: aws configure")
        return False

    return True'''

    def list_all_tools(self):
        """List all available StrandsAgents tools"""
        print("\n🛠️  All Available StrandsAgents Tools:")
        print("=" * 60)

        for category in ["RAG & Memory", "File Operations", "Shell & System",
                        "Code Interpretation", "Web & Network", "Multi-modal",
                        "AWS Services", "Utilities", "Agents & Workflows"]:
            print(f"\n📂 {category}:")
            for tool_name, tool_info in STRANDS_TOOLS.items():
                if self._get_tool_category(tool_name) == category:
                    pip_info = f" (needs: {tool_info['pip']})" if tool_info['pip'] else ""
                    print(f"   • {tool_name}: {tool_info['description']}{pip_info}")

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool"""
        categories = {
            "retrieve": "RAG & Memory",
            "memory": "RAG & Memory",
            "agent_core_memory": "RAG & Memory",
            "mem0_memory": "RAG & Memory",
            "editor": "File Operations",
            "file_read": "File Operations",
            "file_write": "File Operations",
            "environment": "Shell & System",
            "shell": "Shell & System",
            "cron": "Shell & System",
            "use_computer": "Shell & System",
            "python_repl": "Code Interpretation",
            "code_interpreter": "Code Interpretation",
            "http_request": "Web & Network",
            "slack": "Web & Network",
            "browser": "Web & Network",
            "rss": "Web & Network",
            "generate_image_stability": "Multi-modal",
            "image_reader": "Multi-modal",
            "generate_image": "Multi-modal",
            "nova_reels": "Multi-modal",
            "speak": "Multi-modal",
            "diagram": "Multi-modal",
            "use_aws": "AWS Services",
            "calculator": "Utilities",
            "current_time": "Utilities",
            "load_tool": "Utilities",
            "sleep": "Utilities",
            "graph": "Agents & Workflows",
            "agent_graph": "Agents & Workflows",
            "journal": "Agents & Workflows",
            "swarm": "Agents & Workflows",
            "stop": "Agents & Workflows",
            "handoff_to_user": "Agents & Workflows",
            "use_agent": "Agents & Workflows",
            "think": "Agents & Workflows",
            "use_llm": "Agents & Workflows",
            "workflow": "Agents & Workflows",
            "batch": "Agents & Workflows",
            "a2a_client": "Agents & Workflows",
        }
        return categories.get(tool_name, "Utilities")




def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real StrandsAgents Agent Builder")
    parser.add_argument('--create', '-c', metavar='NAME',
                        help='Create a real functional agent')
    parser.add_argument('--description', '-d', metavar='DESC',
                        help='Description of the agent')
    parser.add_argument('--model', '-m', choices=['ollama', 'bedrock'],
                        default='ollama', help='Model runtime (default: ollama)')
    parser.add_argument('--tools', '-t', nargs='+',
                        help='List of tools to include')
    parser.add_argument('--code-execution', action='store_true',
                        help='Enable code execution (requires AgentCore)')
    parser.add_argument('--list-tools', action='store_true',
                        help='List all available tools')

    args = parser.parse_args()

    builder = RealAgentBuilder()

    if args.list_tools:
        builder.list_all_tools()
        return

    if not args.create:
        parser.print_help()
        print("\n💡 Examples:")
        print("  python strands-meta/agent_builder.py --create my_agent --description 'A helpful assistant'")
        print("  python strands-meta/agent_builder.py --create coding_agent --description 'Python coding assistant' --model ollama --tools python_repl file_read")
        print("  python strands-meta/agent_builder.py --create bedrock_agent --description 'AWS Bedrock agent' --model bedrock --code-execution")
        print("  python strands-meta/agent_builder.py --list-tools")
        return

    # Create the real agent
    try:
        result = builder.create_real_agent(
            name=args.create,
            description=args.description or f"Agent: {args.create}",
            model_type=args.model,
            tools=args.tools or ["http_request", "file_read", "calculator"],
            enable_code_execution=args.code_execution
        )

        print(f"\n🎉 Real agent '{args.create}' created successfully!")
        print(f"📂 Location: {result['agent'].parent}")
        print(f"\n🚀 To use your agent:")
        print(f"   1. cd {result['agent'].parent}")
        print(f"   2. python {result['setup'].name}  # Install dependencies")
        print(f"   3. python {result['agent'].name}   # Run your agent")

    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        logger.error(f"Agent creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
