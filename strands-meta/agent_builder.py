#!/usr/bin/env python3
"""
Enhanced Agent Builder for StrandsAgents

A comprehensive script for creating, customizing, and managing AI agents using
the @agent decorator system. This script provides both interactive and
programmatic interfaces for agent creation.

Features:
- Interactive questionnaire for agent creation
- Copy and modify existing agents
- Full customization (model, prompt, tools, etc.)
- Batch agent creation
- Agent management and testing
- Integration with existing strands-meta infrastructure
"""

import json
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Import strands-meta components
try:
    # When run as module from parent directory
    from ..agent.agent_decorator import agent, list_agents, get_agent_info, get_agent_function, AGENT_REGISTRY
    from .meta_agent_builder import MetaAgentBuilder, AgentCreationSpec
    from ..agent.model_selector import get_best_model_for_task, list_available_models
    from ..agent.sandbox_executor import SandboxExecutor
    from ..agent.workflow_templates import list_workflow_templates
except ImportError:
    try:
        # When run as script from strands-meta directory
        from agent.agent_decorator import agent, list_agents, get_agent_info, get_agent_function, AGENT_REGISTRY
        from meta_agent_builder import MetaAgentBuilder, AgentCreationSpec
        from agent.model_selector import get_best_model_for_task, list_available_models
        from agent.sandbox_executor import SandboxExecutor
        from agent.workflow_templates import list_workflow_templates
    except ImportError:
        # Fallback for direct script execution from anywhere
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from agent.agent_decorator import agent, list_agents, get_agent_info, get_agent_function, AGENT_REGISTRY
        from meta_agent_builder import MetaAgentBuilder, AgentCreationSpec
        from agent.model_selector import get_best_model_for_task, list_available_models
        from agent.sandbox_executor import SandboxExecutor
        from agent.workflow_templates import list_workflow_templates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='🤖 %(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agent_builder")


class InteractiveAgentBuilder:
    """Interactive interface for creating agents"""

    def __init__(self):
        self.meta_builder = MetaAgentBuilder()
        self.sandbox = SandboxExecutor()
        self.output_dir = Path("assistants/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_agent_interactive(self) -> Dict[str, Path]:
        """Create an agent through interactive questionnaire"""

        print("\n🤖 Agent Builder - Interactive Mode")
        print("=" * 50)

        # Get agent name
        name = self._get_agent_name()

        # Get agent description
        description = self._get_agent_description()

        # Get agent type
        agent_type = self._get_agent_type()

        # Get model selection
        model_id = self._get_model_selection(agent_type)

        # Get tools configuration
        tools = self._get_tools_configuration()

        # Get system prompt
        system_prompt = self._get_system_prompt(name, description, agent_type)

        # Get code execution settings
        enable_code_execution, sandbox_timeout = self._get_code_execution_settings()

        # Create agent specification
        spec = AgentCreationSpec(
            name=name,
            description=description,
            agent_type=agent_type,
            model_id=model_id,
            tools=tools,
            system_prompt=system_prompt,
            enable_code_execution=enable_code_execution,
            sandbox_timeout=sandbox_timeout
        )

        # Generate and save agent
        return self._generate_and_save_agent(spec)

    def _get_agent_name(self) -> str:
        """Get agent name from user"""
        while True:
            name = input(
                "Enter agent name (function name, no spaces): ").strip()
            if not name:
                print("❌ Agent name is required")
                continue
            if not name.replace('_', '').isalnum():
                print("❌ Agent name must contain only letters, numbers, and underscores")
                continue
            if name in AGENT_REGISTRY:
                print(
                    f"⚠️  Agent '{name}' already exists. Choose a different name.")
                continue
            return name

    def _get_agent_description(self) -> str:
        """Get agent description from user"""
        print("\n📝 Agent Description:")
        print("Describe what this agent should do, its purpose, and capabilities.")
        print("Example: 'A Python coding assistant that can write, debug, and test code'")
        description = input("Description: ").strip()
        return description or "A specialized AI agent"

    def _get_agent_type(self) -> str:
        """Get agent type from user"""
        print("\n🎯 Agent Type:")
        print("Available types:")
        print("  1. research - Information gathering and analysis")
        print("  2. coding - Software development and code generation")
        print("  3. data_analysis - Data processing and analytics")
        print("  4. creative - Content generation and creative tasks")
        print("  5. workflow - Multi-agent process orchestration")
        print("  6. meta - Agent creation and management")
        print("  7. auto - Let the system analyze and choose")

        while True:
            choice = input("Select type (1-7) or 'auto': ").strip().lower()
            if choice == 'auto':
                return 'auto'
            if choice in ['1', '2', '3', '4', '5', '6']:
                types = ['research', 'coding', 'data_analysis',
                         'creative', 'workflow', 'meta']
                return types[int(choice) - 1]
            print("❌ Invalid choice. Please select 1-7 or 'auto'")

    def _get_model_selection(self, agent_type: str) -> str:
        """Get model selection from user"""
        print(f"\n🔧 Model Selection for {agent_type}:")

        # Show available models
        available_models = list_available_models()
        if not available_models:
            print("⚠️  No models available, using default")
            return "llama3.2"

        print("Available models:")
        for i, model in enumerate(available_models, 1):
            print(
                f"  {i}. {model['name']} ({model['size']}) - {', '.join(model['capabilities'][:3])}")

        # Get best model for task type
        best_model = get_best_model_for_task(agent_type)
        print(f"\n💡 Recommended model for {agent_type}: {best_model}")

        while True:
            choice = input(
                f"Select model (1-{len(available_models)}) or press Enter for recommended: ").strip()
            if not choice:
                return best_model
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                return available_models[int(choice) - 1]['name']
            print(
                f"❌ Invalid choice. Please select 1-{len(available_models)} or press Enter")

    def _get_tools_configuration(self) -> List[str]:
        """Get tools configuration from user"""
        print("\n🛠️  Tools Configuration:")

        # Common tool categories
        tool_categories = {
            "Web & API": ["http_request", "browser"],
            "File Operations": ["file_read", "file_write", "file_search"],
            "Code Execution": ["python_repl", "shell", "javascript"],
            "Data Processing": ["calculator", "data_analysis"],
            "Communication": ["email", "slack"],
            "Productivity": ["todo", "calendar", "notes"]
        }

        selected_tools = []

        print("Select tools by category (or 'custom' for specific tools):")
        for i, (category, tools) in enumerate(tool_categories.items(), 1):
            print(f"  {i}. {category}: {', '.join(tools)}")

        print("  7. Custom tools")
        print("  8. No tools")

        while True:
            choice = input(
                "Select category (1-8) or 'done' to finish: ").strip().lower()

            if choice == 'done':
                break
            elif choice == '8':
                return []
            elif choice == '7':
                custom_tools = input(
                    "Enter custom tools (comma-separated): ").strip()
                if custom_tools:
                    selected_tools.extend([t.strip()
                                          for t in custom_tools.split(',')])
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                category_tools = list(tool_categories.values())[
                    int(choice) - 1]
                selected_tools.extend(category_tools)
                print(
                    f"✅ Added {len(category_tools)} tools from {list(tool_categories.keys())[int(choice) - 1]}")
            else:
                print("❌ Invalid choice")

        return list(set(selected_tools))  # Remove duplicates

    def _get_system_prompt(self, name: str, description: str, agent_type: str) -> str:
        """Get or generate system prompt"""
        print(f"\n📝 System Prompt for {name}:")

        # Generate default prompt based on type
        default_prompts = {
            "research": f"""You are {name}, a specialized research agent.

{description}

Your capabilities include:
- Web research and information gathering
- Source citation and validation
- Data analysis and pattern recognition
- Report generation with confidence levels

Guidelines:
- Always cite sources for factual claims
- Distinguish between facts and assumptions
- Provide balanced, objective analysis
- Acknowledge uncertainties and limitations""",

            "coding": f"""You are {name}, a specialized software development agent.

{description}

Your capabilities include:
- Full-stack development (Python, JavaScript, etc.)
- Code review and optimization
- Testing and debugging
- Architecture design and documentation

Guidelines:
- Write clean, maintainable code
- Follow best practices and conventions
- Include comprehensive documentation
- Test thoroughly before deployment""",

            "data_analysis": f"""You are {name}, a specialized data analysis agent.

{description}

Your capabilities include:
- Data cleaning and preprocessing
- Statistical analysis and modeling
- Data visualization and reporting
- Pattern recognition and trend analysis

Guidelines:
- Validate data quality and integrity
- Use appropriate statistical methods
- Create clear, informative visualizations
- Provide actionable insights""",

            "creative": f"""You are {name}, a specialized creative agent.

{description}

Your capabilities include:
- Creative writing and content generation
- Brainstorming and ideation
- Style adaptation and tone matching
- Artistic and design concepts

Guidelines:
- Generate original, engaging content
- Adapt to specified styles and tones
- Provide multiple creative options
- Balance innovation with practicality""",

            "workflow": f"""You are {name}, a specialized workflow orchestration agent.

{description}

Your capabilities include:
- Process planning and optimization
- Task decomposition and sequencing
- Resource allocation and scheduling
- Quality control and validation

Guidelines:
- Break complex tasks into manageable steps
- Identify dependencies and critical paths
- Monitor progress and adjust as needed
- Ensure quality at each stage""",

            "meta": f"""You are {name}, a specialized meta-agent for creating and managing other agents.

{description}

Your capabilities include:
- Agent design and architecture
- Tool selection and integration
- Performance optimization
- System integration and testing

Guidelines:
- Design agents for specific, well-defined purposes
- Select appropriate models and tools
- Ensure robust error handling
- Create comprehensive documentation"""
        }

        default_prompt = default_prompts.get(agent_type, f"""You are {name}, a specialized AI agent.

{description}

Focus on delivering high-quality, specialized responses for your domain.""")

        print(f"💡 Generated default prompt for {agent_type}:")
        print("-" * 40)
        print(default_prompt[:200] + "..." if len(default_prompt) > 200 else default_prompt)
        print("-" * 40)

        use_custom = input("Use this prompt? (y/n): ").strip().lower()
        if use_custom == 'n':
            print("Enter your custom system prompt:")
            return input().strip()
        return default_prompt

    def _get_code_execution_settings(self) -> tuple[bool, int]:
        """Get code execution settings from user"""
        print("\n⚡ Code Execution Settings:")

        enable_code = input("Enable code execution? (y/n): ").strip().lower() == 'y'
        if not enable_code:
            return False, 30

        timeout = input("Sandbox timeout in seconds (default 30): ").strip()
        timeout = int(timeout) if timeout.isdigit() else 30

        return True, timeout

    def _generate_and_save_agent(self, spec: AgentCreationSpec) -> Dict[str, Path]:
        """Generate and save the agent"""
        print(f"\n🔧 Generating agent '{spec.name}'...")

        # Use the meta builder to create the agent
        result = self.meta_builder.create_agent(
            spec.name, spec.description, spec.agent_type
        )

        # Also test the meta agent builder function directly
        try:
            meta_response = meta_agent_builder(f"Create a {spec.agent_type} agent for {spec.description}")
            print(f"🤖 Meta-agent response: {meta_response[:100]}...")
        except Exception as e:
            print(f"⚠️  Meta-agent test failed: {str(e)}")

        print("✅ Agent generated successfully!")
        print(f"📁 Files created:")
        for file_type, file_path in result.items():
            print(f"   • {file_type}: {file_path}")

        return result

    def copy_and_modify_agent(self, existing_name: str) -> Dict[str, Path]:
        """Copy and modify an existing agent"""
        print(f"\n📋 Copy and Modify Agent: {existing_name}")
        print("=" * 50)

        if existing_name not in AGENT_REGISTRY:
            print(f"❌ Agent '{existing_name}' not found")
            return {}

        # Get existing agent info
        existing_info = get_agent_info(existing_name)
        if not existing_info:
            print(f"❌ Could not get info for agent '{existing_name}'")
            return {}

        # Get new name
        new_name = input(f"Enter new agent name (current: {existing_name}): ").strip()
        if not new_name:
            new_name = f"{existing_name}_copy"

        # Get new description
        current_desc = existing_info.get('description', 'No description')
        print(f"Current description: {current_desc}")
        new_description = input("Enter new description (press Enter to keep current): ").strip()
        if not new_description:
            new_description = current_desc

        # Get new model
        current_model = existing_info.get('model_id', 'unknown')
        print(f"Current model: {current_model}")
        new_model = input("Enter new model (press Enter to keep current): ").strip()
        if not new_model:
            new_model = current_model

        # Get new tools
        current_tools = existing_info.get('tools', [])
        print(f"Current tools: {', '.join(current_tools) if current_tools else 'None'}")
        modify_tools = input("Modify tools? (y/n): ").strip().lower() == 'y'
        new_tools = current_tools.copy()
        if modify_tools:
            new_tools = self._get_tools_configuration()

        # Get new system prompt
        print(f"Current system prompt: {existing_info.get('system_prompt', 'Default')[:100]}...")
        modify_prompt = input("Modify system prompt? (y/n): ").strip().lower() == 'y'
        if modify_prompt:
            new_prompt = self._get_system_prompt(new_name, new_description, 'auto')
        else:
            new_prompt = existing_info.get('system_prompt', '')

        # Create new specification
        spec = AgentCreationSpec(
            name=new_name,
            description=new_description,
            agent_type='auto',  # Will be auto-detected
            model_id=new_model,
            tools=new_tools,
            system_prompt=new_prompt,
            enable_code_execution=existing_info.get('enable_code_execution', True),
            sandbox_timeout=existing_info.get('sandbox_timeout', 30)
        )

        # Generate and save agent
        return self._generate_and_save_agent(spec)

    def list_agents(self):
        """List all available agents"""
        print("\n🤖 Available Agents:")
        print("=" * 50)

        agents = list_agents()
        if not agents:
            print("No agents registered")
            return

        for agent_name in agents:
            info = get_agent_info(agent_name)
            if info:
                print(f"\n🔧 {agent_name}:")
                print(f"   Model: {info.get('model_id', 'Unknown')}")
                print(f"   Tools: {info.get('tools', [])}")
                print(f"   Code Execution: {info.get('enable_code_execution', False)}")
                print(f"   Description: {info.get('description', 'No description')[:100]}...")

    def test_agent(self, agent_name: str):
        """Test an agent with sample queries"""
        print(f"\n🧪 Testing Agent: {agent_name}")
        print("=" * 50)

        if agent_name not in AGENT_REGISTRY:
            print(f"❌ Agent '{agent_name}' not found")
            return

        # Get test queries based on agent type
        agent_info = get_agent_info(agent_name)
        agent_type = 'general'
        if agent_info and 'description' in agent_info:
            desc = agent_info['description'].lower()
            if any(word in desc for word in ['code', 'program', 'develop']):
                agent_type = 'coding'
            elif any(word in desc for word in ['research', 'analyze', 'data']):
                agent_type = 'research'

        # Sample queries for different agent types
        test_queries = {
            'coding': [
                "Write a Python function to calculate fibonacci numbers",
                "Debug this code: print('hello world'",
                "Create a simple web server in Python"
            ],
            'research': [
                "What are the latest developments in AI?",
                "Explain quantum computing in simple terms",
                "What are the benefits of renewable energy?"
            ],
            'general': [
                "Hello, how are you?",
                "What can you help me with?",
                "Explain your capabilities"
            ]
        }

        queries = test_queries.get(agent_type, test_queries['general'])

        for i, query in enumerate(queries, 1):
            print(f"\n📝 Test {i}: {query}")
            try:
                agent_func = get_agent_function(agent_name)
                if agent_func:
                    result = agent_func(query)
                    print(f"✅ Response: {result[:200]}{'...' if len(result) > 200 else ''}")
                else:
                    print("❌ Could not get agent function")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def run_batch_creation(self, config_file: str):
        """Run batch agent creation from config file"""
        print(f"\n📦 Batch Agent Creation from: {config_file}")
        print("=" * 50)

        if not os.path.exists(config_file):
            print(f"❌ Config file '{config_file}' not found")
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            agents_config = config.get('agents', [])
            if not agents_config:
                print("❌ No agents defined in config file")
                return

            print(f"Creating {len(agents_config)} agents...")

            for agent_config in agents_config:
                try:
                    spec = AgentCreationSpec(**agent_config)
                    result = self._generate_and_save_agent(spec)
                    print(f"✅ Created agent: {spec.name}")
                except Exception as e:
                    print(f"❌ Failed to create agent {agent_config.get('name', 'unknown')}: {str(e)}")

        except Exception as e:
            print(f"❌ Error reading config file: {str(e)}")


class AgentManager:
    """Agent management utilities"""

    def __init__(self):
        self.output_dir = Path("assistants/generated")

    def list_generated_agents(self):
        """List all generated agents"""
        print("\n📋 Generated Agents:")
        print("=" * 50)

        if not self.output_dir.exists():
            print("No generated agents directory found")
            return

        agent_files = list(self.output_dir.glob("*_metadata.json"))
        if not agent_files:
            print("No generated agents found")
            return

        for metadata_file in sorted(agent_files):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                print(f"\n🤖 {metadata['name']}:")
                print(f"   Type: {metadata['agent_type']}")
                print(f"   Model: {metadata['model_id']}")
                print(f"   Tools: {', '.join(metadata['tools'])}")
                print(f"   Created: {metadata['created_at']}")
                print(f"   File: {metadata['file_path']}")

            except Exception as e:
                print(f"❌ Error reading {metadata_file}: {str(e)}")

    def validate_agent(self, agent_name: str):
        """Validate a generated agent"""
        print(f"\n🔍 Validating Agent: {agent_name}")
        print("=" * 50)

        # Find agent file
        agent_file = self.output_dir / f"{agent_name}.py"
        metadata_file = self.output_dir / f"{agent_name}_metadata.json"

        if not agent_file.exists():
            print(f"❌ Agent file not found: {agent_file}")
            return False

        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return False

        # Check if agent can be imported and used
        try:
            # Read and check syntax
            with open(agent_file, 'r') as f:
                code = f.read()

            # Basic syntax check
            compile(code, agent_file, 'exec')

            # Check for required components
            required_parts = ['@agent', 'def ' + agent_name]
            for part in required_parts:
                if part not in code:
                    print(f"❌ Missing required component: {part}")
                    return False

            print("✅ Agent file syntax is valid")
            print("✅ Required components found")

            # Try to load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print("✅ Metadata is valid JSON")
            print(f"✅ Agent type: {metadata.get('agent_type', 'unknown')}")
            print(f"✅ Model: {metadata.get('model_id', 'unknown')}")

            return True

        except SyntaxError as e:
            print(f"❌ Syntax error in agent file: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Error validating agent: {str(e)}")
            return False


def main():
    """Main entry point for the agent builder"""
    parser = argparse.ArgumentParser(description="StrandsAgents Agent Builder")
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive agent creation')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available agents')
    parser.add_argument('--list-generated', action='store_true',
                       help='List all generated agents')
    parser.add_argument('--copy', '-c', metavar='AGENT_NAME',
                       help='Copy and modify existing agent')
    parser.add_argument('--test', '-t', metavar='AGENT_NAME',
                       help='Test an agent with sample queries')
    parser.add_argument('--validate', '-v', metavar='AGENT_NAME',
                       help='Validate a generated agent')
    parser.add_argument('--batch', '-b', metavar='CONFIG_FILE',
                       help='Create agents from batch config file')
    parser.add_argument('--create', '-n', metavar='NAME',
                       help='Create agent with name (non-interactive)')

    args = parser.parse_args()

    # Initialize components
    builder = InteractiveAgentBuilder()
    manager = AgentManager()

    try:
        if args.interactive:
            builder.create_agent_interactive()

        elif args.list:
            builder.list_agents()

        elif args.list_generated:
            manager.list_generated_agents()

        elif args.copy:
            builder.copy_and_modify_agent(args.copy)

        elif args.test:
            builder.test_agent(args.test)

        elif args.validate:
            success = manager.validate_agent(args.validate)
            sys.exit(0 if success else 1)

        elif args.batch:
            builder.run_batch_creation(args.batch)

        elif args.create:
            print(f"Creating agent '{args.create}' (non-interactive mode)")
            print("Note: Use --interactive for full customization")
            # Simple creation - would need more parameters for full customization
            result = builder.meta_builder.create_agent(args.create, f"Agent created: {args.create}")
            print(f"✅ Created agent: {args.create}")

        else:
            parser.print_help()
            print("\n💡 Examples:")
            print("  python agent_builder.py --interactive")
            print("  python agent_builder.py --list")
            print("  python agent_builder.py --copy my_agent")
            print("  python agent_builder.py --test my_agent")
            print("  python agent_builder.py --batch agents.json")

    except KeyboardInterrupt:
        print("\n\n👋 Agent builder interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error in agent builder: {str(e)}")
        logger.error(f"Agent builder error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
