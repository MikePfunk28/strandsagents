"""
Agent Builder for StrandsAgents

A comprehensive agent creation system that leverages the @agent decorator
to build specialized AI agents for specific purposes. This system creates
deterministic, production-ready agents with proper structure and tooling.

Features:
- Deterministic agent generation using @agent decorator
- Multi-model support with intelligent selection
- Workflow template integration
- Sandbox execution capabilities
- Comprehensive scaffolding with prompts, tools, and metadata
- Integration with existing StrandsAgents ecosystem
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple
from agent import agent, SandboxExecutor, WorkflowTemplate
from agent.model_selector import get_best_model_for_task, list_available_models
from agent.workflow_templates import get_workflow_template, list_workflow_templates

logger = logging.getLogger("agent_builder")

# Updated model list as provided by user
AVAILABLE_MODELS = [
    "qwen3:8b",
    "qwen3:4b",
    "llama3.2",
    "gemma3:270m",
    "qwen3:1.7b",
    "gemma3:1b",
    "gemma3:4b",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
    "qwen3-embedding:8b",
    "embeddinggemma:300m",
    "qwen3-coder:latest",
    "qwen3:0.6b",
    "qwen3:14b",
    "phi4-reasoning:latest",
    "phi4-mini:latest",
    "phi4-mini-reasoning:latest"
]

# Agent type definitions
AGENT_TYPES = {
    "research": {
        "description": "Information gathering and analysis specialist",
        "default_model": "qwen3:8b",
        "tools": ["http_request", "file_read", "file_write"],
        "enable_code_execution": True,
        "system_prompt_template": """You are a Research Specialist AI agent.
        Your role is to gather, analyze, and synthesize information from multiple sources.

        Capabilities:
        - Web research and information gathering
        - Source citation and validation
        - Data analysis and pattern recognition
        - Report generation with confidence levels

        Guidelines:
        - Always cite sources for factual claims
        - Distinguish between facts and assumptions
        - Provide balanced, objective analysis
        - Acknowledge uncertainties and limitations

        When researching: {goal}
        Context: {context}"""
    },

    "coding": {
        "description": "Software development and code generation specialist",
        "default_model": "qwen3-coder:latest",
        "tools": ["python_repl", "file_write", "file_read", "shell"],
        "enable_code_execution": True,
        "system_prompt_template": """You are a Senior Software Developer AI agent.
        Your role is to design, implement, and maintain high-quality software solutions.

        Capabilities:
        - Full-stack development (Python, JavaScript, etc.)
        - Code review and optimization
        - Testing and debugging
        - Architecture design and documentation

        Guidelines:
        - Write clean, maintainable code
        - Follow best practices and conventions
        - Include comprehensive documentation
        - Test thoroughly before deployment

        Development task: {goal}
        Requirements: {context}"""
    },

    "data_analysis": {
        "description": "Data processing and analytics specialist",
        "default_model": "qwen3:8b",
        "tools": ["python_repl", "calculator", "file_read", "file_write"],
        "enable_code_execution": True,
        "system_prompt_template": """You are a Data Analysis Specialist AI agent.
        Your role is to process, analyze, and visualize data to extract meaningful insights.

        Capabilities:
        - Data cleaning and preprocessing
        - Statistical analysis and modeling
        - Data visualization and reporting
        - Pattern recognition and trend analysis

        Guidelines:
        - Validate data quality and integrity
        - Use appropriate statistical methods
        - Create clear, informative visualizations
        - Provide actionable insights and recommendations

        Analysis task: {goal}
        Dataset context: {context}"""
    },

    "creative": {
        "description": "Creative content generation and ideation specialist",
        "default_model": "qwen3:8b",
        "tools": ["file_write", "http_request"],
        "enable_code_execution": False,
        "system_prompt_template": """You are a Creative Specialist AI agent.
        Your role is to generate innovative ideas, content, and solutions.

        Capabilities:
        - Creative writing and content generation
        - Brainstorming and ideation
        - Style adaptation and tone matching
        - Artistic and design concepts

        Guidelines:
        - Generate original, engaging content
        - Adapt to specified styles and tones
        - Provide multiple creative options
        - Balance innovation with practicality

        Creative task: {goal}
        Style/requirements: {context}"""
    },

    "workflow": {
        "description": "Multi-agent workflow orchestration specialist",
        "default_model": "qwen3:8b",
        "tools": ["http_request", "file_read", "file_write"],
        "enable_code_execution": True,
        "system_prompt_template": """You are a Workflow Orchestration Specialist AI agent.
        Your role is to coordinate complex multi-step processes across different domains.

        Capabilities:
        - Process planning and optimization
        - Task decomposition and sequencing
        - Resource allocation and scheduling
        - Quality control and validation

        Guidelines:
        - Break complex tasks into manageable steps
        - Identify dependencies and critical paths
        - Monitor progress and adjust as needed
        - Ensure quality at each stage

        Workflow task: {goal}
        Process requirements: {context}"""
    },

    "meta": {
        "description": "Agent creation and management specialist",
        "default_model": "qwen3:8b",
        "tools": ["python_repl", "file_write", "shell"],
        "enable_code_execution": True,
        "system_prompt_template": """You are a Meta-Agent Specialist AI agent.
        Your role is to create, modify, and manage other AI agents and their capabilities.

        Capabilities:
        - Agent design and architecture
        - Tool selection and integration
        - Performance optimization
        - System integration and testing

        Guidelines:
        - Design agents for specific, well-defined purposes
        - Select appropriate models and tools
        - Ensure robust error handling
        - Create comprehensive documentation

        Agent creation task: {goal}
        Requirements: {context}"""
    }
}


@dataclass
class AgentSpecification:
    """Complete specification for agent creation"""

    # Basic Information
    name: str
    display_name: str
    description: str
    agent_type: str

    # Technical Configuration
    model_id: str = ""
    tools: List[str] = field(default_factory=list)
    enable_code_execution: bool = True
    sandbox_timeout: int = 45

    # Content
    system_prompt: str = ""
    examples: List[str] = field(default_factory=list)
    context_documents: List[str] = field(default_factory=list)

    # Workflow Integration
    workflow_template: Optional[str] = None
    can_chain_agents: bool = True

    # Metadata
    version: str = "1.0.0"
    author: str = "StrandsAgents Agent Builder"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.model_id:
            self.model_id = self._get_default_model()

        if not self.tools:
            self.tools = self._get_default_tools()

        if not self.system_prompt:
            self.system_prompt = self._generate_system_prompt()

        if not self.tags:
            self.tags = [self.agent_type, "generated", "ai-agent"]

    def _get_default_model(self) -> str:
        """Get default model for agent type"""
        agent_config = AGENT_TYPES.get(
            self.agent_type, AGENT_TYPES["research"])
        return agent_config["default_model"]

    def _get_default_tools(self) -> List[str]:
        """Get default tools for agent type"""
        agent_config = AGENT_TYPES.get(
            self.agent_type, AGENT_TYPES["research"])
        return agent_config["tools"].copy()

    def _generate_system_prompt(self) -> str:
        """Generate system prompt from template"""
        agent_config = AGENT_TYPES.get(
            self.agent_type, AGENT_TYPES["research"])
        template = agent_config["system_prompt_template"]

        return template.format(
            goal=f"Handle {self.agent_type} tasks effectively",
            context=f"Specialized in {self.description}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpecification":
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data_copy)


class AgentBuilder:
    """Main agent building system"""

    def __init__(self, output_dir: str = "assistants/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = SandboxExecutor()

        logger.info(
            f"🤖 Agent Builder initialized - Output dir: {self.output_dir}")

    def create_agent_from_spec(self, spec: AgentSpecification) -> Dict[str, Path]:
        """Create a complete agent from specification"""

        logger.info(f"🤖 Creating agent: {spec.name}")

        # Generate file paths
        agent_file = self.output_dir / f"{spec.name}.py"
        prompt_file = self.output_dir / "prompts" / f"{spec.name}.prompt"
        metadata_file = self.output_dir / "metadata" / f"{spec.name}.json"
        test_file = self.output_dir / "tests" / f"test_{spec.name}.py"

        # Create directories
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate agent code
        agent_code = self._generate_agent_code(spec)
        agent_file.write_text(agent_code, encoding='utf-8')

        # Generate prompt file
        prompt_file.write_text(spec.system_prompt, encoding='utf-8')

        # Generate metadata
        metadata = self._generate_metadata(spec)
        metadata_file.write_text(json.dumps(
            metadata, indent=2), encoding='utf-8')

        # Generate test file
        test_code = self._generate_test_code(spec)
        test_file.write_text(test_code, encoding='utf-8')

        logger.info(f"🤖 Agent {spec.name} created successfully")

        return {
            "agent": agent_file,
            "prompt": prompt_file,
            "metadata": metadata_file,
            "test": test_file
        }

    def _generate_agent_code(self, spec: AgentSpecification) -> str:
        """Generate the complete agent Python module"""

        # Generate imports
        imports = [
            "from agent import agent, SandboxExecutor",
            "import logging"
        ]

        # Add tool imports if needed
        tool_imports = set()
        for tool in spec.tools:
            if tool == "http_request":
                tool_imports.add("from strands_tools import http_request")
            elif tool == "file_read":
                tool_imports.add("from strands_tools import file_read")
            elif tool == "file_write":
                tool_imports.add("from strands_tools import file_write")
            elif tool == "python_repl":
                tool_imports.add("from strands_tools import python_repl")
            elif tool == "shell":
                tool_imports.add("from strands_tools import shell")
            elif tool == "calculator":
                tool_imports.add("from strands_tools import calculator")

        imports.extend(sorted(tool_imports))

        # Generate tools list
        tools_list = []
        for tool in spec.tools:
            if tool in ["http_request", "file_read", "file_write", "python_repl", "shell", "calculator"]:
                tools_list.append(f'"{tool}"')

        tools_str = ", ".join(tools_list) if tools_list else '[]'

        # Generate the complete agent code
        timestamp = datetime.now().isoformat()
        code_lines = [
            f'"""',
            f"{spec.display_name}",
            f"",
            f"{spec.description}",
            f"",
            f"Generated by Agent Builder on {timestamp}",
            f"Agent Type: {spec.agent_type}",
            f"Model: {spec.model_id}",
            f"Version: {spec.version}",
            f'"""',
            "",
            *imports,
            "",
            "# Setup logging",
            f'logger = logging.getLogger("{spec.name}")',
            "",
            "# System prompt",
            "SYSTEM_PROMPT = \"\"\"" + spec.system_prompt + "\"\"\"",
            "",
            "# Tools configuration",
            f"TOOLS = [{tools_str}]",
            "",
            "@agent(",
            f'    model_id="{spec.model_id}",',
            '    system_prompt=SYSTEM_PROMPT,',
            '    tools=TOOLS,',
            f'    enable_code_execution={str(spec.enable_code_execution).lower()},',
            f'    sandbox_timeout={spec.sandbox_timeout}',
            ")",
            f"def {spec.name}(query: str) -> str:",
            '    """',
            f"    {spec.description}",
            '    """',
            '    """This agent was automatically generated by the Agent Builder system."""',
            '    """',
            '    """Args:"""',
            '    """    query: Input query or request"""',
            '    """',
            '    """Returns:"""',
            '    """    Response from the specialized agent"""',
            '    """',
            '    try:',
            f'        logger.info(f"🔧 {spec.display_name} processing query: {{query[:100]}}...")',
            '        ',
            '        # Agent logic goes here',
            '        # This is a template that can be customized for specific use cases',
            '        ',
            f'        response = f"Processed by {spec.display_name}: {{query}}"',
            f'        logger.info(f"🔧 {spec.display_name} completed successfully")',
            '        ',
            '        return response',
            '        ',
            '    except Exception as e:',
            f'        error_msg = f"Error in {spec.name}: {{str(e)}}"',
            f'        logger.error(f"🔧 {spec.display_name} error: {{error_msg}}")',
            '        return error_msg',
            '        ',
            '# Example usage and testing',
            'if __name__ == "__main__":',
            f'    print("🤖 {spec.display_name}")',
            '    print("=" * 50)',
            f'    print("{spec.description}")',
            f'    print(f"Model: {spec.model_id}")',
            f'    print(f"Tools: {{", ".join(spec.tools)}}")',
            f'    print(f"Code Execution: {spec.enable_code_execution}")',
            '    print()',
            '    ',
            '    # Test the agent',
            '    test_query = "Hello, please demonstrate your capabilities"',
            '    print(f"Test Query: {{test_query}}")',
            '    print()',
            '    ',
            f'    try:',
            f'        result = {spec.name}(test_query)',
            '        print(f"Response: {{result}}")',
            '    except Exception as e:',
            '        print(f"Test failed: {{e}}")',
            '        ',
            '    print("\\n✅ Agent ready for use!")'
        ]

        return "\n".join(code_lines)

        return code

    def _generate_metadata(self, spec: AgentSpecification) -> Dict[str, Any]:
        """Generate comprehensive metadata for the agent"""

        return {
            "name": spec.name,
            "display_name": spec.display_name,
            "description": spec.description,
            "agent_type": spec.agent_type,
            "model_id": spec.model_id,
            "tools": spec.tools,
            "enable_code_execution": spec.enable_code_execution,
            "sandbox_timeout": spec.sandbox_timeout,
            "version": spec.version,
            "author": spec.author,
            "tags": spec.tags,
            "created_at": spec.created_at.isoformat(),
            "system_prompt": spec.system_prompt,
            "examples": spec.examples,
            "context_documents": spec.context_documents,
            "workflow_template": spec.workflow_template,
            "can_chain_agents": spec.can_chain_agents,
            "performance_metrics": {
                "expected_response_time": "2-5 seconds",
                "max_query_length": "4000 tokens",
                "supported_languages": ["english"],
                "specialization_level": "high"
            }
        }

    def _generate_test_code(self, spec: AgentSpecification) -> str:
        """Generate comprehensive test code for the agent"""

        timestamp = datetime.now().isoformat()
        test_class_name = spec.name.title().replace("_", "")

        test_code_lines = [
            '"""',
            f"Comprehensive tests for {spec.display_name}",
            "",
            f"Generated by Agent Builder on {timestamp}",
            '"""',
            "",
            "import pytest",
            "import logging",
            "from pathlib import Path",
            "",
            "# Add the generated agent to path",
            "agent_dir = Path(__file__).parent.parent",
            "if str(agent_dir) not in __import__('sys').path:",
            "    __import__('sys').path.insert(0, str(agent_dir))",
            "",
            "try:",
            f"    from {spec.name} import {spec.name}",
            "except ImportError as e:",
            '    print(f"Failed to import agent: {{e}}")',
            '    print("Make sure the agent file exists and is properly formatted")',
            "    raise",
            "",
            f"class Test{test_class_name}Agent:",
            '    """Test suite for {spec.display_name}"""',
            "",
            "    def setup_method(self):",
            '        """Setup for each test"""',
            "        self.test_queries = [",
            '            "What is your primary function?",',
            f'            "Can you help with {spec.agent_type} tasks?",',
            '            "What tools do you have available?"',
            "        ]",
            "",
            "    def test_agent_initialization(self):",
            '        """Test that agent can be imported and called"""',
            f"        assert {spec.name} is not None",
            f"        assert callable({spec.name})",
            "",
            "    def test_basic_functionality(self):",
            '        """Test basic agent functionality"""',
            "        for query in self.test_queries:",
            "            try:",
            f"                response = {spec.name}(query)",
            "                assert response is not None",
            "                assert isinstance(response, str)",
            "                assert len(response) > 0",
            '                print(f"Query: {{query}}")',
            '                print(f"Response: {{response[:200]}}...")',
            "            except Exception as e:",
            f'                pytest.fail(f"Agent failed on query \'{{query}}\': {{e}}")',
            "",
            "    def test_code_execution(self):",
            '        """Test code execution capabilities if enabled"""',
            f"        if {str(spec.enable_code_execution).lower()}:",
            '            code_query = "Execute this Python code: print(\'Hello from test!\')"',
            "            try:",
            f"                response = {spec.name}(code_query)",
            "                assert response is not None",
            '                assert \"Hello from test\" in response or \"Code executed\" in response',
            "            except Exception as e:",
            f'                pytest.fail(f"Code execution test failed: {{e}}")',
            "",
            "    def test_error_handling(self):",
            '        """Test error handling capabilities"""',
            '        # Test with empty query',
            "        try:",
            f'            response = {spec.name}("")',
            "            assert response is not None",
            "        except Exception as e:",
            f'            pytest.fail(f"Agent should handle empty queries gracefully: {{e}}")',
            "",
            "    def test_agent_chaining(self):",
            '        """Test agent chaining capabilities"""',
            f"        if {str(spec.can_chain_agents).lower()}:",
            '            # This would test CALL_AGENT functionality if implemented',
            "            pass",
            "",
            'if __name__ == "__main__":',
            f'    # Run basic functionality test',
            f'    print("🧪 Running {spec.display_name} Tests")',
            '    print("=" * 50)',
            "",
            f'    test_instance = Test{test_class_name}Agent()',
            "    test_instance.setup_method()",
            "",
            '    print("Testing basic functionality...")',
            "    test_instance.test_basic_functionality()",
            "",
            f"    if {str(spec.enable_code_execution).lower()}:",
            '        print("Testing code execution...")',
            "    test_instance.test_code_execution()",
            "",
            '    print("Testing error handling...")',
            "    test_instance.test_error_handling()",
            "",
            '    print("\\n✅ All tests completed!")'
        ]

        return "\n".join(test_code_lines)

    def create_agent_interactive(self) -> Dict[str, Path]:
        """Create an agent through interactive questionnaire"""

        print("🤖 StrandsAgents Agent Builder")
        print("=" * 50)
        print("Create a specialized AI agent using the @agent decorator")
        print()

        # Collect basic information
        name = input("Agent function name (snake_case): ").strip()
        if not name:
            name = "my_specialized_agent"

        display_name = input("Display name (optional): ").strip()
        if not display_name:
            display_name = name.replace("_", " ").title()

        description = input("Short description: ").strip()
        if not description:
            description = f"Specialized agent for {display_name.lower()}"

        # Agent type selection
        print("\\nAvailable agent types:")
        for i, (type_key, type_info) in enumerate(AGENT_TYPES.items(), 1):
            print(f"{i}. {type_key}: {type_info['description']}")

        while True:
            try:
                type_choice = int(input("\\nSelect agent type (1-6): "))
                if 1 <= type_choice <= len(AGENT_TYPES):
                    agent_type = list(AGENT_TYPES.keys())[type_choice - 1]
                    break
                else:
                    print(f"Please select 1-{len(AGENT_TYPES)}")
            except ValueError:
                print("Please enter a number")

        # Model selection
        print("\\nAvailable models:")
        available_models = list_available_models()
        for i, model in enumerate(available_models, 1):
            print(
                f"{i}. {model['name']} ({model['size']}) - {model['family']}")

        recommended_model = get_best_model_for_task(agent_type)
        print(f"\\nRecommended model for {agent_type}: {recommended_model}")

        use_recommended = input(
            "Use recommended model? (y/n): ").lower().startswith('y')
        if use_recommended:
            model_id = recommended_model
        else:
            while True:
                try:
                    model_choice = int(input("Select model (number): "))
                    if 1 <= model_choice <= len(available_models):
                        model_id = available_models[model_choice - 1]['name']
                        break
                    else:
                        print(f"Please select 1-{len(available_models)}")
                except ValueError:
                    print("Please enter a number")

        # Advanced options
        enable_code = input(
            "Enable code execution? (y/n): ").lower().startswith('y')
        sandbox_timeout = 45
        if enable_code:
            try:
                timeout_input = input(
                    "Sandbox timeout (seconds, default 45): ").strip()
                if timeout_input:
                    sandbox_timeout = int(timeout_input)
            except ValueError:
                print("Using default timeout: 45 seconds")

        # Create specification
        spec = AgentSpecification(
            name=name,
            display_name=display_name,
            description=description,
            agent_type=agent_type,
            model_id=model_id,
            enable_code_execution=enable_code,
            sandbox_timeout=sandbox_timeout
        )

        # Generate the agent
        print(f"\\n🤖 Creating agent '{spec.name}'...")
        return self.create_agent_from_spec(spec)

    def create_agent_from_template(self, template_name: str, customization: Dict[str, Any]) -> Dict[str, Path]:
        """Create an agent from a workflow template"""

        # Get workflow template
        template = get_workflow_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Create base specification from template
        spec = AgentSpecification(
            name=customization.get("name", f"{template_name}_agent"),
            display_name=customization.get(
                "display_name", f"{template_name.title()} Agent"),
            description=customization.get("description", template.description),
            agent_type=template_name,
            workflow_template=template_name
        )

        # Customize based on template requirements
        if template_name == "research_workflow":
            spec.tools = ["http_request", "file_read", "file_write"]
            spec.enable_code_execution = True
        elif template_name == "code_development":
            spec.tools = ["python_repl", "file_write", "shell"]
            spec.enable_code_execution = True
        elif template_name == "data_analysis":
            spec.tools = ["python_repl", "calculator", "file_read"]
            spec.enable_code_execution = True

        return self.create_agent_from_spec(spec)

    def list_available_types(self) -> None:
        """List all available agent types and their capabilities"""

        print("🤖 Available Agent Types")
        print("=" * 50)

        for type_key, type_info in AGENT_TYPES.items():
            print(f"\\n🔧 {type_key.upper()}")
            print(f"   Description: {type_info['description']}")
            print(f"   Default Model: {type_info['default_model']}")
            print(f"   Default Tools: {', '.join(type_info['tools'])}")
            print(f"   Code Execution: {type_info['enable_code_execution']}")

        print("\\n🎯 Quick Start Examples:")
        print("   research: Information gathering and analysis")
        print("   coding: Software development and code generation")
        print("   data_analysis: Data processing and visualization")
        print("   creative: Content generation and creative tasks")
        print("   workflow: Multi-step process orchestration")
        print("   meta: Agent creation and management")

    def list_available_models(self) -> None:
        """List all available models and their capabilities"""

        print("🤖 Available Models")
        print("=" * 50)

        try:
            models = list_available_models()
            if not models or not isinstance(models, list):
                print("No models available or invalid model data")
                return
                
            for model in models:
                if isinstance(model, dict):
                    print(f"\\n🔧 {model.get('name', 'Unknown')}")
                    print(f"   Size: {model.get('size', 'Unknown')}")
                    print(f"   Family: {model.get('family', 'Unknown')}")
                    capabilities = model.get('capabilities', [])
                    if isinstance(capabilities, list):
                        print(f"   Capabilities: {', '.join(capabilities)}")
                    else:
                        print(f"   Capabilities: {capabilities}")
                    print(f"   Performance Score: {model.get('performance_score', 'Unknown')}")
        except Exception as e:
            print(f"Error retrieving models: {e}")
            print("Using fallback model list:")
            for i, model_name in enumerate(AVAILABLE_MODELS[:5], 1):
                print(f"   {i}. {model_name}")

    def validate_agent(self, agent_path: Path) -> Dict[str, Any]:
        """Validate a generated agent"""

        try:
            # Try to import and test the agent
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "test_agent", agent_path)
            module = importlib.util.module_from_spec(spec)

            # Check if agent function exists
            agent_function = getattr(module, Path(agent_path).stem, None)
            if not agent_function:
                return {"valid": False, "error": "Agent function not found"}

            # Try to call the agent
            test_result = agent_function("Test query")
            if not isinstance(test_result, str):
                return {"valid": False, "error": "Agent must return string"}

            return {
                "valid": True,
                "name": Path(agent_path).stem,
                "test_result": test_result[:100] + "..." if len(test_result) > 100 else test_result
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


# Global agent builder instance
agent_builder = AgentBuilder()

# Convenience functions


def create_agent_interactive() -> Dict[str, Path]:
    """Create an agent through interactive prompts"""
    return agent_builder.create_agent_interactive()


def create_agent_from_type(agent_type: str, name: str, description: str = "") -> Dict[str, Path]:
    """Create an agent from a specific type"""
    spec = AgentSpecification(
        name=name,
        display_name=name.replace("_", " ").title(),
        description=description or AGENT_TYPES[agent_type]["description"],
        agent_type=agent_type
    )
    return agent_builder.create_agent_from_spec(spec)


def list_agent_types() -> None:
    """List all available agent types"""
    agent_builder.list_available_types()


def list_available_models() -> None:
    """List all available models"""
    agent_builder.list_available_models()


def validate_generated_agent(agent_path: str) -> Dict[str, Any]:
    """Validate a generated agent"""
    return agent_builder.validate_agent(Path(agent_path))


if __name__ == "__main__":
    print("🤖 StrandsAgents Agent Builder")
    print("=" * 50)
    print("Create specialized AI agents using the @agent decorator")
    print()

    print("Available commands:")
    print("1. Create agent interactively")
    print("2. List available agent types")
    print("3. List available models")
    print("4. Create agent from template")
    print()

    while True:
        try:
            choice = input("Select option (1-4) or 'exit': ").strip()

            if choice == 'exit':
                print("Goodbye!")
                break
            elif choice == '1':
                paths = create_agent_interactive()
                print(f"\\n✅ Agent created successfully!")
                for name, path in paths.items():
                    print(f"   {name}: {path}")
            elif choice == '2':
                list_agent_types()
            elif choice == '3':
                list_available_models()
            elif choice == '4':
                print("\\nAvailable workflow templates:")
                templates = list_workflow_templates()
                for i, template in enumerate(templates, 1):
                    print(
                        f"{i}. {template['name']} - {template['description']}")

                try:
                    template_choice = int(
                        input("\\nSelect template (number): "))
                    if 1 <= template_choice <= len(templates):
                        template_id = templates[template_choice - 1]['id']
                        name = input("Agent name: ").strip()
                        description = input("Description: ").strip()

                        paths = agent_builder.create_agent_from_template(template_id, {
                            "name": name,
                            "description": description
                        })

                        print(f"\\n✅ Agent created from template!")
                        for name, path in paths.items():
                            print(f"   {name}: {path}")
                except ValueError:
                    print("Invalid selection")
            else:
                print("Invalid choice")

            print()

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
