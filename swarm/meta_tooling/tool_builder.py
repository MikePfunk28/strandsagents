"""Meta-tooling system for dynamic tool creation using StrandsAgents patterns."""

import os
import importlib.util
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from strands import Agent
from strands.models import OllamaModel
from strands_tools import shell, editor, load_tool
from strands.types.tool_types import ToolUse, ToolResult

@dataclass
class ToolSpec:
    """Specification for a dynamically created tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    function_code: str
    agent_type: Optional[str] = None

# Meta-tooling system prompt following your DESIGN.md
TOOL_BUILDER_SYSTEM_PROMPT = """You are an advanced Meta-Tooling Agent that creates custom tools for the Swarm system.

CRITICAL REQUIREMENTS - NO AWS OR CLOUD SERVICES:
- Use ONLY OllamaModel for all agents
- NO AWS, Bedrock, or any paid cloud services
- Everything must be completely local

## TOOL NAMING CONVENTION:
- Tool name (function name) MUST match the file name without extension
- Example: For file "research_tool.py", use tool name "research_tool"

## TOOL CREATION PROCESS:
1. Name the file "tool_name.py" where tool_name is human readable
2. Function name must match file name (without extension)
3. TOOL_SPEC "name" parameter must match file name
4. Include detailed docstrings
5. After creating, announce "TOOL_CREATED: <filename>"

## TOOL STRUCTURE (exact format required):
```python
from typing import Any
from strands.types.tool_types import ToolUse, ToolResult

TOOL_SPEC = {
    "name": "tool_name",  # Must match function name
    "description": "What the tool does",
    "inputSchema": {  # Exact capitalization required
        "json": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
    }
}

def tool_name(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    \"\"\"Tool function docstring\"\"\"
    tool_use_id = tool_use["toolUseId"]
    param_value = tool_use["input"]["param_name"]

    # Process inputs - NO CLOUD SERVICES
    result = param_value  # Replace with local processing only

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": f"Result: {result}"}]
    }
```

## AUTONOMOUS WORKFLOW:
1. Generate complete Python code following structure above
2. Use editor tool to write code to "tool_name.py"
3. Use load_tool to dynamically load the new tool
4. Report tool name and confirm creation
5. All processing must be LOCAL ONLY

Use tools implicitly without being told. Create tools for swarm agents.
NO AWS, NO CLOUD, LOCAL OLLAMA ONLY.
"""

class ToolBuilder:
    """Dynamic tool builder for the swarm system."""

    def __init__(self, tools_dir: str = "./swarm/tools", host: str = "localhost:11434"):
        self.tools_dir = tools_dir
        self.host = host
        self.created_tools = {}

        # Create tools directory
        os.makedirs(tools_dir, exist_ok=True)

        # Initialize tool builder agent with LOCAL OLLAMA ONLY
        self.agent = Agent(
            model=OllamaModel(model="llama3.2:3b", host=host),  # LOCAL ONLY
            system_prompt=TOOL_BUILDER_SYSTEM_PROMPT,
            tools=[load_tool, shell, editor]
        )

    async def create_tool(self, description: str, agent_type: Optional[str] = None) -> ToolSpec:
        """Create a new tool based on description."""
        prompt = f"""
        Create a Python tool based on this description: "{description}"

        Agent type: {agent_type if agent_type else "general"}

        Requirements:
        - Use ONLY local processing (NO AWS, NO CLOUD)
        - Follow the exact TOOL_SPEC structure
        - Save to tools directory
        - Load the tool after creation
        - All models must use OllamaModel with local host

        Handle all steps autonomously including naming and file creation.
        """

        response = await self.agent.run_async(prompt)

        # Parse response to extract tool name (simplified - would be more robust in production)
        if "TOOL_CREATED:" in response:
            tool_name = response.split("TOOL_CREATED:")[1].strip().split()[0].replace(".py", "")
            tool_spec = ToolSpec(
                name=tool_name,
                description=description,
                input_schema={},  # Would parse from created file
                function_code=response,
                agent_type=agent_type
            )
            self.created_tools[tool_name] = tool_spec
            return tool_spec

        raise Exception(f"Tool creation failed: {response}")

    def create_agent_tool(self, agent_name: str, agent_prompt: str, model_name: str = "llama3.2:1b") -> ToolSpec:
        """Create a tool that wraps an agent as a tool."""
        tool_name = f"{agent_name}_tool"
        tool_file = os.path.join(self.tools_dir, f"{tool_name}.py")

        # Generate agent-as-tool code
        tool_code = f'''"""Agent tool for {agent_name}."""

from typing import Any
from strands import Agent
from strands.models import OllamaModel
from strands.types.tool_types import ToolUse, ToolResult

TOOL_SPEC = {{
    "name": "{tool_name}",
    "description": "Execute {agent_name} agent tasks",
    "inputSchema": {{
        "json": {{
            "type": "object",
            "properties": {{
                "task": {{
                    "type": "string",
                    "description": "Task for the {agent_name} agent"
                }},
                "context": {{
                    "type": "string",
                    "description": "Optional context for the task"
                }}
            }},
            "required": ["task"]
        }}
    }}
}}

# Agent instance (LOCAL OLLAMA ONLY)
_agent = Agent(
    model=OllamaModel(model="{model_name}", host="localhost:11434"),
    system_prompt="""{agent_prompt}"""
)

def {tool_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Execute {agent_name} agent task."""
    tool_use_id = tool_use["toolUseId"]
    task = tool_use["input"]["task"]
    context = tool_use["input"].get("context", "")

    try:
        # Execute agent task locally
        if context:
            prompt = f"Context: {{context}}\\n\\nTask: {{task}}"
        else:
            prompt = task

        result = _agent(prompt)

        return {{
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{{"text": result}}]
        }}
    except Exception as e:
        return {{
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{{"text": f"Error: {{str(e)}}"}}]
        }}
'''

        # Write tool file
        with open(tool_file, 'w') as f:
            f.write(tool_code)

        # Create tool spec
        tool_spec = ToolSpec(
            name=tool_name,
            description=f"Execute {agent_name} agent tasks",
            input_schema={"task": "string", "context": "string"},
            function_code=tool_code,
            agent_type=agent_name
        )

        self.created_tools[tool_name] = tool_spec
        return tool_spec

    def list_tools(self) -> List[ToolSpec]:
        """List all created tools."""
        return list(self.created_tools.values())

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get tool specification by name."""
        return self.created_tools.get(name)

    def load_all_tools(self) -> List[str]:
        """Load all tools from the tools directory."""
        loaded_tools = []

        if not os.path.exists(self.tools_dir):
            return loaded_tools

        for filename in os.listdir(self.tools_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                tool_path = os.path.join(self.tools_dir, filename)
                try:
                    # Use StrandsAgents load_tool
                    tool_name = filename[:-3]  # Remove .py
                    # In real implementation, would use proper load_tool function
                    loaded_tools.append(tool_name)
                except Exception as e:
                    print(f"Failed to load tool {filename}: {e}")

        return loaded_tools

# Example usage functions
async def create_research_tools(builder: ToolBuilder):
    """Create specialized research tools."""

    # Web search tool (local only)
    await builder.create_tool(
        "Search local documents and knowledge base for information",
        agent_type="research"
    )

    # Document analysis tool
    await builder.create_tool(
        "Analyze document structure and extract key information",
        agent_type="research"
    )

    # Data synthesis tool
    await builder.create_tool(
        "Synthesize information from multiple sources into coherent summary",
        agent_type="research"
    )

async def create_creative_tools(builder: ToolBuilder):
    """Create specialized creative tools."""

    # Brainstorming tool
    await builder.create_tool(
        "Generate creative ideas and innovative solutions for problems",
        agent_type="creative"
    )

    # Ideation tool
    await builder.create_tool(
        "Develop and expand on initial concepts with creative variations",
        agent_type="creative"
    )

    # Innovation tool
    await builder.create_tool(
        "Propose novel approaches and unconventional solutions",
        agent_type="creative"
    )

async def create_critical_tools(builder: ToolBuilder):
    """Create specialized critical analysis tools."""

    # Evaluation tool
    await builder.create_tool(
        "Critically evaluate proposals and identify strengths and weaknesses",
        agent_type="critical"
    )

    # Risk assessment tool
    await builder.create_tool(
        "Analyze potential risks and negative consequences of proposed solutions",
        agent_type="critical"
    )

    # Improvement tool
    await builder.create_tool(
        "Suggest specific improvements to address identified issues",
        agent_type="critical"
    )

# Main tool builder initialization
async def initialize_swarm_tools(tools_dir: str = "./swarm/tools") -> ToolBuilder:
    """Initialize the tool builder and create essential swarm tools."""
    builder = ToolBuilder(tools_dir)

    # Create tools for different agent types
    await create_research_tools(builder)
    await create_creative_tools(builder)
    await create_critical_tools(builder)

    print(f"Initialized tool builder with {len(builder.created_tools)} tools")
    return builder

if __name__ == "__main__":
    import asyncio

    async def demo():
        builder = await initialize_swarm_tools()

        # Create a custom tool
        tool = await builder.create_tool(
            "Analyze code for potential improvements and optimizations"
        )

        print(f"Created tool: {tool.name}")
        print(f"Description: {tool.description}")

    asyncio.run(demo())