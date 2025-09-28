"""StrandsAgents Meta-Tooling System for Dynamic Tool Creation.

This module implements a meta-tooling system that follows StrandsAgents patterns
for creating tools dynamically and wrapping agents as tools.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import uuid

from strands import Agent
from strands.models import OllamaModel
from strands.tools import tool

logger = logging.getLogger(__name__)

# Meta-tooling system prompt optimized for tool creation
TOOL_BUILDER_SYSTEM_PROMPT = """You are an advanced Meta-Tooling Agent that creates custom tools for the Swarm system.

CRITICAL REQUIREMENTS - NO AWS OR CLOUD SERVICES:
- Use ONLY OllamaModel for all agents
- NO AWS, Bedrock, or any paid cloud services
- All models must use localhost:11434 (Ollama)

CORE CAPABILITIES:
1. Create StrandsAgents tool specifications with @tool decorators
2. Generate tool functions following exact StrandsAgents patterns
3. Wrap existing agents as callable tools
4. Design tool inputSchemas with proper JSON schema
5. Build tool collections for specific domains

TOOL CREATION PATTERNS (Follow exactly):

```python
from strands.tools import tool

@tool
def my_custom_tool(parameter: str, optional_param: str = "default") -> str:
    '''Tool description for the agent.

    Args:
        parameter: Description of required parameter
        optional_param: Description of optional parameter with default

    Returns:
        Result description
    '''
    # Tool implementation
    result = f"Processed {parameter} with {optional_param}"
    return result
```

AGENT-AS-TOOL WRAPPER PATTERN:

```python
@tool
def agent_wrapper_tool(query: str, context: str = "") -> str:
    '''Wrap agent as a tool for other agents to use.

    Args:
        query: The task or question for the agent
        context: Optional context information

    Returns:
        Agent's response
    '''
    # Create agent with OllamaModel ONLY
    agent = Agent(
        model=OllamaModel(model="gemma:270m", host="localhost:11434"),
        system_prompt="Specialized agent prompt"
    )

    full_query = f"Context: {context}\nQuery: {query}" if context else query
    response = agent.run(full_query)
    return response
```

TOOL SPECIFICATION PATTERN:

```python
TOOL_SPEC = {
    "name": "tool_name",
    "description": "Clear tool description",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                },
                "param2": {
                    "type": "integer",
                    "description": "Optional parameter",
                    "default": 10
                }
            },
            "required": ["param1"]
        }
    }
}
```

RESPONSE FORMAT:
Always provide complete, working tool code that:
1. Uses proper @tool decorator
2. Has clear docstrings
3. Uses OllamaModel ONLY (no cloud services)
4. Follows StrandsAgents patterns exactly
5. Includes proper error handling
6. Returns appropriate types

Focus on creating practical, efficient tools for the microservices swarm."""

@dataclass
class ToolSpec:
    """Specification for a tool to be created."""
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str
    category: str = "general"

class ToolBuilder:
    """Meta-tooling agent that creates custom tools dynamically."""

    def __init__(self, model_name: str = "llama3.2:3b", host: str = "localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.agent = Agent(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=TOOL_BUILDER_SYSTEM_PROMPT
        )
        self.created_tools: Dict[str, Any] = {}

    async def create_tool(self, tool_spec: ToolSpec) -> str:
        """Create a new tool based on specification."""
        prompt = f"""Create a StrandsAgents tool with these specifications:

Tool Name: {tool_spec.name}
Description: {tool_spec.description}
Parameters: {tool_spec.parameters}
Return Type: {tool_spec.return_type}
Category: {tool_spec.category}

Generate complete Python code with:
1. Proper @tool decorator
2. Type hints and docstring
3. Input validation
4. Error handling
5. Working implementation

Use ONLY OllamaModel (localhost:11434) - NO cloud services."""

        try:
            tool_code = await self.agent.run_async(prompt)
            self.created_tools[tool_spec.name] = tool_code
            return tool_code
        except Exception as e:
            logger.error(f"Failed to create tool {tool_spec.name}: {e}")
            raise

    async def create_agent_wrapper_tool(self, agent_name: str, agent_prompt: str,
                                      model_name: str = "gemma:270m") -> str:
        """Create a tool that wraps an agent for use by other agents."""
        prompt = f"""Create an agent wrapper tool with these specifications:

Agent Name: {agent_name}
Agent Prompt: {agent_prompt}
Model: {model_name} (use OllamaModel with localhost:11434)

Generate a @tool decorated function that:
1. Creates an agent with the specified prompt
2. Accepts query and optional context parameters
3. Returns the agent's response
4. Handles errors properly
5. Uses ONLY OllamaModel

The tool should allow other agents to use this agent as a service."""

        try:
            wrapper_code = await self.agent.run_async(prompt)
            tool_name = f"{agent_name}_tool"
            self.created_tools[tool_name] = wrapper_code
            return wrapper_code
        except Exception as e:
            logger.error(f"Failed to create agent wrapper for {agent_name}: {e}")
            raise

    async def create_tool_collection(self, domain: str, capabilities: List[str]) -> Dict[str, str]:
        """Create a collection of related tools for a specific domain."""
        prompt = f"""Create a collection of StrandsAgents tools for the {domain} domain.

Required Capabilities: {capabilities}

For each capability, create a @tool decorated function that:
1. Has clear purpose and description
2. Uses appropriate parameters
3. Implements the functionality
4. Uses OllamaModel if agent interaction needed
5. Follows StrandsAgents patterns

Generate complete, working tools as a collection."""

        try:
            collection_code = await self.agent.run_async(prompt)
            collection_name = f"{domain}_tools"
            self.created_tools[collection_name] = collection_code
            return {collection_name: collection_code}
        except Exception as e:
            logger.error(f"Failed to create tool collection for {domain}: {e}")
            raise

    def get_created_tools(self) -> Dict[str, str]:
        """Get all tools created by this builder."""
        return self.created_tools.copy()

    async def optimize_tool(self, tool_name: str, optimization_goals: List[str]) -> str:
        """Optimize an existing tool based on goals."""
        if tool_name not in self.created_tools:
            raise ValueError(f"Tool {tool_name} not found")

        current_tool = self.created_tools[tool_name]

        prompt = f"""Optimize this StrandsAgents tool:

Current Tool Code:
{current_tool}

Optimization Goals: {optimization_goals}

Improve the tool while maintaining:
1. StrandsAgents compatibility
2. @tool decorator usage
3. Type safety
4. Error handling
5. OllamaModel usage (no cloud services)

Provide the optimized version."""

        try:
            optimized_code = await self.agent.run_async(prompt)
            self.created_tools[f"{tool_name}_optimized"] = optimized_code
            return optimized_code
        except Exception as e:
            logger.error(f"Failed to optimize tool {tool_name}: {e}")
            raise

# Factory functions for common tool patterns
async def create_research_tool(tool_builder: ToolBuilder, research_domain: str) -> str:
    """Create a research tool for a specific domain."""
    tool_spec = ToolSpec(
        name=f"{research_domain}_researcher",
        description=f"Research and analyze information in the {research_domain} domain",
        parameters={
            "query": {"type": "string", "description": "Research query"},
            "depth": {"type": "string", "description": "Research depth", "default": "standard"},
            "sources": {"type": "array", "description": "Source preferences", "default": []}
        },
        return_type="str",
        category="research"
    )
    return await tool_builder.create_tool(tool_spec)

async def create_analysis_tool(tool_builder: ToolBuilder, analysis_type: str) -> str:
    """Create an analysis tool for specific analysis type."""
    tool_spec = ToolSpec(
        name=f"{analysis_type}_analyzer",
        description=f"Perform {analysis_type} analysis on provided data",
        parameters={
            "data": {"type": "string", "description": "Data to analyze"},
            "metrics": {"type": "array", "description": "Analysis metrics", "default": []},
            "format": {"type": "string", "description": "Output format", "default": "structured"}
        },
        return_type="str",
        category="analysis"
    )
    return await tool_builder.create_tool(tool_spec)

async def create_synthesis_tool(tool_builder: ToolBuilder) -> str:
    """Create a tool for synthesizing multiple inputs."""
    tool_spec = ToolSpec(
        name="multi_input_synthesizer",
        description="Synthesize information from multiple sources into coherent output",
        parameters={
            "sources": {"type": "array", "description": "Multiple input sources"},
            "synthesis_type": {"type": "string", "description": "Type of synthesis", "default": "comprehensive"},
            "focus": {"type": "string", "description": "Synthesis focus area", "default": ""}
        },
        return_type="str",
        category="synthesis"
    )
    return await tool_builder.create_tool(tool_spec)

# Example usage and testing
async def demo_tool_builder():
    """Demonstrate the meta-tooling system."""
    print("StrandsAgents Meta-Tooling Demo")
    print("=" * 40)

    # Create tool builder with larger model for complex tool creation
    builder = ToolBuilder(model_name="llama3.2:3b")

    try:
        # Create a custom research tool
        print("\n1. Creating Research Tool:")
        research_tool = await create_research_tool(builder, "renewable_energy")
        print(f"Created research tool: {len(research_tool)} characters")

        # Create agent wrapper tool
        print("\n2. Creating Agent Wrapper Tool:")
        wrapper_tool = await builder.create_agent_wrapper_tool(
            "fact_checker",
            "You are a fact-checking agent. Verify claims and provide evidence-based assessments.",
            "gemma:270m"
        )
        print(f"Created wrapper tool: {len(wrapper_tool)} characters")

        # Create tool collection
        print("\n3. Creating Tool Collection:")
        data_tools = await builder.create_tool_collection(
            "data_processing",
            ["data_validation", "format_conversion", "quality_assessment"]
        )
        print(f"Created tool collection: {list(data_tools.keys())}")

        # Create synthesis tool
        print("\n4. Creating Synthesis Tool:")
        synthesis_tool = await create_synthesis_tool(builder)
        print(f"Created synthesis tool: {len(synthesis_tool)} characters")

        # Show all created tools
        all_tools = builder.get_created_tools()
        print(f"\nTotal tools created: {len(all_tools)}")
        for tool_name in all_tools.keys():
            print(f"  - {tool_name}")

    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(demo_tool_builder())