I mean this is a great implementation, and I would love for you to be able to come up with a plan to improve agent.py, adding the swarm to both agent.py and thought_agent.py or any system by creating a swarm, that can access all different types of assistants, that we define with a separate prompt file, then we import the prompt into the assistant, and define the tools and such and then any other code needed.  We use strands workflows and async iterators as well as callback handlers, using mcp servers and clients for communication, and whatever else, we can define agents as tools in the mcp's.  Then use the assistants to make agents and from the agents we use agent2agent communication.  Forget about adding the swarm to anything, but making it.  Make sure it uses the meta-tooling like

https://github.com/strands-agents/docs/blob/main/docs/examples/python/meta_tooling.md  on that website.

## SWARM

https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/02-swarm-agent/swarm.ipynb

We essentially use meta-tooling to create the tools we need, agents as tools, and then make MCPs.  Use only OllamaModels, do not use AWS, BEDROCK, or any other service that is paid, this is entirely local.  We already have the coding agent, but I like the ones in swarm.  Then was thinking we extend that and define a bunch of system prompts, and then use those in the assistant file where we define the tools and code needed and the model, and set it up.  From there we can stitch many together using the Workflows in strands, to make agents, which we can use as tools, in the mcps or wherever.  Then we can define how it flows, the tools used and how many iterations, and using orchestrator agent, as well as executor, async iterators, and callback handlers we define exactly how it works, repeats, or doesnt, etc.

We can use have utils folder that just defines tools and prompts which can be for that agent.  prompts.py and tools.py

## Meta-tooling
```python
agent = Agent(
    system_prompt=TOOL_BUILDER_SYSTEM_PROMPT, tools=[load_tool, shell, editor]
)
```

```python
from typing import Any
from strands.types.tool_types import ToolUse, ToolResult

TOOL_SPEC = {
    "name": "tool_name",
    "description": "What the tool does",
    "inputSchema": {
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
    # Tool function docstring
    tool_use_id = tool_use["toolUseId"]
    param_value = tool_use["input"]["param_name"]

    # Process inputs
    result = param_value  # Replace with actual processing

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": f"Result: {result}"}]
    }
```
## Example implementation
> Create a tool that counts characters in text

Creating a Python tool based on this description: "create a tool that counts characters in text". Load the tool after it is created Handle all steps autonomously including naming and file creation.

I'll create a custom tool that counts characters in text. I'll follow these steps:
1. Determine the next available tool index
2. Create the tool with proper structure
3. Write the file
4. Load the tool

First, let's check if there are any existing custom tools to determine the next index:

I'll create a tool named custom_tool_0.py:

Tool created! Here's the code for custom_tool_0.py:

```python
from typing import Any
from strands.types.tool_types import ToolUse, ToolResult

TOOL_SPEC = {
    "name": "custom_tool_0",
    "description": "Counts characters in a text string",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count characters in"
                }
            },
            "required": ["text"]
        }
    }
}

def custom_tool_0(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Count the number of characters in the provided text.

    Args:
        tool_use: Contains the input text to analyze

    Returns:
        A ToolResult with the character count statistics
    """
    tool_use_id = tool_use["toolUseId"]
    text = tool_use["input"]["text"]

    # Count different types of characters
    total_chars = len(text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)
    punctuation = sum(not c.isalnum() and not c.isspace() for c in text)

    result = f"The text \"{text}\" contains:\n"
    result += f"- Total characters: {total_chars}\n"
    result += f"- Letters: {letters}\n"
    result += f"- Digits: {digits}\n"
    result += f"- Spaces: {spaces}\n"
    result += f"- Punctuation: {punctuation}"

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": result}]
    }
```
Now I'll load this tool so it's immediately available for use:

TOOL_CREATED: custom_tool_0.py

The tool custom_tool_0 has been successfully created and loaded. You can now use it to count characters in text.

## FULL meta-tooling code
```python
#!/usr/bin/env python3
"""
# Meta Tooling Example

This example demonstrates Strands Agents' advanced meta-tooling capabilities - the ability of an agent
to create, load, and use custom tools dynamically at runtime.

It creates custom tools using the Agent's built-in tools for file operations and implicit tool calling.
"""

import os

from strands import Agent
from strands_tools import shell, editor, load_tool

# Define an enhanced system prompt for our tool builder agent
TOOL_BUILDER_SYSTEM_PROMPT = """You are an advanced agent that creates and uses custom Strands Agents tools.

Use all available tools implicitly as needed without being explicitly told. Always use tools instead of suggesting code
that would perform the same operations. Proactively identify when tasks can be completed using available tools.

## TOOL NAMING CONVENTION:
   - The tool name (function name) MUST match the file name without the extension
   - Example: For file "tool_name.py", use tool name "tool_name"

## TOOL CREATION vs. TOOL USAGE:
   - CAREFULLY distinguish between requests to CREATE a new tool versus USE an existing tool
   - When a user asks a question like "reverse hello world" or "count abc", first check if an appropriate tool already exists before creating a new one
   - If an appropriate tool already exists, use it directly instead of creating a redundant tool
   - Only create a new tool when the user explicitly requests one with phrases like "create", "make a tool", etc.

## TOOL CREATION PROCESS:
   - Name the file "tool_name.py" where "tool_name is a human readable name
   - Name the function in the file the SAME as the file name (without extension)
   - The "name" parameter in the TOOL_SPEC MUST match the name of the file (without extension)
   - Include detailed docstrings explaining the tool's purpose and parameters
   - After creating a tool, announce "TOOL_CREATED: <filename>" to track successful creation

## TOOL USAGE:
   - Use existing tools with appropriate parameters
   - Provide a clear explanation of the result

## TOOL STRUCTURE
When creating a tool, follow this exact structure:

```python
from typing import Any
from strands.types.tools import ToolUse, ToolResult

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
    # Tool function docstring
    tool_use_id = tool_use["toolUseId"]
    param_value = tool_use["input"]["param_name"]

    # Process inputs
    result = param_value  # Replace with actual processing

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": f"Result: {result}"}]
    }
```

Critical requirements:
1. Use "inputSchema" (not input_schema) with "json" wrapper
2. Function must access parameters via tool_use["input"]["param_name"]
3. Return dict must use "toolUseId" (not tool_use_id)
4. Content must be a list of objects: [{"text": "message"}]

## AUTONOMOUS TOOL CREATION WORKFLOW

When asked to create a tool:
1. Generate the complete Python code for the tool following the structure above
2. Use the editor tool to write the code directly to a file named "tool_name.py" where "tool_name" is a human readable name.
3. Use the load_tool tool to dynamically load the newly created tool
4. After loading, report the exact tool name and path you created
5. Confirm when the tool has been created and loaded

Always extract your own code and write it to files without waiting for further instructions or relying on external extraction functions.

Always use the following tools when appropriate:
- editor: For writing code to files and file editing operations
- load_tool: For loading custom tools
- shell: For running shell commands

You should detect user intents to create tools from natural language (like "create a tool that...", "build a tool for...", etc.) and handle the creation process automatically.
"""

# Create our agent with the necessary tools and implicit tool calling enabled
agent = Agent(
    system_prompt=TOOL_BUILDER_SYSTEM_PROMPT, tools=[load_tool, shell, editor]
)


# Example usage
if __name__ == "__main__":
    print("\nMeta-Tooling Demonstration (Improved)")
    print("==================================")
    print("Commands:")
    print("  • create <description> - Create a new tool")
    print("  • make a tool that <description>")
    print("  • list tools - Show created tools")
    print("  • exit - Exit the program")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ")

            # Handle exit command
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            # Regular interaction - let the agent's system prompt handle tool creation detection
            else:
                response = agent(
                    f'Create a Python tool based on this description: "{user_input}". Load the tool after it is created '
                    f"Handle all steps autonomously including naming and file creation."
                )

        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try a different request.")
```

## prompts.py
```python
# Create specialized agents with different expertise
research_agent = Agent(system_prompt=("""You are a Research Agent specializing in gathering and analyzing information.
Your role in the swarm is to provide factual information and research insights on the topic.
You should focus on providing accurate data and identifying key aspects of the problem.
When receiving input from other agents, evaluate if their information aligns with your research.
"""),
name="research_agent",model="ollama_model"

creative_agent = Agent(system_prompt=("""You are a Creative Agent specializing in generating innovative solutions.
Your role in the swarm is to think outside the box and propose creative approaches.
You should build upon information from other agents while adding your unique creative perspective.
Focus on novel approaches that others might not have considered.
"""),
name="creative_agent",model="ollama_model"

critical_agent = Agent(system_prompt=("""You are a Critical Agent specializing in analyzing proposals and finding flaws.
Your role in the swarm is to evaluate solutions proposed by other agents and identify potential issues.
You should carefully examine proposed solutions, find weaknesses or oversights, and suggest improvements.
Be constructive in your criticism while ensuring the final solution is robust.
"""),
name="critical_agent",model="ollama_model

summarizer_agent = Agent(system_prompt=("""You are a Summarizer Agent specializing in synthesizing information.
Your role in the swarm is to gather insights from all agents and create a cohesive final solution.
You should combine the best ideas and address the criticisms to create a comprehensive response.
Focus on creating a clear, actionable summary that addresses the original query effectively.
"""),name="summarizer_agent",model=ollama_model
```
