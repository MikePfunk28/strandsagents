import json
from strands import Agent
from strands.models import BedrockModel
from strands_tools import shell
"""
Creating a strands agent
"""
bedrock_model = BedrockModel(
    model_id="anthropic.claude-2",
    region_name="us-east-1",
    guardrails=[
        {"type": "input", "value": "What is the capital of France?"},
        {"type": "output", "value": "Paris"}
    ]
    temperature=0.3,
    top_p=0.8,
)

agent = Agent(
    system_prompt="You are a helpful assistant.",
    model=bedrock_model,
    tools=[shell, letter_counter],
)
response = agent("what OS am I using?")
print(response)

# Use models to run code you define, making it deterministic,
# as you can have the model make the deterministic code and
# then running it.

@tools([shell])
def get_os():
    return shell("uname -a")

@tool
def letter_counter(text: str) -> int:
    """Counts the number of letters in a given text."""
    return len([c for c in text if c.isalpha()])
if not isinstance(word, str):
    return 0
# comment well as the tools, ai models using tools, will
# read the file.  SO have a file description, maybe define
# variables for the tool code, to direct it and be clear,
# also add SCOPE of file built-in, Global, Local, nested

stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx" # combines python and php
        args=["awslabds."]
    )
))

with stdio_mcp_client:
    tools = stdio_mcp_client.list_tools_sync()
    print(f"Tools: {tools}")
    agent = Agent(tools=tools)
    agent("tell me about bedrock agent builder")

# https://github.com/strands-agents/tools