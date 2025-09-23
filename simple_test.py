from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import handoff_to_user, browser

# Test without browser first
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini-reasoning:latest"
)
# Defing agent
agent = Agent(
    model=ollama_model,
    tools=[handoff_to_user, browser],
)

response = agent.tool.browser("Go to strandsagents.com and look up the documentation, then tell me about strandsagents.")
print(response)