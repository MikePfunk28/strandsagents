from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools.browser import browser

# Test without browser first
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini-reasoning:latest"
)

agent = Agent(
    model=ollama_model,
    tools=[browser],
)
response = agent("Go to strandsagents.com and look up the documentation, then tell me about strandsagents.")
print(response)