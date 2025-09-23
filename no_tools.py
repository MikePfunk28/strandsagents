from strands import Agent
from strands.models.ollama import OllamaModel

# Use phi4 without tools
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini-reasoning:latest"
)

# Create agent without tools
agent = Agent(model=ollama_model)

# Just ask about Strands agents (no web access)
response = agent("Tell me what you know about Strands agents framework")
print(response)