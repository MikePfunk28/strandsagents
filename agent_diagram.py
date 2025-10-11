from strands import Agent, tools
from strands_tools import http_request, diagram, file_read, editor, file_write, think, use_aws, workflow, swarm, image_reader
from strands.models import OllamaModel, BedrockModel

# Define the agent with the diagram tool
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:8b"
)
# Create a BedrockModel
bedrock_model = BedrockModel(
    model_id="anthropic.claude-sonnet-4.5-20250514-v1:0",
    region_name="us-east-1",
    temperature=0.8,
)

agent = Agent(
    system_prompt="""
    You are an expert at creating AWS architecture diagrams using the Strands diagram tool.

    When asked to create a diagram, you examine the files in the repo, then get the aws icons,
    and create a diagram that matches how the product works.
    """,
    model=ollama_model,
    tools=[http_request, diagram, file_read, editor, file_write, think, use_aws, workflow, image_reader, swarm],
)



def create_diagram() -> tools.tool:
    """
    Create and return the diagram tool with the specified configuration.
    """
    agent.tool.think = thought("Check the files to understand the architecture before diagramming.")
    return tools.tool(
        name="diagram",
        description="Create AWS architecture diagrams using the Strands diagram tool.",
        func=diagram,
        input_schema={
            "json": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A detailed description of the architecture to be diagrammed.",
                    },
                    "style": {
                        "type": "string",
                        "description": "The style of the diagram (e.g., 'aws', 'gcp', 'azure'). Default is 'aws'.",
                        "default": "aws",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "The output format of the diagram (e.g., 'png', 'svg'). Default is 'png'.",
                        "default": "png",
                    },
                },
                "required": ["description"],
            }
        },
    )


agent.tool.diagram = create_diagram()
