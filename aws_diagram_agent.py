"""
AWS Infrastructure Diagram Agent using Strands and Ollama qwen3:4b
Generates AWS architecture diagrams using the diagram tool from strands_tools.
"""

from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import diagram
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create Ollama model instance
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:4b"
)

# System prompt for the AWS diagram agent
AWS_DIAGRAM_SYSTEM_PROMPT = """You are an AWS Infrastructure Diagram Expert.
Your job is to create detailed, accurate AWS architecture diagrams using the diagram tool.

You have access to the `diagram` tool. When creating AWS cloud diagrams, you MUST call it with:

diagram_type="cloud"
nodes=[list of nodes with id, type, and label]
edges=[list of edges with from and to]
title="your diagram title"

Each node must have:
- id: unique identifier (e.g., "user", "igw", "vpc")
- type: AWS service type (e.g., "Users", "InternetGateway", "VPC", "ECS", "Fargate", "ALB", "S3", "SecretsManager", "CloudWatch", "IAM", "KMS")
- label: display name

Each edge must have:
- from: source node id
- to: target node id

Available AWS service types for the infrastructure:
- Users (for end users/clients)
- InternetGateway (IGW)
- VPC
- PublicSubnet
- ALB (Application Load Balancer)
- ECS
- Fargate
- S3
- SecretsManager
- CloudWatch
- IAM
- KMS

Example call:
diagram(
    diagram_type="cloud",
    nodes=[
        {"id": "user", "type": "Users", "label": "End Users"},
        {"id": "igw", "type": "InternetGateway", "label": "Internet Gateway"}
    ],
    edges=[
        {"from": "user", "to": "igw"}
    ],
    title="AWS Infrastructure"
)

You MUST use the diagram tool to create the visualization!
"""


def create_aws_infrastructure_diagram():
    """
    Create an AWS infrastructure diagram agent using Strands and Ollama.
    """
    logger.info("Initializing AWS Diagram Agent with Ollama qwen3:4b")

    # Create the agent with the diagram tool explicitly passed
    aws_diagram_agent = Agent(
        name="AWS Infrastructure Diagram Agent",
        model=ollama_model,
        tools=[diagram],  # Explicitly provide the diagram tool
        system_prompt=AWS_DIAGRAM_SYSTEM_PROMPT
    )

    # Define the infrastructure to diagram
    infrastructure_description = """
Create an AWS cloud architecture diagram for this production infrastructure:

**Production Features:**
• S3 encryption at rest (AWS KMS)
• Least-privilege IAM policies
• Cost monitoring alarms (CloudWatch)
• Secrets management (AWS Secrets Manager)
• VPC with public subnet for Fargate
• ECS Fargate cluster for containers

**Components to include:**
1. Users (end users/clients)
2. InternetGateway (IGW for VPC)
3. VPC (Virtual Private Cloud)
4. PublicSubnet (in VPC)
5. ALB (Application Load Balancer)
6. ECS (Elastic Container Service)
7. Fargate (serverless container compute)
8. S3 (encrypted storage buckets)
9. SecretsManager (credentials storage)
10. CloudWatch (monitoring & alarms)
11. IAM (access control)
12. KMS (encryption keys)

**Data Flow:**
Users → Internet Gateway → VPC → Public Subnet → ALB → ECS/Fargate
ECS/Fargate → S3 (encrypted data)
ECS/Fargate → Secrets Manager (retrieve secrets)
All services → CloudWatch (monitoring)
All services governed by IAM policies
S3 encrypted by KMS

Use the diagram tool to create this AWS cloud diagram with title "AWS Production Infrastructure".
Call the diagram tool NOW to generate the visualization!
    """

    logger.info("Sending infrastructure description to agent")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Execute the agent with the infrastructure description
    print("\n" + "="*80)
    print("AWS INFRASTRUCTURE DIAGRAM GENERATION")
    print("="*80)
    print(f"\nAgent: {aws_diagram_agent.name}")
    print(f"Model: Ollama qwen3:4b")
    print(f"Task: Generate production AWS infrastructure diagram")
    print("\n" + "="*80 + "\n")

    try:
        response = aws_diagram_agent(infrastructure_description)

        print("\n" + "="*80)
        print("AGENT RESPONSE:")
        print("="*80)
        print(response)
        print("\n" + "="*80)

        logger.info("Diagram generation completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error during diagram generation: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("\n🚀 Starting AWS Infrastructure Diagram Agent")
    print("=" * 80)
    print("Using Strands Agent Framework with Ollama qwen3:4b model")
    print("=" * 80 + "\n")

    result = create_aws_infrastructure_diagram()

    if result:
        print("\n✅ Diagram generation completed!")
        print("\nCheck the workspace for 'aws_production_infrastructure.png'")
    else:
        print("\n❌ Diagram generation failed. Check logs for details.")
