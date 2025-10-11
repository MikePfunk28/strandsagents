#!/usr/bin/env python3
"""
Unified Deterministic Agent Builder for StrandsAgents

A complete, production-ready agent builder that creates working agents
with proper imports, infrastructure setup, and deployment capabilities.

Features:
- Deterministic agent generation (same input = same output)
- Complete @agent decorator implementation
- Multi-runtime support (Ollama + AWS Bedrock)
- Infrastructure as code generation
- Agent validation and testing
- Token usage tracking and billing
- User request tracking
- Production deployment scripts
"""

import json
import logging
import argparse
import sys
import os
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

# Import strands-meta components
from .agent_decorator import agent, list_agents, get_agent_info, AGENT_REGISTRY
from .model_selector import get_best_model_for_task, list_available_models
from .sandbox_executor import SandboxExecutor
from .workflow_templates import get_workflow_template, list_workflow_templates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[AGENT_BUILDER] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("unified_agent_builder")


class TokenTracker:
    """Track token usage for billing purposes"""

    def __init__(self):
        self.usage_log = Path("usage_logs")
        self.usage_log.mkdir(exist_ok=True)
        self.current_session = str(uuid.uuid4())

    def track_request(self, user_id: str, agent_name: str, tokens_used: int, cost: float):
        """Track a user request with token usage"""
        log_entry = {
            "session_id": self.current_session,
            "user_id": user_id,
            "agent_name": agent_name,
            "tokens_used": tokens_used,
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }

        log_file = self.usage_log / f"{user_id}_usage.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(
            f"Token usage tracked: {user_id} used {tokens_used} tokens (${cost:.4f})")

    def get_user_usage(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for a user"""
        log_file = self.usage_log / f"{user_id}_usage.jsonl"

        if not log_file.exists():
            return {"total_requests": 0, "total_tokens": 0, "total_cost": 0.0}

        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

        total_requests = 0
        total_tokens = 0
        total_cost = 0.0

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("timestamp"):
                        entry_time = datetime.fromisoformat(
                            entry["timestamp"]).timestamp()
                        if entry_time >= cutoff_date:
                            total_requests += 1
                            total_tokens += entry.get("tokens_used", 0)
                            total_cost += entry.get("cost", 0.0)
        except Exception as e:
            logger.error(f"Error reading usage log: {e}")

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "period_days": days
        }


class AgentInfrastructureGenerator:
    """Generate infrastructure code for different runtimes"""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "infrastructure_templates"
        self.templates_dir.mkdir(exist_ok=True)

    def generate_ollama_infrastructure(self, agent_name: str) -> Dict[str, str]:
        """Generate Ollama runtime infrastructure"""
        return {
            "docker_compose.yml": self._generate_docker_compose(agent_name),
            "requirements.txt": self._generate_requirements(),
            "runtime_config.json": self._generate_runtime_config("ollama"),
            "deployment_script.sh": self._generate_deployment_script("ollama", agent_name)
        }

    def generate_bedrock_infrastructure(self, agent_name: str) -> Dict[str, str]:
        """Generate AWS Bedrock runtime infrastructure"""
        return {
            "cloudformation.yml": self._generate_cloudformation(agent_name),
            "requirements.txt": self._generate_requirements(),
            "runtime_config.json": self._generate_runtime_config("bedrock"),
            "deployment_script.sh": self._generate_deployment_script("bedrock", agent_name),
            "lambda_function.py": self._generate_lambda_function(agent_name)
        }

    def _generate_docker_compose(self, agent_name: str) -> str:
        """Generate Docker Compose for Ollama runtime"""
        return f'''version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  {agent_name}_runtime:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - AGENT_NAME={agent_name}
    depends_on:
      ollama:
        condition: service_healthy
    volumes:
      - ./agents:/app/agents

volumes:
  ollama_data:
'''

    def _generate_cloudformation(self, agent_name: str) -> str:
        """Generate CloudFormation template for Bedrock"""
        return f'''AWSTemplateFormatVersion: '2010-09-09'
Description: 'StrandsAgents Bedrock Infrastructure for {agent_name}'

Parameters:
  AgentName:
    Type: String
    Default: '{agent_name}'
  ModelId:
    Type: String
    Default: 'anthropic.claude-3-sonnet-20240229'
    AllowedValues:
      - 'anthropic.claude-3-sonnet-20240229'
      - 'anthropic.claude-3-haiku-20240307'
      - 'anthropic.claude-3-opus-20240229'

Resources:
  BedrockAgentRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '{{{{AgentName}}}}BedrockRole'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: bedrock.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess

  {agent_name}Agent:
    Type: AWS::Bedrock::Agent
    Properties:
      AgentName: !Ref AgentName
      FoundationModel: !Ref ModelId
      AgentResourceRoleArn: !GetAtt BedrockAgentRole.Arn
      Instruction: 'You are a helpful AI assistant specialized in various tasks.'
      AutoPrepare: true

Outputs:
  AgentId:
    Description: 'Bedrock Agent ID'
    Value: !GetAtt {agent_name}Agent.AgentId
    Export:
      Name: !Sub '{{{{AgentName}}}}AgentId'

  AgentArn:
    Description: 'Bedrock Agent ARN'
    Value: !GetAtt {agent_name}Agent.AgentArn
    Export:
      Name: !Sub '{{{{AgentName}}}}AgentArn'
'''

    def _generate_requirements(self) -> str:
        """Generate Python requirements"""
        return '''boto3>=1.34.0
requests>=2.31.0
python-dotenv>=1.0.0
pydantic>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
'''

    def _generate_runtime_config(self, runtime: str) -> str:
        """Generate runtime configuration"""
        config = {
            "runtime": runtime,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "token_tracking": True,
                "user_management": True,
                "billing": True,
                "monitoring": True
            }
        }

        if runtime == "ollama":
            config.update({
                "ollama_host": "http://localhost:11434",
                "default_model": "llama3.2",
                "supported_models": ["llama3.2", "qwen3:8b", "gemma3:4b"]
            })
        elif runtime == "bedrock":
            config.update({
                "region": "us-east-1",
                "default_model": "anthropic.claude-3-sonnet-20240229",
                "supported_models": [
                    "anthropic.claude-3-sonnet-20240229",
                    "anthropic.claude-3-haiku-20240307"
                ]
            })

        return json.dumps(config, indent=2)

    def _generate_deployment_script(self, runtime: str, agent_name: str) -> str:
        """Generate deployment script"""
        if runtime == "ollama":
            return f'''#!/bin/bash
# Deploy {agent_name} with Ollama runtime

echo "Deploying {agent_name} with Ollama runtime..."

# Check if Ollama is running
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Pull required models
echo "Pulling models..."
ollama pull llama3.2
ollama pull qwen3:8b

# Start the agent service
echo "Starting {agent_name} service..."
python -m agents.{agent_name}_service &

echo "Deployment complete!"
echo "Agent available at: http://localhost:8000"
'''
        else:  # bedrock
            return f'''#!/bin/bash
# Deploy {agent_name} with AWS Bedrock

echo "Deploying {agent_name} with AWS Bedrock..."

# Check AWS credentials
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "AWS credentials not configured"
    exit 1
fi

# Deploy CloudFormation stack
aws cloudformation deploy \\
  --template-file infrastructure/cloudformation.yml \\
  --stack-name {agent_name}-stack \\
  --parameter-overrides \\
    AgentName={agent_name} \\
    ModelId=anthropic.claude-3-sonnet-20240229 \\
  --capabilities CAPABILITY_IAM

if [ $? -eq 0 ]; then
    echo "CloudFormation deployment successful!"
    echo "Agent will be available in AWS Bedrock console"
else
    echo "CloudFormation deployment failed"
    exit 1
fi
'''


class UnifiedAgentBuilder:
    """Unified deterministic agent builder"""

    def __init__(self):
        self.token_tracker = TokenTracker()
        self.infrastructure_generator = AgentInfrastructureGenerator()
        self.output_dir = Path("generated_agents")
        self.output_dir.mkdir(exist_ok=True)

    def create_agent_deterministic(
        self,
        name: str,
        description: str,
        agent_type: str = "auto",
        runtime: str = "ollama",
        user_id: str = "anonymous",
        tools: Optional[List[str]] = None,
        enable_code_execution: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an agent deterministically with full infrastructure

        Args:
            name: Agent function name
            description: What the agent should do
            agent_type: Type of agent (research, coding, etc.)
            runtime: Runtime to use (ollama or bedrock)
            user_id: User identifier for tracking
            tools: List of tools to provide
            enable_code_execution: Enable code execution
            system_prompt: Custom system prompt

        Returns:
            Dictionary with generated files and metadata
        """

        logger.info(
            f"Creating agent '{name}' for user '{user_id}' with runtime '{runtime}'")

        # Validate inputs
        if not self._validate_inputs(name, description, runtime):
            raise ValueError("Invalid input parameters")

        # Generate deterministic specification
        spec = self._generate_agent_spec(
            name, description, agent_type, runtime, tools, enable_code_execution, system_prompt
        )

        # Generate agent code
        agent_code = self._generate_deterministic_agent_code(spec)

        # Generate infrastructure
        infrastructure_files = self._generate_infrastructure(spec, runtime)

        # Create output structure
        agent_dir = self.output_dir / f"{name}_{int(time.time())}"
        agent_dir.mkdir(exist_ok=True)

        # Write files
        generated_files = {}

        # Write agent code
        agent_file = agent_dir / f"{name}.py"
        agent_file.write_text(agent_code, encoding='utf-8')
        generated_files["agent_code"] = str(agent_file)

        # Write infrastructure files
        infra_dir = agent_dir / "infrastructure"
        infra_dir.mkdir(exist_ok=True)

        for filename, content in infrastructure_files.items():
            file_path = infra_dir / filename
            file_path.write_text(content, encoding='utf-8')
            generated_files[f"infrastructure_{filename}"] = str(file_path)

        # Write metadata
        metadata = self._generate_metadata(spec, user_id, runtime)
        metadata_file = agent_dir / "metadata.json"
        metadata_file.write_text(json.dumps(
            metadata, indent=2), encoding='utf-8')
        generated_files["metadata"] = str(metadata_file)

        # Write usage tracking
        usage_file = agent_dir / "usage_tracking.json"
        usage_file.write_text(json.dumps({
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "estimated_monthly_cost": self._estimate_cost(spec),
            "token_tracking_enabled": True
        }, indent=2), encoding='utf-8')
        generated_files["usage_tracking"] = str(usage_file)

        logger.info(
            f"Agent '{name}' created successfully for user '{user_id}'")

        return {
            "success": True,
            "agent_name": name,
            "user_id": user_id,
            "runtime": runtime,
            "files": generated_files,
            "metadata": metadata
        }

    def _validate_inputs(self, name: str, description: str, runtime: str) -> bool:
        """Validate input parameters"""
        if not name or not re.match(r'^[a-zA-Z0-9_]+$', name):
            logger.error("Invalid agent name")
            return False

        if not description or len(description) < 10:
            logger.error("Description too short")
            return False

        if runtime not in ["ollama", "bedrock"]:
            logger.error("Invalid runtime")
            return False

        return True

    def _generate_agent_spec(
        self,
        name: str,
        description: str,
        agent_type: str,
        runtime: str,
        tools: Optional[List[str]],
        enable_code_execution: bool,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate deterministic agent specification"""

        # Auto-detect agent type if not specified
        if agent_type == "auto":
            agent_type = self._detect_agent_type(description)

        # Get best model for task
        model_id = get_best_model_for_task(
            agent_type, require_code_execution=enable_code_execution)

        # Default tools based on agent type
        if not tools:
            tools = self._get_default_tools(agent_type)

        # Generate system prompt if not provided
        if not system_prompt:
            system_prompt = self._generate_system_prompt(
                name, description, agent_type)

        return {
            "name": name,
            "description": description,
            "agent_type": agent_type,
            "runtime": runtime,
            "model_id": model_id,
            "tools": tools,
            "system_prompt": system_prompt,
            "enable_code_execution": enable_code_execution,
            "sandbox_timeout": 45,
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    def _detect_agent_type(self, description: str) -> str:
        """Detect agent type from description"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["code", "program", "develop", "debug"]):
            return "coding"
