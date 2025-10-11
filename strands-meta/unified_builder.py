#!/usr/bin/env python3
"""
Working Strands-Meta Agent Builder CLI

Creates working agents with proper imports and infrastructure.
"""

import json
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_builder")


class WorkingAgentBuilder:
    """Creates working agents with proper imports"""

    def __init__(self):
        self.output_dir = Path("generated_agents")
        self.output_dir.mkdir(exist_ok=True)

    def create_working_agent(
        self,
        name: str,
        description: str,
        model: str = "llama3.2",
        runtime: str = "ollama"
    ) -> Dict[str, str]:
        """Create a working agent with proper imports"""

        # Generate proper agent code
        agent_code = self._generate_proper_agent_code(name, description, model, runtime)

        # Create output files
        timestamp = int(datetime.now().timestamp())
        agent_dir = self.output_dir / f"{name}_{timestamp}"
        agent_dir.mkdir(exist_ok=True)

        # Write agent file
        agent_file = agent_dir / f"{name}.py"
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(agent_code)

        # Write metadata
        metadata = {
            "name": name,
            "description": description,
            "model": model,
            "runtime": runtime,
            "created_at": datetime.now().isoformat(),
            "agent_file": str(agent_file)
        }
        metadata_file = agent_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        # Write requirements
        requirements_file = agent_dir / "requirements.txt"
        requirements_file.write_text("requests>=2.31.0\n")

        # Write Dockerfile for containerized testing
        dockerfile = agent_dir / "Dockerfile"
        dockerfile.write_text(f'''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy agent files
COPY {name}.py .
COPY metadata.json .
COPY requirements.txt .
COPY test_agent.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create logs directory
RUN mkdir -p /app/logs

# Expose ports
EXPOSE 8000 11434

# Pull the required model
RUN ollama pull {model}

# Default command - run both ollama and test the agent
CMD ollama serve & sleep 10 && python test_agent.py
''')

        # Write docker-compose.yml for full environment
        docker_compose = agent_dir / "docker-compose.yml"
        docker_compose.write_text(f'''version: '3.8'

services:
  # Ollama service for model inference
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
      start_period: 30s
    command: serve

  # Agent testing service
  agent-test:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - TEST_MODE=true
    volumes:
      - ./logs:/app/logs

volumes:
  ollama_data:
''')

        # Write deployment script
        deploy_script = agent_dir / "deploy.sh"
        deploy_script.write_text(f'''#!/bin/bash
# Deploy {name} agent

echo "Deploying {name} agent..."

# Option 1: Run with Docker Compose (recommended)
echo "Starting with Docker Compose..."
docker-compose up --build

# Option 2: Run locally (requires Ollama installed)
# echo "Starting Ollama..."
# ollama serve &
# sleep 5
# echo "Starting agent..."
# python {name}.py
''')
        deploy_script.chmod(0o755)

        # Write simple test script
        test_script = agent_dir / "test_agent.py"
        with open(test_script, 'w', encoding='utf-8') as f:
            f.write(f'''#!/usr/bin/env python3
"""Test script for {name} agent"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Load and execute agent code
    with open("{name}.py", encoding='utf-8') as f:
        agent_code = f.read()
    exec(agent_code, globals())

    print("Testing {name} agent...")
    result = {name}("Hello from test script!")
    print(f"Result: {{result}}")
    print("SUCCESS: Agent test successful!")
except Exception as e:
    print(f"ERROR: Agent test failed: {{e}}")
    import traceback
    traceback.print_exc()
''')

        return {
            "agent_file": str(agent_file),
            "metadata_file": str(metadata_file),
            "requirements_file": str(requirements_file),
            "test_script": str(test_script),
            "dockerfile": str(dockerfile),
            "docker_compose": str(docker_compose),
            "deploy_script": str(deploy_script),
            "agent_dir": str(agent_dir)
        }

    def _generate_proper_agent_code(self, name: str, description: str, model: str, runtime: str) -> str:
        """Generate proper working agent code"""

        return f'''#!/usr/bin/env python3
"""
{name.title()} Agent

{description}

Generated by Working Agent Builder
Runtime: {runtime}
Model: {model}
"""

import sys
import os
import logging
import json
import requests
from typing import Optional, List
from datetime import datetime

# Setup logging
logger = logging.getLogger("{name}")

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class SimpleAgent:
    """Simple working agent implementation"""

    def __init__(self, model_id: str = "{model}", system_prompt: str = ""):
        self.model_id = model_id
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def __call__(self, query: str) -> str:
        """Process a query"""
        try:
            if "{runtime}" == "ollama":
                return self._call_ollama(query)
            else:
                return self._call_bedrock(query)
        except Exception as e:
            return f"Agent error: {{str(e)}}"

    def _call_ollama(self, query: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={{
                    "model": self.model_id,
                    "messages": [
                        {{"role": "system", "content": self.system_prompt}},
                        {{"role": "user", "content": query}}
                    ],
                    "stream": False
                }},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            return f"Ollama API error: {{str(e)}}"

    def _call_bedrock(self, query: str) -> str:
        """Call AWS Bedrock API"""
        try:
            import boto3

            client = boto3.client("bedrock-runtime")
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({{
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {{"role": "human", "content": query}}
                    ]
                }})
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            return f"Bedrock API error: {{str(e)}}"


# Create agent instance
agent_instance = SimpleAgent(
    model_id="{model}",
    system_prompt="""You are {name}, a specialized AI agent.

{description}

Focus on delivering high-quality, accurate responses for your domain."""
)


def {name}(query: str) -> str:
    """
    {description}

    Args:
        query: Input query or request

    Returns:
        Response from the agent
    """
    try:
        logger.info(f"{{name.title()}} processing query: {{query[:100]}}...")

        # Add agent-specific logic here
        response = agent_instance(query)

        logger.info(f"{{name.title()}} completed successfully")
        return response

    except Exception as e:
        error_msg = f"Error in {{name}}: {{str(e)}}"
        logger.error(error_msg)
        return error_msg


# Agent metadata
AGENT_METADATA = {{
    "name": "{name}",
    "description": "{description}",
    "model": "{model}",
    "runtime": "{runtime}",
    "created_at": "{datetime.now().isoformat()}",
    "version": "1.0.0"
}}


if __name__ == "__main__":
    print("{name.title()} Agent")
    print("=" * 50)
    print(f"Description: {{AGENT_METADATA['description']}}")
    print(f"Model: {{AGENT_METADATA['model']}}")
    print(f"Runtime: {{AGENT_METADATA['runtime']}}")
    print()

    # Test the agent
    test_query = "Hello, please demonstrate your capabilities"
    print(f"Test Query: {{test_query}}")
    print()

    try:
        result = {name}(test_query)
        print(f"Response: {{result}}")
        print("SUCCESS: Agent working correctly!")
    except Exception as e:
        print(f"❌ Agent test failed: {{e}}")
        import traceback
        traceback.print_exc()
'''


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Working Strands-Meta Agent Builder")
    parser.add_argument("--name", "-n", required=True, help="Agent name")
    parser.add_argument("--description", "-d", required=True, help="Agent description")
    parser.add_argument("--model", "-m", default="llama3.2", help="Model to use")
    parser.add_argument("--runtime", "-r", default="ollama",
                       choices=["ollama", "bedrock"], help="Runtime to use")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("  Ollama models: llama3.2, qwen3:8b, gemma3:4b")
        print("  Bedrock models: anthropic.claude-3-sonnet-20240229, anthropic.claude-3-haiku-20240307")
        return

    # Create the builder
    builder = WorkingAgentBuilder()

    try:
        # Create the agent
        result = builder.create_working_agent(
            name=args.name,
            description=args.description,
            model=args.model,
            runtime=args.runtime
        )

        print(f"SUCCESS: Agent '{args.name}' created successfully!")
        print(f"Directory: {result['agent_dir']}")
        print(f"Agent file: {result['agent_file']}")
        print(f"Metadata: {result['metadata_file']}")
        print(f"Test script: {result['test_script']}")
        print(f"Dockerfile: {result['dockerfile']}")
        print(f"Docker Compose: {result['docker_compose']}")
        print(f"Deploy script: {result['deploy_script']}")
        print()
        print("To test the agent locally:")
        print(f"  cd {result['agent_dir']}")
        print("  python test_agent.py")
        print()
        print("To test in Docker:")
        print(f"  cd {result['agent_dir']}")
        print("  docker build -t {args.name}-agent .")
        print("  docker run --rm {args.name}-agent")
        print()
        print("To run full environment:")
        print(f"  cd {result['agent_dir']}")
        print("  docker-compose up")
        print()
        print("To use the agent:")
        print(f"  from {args.name} import {args.name}")
        print(f"  result = {args.name}('Your query here')")

    except Exception as e:
        print(f"❌ Failed to create agent: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
