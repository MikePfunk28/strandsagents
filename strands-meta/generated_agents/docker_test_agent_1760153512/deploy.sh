#!/bin/bash
# Deploy docker_test_agent agent

echo "Deploying docker_test_agent agent..."

# Option 1: Run with Docker Compose (recommended)
echo "Starting with Docker Compose..."
docker-compose up --build

# Option 2: Run locally (requires Ollama installed)
# echo "Starting Ollama..."
# ollama serve &
# sleep 5
# echo "Starting agent..."
# python docker_test_agent.py
