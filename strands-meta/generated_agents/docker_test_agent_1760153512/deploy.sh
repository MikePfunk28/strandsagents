#!/bin/bash
# Deploy docker_test_agent agent

echo "Deploying docker_test_agent agent..."
echo "=================================="

# Option 1: Run with Docker Compose (recommended)
echo "Starting with Docker Compose..."
echo "The agent will be available at:"
echo "  - Web interface: http://localhost:8001"
echo "  - Health check: http://localhost:8001/health"
echo "  - API endpoint: http://localhost:8001/query"
echo "  - Ollama API: http://localhost:11434"
echo ""

docker-compose up --build

# Option 2: Run locally (requires Ollama installed)
# echo "Starting Ollama..."
# ollama serve &
# sleep 5
# echo "Starting agent..."
# python docker_test_agent.py
