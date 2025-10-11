FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy agent files
COPY docker_test_agent.py .
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

# Default command - run both ollama and test the agent
CMD ollama serve & sleep 15 && ollama pull llama3.2 && python test_agent.py
