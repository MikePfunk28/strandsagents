FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "No requirements.txt found"

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create logs directory
RUN mkdir -p /app/logs

# Expose ports
EXPOSE 8000 11434

# Pull default models
RUN ollama pull llama3.2 2>/dev/null || echo "Ollama not available during build"
RUN ollama pull qwen3:8b 2>/dev/null || echo "Ollama not available during build"

# Default command
CMD ollama serve & sleep 15 && python app/main.py
