# Agent Builder Platform - Development Sandbox
# Multi-stage Docker container for agent development and testing
# Enhanced with Python 3.11, Node.js, AWS CLI, and development tools

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    jq \
    unzip \
    telnet \
    net-tools \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g yarn

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

# Install Docker CLI (for Docker-in-Docker if needed)
RUN curl -fsSL https://get.docker.com | sh

# Create app user
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app \
    && chown -R app:app /app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter \
    notebook

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/generated_agents /app/memory_vectors /app/logs \
    && chown -R app:app /app

# Switch to app user
USER app

# Expose ports
EXPOSE 8000 3000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('Agent Builder Platform ready')" || exit 1

# Default command
CMD ["python", "main.py"]

# Development stage with hot reload
FROM base as development

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    watchdog \
    reloadium

USER app

# Development command with auto-reload
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "main.py"]

# Production stage
FROM base as production

# Production optimizations
USER root
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip cache purge

USER app

# Production command
CMD ["python", "main.py"]

# Testing stage
FROM base as testing

# Install testing dependencies
USER root
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-benchmark \
    coverage \
    moto \
    responses

USER app

# Testing command
CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=html"]
