FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for tools
RUN pip install requests beautifulsoup4 httpx numpy pandas matplotlib seaborn sqlite3

# Copy agent code
COPY . .

# Expose port
EXPOSE 8000

# Run the agent
CMD ["python", "agent.py"]