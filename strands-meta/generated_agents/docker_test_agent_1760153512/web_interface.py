#!/usr/bin/env python3
"""
Web interface for docker_test_agent
Provides HTTP API endpoints for interacting with the agent
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from typing import Optional
from datetime import datetime

# Import the agent
try:
    from docker_test_agent import docker_test_agent, AGENT_METADATA
except ImportError:
    # Fallback if import fails
    def docker_test_agent(query: str) -> str:
        return f"Agent not available: {query}"

    AGENT_METADATA = {
        "name": "docker_test_agent",
        "description": "Docker test agent (import failed)",
        "version": "1.0.0"
    }

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Docker Test Agent API",
    description="API for interacting with the docker_test_agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 1000

class QueryResponse(BaseModel):
    response: str
    agent_name: str
    timestamp: str
    version: str

class HealthResponse(BaseModel):
    status: str
    agent_name: str
    version: str
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Docker Test Agent API",
        "agent": AGENT_METADATA["name"],
        "version": AGENT_METADATA["version"],
        "description": AGENT_METADATA["description"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent_name=AGENT_METADATA["name"],
        version=AGENT_METADATA["version"],
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the agent"""
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        # Call the agent
        response = docker_test_agent(request.query)

        logger.info("Query processed successfully")

        return QueryResponse(
            response=response,
            agent_name=AGENT_METADATA["name"],
            timestamp=datetime.utcnow().isoformat(),
            version=AGENT_METADATA["version"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/info")
async def agent_info():
    """Get agent information"""
    return {
        "metadata": AGENT_METADATA,
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "info": "/info"
        },
        "ollama_available": os.getenv("OLLAMA_HOST", "not configured")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
