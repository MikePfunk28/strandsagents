"""Swarm system for coordinating multiple AI agents using strands framework.

This package provides a comprehensive swarm intelligence system that coordinates
multiple AI agents using local Ollama models. The system includes:

- SwarmSystem: Main coordination class
- Database management with SQLite and embeddings
- MCP-based agent communication
- Lightweight assistant microservices
- Task orchestration and distribution

Usage:
    from swarm import SwarmSystem, create_research_swarm

    # Create and run a basic swarm
    swarm = SwarmSystem()
    await swarm.start()
    result = await swarm.process_task("Analyze AI trends")
    await swarm.stop()

    # Or use a specialized swarm
    research_swarm = create_research_swarm()
    await research_swarm.run_until_shutdown()
"""

from .main import (
    SwarmSystem,
    create_research_swarm,
    create_development_swarm,
    create_creative_swarm
)

from .storage.database_manager import DatabaseManager
from .coordinator.orchestrator import SwarmOrchestrator
from .agents.base_assistant import BaseAssistant, create_lightweight_assistant

__version__ = "1.0.0"
__author__ = "Swarm Development Team"

__all__ = [
    "SwarmSystem",
    "create_research_swarm",
    "create_development_swarm",
    "create_creative_swarm",
    "DatabaseManager",
    "SwarmOrchestrator",
    "BaseAssistant",
    "create_lightweight_assistant"
]