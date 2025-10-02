"""
Swarm System Package

Hierarchical assistant → agent → swarm architecture with meta-tooling capabilities.
Built on StrandsAgents framework with local Ollama models.
"""

__version__ = "1.0.0"
__author__ = "StrandsAgents"

# Package-level imports and configuration
from .utils.database_manager import db_manager

__all__ = ["db_manager"]
