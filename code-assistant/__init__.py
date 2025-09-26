"""Coding Assistant package with Ollama integration and semantic memory."""

from .coding_agent import CodingAgent, create_coding_agent
from .coding_assistant import CodingAssistant
from .database_manager import DatabaseManager
from .memory_manager import MemoryManager
from .ollama_model import OllamaModel, create_ollama_model

__all__ = [
    "CodingAgent",
    "create_coding_agent",
    "CodingAssistant",
    "DatabaseManager",
    "MemoryManager",
    "OllamaModel",
    "create_ollama_model",
]