"""
Swarm System Assistants Package

This package contains the foundational assistant classes that serve as building blocks
for the hierarchical assistant → agent → swarm architecture.

Core Concepts:
- Assistant: Simple, focused building block with prompt + tools + model
- Agent: Composition of multiple assistants with additional code/logic
- Swarm: Collection of lightweight agents coordinated by orchestrator
"""

from .registry import AssistantRegistry
from .base_assistant import BaseAssistant

__all__ = ['AssistantRegistry', 'BaseAssistant']
