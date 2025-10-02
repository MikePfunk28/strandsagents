"""
Base Assistant Class

Defines the fundamental interface and functionality for all assistants in the swarm system.
Assistants are the basic building blocks that provide focused capabilities.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field

from strands import Agent
from strands.models.ollama import OllamaModel

logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """Configuration for an assistant instance."""
    name: str
    description: str
    model_id: str = "llama3.2"  # Default model
    host: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    tools: List[Any] = field(default_factory=list)
    max_tokens: int = 1000
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAssistant(ABC):
    """
    Abstract base class for all assistants in the swarm system.

    An assistant is a focused, single-purpose AI component that:
    - Has one primary responsibility
    - Uses a specific prompt and tool set
    - Operates with a dedicated model instance
    - Can be composed into more complex agents
    """

    def __init__(self, config: AssistantConfig):
        """
        Initialize the assistant.

        Args:
            config: Configuration for this assistant instance
        """
        self.config = config
        self.name = config.name
        self.description = config.description

        # Create the underlying model
        self.model = OllamaModel(
            host=config.host,
            model_id=config.model_id
        )

        # Create the underlying agent
        self.agent = Agent(
            model=self.model,
            tools=config.tools,
            system_prompt=config.system_prompt or self.get_default_prompt()
        )

        # State management
        self._is_active = False
        self._execution_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized assistant: {self.name}")

    @abstractmethod
    def get_default_prompt(self) -> str:
        """
        Get the default system prompt for this assistant type.

        Returns:
            The system prompt string
        """
        pass

    @abstractmethod
    async def execute_async(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the assistant's primary function asynchronously.

        Args:
            input_data: Input data for the assistant
            **kwargs: Additional arguments

        Returns:
            The assistant's output
        """
        pass

    def execute(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the assistant's primary function synchronously.

        Args:
            input_data: Input data for the assistant
            **kwargs: Additional arguments

        Returns:
            The assistant's output
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to create a task
                task = loop.create_task(self.execute_async(input_data, **kwargs))
                return task
            else:
                return loop.run_until_complete(self.execute_async(input_data, **kwargs))
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(self.execute_async(input_data, **kwargs))

    async def stream_async(self, input_data: Any, **kwargs) -> AsyncIterator[Any]:
        """
        Stream the assistant's execution asynchronously.

        Args:
            input_data: Input data for the assistant
            **kwargs: Additional arguments

        Yields:
            Streaming output from the assistant
        """
        # Default implementation - can be overridden by subclasses
        result = await self.execute_async(input_data, **kwargs)
        yield result

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about this assistant's capabilities.

        Returns:
            Dictionary describing the assistant's capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "model": self.config.model_id,
            "tools_count": len(self.config.tools),
            "tools": [getattr(tool, 'name', str(tool)) for tool in self.config.tools],
            "metadata": self.config.metadata
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this assistant.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_executions": len(self._execution_history),
            "is_active": self._is_active,
            "average_execution_time": self._calculate_average_execution_time()
        }

    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time from history."""
        if not self._execution_history:
            return 0.0

        total_time = sum(entry.get("execution_time", 0)
                        for entry in self._execution_history)
        return total_time / len(self._execution_history)

    def _record_execution(self, input_data: Any, output: Any, execution_time: float):
        """Record an execution in the history."""
        self._execution_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "input_size": len(str(input_data)),
            "output_size": len(str(output)),
            "execution_time": execution_time
        })

        # Keep only last 100 executions to prevent memory bloat
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

    def activate(self) -> None:
        """Activate the assistant."""
        self._is_active = True
        logger.info(f"Activated assistant: {self.name}")

    def deactivate(self) -> None:
        """Deactivate the assistant."""
        self._is_active = False
        logger.info(f"Deactivated assistant: {self.name}")

    def cleanup(self) -> None:
        """Clean up resources used by the assistant."""
        self.deactivate()
        logger.info(f"Cleaned up assistant: {self.name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', active={self._is_active})"
