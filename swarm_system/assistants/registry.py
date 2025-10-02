"""
Assistant Registry System

Manages the registration, discovery, and instantiation of assistant classes.
Provides a centralized way to manage all available assistants in the swarm system.
"""

import logging
from typing import Dict, Type, Any, Optional, List
from .base_assistant import BaseAssistant

logger = logging.getLogger(__name__)


class AssistantRegistry:
    """
    Registry for managing assistant types and instances.

    Provides functionality to:
    - Register new assistant types
    - Retrieve assistant instances by name
    - List available assistants
    - Manage assistant lifecycle
    """

    def __init__(self):
        self._assistants: Dict[str, Type[BaseAssistant]] = {}
        self._instances: Dict[str, BaseAssistant] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        assistant_class: Type[BaseAssistant],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new assistant type.

        Args:
            name: Unique name for the assistant type
            assistant_class: The assistant class to register
            metadata: Optional metadata about the assistant
        """
        if not issubclass(assistant_class, BaseAssistant):
            raise ValueError(f"Assistant class must inherit from BaseAssistant")

        self._assistants[name] = assistant_class
        self._metadata[name] = metadata or {}

        logger.info(f"Registered assistant type: {name}")

    def get_class(self, name: str) -> Type[BaseAssistant]:
        """
        Get an assistant class by name.

        Args:
            name: Name of the assistant type

        Returns:
            The assistant class

        Raises:
            KeyError: If assistant type not found
        """
        if name not in self._assistants:
            raise KeyError(f"Assistant type '{name}' not found")

        return self._assistants[name]

    def create_instance(
        self,
        name: str,
        instance_name: Optional[str] = None,
        **kwargs
    ) -> BaseAssistant:
        """
        Create a new instance of an assistant.

        Args:
            name: Name of the assistant type
            instance_name: Optional unique name for this instance
            **kwargs: Arguments to pass to assistant constructor

        Returns:
            New assistant instance
        """
        assistant_class = self.get_class(name)
        instance = assistant_class(**kwargs)

        if instance_name:
            if instance_name in self._instances:
                raise ValueError(f"Instance name '{instance_name}' already exists")
            self._instances[instance_name] = instance
            instance.name = instance_name

        logger.info(f"Created assistant instance: {name} ({instance_name or 'unnamed'})")
        return instance

    def get_instance(self, name: str) -> BaseAssistant:
        """
        Get an existing assistant instance by name.

        Args:
            name: Name of the instance

        Returns:
            The assistant instance

        Raises:
            KeyError: If instance not found
        """
        if name not in self._instances:
            raise KeyError(f"Assistant instance '{name}' not found")

        return self._instances[name]

    def list_available_types(self) -> List[str]:
        """Get list of all registered assistant types."""
        return list(self._assistants.keys())

    def list_instances(self) -> List[str]:
        """Get list of all assistant instance names."""
        return list(self._instances.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for an assistant type.

        Args:
            name: Name of the assistant type

        Returns:
            Metadata dictionary
        """
        return self._metadata.get(name, {})

    def unregister(self, name: str) -> None:
        """
        Unregister an assistant type.

        Args:
            name: Name of the assistant type to remove
        """
        if name in self._assistants:
            del self._assistants[name]
            del self._metadata[name]
            logger.info(f"Unregistered assistant type: {name}")

    def destroy_instance(self, name: str) -> None:
        """
        Destroy an assistant instance.

        Args:
            name: Name of the instance to destroy
        """
        if name in self._instances:
            instance = self._instances[name]
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
            del self._instances[name]
            logger.info(f"Destroyed assistant instance: {name}")

    def clear(self) -> None:
        """Clear all registered assistants and instances."""
        # Cleanup instances first
        for instance in self._instances.values():
            if hasattr(instance, 'cleanup'):
                instance.cleanup()

        self._assistants.clear()
        self._instances.clear()
        self._metadata.clear()
        logger.info("Cleared all assistants and instances")


# Global registry instance
global_registry = AssistantRegistry()
