"""Main swarm system launcher with real functionality.

This module provides the SwarmSystem class for coordinating multiple AI agents
using the strands framework with local Ollama models.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from strands import Agent
from .storage.database_manager import DatabaseManager
from .communication.mcp_client import SwarmMCPClient
from .coordinator.orchestrator import SwarmOrchestrator
from .agents.base_assistant import create_lightweight_assistant

logger = logging.getLogger(__name__)

class SwarmSystem:
    """Main swarm coordination system using local Ollama models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize swarm system with configurable local models.

        Args:
            config: Optional configuration dict with model settings
        """
        self.config = config or self._default_config()
        self.running = False
        self.shutdown_requested = False

        # Core components
        self.database_manager: Optional[DatabaseManager] = None
        self.orchestrator: Optional[SwarmOrchestrator] = None
        self.assistants: Dict[str, Any] = {}
        self.mcp_clients: List[SwarmMCPClient] = []

        # Model configuration
        self.orchestrator_model = self.config.get("orchestrator_model", "llama3.2:3b")
        self.assistant_model = self.config.get("assistant_model", "gemma:270m")
        self.available_models = self.config.get("available_models", [
            "gemma:270m", "llama3.2:3b", "qwen:3b", "phi:4b"
        ])

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for local development."""
        return {
            "orchestrator_model": "llama3.2:3b",
            "assistant_model": "gemma:270m",
            "available_models": ["gemma:270m", "llama3.2:3b", "qwen:3b", "phi:4b"],
            "host": "localhost:11434",
            "database_path": "swarm_data",
            "max_assistants": 10,
            "log_level": "INFO"
        }

    async def initialize(self):
        """Initialize all swarm components."""
        try:
            logger.info("Initializing SwarmSystem...")

            # Initialize database manager
            self.database_manager = DatabaseManager(
                db_path=self.config.get("database_path", "swarm_data")
            )
            await self.database_manager.initialize()
            logger.info("Database manager initialized")

            # Initialize orchestrator with real model
            self.orchestrator = SwarmOrchestrator(
                orchestrator_id="main_orchestrator",
                model_name=self.orchestrator_model,
                host=self.config.get("host", "localhost:11434"),
                database_manager=self.database_manager
            )
            await self.orchestrator.initialize()
            logger.info(f"Orchestrator initialized with {self.orchestrator_model}")

            # Create initial set of lightweight assistants
            await self._create_initial_assistants()

            logger.info("SwarmSystem initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize SwarmSystem: {e}")
            raise

    async def _create_initial_assistants(self):
        """Create initial set of lightweight assistants."""
        assistant_types = ["research", "creative", "critical", "summarizer", "code_feedback"]

        for assistant_type in assistant_types:
            try:
                assistant = create_lightweight_assistant(assistant_type)
                assistant.model_name = self.assistant_model
                assistant.host = self.config.get("host", "localhost:11434")

                await assistant.initialize()
                self.assistants[assistant.assistant_id] = assistant

                # Track MCP client for cleanup
                if assistant.mcp_client:
                    self.mcp_clients.append(assistant.mcp_client)

                logger.info(f"Created {assistant_type} assistant: {assistant.assistant_id}")

            except Exception as e:
                logger.error(f"Failed to create {assistant_type} assistant: {e}")

    async def start(self):
        """Start the swarm system."""
        if self.running:
            logger.warning("SwarmSystem is already running")
            return

        try:
            await self.initialize()

            # Start orchestrator
            if self.orchestrator:
                await self.orchestrator.start_service()

            # Start all assistants
            for assistant in self.assistants.values():
                await assistant.start_service()

            self.running = True
            logger.info("SwarmSystem started successfully")

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        except Exception as e:
            logger.error(f"Failed to start SwarmSystem: {e}")
            await self.stop()
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    async def stop(self):
        """Stop the swarm system gracefully."""
        if not self.running:
            return

        logger.info("Stopping SwarmSystem...")
        self.running = False

        try:
            # Stop all assistants
            for assistant in self.assistants.values():
                try:
                    await assistant.stop_service()
                except Exception as e:
                    logger.error(f"Error stopping assistant: {e}")

            # Stop orchestrator
            if self.orchestrator:
                try:
                    await self.orchestrator.stop_service()
                except Exception as e:
                    logger.error(f"Error stopping orchestrator: {e}")

            # Close database connections
            if self.database_manager:
                try:
                    await self.database_manager.close()
                except Exception as e:
                    logger.error(f"Error closing database: {e}")

            logger.info("SwarmSystem stopped")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a task using the swarm system.

        Args:
            task_description: Description of the task to process
            context: Optional context for the task

        Returns:
            Task result dictionary
        """
        if not self.running:
            raise RuntimeError("SwarmSystem is not running")

        if not self.orchestrator:
            raise RuntimeError("Orchestrator not available")

        task = {
            "description": task_description,
            "context": context or {},
            "timestamp": asyncio.get_event_loop().time()
        }

        logger.info(f"Processing task: {task_description}")
        result = await self.orchestrator.distribute_task(task)

        # Store task and result in database
        if self.database_manager:
            await self.database_manager.store_code_entry(
                file_path=f"task_{task.get('timestamp', 'unknown')}",
                content=task_description,
                tags=["task"],
                metadata={"result": result}
            )

        return result

    async def switch_model(self, component: str, new_model: str):
        """Switch model for a specific component.

        Args:
            component: 'orchestrator' or 'assistants'
            new_model: Model name to switch to
        """
        if new_model not in self.available_models:
            raise ValueError(f"Model {new_model} not in available models: {self.available_models}")

        if component == "orchestrator" and self.orchestrator:
            logger.info(f"Switching orchestrator model from {self.orchestrator_model} to {new_model}")
            self.orchestrator_model = new_model
            self.orchestrator.model_name = new_model
            # Orchestrator will use new model on next task

        elif component == "assistants":
            logger.info(f"Switching assistant model from {self.assistant_model} to {new_model}")
            self.assistant_model = new_model
            for assistant in self.assistants.values():
                assistant.model_name = new_model
                # Assistants will use new model on next task

        else:
            raise ValueError(f"Unknown component: {component}")

    def get_status(self) -> Dict[str, Any]:
        """Get current swarm system status."""
        return {
            "running": self.running,
            "orchestrator_model": self.orchestrator_model,
            "assistant_model": self.assistant_model,
            "available_models": self.available_models,
            "assistants": {aid: assistant.get_status() for aid, assistant in self.assistants.items()},
            "orchestrator_status": self.orchestrator.get_status() if self.orchestrator else None,
            "database_connected": self.database_manager is not None
        }

    async def run_until_shutdown(self):
        """Run the swarm system until shutdown is requested."""
        await self.start()

        try:
            while self.running and not self.shutdown_requested:
                await asyncio.sleep(1)
        finally:
            await self.stop()

# Factory functions for different swarm configurations

def create_research_swarm(config: Optional[Dict[str, Any]] = None) -> SwarmSystem:
    """Create a swarm optimized for research tasks."""
    default_config = {
        "orchestrator_model": "llama3.2:3b",
        "assistant_model": "gemma:270m",
        "max_assistants": 5
    }
    if config:
        default_config.update(config)
    return SwarmSystem(default_config)

def create_development_swarm(config: Optional[Dict[str, Any]] = None) -> SwarmSystem:
    """Create a swarm optimized for development tasks."""
    default_config = {
        "orchestrator_model": "qwen:3b",
        "assistant_model": "phi:4b",
        "max_assistants": 8
    }
    if config:
        default_config.update(config)
    return SwarmSystem(default_config)

def create_creative_swarm(config: Optional[Dict[str, Any]] = None) -> SwarmSystem:
    """Create a swarm optimized for creative tasks."""
    default_config = {
        "orchestrator_model": "llama3.2:3b",
        "assistant_model": "gemma:270m",
        "max_assistants": 6
    }
    if config:
        default_config.update(config)
    return SwarmSystem(default_config)

# CLI and demo functions (clearly marked as examples)

async def demo_basic_task():
    """Demo function: Shows basic task processing."""
    print("🔄 Demo: Basic task processing")

    swarm = SwarmSystem()
    try:
        await swarm.start()

        result = await swarm.process_task(
            "Analyze the benefits of using local AI models",
            context={"domain": "ai_research"}
        )

        print(f"✅ Task completed: {result.get('status', 'unknown')}")
        print(f"📊 Result: {result.get('result', 'No result')}")

    finally:
        await swarm.stop()

async def demo_model_switching():
    """Demo function: Shows model switching capabilities."""
    print("🔄 Demo: Model switching")

    swarm = SwarmSystem()
    try:
        await swarm.start()

        print(f"Initial orchestrator model: {swarm.orchestrator_model}")
        print(f"Initial assistant model: {swarm.assistant_model}")

        # Switch models
        await swarm.switch_model("orchestrator", "qwen:3b")
        await swarm.switch_model("assistants", "phi:4b")

        print(f"New orchestrator model: {swarm.orchestrator_model}")
        print(f"New assistant model: {swarm.assistant_model}")

        # Process a task with new models
        result = await swarm.process_task("Test with new models")
        print(f"✅ Task with new models: {result.get('status', 'unknown')}")

    finally:
        await swarm.stop()

def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Swarm System CLI")
    parser.add_argument("--mode", choices=["run", "demo-basic", "demo-switching"],
                       default="run", help="Operation mode")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--orchestrator-model", default="llama3.2:3b",
                       help="Model for orchestrator")
    parser.add_argument("--assistant-model", default="gemma:270m",
                       help="Model for assistants")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = {
        "orchestrator_model": args.orchestrator_model,
        "assistant_model": args.assistant_model
    }

    if args.config and os.path.exists(args.config):
        import json
        with open(args.config) as f:
            file_config = json.load(f)
            config.update(file_config)

    # Run based on mode
    if args.mode == "run":
        swarm = SwarmSystem(config)
        asyncio.run(swarm.run_until_shutdown())
    elif args.mode == "demo-basic":
        asyncio.run(demo_basic_task())
    elif args.mode == "demo-switching":
        asyncio.run(demo_model_switching())

if __name__ == "__main__":
    main()