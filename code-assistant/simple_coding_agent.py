"""Simplified coding agent that works with current StrandsAgents version."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from strands import Agent

from coding_assistant import CodingAssistant

logger = logging.getLogger(__name__)


class SimpleCodingAgent:
    """Simplified coding agent without complex workflow orchestration."""

    def __init__(
        self,
        db_dir: Path = Path("./data"),
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        session_id: Optional[str] = None,
    ):
        # Initialize the core assistant
        self.assistant = CodingAssistant(db_dir, ollama_host, model_name, session_id)

        # Simple workflow state tracking
        self.current_task = None
        self.task_history = []

        logger.info("SimpleCodingAgent initialized with session: %s", self.assistant.session_id)

    async def execute_task(
        self,
        task_description: str,
        task_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a coding task with simple workflow management."""
        task = {
            "id": str(uuid4()),
            "description": task_description,
            "type": task_type,
            "timestamp": time.time(),
            "status": "started",
            **kwargs
        }

        self.current_task = task
        logger.info("Starting task: %s (%s)", task_description[:50], task_type)

        try:
            # Execute based on task type
            if task_type == "analysis":
                result = await self._execute_analysis_task(task)
            elif task_type == "feature":
                result = await self._execute_feature_task(task)
            elif task_type == "debug":
                result = await self._execute_debug_task(task)
            elif task_type == "testing":
                result = await self._execute_testing_task(task)
            else:
                result = await self._execute_general_task(task)

            task["status"] = "completed"
            task["result"] = result

            # Store task in memory
            self.assistant.memory_manager.add_memory(
                content=f"Task completed: {task_description}\nResult: {str(result)[:500]}",
                session_id=self.assistant.session_id,
                memory_type="task_completion",
                importance=0.8,
                metadata={
                    "task_id": task["id"],
                    "task_type": task_type,
                    "success": result.get("success", True),
                }
            )

            self.task_history.append(task)

            return {
                "success": True,
                "task": task,
                "result": result,
                "session_id": self.assistant.session_id,
            }

        except Exception as e:
            logger.error("Task execution failed: %s", e)
            task["status"] = "failed"
            task["error"] = str(e)

            return {
                "success": False,
                "task": task,
                "error": str(e),
                "session_id": self.assistant.session_id,
            }

    async def _execute_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis task."""
        description = task["description"]

        # Check if we have a specific file or project path
        if "file_path" in task:
            analysis = self.assistant.analyze_code_context(task["file_path"])
            return {"type": "file_analysis", "analysis": analysis}
        elif "project_path" in task:
            summary = self.assistant.get_project_summary(task["project_path"])
            return {"type": "project_analysis", "summary": summary}
        else:
            # General analysis using chat
            response = self.assistant.chat(f"Please analyze: {description}")
            return {"type": "general_analysis", "analysis": response}

    async def _execute_feature_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature implementation task."""
        description = task["description"]

        # Use structured approach for feature implementation
        steps = [
            f"Plan the implementation for: {description}",
            f"Implement the feature: {description}",
            f"Test the implementation for: {description}",
            f"Review and optimize: {description}"
        ]

        results = []
        for i, step in enumerate(steps):
            logger.info("Feature step %d: %s", i + 1, step[:50])
            response = self.assistant.chat(step)
            results.append({
                "step": i + 1,
                "description": step,
                "result": response
            })

        return {"type": "feature_implementation", "steps": results}

    async def _execute_debug_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute debugging task."""
        description = task["description"]

        # Structured debugging approach
        debug_steps = [
            f"Analyze the error: {description}",
            f"Investigate root cause: {description}",
            f"Implement fix: {description}",
            f"Verify fix works: {description}"
        ]

        results = []
        for i, step in enumerate(debug_steps):
            logger.info("Debug step %d: %s", i + 1, step[:50])
            response = self.assistant.chat(step)
            results.append({
                "step": i + 1,
                "description": step,
                "result": response
            })

        return {"type": "debugging", "steps": results}

    async def _execute_testing_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing task."""
        description = task["description"]

        # Testing workflow
        test_steps = [
            f"Plan test strategy for: {description}",
            f"Write tests for: {description}",
            f"Run and validate tests for: {description}"
        ]

        results = []
        for i, step in enumerate(test_steps):
            logger.info("Test step %d: %s", i + 1, step[:50])
            response = self.assistant.chat(step)
            results.append({
                "step": i + 1,
                "description": step,
                "result": response
            })

        return {"type": "testing", "steps": results}

    async def _execute_general_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general task."""
        description = task["description"]
        response = self.assistant.chat(description)

        return {"type": "general", "response": response}

    def chat(self, message: str) -> str:
        """Simple chat interface."""
        return self.assistant.chat(message)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        base_stats = self.assistant.get_stats()
        base_stats.update({
            "current_task": self.current_task["id"] if self.current_task else None,
            "completed_tasks": len(self.task_history),
            "task_types": list(set(task["type"] for task in self.task_history)),
        })
        return base_stats

    def cleanup(self):
        """Cleanup resources."""
        self.assistant.cleanup()


# Factory function for easy creation
def create_simple_coding_agent(
    db_dir: str = "./data",
    ollama_host: str = "http://localhost:11434",
    model_name: str = "llama3.2",
    session_id: Optional[str] = None,
) -> SimpleCodingAgent:
    """Create a simple coding agent with default configuration."""
    return SimpleCodingAgent(
        db_dir=Path(db_dir),
        ollama_host=ollama_host,
        model_name=model_name,
        session_id=session_id,
    )