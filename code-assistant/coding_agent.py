"""Coding Agent with StrandsAgents workflow orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from strands import Agent
# Note: Using simplified workflow implementation as Workflow is not available in current strands version

from coding_assistant import CodingAssistant
from database_manager import DatabaseManager
from memory_manager import MemoryManager
from ollama_model import create_ollama_model

logger = logging.getLogger(__name__)


class CodingAgentCallbackHandler(CallbackHandler):
    """Custom callback handler for coding agent workflow."""

    def __init__(self, assistant: CodingAssistant):
        self.assistant = assistant
        self.workflow_id = str(uuid4())
        self.step_count = 0

    async def on_workflow_start(self, workflow_data: Dict[str, Any]):
        """Called when workflow starts."""
        logger.info("Coding workflow started: %s", self.workflow_id)

        # Store workflow start in memory
        self.assistant.memory_manager.add_memory(
            content=f"Workflow started: {workflow_data.get('description', 'Coding task')}",
            session_id=self.assistant.session_id,
            memory_type="workflow",
            importance=0.7,
            metadata={
                "workflow_id": self.workflow_id,
                "event": "start",
                "task_type": workflow_data.get("task_type", "general"),
            }
        )

    async def on_workflow_complete(self, workflow_data: Dict[str, Any], result: Any):
        """Called when workflow completes."""
        logger.info("Coding workflow completed: %s", self.workflow_id)

        # Store workflow completion in memory
        self.assistant.memory_manager.add_memory(
            content=f"Workflow completed successfully: {str(result)[:500]}",
            session_id=self.assistant.session_id,
            memory_type="workflow",
            importance=0.8,
            metadata={
                "workflow_id": self.workflow_id,
                "event": "complete",
                "steps_executed": self.step_count,
                "success": True,
            }
        )

    async def on_workflow_error(self, workflow_data: Dict[str, Any], error: Exception):
        """Called when workflow encounters an error."""
        logger.error("Coding workflow error: %s - %s", self.workflow_id, error)

        # Store workflow error in memory
        self.assistant.memory_manager.add_memory(
            content=f"Workflow failed with error: {str(error)}",
            session_id=self.assistant.session_id,
            memory_type="workflow",
            importance=0.9,  # High importance for errors
            metadata={
                "workflow_id": self.workflow_id,
                "event": "error",
                "error_type": type(error).__name__,
                "steps_executed": self.step_count,
                "success": False,
            }
        )

    async def on_step_start(self, step_name: str, step_data: Dict[str, Any]):
        """Called when a workflow step starts."""
        self.step_count += 1
        logger.debug("Step started: %s (%d)", step_name, self.step_count)

    async def on_step_complete(self, step_name: str, step_data: Dict[str, Any], result: Any):
        """Called when a workflow step completes."""
        logger.debug("Step completed: %s", step_name)

        # Store important step results
        if step_name in ["code_analysis", "test_execution", "code_generation"]:
            self.assistant.memory_manager.add_memory(
                content=f"Step '{step_name}' result: {str(result)[:300]}",
                session_id=self.assistant.session_id,
                memory_type="step_result",
                importance=0.6,
                metadata={
                    "workflow_id": self.workflow_id,
                    "step_name": step_name,
                    "step_number": self.step_count,
                }
            )

    async def on_tool_use(self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any):
        """Called when a tool is used."""
        logger.debug("Tool used: %s", tool_name)


class CodingAgentOrchestrator(Orchestrator):
    """Custom orchestrator for coding tasks."""

    def __init__(self, assistant: CodingAssistant):
        self.assistant = assistant
        super().__init__()

    async def plan_workflow(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan workflow steps based on the task."""
        task_type = task.get("type", "general")
        description = task.get("description", "")

        # Retrieve relevant memories for planning
        relevant_memories = self.assistant.memory_manager.retrieve_memory(
            description,
            self.assistant.session_id,
            memory_types=["workflow", "code_analysis", "project_analysis"],
            limit=3
        )

        # Build context for planning
        context = f"Task: {description}\nType: {task_type}"
        if relevant_memories:
            context += "\nRelevant past experience:"
            for memory in relevant_memories:
                context += f"\n- {memory.content[:200]}"

        # Generate workflow plan using LLM
        planning_prompt = f"""
        Plan a workflow for this coding task:

        {context}

        Available workflow steps:
        - analysis: Analyze code/project structure
        - planning: Plan implementation approach
        - implementation: Write/modify code
        - testing: Run tests and validate
        - documentation: Update documentation
        - review: Code review and optimization

        Respond with a JSON array of workflow steps, each with:
        - name: step name
        - description: what this step does
        - dependencies: list of prerequisite steps
        - estimated_time: rough time estimate

        Example:
        [
          {{"name": "analysis", "description": "Analyze existing code", "dependencies": [], "estimated_time": "5 minutes"}},
          {{"name": "implementation", "description": "Implement new feature", "dependencies": ["analysis"], "estimated_time": "15 minutes"}}
        ]
        """

        try:
            plan_response = self.assistant.ollama_model.generate(planning_prompt, max_tokens=500)

            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', plan_response, re.DOTALL)
            if json_match:
                workflow_steps = json.loads(json_match.group())
                logger.info("Generated workflow plan with %d steps", len(workflow_steps))
                return workflow_steps
            else:
                # Fallback to default plan
                return self._default_workflow_plan(task_type)

        except Exception as e:
            logger.error("Error generating workflow plan: %s", e)
            return self._default_workflow_plan(task_type)

    def _default_workflow_plan(self, task_type: str) -> List[Dict[str, Any]]:
        """Generate default workflow plan based on task type."""
        if task_type == "debug":
            return [
                {"name": "analysis", "description": "Analyze error and code", "dependencies": [], "estimated_time": "5 minutes"},
                {"name": "investigation", "description": "Investigate root cause", "dependencies": ["analysis"], "estimated_time": "10 minutes"},
                {"name": "fix", "description": "Implement fix", "dependencies": ["investigation"], "estimated_time": "15 minutes"},
                {"name": "testing", "description": "Test the fix", "dependencies": ["fix"], "estimated_time": "10 minutes"},
            ]
        elif task_type == "feature":
            return [
                {"name": "planning", "description": "Plan feature implementation", "dependencies": [], "estimated_time": "10 minutes"},
                {"name": "implementation", "description": "Implement feature", "dependencies": ["planning"], "estimated_time": "30 minutes"},
                {"name": "testing", "description": "Write and run tests", "dependencies": ["implementation"], "estimated_time": "15 minutes"},
                {"name": "documentation", "description": "Update documentation", "dependencies": ["testing"], "estimated_time": "10 minutes"},
            ]
        elif task_type == "analysis":
            return [
                {"name": "code_analysis", "description": "Analyze code structure", "dependencies": [], "estimated_time": "10 minutes"},
                {"name": "documentation", "description": "Document findings", "dependencies": ["code_analysis"], "estimated_time": "15 minutes"},
            ]
        else:
            return [
                {"name": "analysis", "description": "Analyze the task", "dependencies": [], "estimated_time": "5 minutes"},
                {"name": "implementation", "description": "Execute the task", "dependencies": ["analysis"], "estimated_time": "20 minutes"},
                {"name": "review", "description": "Review results", "dependencies": ["implementation"], "estimated_time": "5 minutes"},
            ]


class CodingAgentExecutor(Executor):
    """Custom executor for coding workflow steps."""

    def __init__(self, assistant: CodingAssistant):
        self.assistant = assistant
        super().__init__()

    async def execute_step(self, step_name: str, step_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a specific workflow step."""
        logger.info("Executing step: %s", step_name)

        try:
            if step_name == "analysis":
                return await self._execute_analysis(step_data, context)
            elif step_name == "code_analysis":
                return await self._execute_code_analysis(step_data, context)
            elif step_name == "planning":
                return await self._execute_planning(step_data, context)
            elif step_name == "implementation":
                return await self._execute_implementation(step_data, context)
            elif step_name == "testing":
                return await self._execute_testing(step_data, context)
            elif step_name == "documentation":
                return await self._execute_documentation(step_data, context)
            elif step_name == "review":
                return await self._execute_review(step_data, context)
            elif step_name == "investigation":
                return await self._execute_investigation(step_data, context)
            elif step_name == "fix":
                return await self._execute_fix(step_data, context)
            else:
                return await self._execute_generic(step_name, step_data, context)

        except Exception as e:
            logger.error("Error executing step %s: %s", step_name, e)
            return {"success": False, "error": str(e)}

    async def _execute_analysis(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis step."""
        task_description = context.get("task", {}).get("description", "")

        # Use assistant's chat interface for analysis
        analysis_query = f"Analyze this task and provide insights: {task_description}"
        analysis_result = self.assistant.chat(analysis_query)

        return {
            "success": True,
            "analysis": analysis_result,
            "step": "analysis"
        }

    async def _execute_code_analysis(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code analysis step."""
        file_path = context.get("file_path") or step_data.get("file_path")

        if file_path:
            analysis_result = self.assistant.analyze_code_context(file_path)
            return {
                "success": analysis_result.get("success", False),
                "analysis": analysis_result,
                "file_path": file_path,
                "step": "code_analysis"
            }
        else:
            # Analyze project directory
            project_path = context.get("project_path", ".")
            project_summary = self.assistant.get_project_summary(project_path)
            return {
                "success": project_summary.get("success", False),
                "summary": project_summary,
                "project_path": project_path,
                "step": "code_analysis"
            }

    async def _execute_planning(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning step."""
        task_description = context.get("task", {}).get("description", "")

        planning_query = f"Create a detailed implementation plan for: {task_description}"
        planning_result = self.assistant.chat(planning_query)

        return {
            "success": True,
            "plan": planning_result,
            "step": "planning"
        }

    async def _execute_implementation(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute implementation step."""
        task_description = context.get("task", {}).get("description", "")

        implementation_query = f"Implement the solution for: {task_description}. Use the appropriate tools to write and test the code."
        implementation_result = self.assistant.chat(implementation_query)

        return {
            "success": True,
            "implementation": implementation_result,
            "step": "implementation"
        }

    async def _execute_testing(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing step."""
        file_path = context.get("file_path") or step_data.get("file_path")

        if file_path:
            testing_query = f"Write and run tests for the code in {file_path}. Use appropriate testing tools."
        else:
            testing_query = "Write and run tests for the recently implemented code. Use appropriate testing tools."

        testing_result = self.assistant.chat(testing_query)

        return {
            "success": True,
            "testing": testing_result,
            "step": "testing"
        }

    async def _execute_documentation(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation step."""
        documentation_query = "Create or update documentation for the recent changes. Include usage examples and explanations."
        documentation_result = self.assistant.chat(documentation_query)

        return {
            "success": True,
            "documentation": documentation_result,
            "step": "documentation"
        }

    async def _execute_review(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute review step."""
        review_query = "Review the implemented code for quality, security, and best practices. Suggest improvements if needed."
        review_result = self.assistant.chat(review_query)

        return {
            "success": True,
            "review": review_result,
            "step": "review"
        }

    async def _execute_investigation(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute investigation step for debugging."""
        task_description = context.get("task", {}).get("description", "")

        investigation_query = f"Investigate this issue thoroughly: {task_description}. Use debugging tools and analysis to find the root cause."
        investigation_result = self.assistant.chat(investigation_query)

        return {
            "success": True,
            "investigation": investigation_result,
            "step": "investigation"
        }

    async def _execute_fix(self, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fix step for debugging."""
        task_description = context.get("task", {}).get("description", "")

        fix_query = f"Implement a fix for this issue: {task_description}. Apply the fix and verify it works."
        fix_result = self.assistant.chat(fix_query)

        return {
            "success": True,
            "fix": fix_result,
            "step": "fix"
        }

    async def _execute_generic(self, step_name: str, step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generic step."""
        task_description = context.get("task", {}).get("description", "")
        step_description = step_data.get("description", f"Execute {step_name}")

        generic_query = f"Execute this step: {step_description} for task: {task_description}"
        generic_result = self.assistant.chat(generic_query)

        return {
            "success": True,
            "result": generic_result,
            "step": step_name
        }


class CodingAgent:
    """Main coding agent with workflow orchestration."""

    def __init__(
        self,
        db_dir: Path = Path("./data"),
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        session_id: Optional[str] = None,
    ):
        # Initialize assistant
        self.assistant = CodingAssistant(db_dir, ollama_host, model_name, session_id)

        # Initialize workflow components
        self.callback_handler = CodingAgentCallbackHandler(self.assistant)
        self.orchestrator = CodingAgentOrchestrator(self.assistant)
        self.executor = CodingAgentExecutor(self.assistant)

        # Create workflow
        self.workflow = Workflow(
            orchestrator=self.orchestrator,
            executor=self.executor,
            callback_handler=self.callback_handler,
        )

        logger.info("CodingAgent initialized with session: %s", self.assistant.session_id)

    async def execute_task(self, task_description: str, task_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Execute a coding task using workflow orchestration."""
        task = {
            "description": task_description,
            "type": task_type,
            "timestamp": time.time(),
            **kwargs
        }

        try:
            result = await self.workflow.execute(task)
            return {
                "success": True,
                "result": result,
                "task": task,
                "session_id": self.assistant.session_id,
            }
        except Exception as e:
            logger.error("Error executing task: %s", e)
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "session_id": self.assistant.session_id,
            }

    def chat(self, message: str) -> str:
        """Simple chat interface."""
        return self.assistant.chat(message)

    async def stream_task(self, task_description: str, task_type: str = "general", **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream task execution progress."""
        task = {
            "description": task_description,
            "type": task_type,
            "timestamp": time.time(),
            **kwargs
        }

        # Start workflow execution
        yield {"type": "start", "task": task}

        try:
            result = await self.workflow.execute(task)
            yield {"type": "complete", "result": result}
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.assistant.get_stats()

    def cleanup(self):
        """Cleanup resources."""
        self.assistant.cleanup()


# Factory function for easy creation
def create_coding_agent(
    db_dir: str = "./data",
    ollama_host: str = "http://localhost:11434",
    model_name: str = "llama3.2",
    session_id: Optional[str] = None,
) -> CodingAgent:
    """Create a coding agent with default configuration."""
    return CodingAgent(
        db_dir=Path(db_dir),
        ollama_host=ollama_host,
        model_name=model_name,
        session_id=session_id,
    )