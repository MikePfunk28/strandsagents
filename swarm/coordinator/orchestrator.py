"""Swarm Orchestrator using llama3.2:3b for complex reasoning and coordination."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict

from strands import Agent
from ..agents.base_assistant import BaseAssistant, create_lightweight_assistant
from ..communication.mcp_client import SwarmMCPClient
from ..storage.database_manager import DatabaseManager, MemoryEntry
from ...security import SecurityManager, SecurityEvent, Severity

logger = logging.getLogger(__name__)

@dataclass
class SwarmTask:
    """Task to be executed by the swarm."""
    task_id: str
    description: str
    task_type: str  # 'research', 'creative', 'analysis', 'synthesis', 'collaboration'
    priority: int  # 1-10, higher is more priority
    required_agents: List[str]
    context: Dict[str, Any]
    deadline: Optional[datetime] = None
    status: str = "pending"  # 'pending', 'in_progress', 'completed', 'failed'
    results: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = {}

@dataclass
class AgentAllocation:
    """Agent allocation and status."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    current_task: Optional[str] = None
    status: str = "available"  # 'available', 'busy', 'offline'
    performance_score: float = 1.0
    last_seen: datetime = None

    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()

class SwarmOrchestrator:
    """Main orchestrator for coordinating swarm of lightweight assistants.

    Uses llama3.2:3b for complex reasoning about task distribution,
    while lightweight agents use gemma:270m for fast execution.
    """

    def __init__(self, orchestrator_id: str = None, host: str = "localhost:11434"):
        self.orchestrator_id = orchestrator_id or f"orchestrator_{str(uuid.uuid4())[:8]}"
        self.host = host

        # Core components
        self.orchestrator_agent: Optional[Agent] = None
        self.mcp_client: Optional[SwarmMCPClient] = None
        self.db_manager = DatabaseManager()
        self.security_manager = SecurityManager(orchestrator_id=self.orchestrator_id)

        # Swarm state
        self.active_agents: Dict[str, AgentAllocation] = {}
        self.pending_tasks: List[SwarmTask] = []
        self.active_tasks: Dict[str, SwarmTask] = {}
        self.completed_tasks: List[SwarmTask] = []

        # Performance tracking
        self.task_success_rate = 0.0
        self.average_completion_time = 0.0
        self.session_id = str(uuid.uuid4())

        self.running = False

    async def initialize(self):
        """Initialize the orchestrator with larger reasoning model."""
        try:
            # Initialize security manager first
            security_initialized = await self.security_manager.initialize_security()
            if not security_initialized:
                raise Exception("Failed to initialize security manager")

            # Create orchestrator agent for complex reasoning
            self.orchestrator_agent = Agent(
                system_prompt=self._get_orchestrator_prompt()
            )

            # Register orchestrator as trusted agent
            await self.security_manager.register_trusted_agent(
                self.orchestrator_id,
                "orchestrator",
                ["coordination", "planning", "synthesis", "evaluation", "system_admin"],
                trust_level=1.0
            )

            # Create MCP client for coordination
            self.mcp_client = SwarmMCPClient(
                agent_id=self.orchestrator_id,
                agent_type="orchestrator",
                capabilities=["coordination", "planning", "synthesis", "evaluation"],
                model_name="llama3.2:3b",
                server_host="localhost",
                server_port=8080
            )

            # Connect to swarm
            await self.mcp_client.connect()

            # Register message handlers
            self.mcp_client.register_message_handler("agent_registration", self.handle_agent_registration)
            self.mcp_client.register_message_handler("task_result", self.handle_task_result)
            self.mcp_client.register_message_handler("agent_status", self.handle_agent_status)

            self.running = True
            logger.info(f"Secure swarm orchestrator {self.orchestrator_id} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def _get_orchestrator_prompt(self) -> str:
        """Get the system prompt for the orchestrator agent."""
        return """You are the Swarm Orchestrator, coordinating a network of specialized lightweight AI assistants.

Your role:
- Analyze complex tasks and break them into subtasks
- Assign optimal agents based on capabilities and current load
- Synthesize results from multiple agents into coherent solutions
- Make strategic decisions about resource allocation
- Evaluate and improve swarm performance

Available agent types and their capabilities:
- Research Agents: Information gathering, fact checking, document analysis
- Creative Agents: Ideation, brainstorming, innovative solutions
- Critical Agents: Analysis, evaluation, risk assessment, improvements
- Summarizer Agents: Information synthesis, coherent summaries

When given a task:
1. Analyze the requirements and complexity
2. Determine which agent types are needed
3. Plan the workflow and dependencies
4. Provide clear, specific subtasks for each agent
5. Synthesize results into a final comprehensive response

Focus on efficiency, accuracy, and leveraging each agent's strengths.
Use data-driven decisions based on agent performance history.
Always provide structured, actionable coordination plans."""

    async def start_orchestration(self):
        """Start the orchestration system."""
        if not self.running:
            await self.initialize()

        # Start background tasks
        asyncio.create_task(self.process_task_queue())
        asyncio.create_task(self.monitor_agents())
        asyncio.create_task(self.performance_tracking())

        logger.info(f"Swarm orchestration started with orchestrator {self.orchestrator_id}")

    async def stop_orchestration(self):
        """Stop the orchestration system."""
        self.running = False

        # Disconnect MCP client
        if self.mcp_client:
            await self.mcp_client.disconnect()

        logger.info(f"Swarm orchestration stopped")

    async def submit_task(self, description: str, task_type: str = "general",
                         priority: int = 5, context: Dict[str, Any] = None,
                         required_agents: List[str] = None) -> str:
        """Submit a task to the swarm."""
        task_id = str(uuid.uuid4())

        # Use orchestrator agent to analyze task and determine requirements
        analysis_prompt = f"""Analyze this task and provide a structured plan:

Task: {description}
Task Type: {task_type}
Context: {context or {}}

Provide:
1. Required agent types (research, creative, critical, summarizer)
2. Estimated complexity (1-10)
3. Suggested workflow steps
4. Expected timeline
5. Dependencies between subtasks

Format as JSON with structure:
{{
    "required_agents": ["agent_type1", "agent_type2"],
    "complexity": 5,
    "workflow_steps": ["step1", "step2"],
    "timeline_minutes": 10,
    "dependencies": {{"step2": ["step1"]}}
}}"""`n`nIf the task references source code or requires annotation, include the 'code_feedback' agent in required_agents and plan for generator -> discriminator -> agitator loop coordination.

        try:
            analysis_result = await self.orchestrator_agent.run_async(analysis_prompt)

            # Parse analysis (simplified - would use more robust parsing)
            import json
            try:
                analysis = json.loads(analysis_result)
            except json.JSONDecodeError:
                analysis = {}

            parsed_required = None
            if isinstance(analysis, dict):
                parsed_required = analysis.get("required_agents")

            if isinstance(parsed_required, str):
                parsed_required = [parsed_required]

            if parsed_required is None:
                parsed_required = required_agents or ["research"]

            if not isinstance(parsed_required, list):
                parsed_required = list(parsed_required)

            required_agents = parsed_required

            if task_type == "code_feedback" or (context and context.get("code")):
                if "code_feedback" not in required_agents:
                    required_agents.append("code_feedback")
                if "summarizer" not in required_agents:
                    required_agents.append("summarizer")

        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            required_agents = required_agents or ["research"]
            if task_type == "code_feedback" or (context and context.get("code")):
                if "code_feedback" not in required_agents:
                    required_agents.append("code_feedback")
                if "summarizer" not in required_agents:
                    required_agents.append("summarizer")

        # Create task
        task = SwarmTask(
            task_id=task_id,
            description=description,
            task_type=task_type,
            priority=priority,
            required_agents=required_agents,
            context=context or {}
        )

        self.pending_tasks.append(task)

        # Store task in memory
        await self._store_task_memory(task, "task_submitted")

        logger.info(f"Task {task_id} submitted: {description[:100]}...")
        return task_id

    async def process_task_queue(self):
        """Process pending tasks."""
        while self.running:
            try:
                if self.pending_tasks:
                    # Sort by priority
                    self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)

                    # Try to process highest priority task
                    task = self.pending_tasks[0]

                    if await self._can_execute_task(task):
                        self.pending_tasks.remove(task)
                        await self._execute_task(task)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in task processing: {e}")
                await asyncio.sleep(5)

    async def _can_execute_task(self, task: SwarmTask) -> bool:
        """Check if task can be executed with available agents."""
        available_agents_by_type = {}

        for agent_id, allocation in self.active_agents.items():
            if allocation.status == "available":
                agent_type = allocation.agent_type
                if agent_type not in available_agents_by_type:
                    available_agents_by_type[agent_type] = []
                available_agents_by_type[agent_type].append(agent_id)

        # Check if we have all required agent types
        for required_type in task.required_agents:
            if required_type not in available_agents_by_type:
                return False
            if len(available_agents_by_type[required_type]) == 0:
                return False

        return True

    async def _execute_task(self, task: SwarmTask):
        """Execute a task using available agents."""
        try:
            task.status = "in_progress"
            self.active_tasks[task.task_id] = task

            # Store task start in memory
            await self._store_task_memory(task, "task_started")

            # Create execution plan using orchestrator
            execution_prompt = f"""Create a detailed execution plan for this task:

Task: {task.description}
Required Agents: {task.required_agents}
Context: {task.context}
Available Agents: {list(self.active_agents.keys())}

Create specific subtasks for each agent type needed.
Each subtask should be clear, actionable, and leverage the agent's strengths.

Format as JSON:
{{
    "subtasks": [
        {{
            "agent_type": "research",
            "task_description": "specific task for research agent",
            "priority": 1
        }},
        {{
            "agent_type": "creative",
            "task_description": "specific task for creative agent",
            "priority": 2
        }}
    ],
    "synthesis_plan": "how to combine results"
}}"""`n`nIf the task references source code or requires annotation, include the 'code_feedback' agent in required_agents and plan for generator -> discriminator -> agitator loop coordination.

            execution_plan = await self.orchestrator_agent.run_async(execution_prompt)

            # Parse and execute subtasks (simplified implementation)
            agent_results = {}

            for required_type in task.required_agents:
                # Find best available agent of this type
                best_agent = self._select_best_agent(required_type)

                if best_agent:
                    # Mark agent as busy
                    self.active_agents[best_agent].status = "busy"
                    self.active_agents[best_agent].current_task = task.task_id

                    # Send task to agent via MCP
                    await self.mcp_client.send_message(
                        best_agent,
                        "task_request",
                        {
                            "task": {
                                "task_id": task.task_id,
                                "description": task.description,
                                "context": task.context,
                                "agent_focus": required_type
                            },
                            "sender_id": self.orchestrator_id
                        }
                    )

            logger.info(f"Task {task.task_id} distributed to agents")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.status = "failed"
            task.results = {"error": str(e)}
            await self._complete_task(task)

    def _select_best_agent(self, agent_type: str) -> Optional[str]:
        """Select the best available agent of the given type."""
        candidates = []

        for agent_id, allocation in self.active_agents.items():
            if (allocation.agent_type == agent_type and
                allocation.status == "available"):
                candidates.append((agent_id, allocation.performance_score))

        if not candidates:
            return None

        # Return agent with highest performance score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def handle_agent_registration(self, message_data: Dict[str, Any]):
        """Handle agent registration with security verification."""
        payload = message_data.get("payload", {})
        agent_id = payload.get("agent_id")
        agent_type = payload.get("agent_type")
        capabilities = payload.get("capabilities", [])

        if not agent_id or not agent_type:
            logger.warning("Invalid agent registration: missing agent_id or agent_type")
            return

        try:
            # Verify message integrity if signature present
            if "signature" in message_data:
                is_valid, _, verification_result = await self.security_manager.verify_secure_message(message_data)
                if not is_valid:
                    logger.warning(f"Agent registration rejected - message verification failed: {agent_id}")
                    await self.security_manager.report_security_incident(
                        "invalid_registration_message",
                        agent_id,
                        Severity.HIGH,
                        {"verification_issues": verification_result.issues}
                    )
                    return

            # Register agent with security manager
            registration_success = await self.security_manager.register_trusted_agent(
                agent_id,
                agent_type,
                capabilities,
                trust_level=0.7  # Default trust level for new agents
            )

            if not registration_success:
                logger.warning(f"Agent registration rejected by security manager: {agent_id}")
                return

            # Create agent allocation
            allocation = AgentAllocation(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities
            )

            self.active_agents[agent_id] = allocation
            logger.info(f"Secure agent {agent_id} ({agent_type}) registered with capabilities: {capabilities}")

        except Exception as e:
            logger.error(f"Agent registration error for {agent_id}: {e}")
            await self.security_manager.report_security_incident(
                "agent_registration_error",
                agent_id,
                Severity.MEDIUM,
                {"error": str(e)}
            )

    async def handle_task_result(self, message_data: Dict[str, Any]):
        """Handle task results from agents with security validation."""
        payload = message_data.get("payload", {})
        task_id = payload.get("task_id")
        agent_id = payload.get("assistant_id")
        result = payload.get("result")

        if not task_id or not agent_id or not result:
            logger.warning("Invalid task result: missing required fields")
            return

        try:
            # Verify message integrity if signature present
            if "signature" in message_data:
                is_valid, _, verification_result = await self.security_manager.verify_secure_message(message_data)
                if not is_valid:
                    logger.warning(f"Task result rejected - message verification failed: {agent_id}")
                    await self.security_manager.report_security_incident(
                        "invalid_task_result_message",
                        agent_id,
                        Severity.HIGH,
                        {"task_id": task_id, "verification_issues": verification_result.issues}
                    )
                    return

            # Authenticate and authorize agent for task submission
            authorized, credentials = await self.security_manager.authenticate_and_authorize(
                agent_id,
                self.active_agents.get(agent_id, {}).get("agent_type", "unknown"),
                "task_execution",
                {"task_id": task_id}
            )

            if not authorized:
                logger.warning(f"Task result rejected - agent not authorized: {agent_id}")
                return

            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]

                # Validate the answer using the security manager
                if isinstance(result, str) and len(result) > 10:  # Only validate substantial text results
                    validation_result = await self.security_manager.validate_agent_answer(
                        task.description,
                        result,
                        agent_id,
                        self.active_agents.get(agent_id, {}).get("agent_type", "unknown"),
                        task.context,
                        cross_check_agents=[]  # Could add other agents for cross-validation
                    )

                    if not validation_result.is_valid:
                        logger.warning(f"Task result validation failed for agent {agent_id}: {validation_result.issues}")
                        await self.security_manager.report_security_incident(
                            "invalid_task_result",
                            agent_id,
                            Severity.MEDIUM,
                            {
                                "task_id": task_id,
                                "validation_issues": validation_result.issues,
                                "confidence_score": validation_result.confidence_score
                            }
                        )
                        # Don't reject the result, but log the concerns
                        task.results[agent_id] = {
                            "content": result,
                            "validation_warning": validation_result.issues,
                            "confidence_score": validation_result.confidence_score
                        }
                    else:
                        task.results[agent_id] = {
                            "content": result,
                            "confidence_score": validation_result.confidence_score,
                            "trust_level": validation_result.trust_level
                        }
                else:
                    # Store non-text results directly
                    task.results[agent_id] = result

                # Mark agent as available
                if agent_id in self.active_agents:
                    self.active_agents[agent_id].status = "available"
                    self.active_agents[agent_id].current_task = None

                # Report task completion to security manager
                await self.security_manager.reporter.report_task_completion(
                    agent_id,
                    {"task_id": task_id, "description": task.description},
                    {"completion_time": datetime.now().isoformat()}
                )

                # Check if all required agents have completed
                completed_agents = len(task.results)
                required_agents = len(task.required_agents)

                if completed_agents >= required_agents:
                    await self._synthesize_and_complete_task(task)

        except Exception as e:
            logger.error(f"Task result handling error for {agent_id}: {e}")
            await self.security_manager.report_security_incident(
                "task_result_error",
                agent_id,
                Severity.MEDIUM,
                {"task_id": task_id, "error": str(e)}
            )

    async def _synthesize_and_complete_task(self, task: SwarmTask):
        """Synthesize results from multiple agents and complete task."""
        try:
            # Use orchestrator to synthesize results
            synthesis_prompt = f"""Synthesize these agent results into a coherent final response:

Original Task: {task.description}
Context: {task.context}

Agent Results:
{json.dumps(task.results, indent=2)}

Provide a comprehensive, well-structured response that:
1. Addresses the original task completely
2. Integrates insights from all agents
3. Resolves any conflicting information
4. Provides actionable conclusions
5. Highlights key findings and recommendations

Format the response clearly with sections and conclusions."""

            synthesized_result = await self.orchestrator_agent.run_async(synthesis_prompt)

            task.results["final_synthesis"] = synthesized_result
            task.status = "completed"

            await self._complete_task(task)

        except Exception as e:
            logger.error(f"Task synthesis failed: {e}")
            task.status = "failed"
            task.results["synthesis_error"] = str(e)
            await self._complete_task(task)

    async def _complete_task(self, task: SwarmTask):
        """Complete a task and clean up."""
        # Move from active to completed
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        self.completed_tasks.append(task)

        # Store completion in memory
        await self._store_task_memory(task, "task_completed")

        logger.info(f"Task {task.task_id} completed with status: {task.status}")

    async def handle_agent_status(self, message_data: Dict[str, Any]):
        """Handle agent status updates."""
        payload = message_data.get("payload", {})
        agent_id = payload.get("agent_id")
        status = payload.get("status")

        if agent_id in self.active_agents:
            self.active_agents[agent_id].status = status
            self.active_agents[agent_id].last_seen = datetime.now()

    async def monitor_agents(self):
        """Monitor agent health and performance."""
        while self.running:
            try:
                current_time = datetime.now()

                for agent_id, allocation in list(self.active_agents.items()):
                    # Check for stale agents (no heartbeat in 2 minutes)
                    time_since_seen = (current_time - allocation.last_seen).total_seconds()

                    if time_since_seen > 120:  # 2 minutes
                        logger.warning(f"Agent {agent_id} appears offline")
                        allocation.status = "offline"

                        # Clean up any stuck tasks
                        if allocation.current_task:
                            if allocation.current_task in self.active_tasks:
                                task = self.active_tasks[allocation.current_task]
                                task.status = "failed"
                                task.results["error"] = f"Agent {agent_id} went offline"
                                await self._complete_task(task)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(60)

    async def performance_tracking(self):
        """Track and update performance metrics."""
        while self.running:
            try:
                if self.completed_tasks:
                    # Calculate success rate
                    successful_tasks = len([t for t in self.completed_tasks if t.status == "completed"])
                    self.task_success_rate = successful_tasks / len(self.completed_tasks)

                    # Calculate average completion time
                    completion_times = []
                    for task in self.completed_tasks:
                        if hasattr(task, 'created_at') and task.status == "completed":
                            # Simplified - would track actual completion time
                            completion_times.append(300)  # Placeholder: 5 minutes

                    if completion_times:
                        self.average_completion_time = sum(completion_times) / len(completion_times)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)

    async def _store_task_memory(self, task: SwarmTask, event_type: str):
        """Store task-related memory."""
        memory_entry = MemoryEntry(
            id=f"{task.task_id}_{event_type}_{int(datetime.now().timestamp())}",
            content=f"{event_type}: {task.description}",
            timestamp=datetime.now(),
            session_id=self.session_id,
            agent_id=self.orchestrator_id,
            memory_type="task",
            importance=0.7 if event_type == "task_completed" else 0.5,
            metadata={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "event_type": event_type,
                "priority": task.priority,
                "status": task.status
            }
        )

        await self.db_manager.store_memory(memory_entry)

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current status of the swarm with security information."""
        status = {
            "orchestrator_id": self.orchestrator_id,
            "running": self.running,
            "active_agents": len([a for a in self.active_agents.values() if a.status != "offline"]),
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "task_success_rate": self.task_success_rate,
            "average_completion_time": self.average_completion_time,
            "agents_by_type": self._get_agents_by_type(),
            "session_id": self.session_id
        }

        # Add security status if security manager is available
        if hasattr(self, 'security_manager') and self.security_manager:
            status["security_status"] = self.security_manager.get_security_status()

        return status

    def get_security_status(self) -> Dict[str, Any]:
        """Get detailed security status of the swarm."""
        if not hasattr(self, 'security_manager') or not self.security_manager:
            return {"error": "Security manager not initialized"}

        return self.security_manager.get_security_status()

    def get_agent_security_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get security profile for a specific agent."""
        if not hasattr(self, 'security_manager') or not self.security_manager:
            return None

        return self.security_manager.get_agent_security_profile(agent_id)

    def _get_agents_by_type(self) -> Dict[str, int]:
        """Get count of agents by type."""
        agent_counts = {}
        for allocation in self.active_agents.values():
            if allocation.status != "offline":
                agent_type = allocation.agent_type
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        return agent_counts

# Factory function for creating orchestrator
def create_orchestrator(orchestrator_id: str = None) -> SwarmOrchestrator:
    """Create a swarm orchestrator instance."""
    return SwarmOrchestrator(orchestrator_id)

# Example usage and testing
async def demo_orchestrator():
    """Demonstrate orchestrator functionality."""
    print("Swarm Orchestrator Demo")
    print("=" * 30)

    # Create orchestrator
    orchestrator = create_orchestrator("demo_orchestrator")

    try:
        # Start orchestration
        await orchestrator.start_orchestration()

        # Simulate agent registrations
        await orchestrator.handle_agent_registration({
            "payload": {
                "agent_id": "research_001",
                "agent_type": "research",
                "capabilities": ["document_search", "fact_checking"]
            }
        })

        await orchestrator.handle_agent_registration({
            "payload": {
                "agent_id": "creative_001",
                "agent_type": "creative",
                "capabilities": ["brainstorming", "ideation"]
            }
        })

        # Submit test tasks
        task_id = await orchestrator.submit_task(
            "Analyze renewable energy trends and propose innovative solutions",
            task_type="research_creative",
            priority=8,
            context={"domain": "renewable_energy", "focus": "innovation"}
        )

        print(f"Submitted task: {task_id}")

        # Show status
        status = orchestrator.get_swarm_status()
        print(f"Swarm status: {json.dumps(status, indent=2)}")

        # Wait a bit for processing
        await asyncio.sleep(5)

        print("Demo completed")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await orchestrator.stop_orchestration()

if __name__ == "__main__":
    import json
    asyncio.run(demo_orchestrator())