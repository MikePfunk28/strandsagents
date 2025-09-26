"""Agent-to-agent communication coordinator for parallel adversarial coding."""

import asyncio
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

class MessageType(Enum):
    """Types of messages between agents."""
    CODE_GENERATED = "code_generated"
    ISSUES_FOUND = "issues_found"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    SECURITY_ALERT = "security_alert"
    TEST_RESULTS = "test_results"
    IMPROVEMENT_REQUEST = "improvement_request"
    VALIDATION_COMPLETE = "validation_complete"
    COORDINATION_REQUEST = "coordination_request"
    STATUS_UPDATE = "status_update"

@dataclass
class AgentMessage:
    """Message structure for agent-to-agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1=low, 5=critical
    requires_response: bool = False
    correlation_id: Optional[str] = None

class Agent2AgentBroker:
    """Message broker for agent-to-agent communication."""

    def __init__(self):
        self.agents: Dict[str, 'CodingAgent'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscriptions: Dict[str, List[MessageType]] = {}
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}

    async def register_agent(self, agent_id: str, agent: 'CodingAgent'):
        """Register an agent with the broker."""
        self.agents[agent_id] = agent
        self.subscriptions[agent_id] = []
        print(f"Agent registered: {agent_id}")

    async def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe an agent to specific message types."""
        if agent_id in self.subscriptions:
            self.subscriptions[agent_id].extend(message_types)
        else:
            self.subscriptions[agent_id] = message_types

    async def send_message(self, message: AgentMessage):
        """Send a message through the broker."""
        await self.message_queue.put(message)

    async def broadcast_message(self, sender: str, message_type: MessageType, content: Dict[str, Any]):
        """Broadcast a message to all interested agents."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient="*",  # Broadcast
            message_type=message_type,
            content=content,
            timestamp=asyncio.get_event_loop().time()
        )
        await self.send_message(message)

    async def start_message_processing(self):
        """Start processing messages from the queue."""
        self.running = True
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue  # Continue checking for messages

    async def _process_message(self, message: AgentMessage):
        """Process a single message."""
        if message.recipient == "*":
            # Broadcast to all subscribed agents
            for agent_id, subscribed_types in self.subscriptions.items():
                if message.message_type in subscribed_types and agent_id != message.sender:
                    if agent_id in self.agents:
                        await self.agents[agent_id].receive_message(message)
        else:
            # Direct message
            if message.recipient in self.agents:
                await self.agents[message.recipient].receive_message(message)

    async def stop(self):
        """Stop the message broker."""
        self.running = False

class CodingAgent:
    """Base class for coding agents with agent2agent communication."""

    def __init__(self, agent_id: str, agent_type: str, broker: Agent2AgentBroker):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.broker = broker
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.state = {}
        self.is_active = True

    async def initialize(self):
        """Initialize the agent and register with broker."""
        await self.broker.register_agent(self.agent_id, self)
        await self._setup_message_handlers()
        await self._subscribe_to_messages()

    async def _setup_message_handlers(self):
        """Setup message handlers - to be overridden by subclasses."""
        pass

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types - to be overridden by subclasses."""
        pass

    async def receive_message(self, message: AgentMessage):
        """Receive and process a message."""
        if message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message)
        else:
            await self._handle_unknown_message(message)

    async def _handle_unknown_message(self, message: AgentMessage):
        """Handle unknown message types."""
        print(f"{self.agent_id}: Received unknown message type {message.message_type}")

    async def send_to_agent(self, recipient: str, message_type: MessageType, content: Dict[str, Any]):
        """Send a message to a specific agent."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=asyncio.get_event_loop().time()
        )
        await self.broker.send_message(message)

    async def broadcast(self, message_type: MessageType, content: Dict[str, Any]):
        """Broadcast a message to all agents."""
        await self.broker.broadcast_message(self.agent_id, message_type, content)

class GeneratorAgent(CodingAgent):
    """Generator agent for creating code."""

    def __init__(self, broker: Agent2AgentBroker):
        super().__init__("generator", "code_generator", broker)

    async def _setup_message_handlers(self):
        """Setup message handlers for generator."""
        self.message_handlers[MessageType.IMPROVEMENT_REQUEST] = self._handle_improvement_request
        self.message_handlers[MessageType.ISSUES_FOUND] = self._handle_issues_found

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types."""
        await self.broker.subscribe(self.agent_id, [
            MessageType.IMPROVEMENT_REQUEST,
            MessageType.ISSUES_FOUND,
            MessageType.OPTIMIZATION_SUGGESTION
        ])

    async def generate_code(self, requirements: str, language: str) -> str:
        """Generate code and broadcast to other agents."""
        # Simulate code generation
        code = f"# Generated {language} code for: {requirements}\n# TODO: Implement functionality"

        # Broadcast generated code
        await self.broadcast(MessageType.CODE_GENERATED, {
            "code": code,
            "language": language,
            "requirements": requirements,
            "quality_score": 6.0  # Initial score
        })

        return code

    async def _handle_improvement_request(self, message: AgentMessage):
        """Handle improvement request from other agents."""
        content = message.content
        print(f"Generator: Received improvement request from {message.sender}")
        print(f"  Suggestions: {content.get('suggestions', [])}")

        # Simulate code improvement
        improved_code = content.get('original_code', '') + "\n# Improved based on feedback"

        await self.broadcast(MessageType.CODE_GENERATED, {
            "code": improved_code,
            "language": content.get('language', 'python'),
            "improvements_applied": content.get('suggestions', []),
            "quality_score": content.get('quality_score', 6.0) + 1.0
        })

    async def _handle_issues_found(self, message: AgentMessage):
        """Handle issues found by discriminator."""
        content = message.content
        print(f"Generator: Issues found by {message.sender}")
        for issue in content.get('issues', []):
            print(f"  - {issue}")

class DiscriminatorAgent(CodingAgent):
    """Discriminator agent for finding code issues."""

    def __init__(self, broker: Agent2AgentBroker):
        super().__init__("discriminator", "code_discriminator", broker)

    async def _setup_message_handlers(self):
        """Setup message handlers for discriminator."""
        self.message_handlers[MessageType.CODE_GENERATED] = self._handle_code_generated

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types."""
        await self.broker.subscribe(self.agent_id, [MessageType.CODE_GENERATED])

    async def _handle_code_generated(self, message: AgentMessage):
        """Analyze generated code and find issues."""
        content = message.content
        code = content.get('code', '')
        language = content.get('language', 'python')

        print(f"Discriminator: Analyzing {language} code from {message.sender}")

        # Simulate code analysis
        issues = [
            "Missing error handling",
            "No input validation",
            "Incomplete implementation",
            "Missing documentation"
        ]

        quality_score = max(1.0, content.get('quality_score', 6.0) - len(issues) * 0.5)

        # Send issues back to generator
        await self.send_to_agent("generator", MessageType.ISSUES_FOUND, {
            "issues": issues,
            "code": code,
            "language": language,
            "quality_score": quality_score
        })

        # Broadcast validation results
        await self.broadcast(MessageType.VALIDATION_COMPLETE, {
            "code": code,
            "issues_count": len(issues),
            "quality_score": quality_score,
            "validation_passed": quality_score >= 7.0
        })

class OptimizerAgent(CodingAgent):
    """Optimizer agent for performance improvements."""

    def __init__(self, broker: Agent2AgentBroker):
        super().__init__("optimizer", "code_optimizer", broker)

    async def _setup_message_handlers(self):
        """Setup message handlers for optimizer."""
        self.message_handlers[MessageType.CODE_GENERATED] = self._handle_code_generated

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types."""
        await self.broker.subscribe(self.agent_id, [MessageType.CODE_GENERATED])

    async def _handle_code_generated(self, message: AgentMessage):
        """Analyze code for optimization opportunities."""
        content = message.content
        code = content.get('code', '')
        language = content.get('language', 'python')

        print(f"Optimizer: Analyzing {language} code from {message.sender}")

        # Simulate optimization analysis
        suggestions = [
            "Use list comprehension instead of loops",
            "Cache repeated calculations",
            "Use more efficient data structures",
            "Avoid unnecessary string concatenation"
        ]

        await self.send_to_agent("generator", MessageType.OPTIMIZATION_SUGGESTION, {
            "suggestions": suggestions,
            "original_code": code,
            "language": language,
            "performance_impact": "medium"
        })

class SecurityAgent(CodingAgent):
    """Security agent for vulnerability analysis."""

    def __init__(self, broker: Agent2AgentBroker):
        super().__init__("security", "security_analyzer", broker)

    async def _setup_message_handlers(self):
        """Setup message handlers for security agent."""
        self.message_handlers[MessageType.CODE_GENERATED] = self._handle_code_generated

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types."""
        await self.broker.subscribe(self.agent_id, [MessageType.CODE_GENERATED])

    async def _handle_code_generated(self, message: AgentMessage):
        """Analyze code for security issues."""
        content = message.content
        code = content.get('code', '')
        language = content.get('language', 'python')

        print(f"Security: Analyzing {language} code from {message.sender}")

        # Simulate security analysis
        security_issues = [
            "Input validation missing",
            "Potential injection vulnerability",
            "Unencrypted data transmission"
        ]

        if security_issues:
            await self.broadcast(MessageType.SECURITY_ALERT, {
                "issues": security_issues,
                "severity": "medium",
                "code": code,
                "language": language
            })

class AdversarialCoordinator:
    """Coordinator for the adversarial coding system."""

    def __init__(self):
        self.broker = Agent2AgentBroker()
        self.agents = {}
        self.current_task = None

    async def initialize(self):
        """Initialize the coordinator and all agents."""
        # Create agents
        self.agents['generator'] = GeneratorAgent(self.broker)
        self.agents['discriminator'] = DiscriminatorAgent(self.broker)
        self.agents['optimizer'] = OptimizerAgent(self.broker)
        self.agents['security'] = SecurityAgent(self.broker)

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize()

        # Start message processing
        asyncio.create_task(self.broker.start_message_processing())

    async def generate_code_adversarially(self, requirements: str, language: str = "python") -> str:
        """Generate code using adversarial process."""
        print(f"Starting adversarial code generation for: {requirements}")

        # Start code generation
        code = await self.agents['generator'].generate_code(requirements, language)

        # Allow time for agent communication
        await asyncio.sleep(2.0)

        return code

    async def shutdown(self):
        """Shutdown the coordinator and all agents."""
        await self.broker.stop()

async def demonstrate_agent2agent():
    """Demonstrate agent-to-agent communication."""
    print("Adversarial Coding System - Agent2Agent Demo")
    print("=" * 50)

    coordinator = AdversarialCoordinator()
    await coordinator.initialize()

    try:
        # Generate code with adversarial feedback
        result = await coordinator.generate_code_adversarially(
            "Create a function to calculate factorial",
            "python"
        )

        print(f"\nFinal result: {result[:100]}...")

    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(demonstrate_agent2agent())