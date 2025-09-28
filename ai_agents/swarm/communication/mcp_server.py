"""MCP Server for agent2agent communication in the swarm system."""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from strands.models import OllamaModel

logger = logging.getLogger(__name__)

@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    model_name: str
    host: str
    status: str = "active"
    last_seen: datetime = None

@dataclass
class SwarmMessage:
    """Message format for agent2agent communication."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=critical

class SwarmMCPServer:
    """MCP Server for coordinating swarm agent communication."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.server = None

        # Message routing
        self.broadcast_subscribers: Dict[str, List[str]] = {}
        self.message_history: List[SwarmMessage] = []
        self.max_history = 1000

    async def start_server(self):
        """Start the MCP server."""
        logger.info(f"Starting Swarm MCP Server on {self.host}:{self.port}")

        self.server = await asyncio.start_server(
            self.handle_agent_connection,
            self.host,
            self.port
        )

        self.running = True
        logger.info("Swarm MCP Server started successfully")

        # Start background tasks
        asyncio.create_task(self.cleanup_inactive_agents())
        asyncio.create_task(self.process_message_queues())

    async def stop_server(self):
        """Stop the MCP server."""
        self.running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("Swarm MCP Server stopped")

    async def handle_agent_connection(self, reader, writer):
        """Handle incoming agent connections."""
        client_address = writer.get_extra_info('peername')
        logger.info(f"Agent connected from {client_address}")

        try:
            while True:
                # Read message from agent
                data = await reader.read(1024)
                if not data:
                    break

                message_data = json.loads(data.decode())
                await self.process_agent_message(message_data, writer)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error handling agent connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_agent_message(self, message_data: Dict[str, Any], writer):
        """Process incoming message from an agent."""
        message_type = message_data.get("type")
        payload = message_data.get("payload", {})

        if message_type == "register":
            await self.handle_agent_registration(payload, writer)
        elif message_type == "send_message":
            await self.handle_send_message(payload)
        elif message_type == "subscribe":
            await self.handle_subscription(payload)
        elif message_type == "get_agents":
            await self.handle_get_agents_request(writer)
        elif message_type == "heartbeat":
            await self.handle_heartbeat(payload)
        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def handle_agent_registration(self, payload: Dict[str, Any], writer):
        """Handle agent registration."""
        agent_info = AgentInfo(
            agent_id=payload["agent_id"],
            agent_type=payload.get("agent_type", "unknown"),
            capabilities=payload.get("capabilities", []),
            model_name=payload.get("model_name", "llama3.2:1b"),
            host=payload.get("host", "localhost:11434"),
            last_seen=datetime.now()
        )

        self.registered_agents[agent_info.agent_id] = agent_info
        self.message_queues[agent_info.agent_id] = asyncio.Queue()

        logger.info(f"Registered agent: {agent_info.agent_id} ({agent_info.agent_type})")

        # Send registration confirmation
        response = {
            "type": "registration_confirmed",
            "agent_id": agent_info.agent_id,
            "server_info": {
                "host": self.host,
                "port": self.port,
                "connected_agents": len(self.registered_agents)
            }
        }

        writer.write(json.dumps(response).encode())
        await writer.drain()

    async def handle_send_message(self, payload: Dict[str, Any]):
        """Handle message sending between agents."""
        message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=payload["sender_id"],
            recipient_id=payload["recipient_id"],
            message_type=payload["message_type"],
            payload=payload["message_payload"],
            timestamp=datetime.now(),
            correlation_id=payload.get("correlation_id"),
            priority=payload.get("priority", 1)
        )

        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)

        # Route message
        if message.recipient_id == "*":
            # Broadcast message
            await self.broadcast_message(message)
        elif message.recipient_id in self.message_queues:
            # Direct message
            await self.message_queues[message.recipient_id].put(message)
        else:
            logger.warning(f"Recipient {message.recipient_id} not found")

    async def broadcast_message(self, message: SwarmMessage):
        """Broadcast message to all agents except sender."""
        for agent_id, queue in self.message_queues.items():
            if agent_id != message.sender_id:
                await queue.put(message)

    async def handle_subscription(self, payload: Dict[str, Any]):
        """Handle agent subscription to message types."""
        agent_id = payload["agent_id"]
        message_types = payload["message_types"]

        if agent_id not in self.broadcast_subscribers:
            self.broadcast_subscribers[agent_id] = []

        self.broadcast_subscribers[agent_id].extend(message_types)
        logger.info(f"Agent {agent_id} subscribed to: {message_types}")

    async def handle_get_agents_request(self, writer):
        """Handle request for list of registered agents."""
        agents_info = {
            agent_id: {
                "agent_type": info.agent_type,
                "capabilities": info.capabilities,
                "status": info.status,
                "model_name": info.model_name
            }
            for agent_id, info in self.registered_agents.items()
        }

        response = {
            "type": "agents_list",
            "agents": agents_info
        }

        writer.write(json.dumps(response).encode())
        await writer.drain()

    async def handle_heartbeat(self, payload: Dict[str, Any]):
        """Handle agent heartbeat."""
        agent_id = payload["agent_id"]
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].last_seen = datetime.now()

    async def process_message_queues(self):
        """Background task to process message queues."""
        while self.running:
            try:
                # Process pending messages for each agent
                for agent_id, queue in self.message_queues.items():
                    if not queue.empty():
                        message = await queue.get()
                        await self.deliver_message_to_agent(agent_id, message)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error processing message queues: {e}")

    async def deliver_message_to_agent(self, agent_id: str, message: SwarmMessage):
        """Deliver message to specific agent."""
        # In a real implementation, this would send the message
        # to the agent's connection. For now, we'll just log it.
        logger.info(f"Delivering message to {agent_id}: {message.message_type}")

    async def cleanup_inactive_agents(self):
        """Remove inactive agents periodically."""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_minutes = 5

                inactive_agents = []
                for agent_id, info in self.registered_agents.items():
                    if info.last_seen:
                        time_diff = current_time - info.last_seen
                        if time_diff.total_seconds() > timeout_minutes * 60:
                            inactive_agents.append(agent_id)

                for agent_id in inactive_agents:
                    await self.unregister_agent(agent_id)
                    logger.info(f"Removed inactive agent: {agent_id}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from the swarm."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]

        if agent_id in self.message_queues:
            del self.message_queues[agent_id]

        if agent_id in self.broadcast_subscribers:
            del self.broadcast_subscribers[agent_id]

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get information about a specific agent."""
        return self.registered_agents.get(agent_id)

    def get_all_agents(self) -> List[AgentInfo]:
        """Get information about all registered agents."""
        return list(self.registered_agents.values())

    def get_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Get agents by type."""
        return [
            info for info in self.registered_agents.values()
            if info.agent_type == agent_type
        ]

    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that have a specific capability."""
        return [
            info for info in self.registered_agents.values()
            if capability in info.capabilities
        ]

    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[SwarmMessage]:
        """Get message history, optionally filtered by agent."""
        messages = self.message_history

        if agent_id:
            messages = [
                msg for msg in messages
                if msg.sender_id == agent_id or msg.recipient_id == agent_id
            ]

        return messages[-limit:]

# Example usage and testing
async def start_swarm_mcp_server(host: str = "localhost", port: int = 8080) -> SwarmMCPServer:
    """Start the swarm MCP server."""
    server = SwarmMCPServer(host, port)
    await server.start_server()
    return server

async def demo_mcp_server():
    """Demonstrate the MCP server functionality."""
    print("Starting Swarm MCP Server Demo")
    print("=" * 50)

    # Start server
    server = await start_swarm_mcp_server()

    # Simulate some agent registrations
    test_agents = [
        {
            "agent_id": "research_001",
            "agent_type": "research",
            "capabilities": ["web_search", "document_analysis"],
            "model_name": "llama3.2:3b"
        },
        {
            "agent_id": "creative_001",
            "agent_type": "creative",
            "capabilities": ["brainstorming", "ideation"],
            "model_name": "llama3.2:1b"
        },
        {
            "agent_id": "critical_001",
            "agent_type": "critical",
            "capabilities": ["evaluation", "risk_assessment"],
            "model_name": "gemma2:2b"
        }
    ]

    # Register test agents
    for agent_data in test_agents:
        agent_info = AgentInfo(**agent_data, last_seen=datetime.now())
        server.registered_agents[agent_info.agent_id] = agent_info
        server.message_queues[agent_info.agent_id] = asyncio.Queue()

    print(f"Registered {len(server.registered_agents)} test agents")

    # Display agent information
    for agent_id, info in server.registered_agents.items():
        print(f"Agent: {agent_id} ({info.agent_type}) - {info.model_name}")

    print("\nMCP Server is running. Press Ctrl+C to stop.")

    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        await server.stop_server()

if __name__ == "__main__":
    asyncio.run(demo_mcp_server())