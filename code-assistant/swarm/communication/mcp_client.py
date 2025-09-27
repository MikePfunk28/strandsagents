"""MCP Client for agent2agent communication in the swarm system."""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from datetime import datetime
import logging

from .mcp_server import SwarmMessage, AgentInfo

logger = logging.getLogger(__name__)

class SwarmMCPClient:
    """MCP Client for agents to communicate with the swarm."""

    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str],
                 model_name: str = "gemma:270m", server_host: str = "localhost",
                 server_port: int = 8080):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.model_name = model_name
        self.server_host = server_host
        self.server_port = server_port

        self.connected = False
        self.reader = None
        self.writer = None
        self.message_handlers: Dict[str, Callable] = {}
        self.incoming_messages: asyncio.Queue = asyncio.Queue()

        # Message subscriptions
        self.subscriptions: List[str] = []

    async def connect(self):
        """Connect to the MCP server."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.server_host, self.server_port
            )
            self.connected = True
            logger.info(f"Agent {self.agent_id} connected to MCP server")

            # Register with server
            await self.register()

            # Start message listening task
            asyncio.create_task(self.listen_for_messages())
            asyncio.create_task(self.send_heartbeats())

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        logger.info(f"Agent {self.agent_id} disconnected from MCP server")

    async def register(self):
        """Register this agent with the MCP server."""
        registration_message = {
            "type": "register",
            "payload": {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": self.capabilities,
                "model_name": self.model_name,
                "host": "localhost:11434"  # Ollama host
            }
        }

        await self.send_to_server(registration_message)

    async def send_to_server(self, message: Dict[str, Any]):
        """Send message to MCP server."""
        if not self.connected or not self.writer:
            logger.error("Not connected to MCP server")
            return

        try:
            message_data = json.dumps(message).encode()
            self.writer.write(message_data)
            await self.writer.drain()
        except Exception as e:
            logger.error(f"Failed to send message to server: {e}")

    async def send_message(self, recipient_id: str, message_type: str,
                          payload: Dict[str, Any], priority: int = 1):
        """Send message to another agent."""
        message = {
            "type": "send_message",
            "payload": {
                "sender_id": self.agent_id,
                "recipient_id": recipient_id,
                "message_type": message_type,
                "message_payload": payload,
                "priority": priority,
                "correlation_id": str(uuid.uuid4())
            }
        }

        await self.send_to_server(message)

    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast message to all agents."""
        await self.send_message("*", message_type, payload)

    async def subscribe_to_messages(self, message_types: List[str]):
        """Subscribe to specific message types."""
        self.subscriptions.extend(message_types)

        subscription_message = {
            "type": "subscribe",
            "payload": {
                "agent_id": self.agent_id,
                "message_types": message_types
            }
        }

        await self.send_to_server(subscription_message)

    async def get_agents(self) -> Dict[str, Any]:
        """Get list of all registered agents."""
        request_message = {
            "type": "get_agents",
            "payload": {
                "agent_id": self.agent_id
            }
        }

        await self.send_to_server(request_message)

        # Wait for response (simplified - would be more robust in production)
        await asyncio.sleep(0.1)
        return {}

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler

    async def listen_for_messages(self):
        """Listen for incoming messages from the server."""
        while self.connected:
            try:
                if self.reader:
                    data = await self.reader.read(1024)
                    if not data:
                        break

                    message_data = json.loads(data.decode())
                    await self.handle_incoming_message(message_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error listening for messages: {e}")

    async def handle_incoming_message(self, message_data: Dict[str, Any]):
        """Handle incoming message from server."""
        message_type = message_data.get("type")

        if message_type in self.message_handlers:
            await self.message_handlers[message_type](message_data)
        else:
            # Queue message for processing
            await self.incoming_messages.put(message_data)

    async def get_next_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next message from queue."""
        try:
            return await asyncio.wait_for(
                self.incoming_messages.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def send_heartbeats(self):
        """Send periodic heartbeats to server."""
        while self.connected:
            try:
                heartbeat_message = {
                    "type": "heartbeat",
                    "payload": {
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }

                await self.send_to_server(heartbeat_message)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

    async def request_collaboration(self, target_agents: List[str],
                                  task_description: str) -> str:
        """Request collaboration from specific agents."""
        collaboration_id = str(uuid.uuid4())

        for target_agent in target_agents:
            await self.send_message(
                target_agent,
                "collaboration_request",
                {
                    "collaboration_id": collaboration_id,
                    "task_description": task_description,
                    "requester": self.agent_id
                }
            )

        return collaboration_id

    async def respond_to_collaboration(self, collaboration_id: str,
                                     response: Dict[str, Any]):
        """Respond to a collaboration request."""
        await self.broadcast_message(
            "collaboration_response",
            {
                "collaboration_id": collaboration_id,
                "response": response,
                "responder": self.agent_id
            }
        )

# Helper functions for creating MCP clients
def create_lightweight_agent_client(agent_id: str, agent_type: str,
                                   capabilities: List[str]) -> SwarmMCPClient:
    """Create MCP client for lightweight 270M agents."""
    return SwarmMCPClient(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        model_name="gemma:270m",  # Lightweight model for swarm agents
        server_host="localhost",
        server_port=8080
    )

def create_orchestrator_client(agent_id: str) -> SwarmMCPClient:
    """Create MCP client for orchestrator with larger model."""
    return SwarmMCPClient(
        agent_id=agent_id,
        agent_type="orchestrator",
        capabilities=["coordination", "planning", "synthesis"],
        model_name="llama3.2:3b",  # Larger model for complex reasoning
        server_host="localhost",
        server_port=8080
    )

# Example usage
async def demo_mcp_client():
    """Demonstrate MCP client functionality."""
    print("Starting MCP Client Demo")
    print("=" * 30)

    # Create lightweight research agent client
    research_client = create_lightweight_agent_client(
        "research_agent_001",
        "research",
        ["document_search", "fact_checking"]
    )

    # Create creative agent client
    creative_client = create_lightweight_agent_client(
        "creative_agent_001",
        "creative",
        ["brainstorming", "ideation"]
    )

    try:
        # Connect clients
        await research_client.connect()
        await creative_client.connect()

        # Research agent sends findings to creative agent
        await research_client.send_message(
            "creative_agent_001",
            "research_findings",
            {
                "topic": "renewable energy",
                "findings": ["Solar panels efficiency improved 20%", "Wind energy costs decreased"],
                "confidence": 0.85
            }
        )

        # Creative agent responds with ideas
        await creative_client.send_message(
            "research_agent_001",
            "creative_ideas",
            {
                "ideas": ["Hybrid solar-wind farms", "Smart grid integration"],
                "inspiration_source": "research_findings"
            }
        )

        print("Agents exchanged messages successfully")

        # Wait a bit for message processing
        await asyncio.sleep(2)

    finally:
        # Disconnect clients
        await research_client.disconnect()
        await creative_client.disconnect()

if __name__ == "__main__":
    asyncio.run(demo_mcp_client())