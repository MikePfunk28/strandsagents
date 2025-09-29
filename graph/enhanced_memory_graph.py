 """Enhanced memory graph integration with existing swarm system.

This module extends the basic graph storage to integrate with the swarm's
memory systems, MCP communication, and agent-to-agent coordination.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from graph.graph_storage import GraphNode, GraphEdge, create_graph_storage
from graph.embedding_integration import GraphEmbeddingManager, create_graph_embedding_manager
from swarm.communication.mcp_client import SwarmMCPClient
from swarm.communication.mcp_server import SwarmMCPServer

logger = logging.getLogger(__name__)

class EnhancedMemoryGraph:
    """Enhanced graph system integrated with swarm memory and communication."""

    def __init__(self, storage_type: str = "parquet", base_path: str = "graph/swarm_memory"):
        self.storage_type = storage_type
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Core components
        self.graph_manager = create_graph_embedding_manager(storage_type, base_path)
        self.mcp_client: Optional[SwarmMCPClient] = None
        self.mcp_server: Optional[SwarmMCPServer] = None

        # Swarm integration
        self.connected_agents: Set[str] = set()
        self.memory_nodes: Dict[str, str] = {}  # memory_content -> node_id mapping
        self.agent_nodes: Dict[str, str] = {}   # agent_id -> node_id mapping
        self.task_nodes: Dict[str, str] = {}    # task_id -> node_id mapping

        # Graph enhancement settings
        self.auto_create_relationships = True
        self.memory_similarity_threshold = 0.8
        self.max_related_memories = 10

        self.initialized = False

    async def initialize(self, mcp_host: str = "localhost", mcp_port: int = 8080):
        """Initialize the enhanced memory graph with MCP integration."""
        if not self.initialized:
            # Initialize graph manager
            await self.graph_manager.initialize()

            # Initialize MCP components
            await self._initialize_mcp(mcp_host, mcp_port)

            # Register graph-specific message handlers
            self._register_message_handlers()

            self.initialized = True
            logger.info("Enhanced memory graph initialized with MCP integration")

    async def _initialize_mcp(self, host: str, port: int):
        """Initialize MCP client and server for graph communication."""
        try:
            # Create MCP client for this graph system
            self.mcp_client = SwarmMCPClient(
                agent_id="graph_memory_system",
                agent_type="memory_graph",
                capabilities=["memory_storage", "context_retrieval", "relationship_mapping"],
                model_name="llama3.2:3b",
                server_host=host,
                server_port=port
            )

            await self.mcp_client.connect()
            logger.info("Graph MCP client connected")

            # Create MCP server for graph operations
            self.mcp_server = SwarmMCPServer(host, port + 1)  # Use different port
            await self.mcp_server.start_server()
            logger.info(f"Graph MCP server started on {host}:{port + 1}")

        except Exception as e:
            logger.error(f"Failed to initialize MCP components: {e}")
            raise

    def _register_message_handlers(self):
        """Register message handlers for graph operations."""
        if self.mcp_client:
            self.mcp_client.register_message_handler("store_memory", self.handle_store_memory)
            self.mcp_client.register_message_handler("retrieve_context", self.handle_retrieve_context)
            self.mcp_client.register_message_handler("agent_registration", self.handle_agent_registration)
            self.mcp_client.register_message_handler("task_update", self.handle_task_update)
            self.mcp_client.register_message_handler("find_similar_memories", self.handle_find_similar_memories)

    async def store_swarm_memory(self, memory_content: str, memory_type: str,
                                agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Store swarm memory in the graph with embeddings and relationships."""
        try:
            # Create knowledge node for the memory
            node_metadata = metadata or {}
            node_metadata.update({
                "memory_type": memory_type,
                "agent_id": agent_id,
                "stored_by": "enhanced_graph_system",
                "importance": self._calculate_memory_importance(memory_content, memory_type)
            })

            node_id = await self.graph_manager.create_knowledge_node(
                knowledge_content=memory_content,
                source=agent_id,
                metadata=node_metadata
            )

            # Track memory node
            self.memory_nodes[memory_content] = node_id

            # Create relationship to agent if agent exists in graph
            if agent_id in self.agent_nodes:
                await self.graph_manager.create_relationship(
                    source_id=self.agent_nodes[agent_id],
                    target_id=node_id,
                    relationship_type="created_memory",
                    metadata={"memory_type": memory_type}
                )

            # Auto-create relationships with similar existing memories
            if self.auto_create_relationships:
                await self._create_similarity_relationships(node_id, memory_content)

            logger.info(f"Stored swarm memory: {node_id} ({memory_type})")
            return node_id

        except Exception as e:
            logger.error(f"Failed to store swarm memory: {e}")
            raise

    async def _create_similarity_relationships(self, new_node_id: str, content: str):
        """Create relationships between similar memory nodes."""
        try:
            # Find similar memories
            similar_nodes = await self.graph_manager.find_similar_nodes(
                content,
                node_types=["knowledge", "task"],
                limit=self.max_related_memories
            )

            # Create relationships with high-similarity nodes
            for similar in similar_nodes:
                if similar['similarity'] >= self.memory_similarity_threshold:
                    await self.graph_manager.create_relationship(
                        source_id=new_node_id,
                        target_id=similar['node_id'],
                        relationship_type="similar_to",
                        weight=similar['similarity'],
                        metadata={"similarity_score": similar['similarity']}
                    )

        except Exception as e:
            logger.error(f"Failed to create similarity relationships: {e}")

    def _calculate_memory_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score for memory content."""
        base_importance = {
            "task_result": 0.9,
            "agent_learning": 0.8,
            "error": 0.7,
            "context": 0.6,
            "conversation": 0.5
        }

        importance = base_importance.get(memory_type, 0.5)

        # Adjust based on content length (longer = potentially more important)
        if len(content) > 1000:
            importance += 0.1
        elif len(content) < 100:
            importance -= 0.1

        return min(1.0, max(0.0, importance))

    async def retrieve_context_for_task(self, task_description: str,
                                      agent_id: str, max_memories: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context memories for a task."""
        try:
            # Find similar task and knowledge nodes
            similar_nodes = await self.graph_manager.find_similar_nodes(
                task_description,
                node_types=["task", "knowledge"],
                limit=max_memories * 2
            )

            # Filter and rank by relevance
            relevant_memories = []
            for node_info in similar_nodes:
                node = self.graph_manager.storage.get_node(node_info['node_id'])
                if node and node.metadata:
                    # Check if memory is accessible to this agent
                    memory_agent_id = node.metadata.get("agent_id", "public")
                    if memory_agent_id == "public" or memory_agent_id == agent_id:
                        relevant_memories.append({
                            "node_id": node.node_id,
                            "content": node.content,
                            "memory_type": node.metadata.get("memory_type", "unknown"),
                            "similarity": node_info['similarity'],
                            "importance": node.metadata.get("importance", 0.5),
                            "created_at": node.created_at.isoformat() if node.created_at else None
                        })

            # Sort by combined relevance score
            relevant_memories.sort(
                key=lambda x: (x['similarity'] * 0.6 + x['importance'] * 0.4),
                reverse=True
            )

            return relevant_memories[:max_memories]

        except Exception as e:
            logger.error(f"Failed to retrieve context for task: {e}")
            return []

    async def register_swarm_agent(self, agent_id: str, capabilities: List[str],
                                  model_name: str, metadata: Dict[str, Any] = None) -> str:
        """Register a swarm agent in the graph."""
        try:
            # Create agent node
            node_id = await self.graph_manager.create_agent_node(
                agent_id=agent_id,
                capabilities=capabilities,
                model_name=model_name,
                metadata=metadata
            )

            # Track agent node
            self.agent_nodes[agent_id] = node_id
            self.connected_agents.add(agent_id)

            # Create relationships with agents that have similar capabilities
            await self._create_agent_relationships(node_id, capabilities)

            logger.info(f"Registered swarm agent in graph: {agent_id} -> {node_id}")
            return node_id

        except Exception as e:
            logger.error(f"Failed to register swarm agent: {e}")
            raise

    async def _create_agent_relationships(self, agent_node_id: str, capabilities: List[str]):
        """Create relationships between agents with similar capabilities."""
        try:
            for existing_agent_id, existing_node_id in self.agent_nodes.items():
                if existing_node_id == agent_node_id:
                    continue

                # Get existing agent capabilities
                existing_node = self.graph_manager.storage.get_node(existing_node_id)
                if existing_node and existing_node.metadata:
                    existing_capabilities = existing_node.metadata.get("capabilities", [])

                    # Calculate capability overlap
                    overlap = set(capabilities) & set(existing_capabilities)
                    if overlap:
                        similarity = len(overlap) / len(set(capabilities + existing_capabilities))

                        await self.graph_manager.create_relationship(
                            source_id=agent_node_id,
                            target_id=existing_node_id,
                            relationship_type="capability_overlap",
                            weight=similarity,
                            metadata={
                                "overlapping_capabilities": list(overlap),
                                "similarity_score": similarity
                            }
                        )

        except Exception as e:
            logger.error(f"Failed to create agent relationships: {e}")

    async def track_task_lifecycle(self, task_id: str, task_description: str,
                                  assigned_agent: str, status: str = "created",
                                  metadata: Dict[str, Any] = None) -> str:
        """Track the complete lifecycle of a swarm task in the graph."""
        try:
            # Create or update task node
            task_metadata = metadata or {}
            task_metadata.update({
                "task_id": task_id,
                "assigned_agent": assigned_agent,
                "status": status,
                "lifecycle_stage": status
            })

            if task_id in self.task_nodes:
                # Update existing task node
                existing_node = self.graph_manager.storage.get_node(self.task_nodes[task_id])
                if existing_node:
                    await self.graph_manager.update_node_embedding(
                        self.task_nodes[task_id],
                        f"{task_description} (Status: {status})"
                    )
                    # Update metadata would require storage update
            else:
                # Create new task node
                node_id = await self.graph_manager.create_task_node(
                    task_description,
                    metadata=task_metadata
                )
                self.task_nodes[task_id] = node_id

            # Create relationship with assigned agent
            if assigned_agent in self.agent_nodes:
                await self.graph_manager.create_relationship(
                    source_id=self.task_nodes[task_id],
                    target_id=self.agent_nodes[assigned_agent],
                    relationship_type=f"assigned_to",
                    metadata={"assignment_status": status}
                )

            logger.info(f"Tracked task lifecycle: {task_id} -> {status}")
            return self.task_nodes[task_id]

        except Exception as e:
            logger.error(f"Failed to track task lifecycle: {e}")
            raise

    async def find_optimal_agent_for_task(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Find the optimal agent for a task using graph relationships."""
        try:
            # Create temporary task node to find relationships
            temp_task_id = await self.graph_manager.create_task_node(task_description)

            # Find agents connected to similar tasks
            related_nodes = await self.graph_manager.get_related_nodes(
                temp_task_id,
                relationship_types=["assigned_to", "suitable_for"],
                max_depth=2
            )

            # Analyze agent connections and scores
            agent_scores = {}
            for node_id, node_info in related_nodes.items():
                node = node_info["node"]
                if node.node_type == "agent" and node.metadata:
                    agent_id = node.metadata.get("agent_id")
                    if agent_id:
                        # Calculate score based on relationship weight and depth
                        score = 1.0 / (node_info["depth"] + 1)  # Closer relationships = higher score
                        if agent_id not in agent_scores:
                            agent_scores[agent_id] = {
                                "score": 0,
                                "relationships": [],
                                "capabilities": node.metadata.get("capabilities", [])
                            }
                        agent_scores[agent_id]["score"] += score
                        agent_scores[agent_id]["relationships"].append(node_info["relationship"])

            # Clean up temporary task node
            # (In a real implementation, you might want to keep it for learning)

            if agent_scores:
                # Return best agent
                best_agent = max(agent_scores.items(), key=lambda x: x[1]["score"])
                return {
                    "agent_id": best_agent[0],
                    "score": best_agent[1]["score"],
                    "capabilities": best_agent[1]["capabilities"],
                    "relationships": best_agent[1]["relationships"]
                }

            return None

        except Exception as e:
            logger.error(f"Failed to find optimal agent: {e}")
            return None

    async def get_graph_context_summary(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a summary of context around a graph node."""
        try:
            node = self.graph_manager.storage.get_node(node_id)
            if not node:
                return {"error": "Node not found"}

            # Get related nodes
            related_nodes = await self.graph_manager.get_related_nodes(
                node_id,
                max_depth=max_depth
            )

            # Get edges for relationship analysis
            edges = self.graph_manager.storage.get_edges(node_id)

            # Analyze relationship patterns
            relationship_types = {}
            for edge in edges:
                rel_type = edge.edge_type
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

            return {
                "node_id": node_id,
                "node_type": node.node_type,
                "content_preview": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                "related_nodes_count": len(related_nodes),
                "relationship_types": relationship_types,
                "embedding_coverage": node.embedding is not None,
                "created_at": node.created_at.isoformat() if node.created_at else None
            }

        except Exception as e:
            logger.error(f"Failed to get context summary: {e}")
            return {"error": str(e)}

    # MCP Message Handlers

    async def handle_store_memory(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory storage requests from swarm agents."""
        try:
            payload = message_data.get("payload", {})
            memory_content = payload.get("content")
            memory_type = payload.get("memory_type", "general")
            agent_id = payload.get("agent_id", "unknown")
            metadata = payload.get("metadata", {})

            node_id = await self.store_swarm_memory(memory_content, memory_type, agent_id, metadata)

            return {
                "status": "success",
                "node_id": node_id,
                "message": "Memory stored in graph successfully"
            }

        except Exception as e:
            logger.error(f"Failed to handle store memory: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def handle_retrieve_context(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context retrieval requests."""
        try:
            payload = message_data.get("payload", {})
            task_description = payload.get("task_description")
            agent_id = payload.get("agent_id")
            max_memories = payload.get("max_memories", 5)

            if not task_description or not agent_id:
                return {
                    "status": "error",
                    "message": "Missing task_description or agent_id"
                }

            memories = await self.retrieve_context_for_task(task_description, agent_id, max_memories)

            return {
                "status": "success",
                "memories": memories,
                "count": len(memories)
            }

        except Exception as e:
            logger.error(f"Failed to handle retrieve context: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def handle_agent_registration(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration in the graph."""
        try:
            payload = message_data.get("payload", {})
            agent_id = payload.get("agent_id")
            capabilities = payload.get("capabilities", [])
            model_name = payload.get("model_name", "unknown")
            metadata = payload.get("metadata", {})

            node_id = await self.register_swarm_agent(agent_id, capabilities, model_name, metadata)

            return {
                "status": "success",
                "node_id": node_id,
                "message": "Agent registered in graph successfully"
            }

        except Exception as e:
            logger.error(f"Failed to handle agent registration: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def handle_task_update(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task lifecycle updates."""
        try:
            payload = message_data.get("payload", {})
            task_id = payload.get("task_id")
            task_description = payload.get("description")
            assigned_agent = payload.get("assigned_agent")
            status = payload.get("status", "created")
            metadata = payload.get("metadata", {})

            node_id = await self.track_task_lifecycle(task_id, task_description, assigned_agent, status, metadata)

            return {
                "status": "success",
                "node_id": node_id,
                "message": "Task lifecycle tracked successfully"
            }

        except Exception as e:
            logger.error(f"Failed to handle task update: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def handle_find_similar_memories(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests to find similar memories."""
        try:
            payload = message_data.get("payload", {})
            query = payload.get("query")
            node_types = payload.get("node_types")
            limit = payload.get("limit", 10)

            if not query:
                return {
                    "status": "error",
                    "message": "Missing query parameter"
                }

            similar_nodes = await self.graph_manager.find_similar_nodes(query, node_types, limit)

            return {
                "status": "success",
                "similar_nodes": similar_nodes,
                "count": len(similar_nodes)
            }

        except Exception as e:
            logger.error(f"Failed to handle find similar memories: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including swarm integration metrics."""
        base_stats = self.graph_manager.get_graph_statistics()

        # Add swarm-specific metrics
        swarm_metrics = {
            "connected_agents": len(self.connected_agents),
            "tracked_memories": len(self.memory_nodes),
            "tracked_tasks": len(self.task_nodes),
            "agent_relationships": self._count_relationships_by_type("agent"),
            "memory_relationships": self._count_relationships_by_type("knowledge"),
            "task_relationships": self._count_relationships_by_type("task")
        }

        return {**base_stats, "swarm_metrics": swarm_metrics}

    def _count_relationships_by_type(self, node_type: str) -> int:
        """Count relationships involving nodes of a specific type."""
        count = 0

        try:
            if hasattr(self.graph_manager.storage.storage, 'edges_data'):
                for edge_data in self.graph_manager.storage.storage.edges_data:
                    source_node = self.graph_manager.storage.get_node(edge_data['source_id'])
                    target_node = self.graph_manager.storage.get_node(edge_data['target_id'])

                    if source_node and source_node.node_type == node_type:
                        count += 1
                    if target_node and target_node.node_type == node_type:
                        count += 1

        except Exception as e:
            logger.error(f"Failed to count relationships: {e}")

        return count

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.mcp_client:
                await self.mcp_client.disconnect()

            if self.mcp_server:
                await self.mcp_server.stop_server()

            logger.info("Enhanced memory graph cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function
def create_enhanced_memory_graph(storage_type: str = "parquet",
                               base_path: str = "graph/swarm_memory") -> EnhancedMemoryGraph:
    """Create an enhanced memory graph with swarm integration."""
    return EnhancedMemoryGraph(storage_type, base_path)

# Example usage and testing
async def demo_enhanced_memory_graph():
    """Demonstrate the enhanced memory graph system."""
    print("Enhanced Memory Graph Demo")
    print("=" * 50)

    # Create enhanced graph
    graph = create_enhanced_memory_graph("parquet", "graph/enhanced_demo")

    try:
        # Initialize with MCP
        await graph.initialize()

        # Register sample agents
        agents = [
            ("research_agent", ["research", "analysis"], "llama3.2:3b"),
            ("creative_agent", ["ideation", "brainstorming"], "gemma:270m"),
            ("critical_agent", ["evaluation", "testing"], "phi:4b")
        ]

        for agent_id, capabilities, model in agents:
            await graph.register_swarm_agent(agent_id, capabilities, model)
            print(f"Registered agent: {agent_id}")

        # Store some memories
        memories = [
            ("Completed analysis of renewable energy trends", "task_result", "research_agent"),
            ("Brainstormed 5 innovative solar panel designs", "task_result", "creative_agent"),
            ("Found critical flaw in previous energy model", "error", "critical_agent"),
            ("Context: Energy storage is crucial for renewable adoption", "context", "research_agent")
        ]

        for content, mem_type, agent_id in memories:
            node_id = await graph.store_swarm_memory(content, mem_type, agent_id)
            print(f"Stored memory: {node_id}")

        # Track task lifecycle
        task_id = "task_001"
        await graph.track_task_lifecycle(
            task_id,
            "Analyze renewable energy trends and propose solutions",
            "research_agent",
            "in_progress"
        )
        print(f"Tracked task: {task_id}")

        # Retrieve context for a new task
        context = await graph.retrieve_context_for_task(
            "Design a sustainable energy solution for urban areas",
            "creative_agent",
            max_memories=3
        )
        print(f"Retrieved {len(context)} context memories")

        # Find optimal agent
        optimal_agent = await graph.find_optimal_agent_for_task(
            "Evaluate the environmental impact of renewable energy solutions"
        )
        if optimal_agent:
            print(f"Optimal agent: {optimal_agent['agent_id']} (score: {optimal_agent['score']:.3f})")

        # Get enhanced statistics
        stats = graph.get_enhanced_statistics()
        print(f"Enhanced statistics: {json.dumps(stats, indent=2)}")

        print("Demo completed successfully!")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await graph.cleanup()

if __name__ == "__main__":
    asyncio.run(demo_enhanced_memory_graph())
