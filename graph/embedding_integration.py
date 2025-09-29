"""Embedding integration for graph nodes using existing embedding assistant.

This module integrates with the existing embedding_assistant.py to generate
embeddings for graph nodes and store them in the graph storage system.
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from .graph_storage import GraphNode, GraphEdge, GraphStorageManager, create_graph_storage
from embedding_assistant import EmbeddingAssistant

logger = logging.getLogger(__name__)

class GraphEmbeddingManager:
    """Manages embeddings for graph nodes using existing embedding system."""

    def __init__(self, storage_type: str = "parquet", base_path: str = "graph/data"):
        self.storage = create_graph_storage(storage_type, base_path)
        self.embedding_assistant = EmbeddingAssistant()
        self.initialized = False

    async def initialize(self):
        """Initialize the embedding assistant."""
        if not self.initialized:
            # EmbeddingAssistant doesn't need async initialization
            self.initialized = True
            logger.info("Graph embedding manager initialized")

    async def create_node_with_embedding(self,
                                       content: str,
                                       node_type: str,
                                       node_id: str = None,
                                       metadata: Dict[str, Any] = None) -> str:
        """Create a graph node with automatically generated embedding."""
        if not self.initialized:
            await self.initialize()

        # Generate node ID if not provided
        if node_id is None:
            node_id = f"{node_type}_{str(uuid.uuid4())[:8]}"

        # Generate embedding using existing assistant
        try:
            result = self.embedding_assistant.embed_text(content)
            embedding = result.vector
            logger.info(f"Generated embedding for node {node_id}: {len(embedding)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding for {node_id}: {e}")
            embedding = None

        # Create graph node
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Store in graph
        stored_id = self.storage.add_node(node)
        logger.info(f"Created graph node: {stored_id}")

        return stored_id

    async def create_task_node(self, task_description: str, priority: int = 5,
                              metadata: Dict[str, Any] = None) -> str:
        """Create a task node with embedding."""
        task_metadata = metadata or {}
        task_metadata.update({
            "task_type": "swarm_task",
            "priority": priority,
            "created_by": "graph_system"
        })

        return await self.create_node_with_embedding(
            content=task_description,
            node_type="task",
            metadata=task_metadata
        )

    async def create_agent_node(self, agent_id: str, capabilities: List[str],
                               model_name: str, metadata: Dict[str, Any] = None) -> str:
        """Create an agent node with embedding."""
        content = f"Agent {agent_id} with capabilities: {', '.join(capabilities)} using {model_name}"

        agent_metadata = metadata or {}
        agent_metadata.update({
            "agent_id": agent_id,
            "capabilities": capabilities,
            "model_name": model_name,
            "status": "active"
        })

        return await self.create_node_with_embedding(
            content=content,
            node_type="agent",
            node_id=f"agent_{agent_id}",
            metadata=agent_metadata
        )

    async def create_knowledge_node(self, knowledge_content: str, source: str = "unknown",
                                   metadata: Dict[str, Any] = None) -> str:
        """Create a knowledge node with embedding."""
        knowledge_metadata = metadata or {}
        knowledge_metadata.update({
            "knowledge_type": "extracted",
            "source": source,
            "confidence": 0.8
        })

        return await self.create_node_with_embedding(
            content=knowledge_content,
            node_type="knowledge",
            metadata=knowledge_metadata
        )

    async def create_relationship(self, source_id: str, target_id: str,
                                 relationship_type: str, weight: float = 1.0,
                                 metadata: Dict[str, Any] = None) -> str:
        """Create a relationship edge between two nodes."""
        edge_id = f"edge_{str(uuid.uuid4())[:8]}"

        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=relationship_type,
            weight=weight,
            metadata=metadata or {}
        )

        stored_id = self.storage.add_edge(edge)
        logger.info(f"Created relationship: {source_id} --[{relationship_type}]--> {target_id}")

        return stored_id

    async def find_similar_nodes(self, query: str, node_types: List[str] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar nodes based on content similarity."""
        if not self.initialized:
            await self.initialize()

        # Generate embedding for query
        query_result = self.embedding_assistant.embed_text(query)
        query_embedding = query_result.vector

        # Search in graph storage
        all_similar = self.storage.search_similar(query_embedding, limit * 2)

        # Filter by node types if specified
        if node_types:
            filtered_similar = [
                result for result in all_similar
                if result['node_type'] in node_types
            ]
        else:
            filtered_similar = all_similar

        return filtered_similar[:limit]

    async def get_related_nodes(self, node_id: str, relationship_types: List[str] = None,
                               max_depth: int = 2) -> Dict[str, Any]:
        """Get nodes related to the given node within specified depth."""
        visited = {node_id}
        current_level = {node_id}
        related_nodes = {}

        for depth in range(max_depth):
            next_level = set()

            for current_node_id in current_level:
                # Get edges for current node
                edges = self.storage.get_edges(current_node_id)

                for edge in edges:
                    # Add related nodes
                    for related_id in [edge.source_id, edge.target_id]:
                        if related_id != current_node_id and related_id not in visited:
                            if relationship_types is None or edge.edge_type in relationship_types:
                                next_level.add(related_id)
                                visited.add(related_id)

                                # Get node details
                                node = self.storage.get_node(related_id)
                                if node:
                                    related_nodes[related_id] = {
                                        "node": node,
                                        "relationship": edge.edge_type,
                                        "depth": depth + 1
                                    }

            current_level = next_level
            if not current_level:
                break

        return related_nodes

    async def build_agent_task_graph(self, task_description: str, available_agents: List[Dict]) -> Dict[str, Any]:
        """Build a graph connecting a task to suitable agents."""
        # Create task node
        task_id = await self.create_task_node(task_description)

        # Create agent nodes
        agent_ids = []
        for agent_info in available_agents:
            agent_id = await self.create_agent_node(
                agent_id=agent_info["agent_id"],
                capabilities=agent_info["capabilities"],
                model_name=agent_info["model_name"]
            )
            agent_ids.append(agent_id)

        # Create relationships based on capability matching
        task_content = task_description.lower()
        relationships = []

        for agent_id in agent_ids:
            agent_node = self.storage.get_node(agent_id)
            if agent_node and agent_node.metadata:
                agent_capabilities = agent_node.metadata.get("capabilities", [])

                # Simple capability matching
                matching_capabilities = []
                for capability in agent_capabilities:
                    if capability.lower() in task_content:
                        matching_capabilities.append(capability)

                if matching_capabilities:
                    # Create relationship with weight based on match quality
                    weight = len(matching_capabilities) / len(agent_capabilities)
                    rel_id = await self.create_relationship(
                        source_id=task_id,
                        target_id=agent_id,
                        relationship_type="suitable_for",
                        weight=weight,
                        metadata={"matching_capabilities": matching_capabilities}
                    )
                    relationships.append(rel_id)

        return {
            "task_id": task_id,
            "agent_ids": agent_ids,
            "relationship_ids": relationships,
            "total_relationships": len(relationships)
        }

    async def update_node_embedding(self, node_id: str, new_content: str) -> bool:
        """Update a node's embedding based on new content."""
        try:
            # Generate new embedding
            new_result = self.embedding_assistant.embed_text(new_content)
            new_embedding = new_result.vector

            # Get existing node
            existing_node = self.storage.get_node(node_id)
            if not existing_node:
                logger.error(f"Node {node_id} not found")
                return False

            # Update node
            updated_node = GraphNode(
                node_id=existing_node.node_id,
                node_type=existing_node.node_type,
                content=new_content,
                embedding=new_embedding,
                metadata=existing_node.metadata,
                created_at=existing_node.created_at,
                updated_at=datetime.now()
            )

            self.storage.add_node(updated_node)
            logger.info(f"Updated embedding for node {node_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to update embedding for {node_id}: {e}")
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph."""
        storage_info = self.storage.get_storage_info()

        # Count nodes by type
        nodes_by_type = {}
        if hasattr(self.storage.storage, 'nodes_data'):
            for node_data in self.storage.storage.nodes_data:
                node_type = node_data.get('node_type', 'unknown')
                nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

        # Count edges by type
        edges_by_type = {}
        if hasattr(self.storage.storage, 'edges_data'):
            for edge_data in self.storage.storage.edges_data:
                edge_type = edge_data.get('edge_type', 'unknown')
                edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1

        return {
            **storage_info,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "embedding_coverage": self._calculate_embedding_coverage()
        }

    def _calculate_embedding_coverage(self) -> float:
        """Calculate what percentage of nodes have embeddings."""
        if hasattr(self.storage.storage, 'nodes_data'):
            total_nodes = len(self.storage.storage.nodes_data)
            if total_nodes == 0:
                return 0.0

            nodes_with_embeddings = 0
            for node_data in self.storage.storage.nodes_data:
                if node_data.get('embedding'):
                    nodes_with_embeddings += 1

            return nodes_with_embeddings / total_nodes

        return 0.0

# Factory function
def create_graph_embedding_manager(storage_type: str = "parquet",
                                 base_path: str = "graph/data") -> GraphEmbeddingManager:
    """Create a graph embedding manager."""
    return GraphEmbeddingManager(storage_type, base_path)

# Example usage and testing
async def demo_embedding_integration():
    """Demonstrate the embedding integration system."""
    print("Graph Embedding Integration Demo")
    print("=" * 50)

    # Create embedding manager
    manager = create_graph_embedding_manager("parquet", "graph/embedding_demo")

    # Create sample task
    task_description = "Analyze the benefits of renewable energy for sustainable development"
    task_id = await manager.create_task_node(
        task_description,
        priority=8,
        metadata={"domain": "sustainability", "complexity": "high"}
    )

    print(f"Created task node: {task_id}")

    # Create sample agents
    agents = [
        {
            "agent_id": "research_agent_001",
            "capabilities": ["research", "analysis", "data_collection"],
            "model_name": "llama3.2:3b"
        },
        {
            "agent_id": "creative_agent_001",
            "capabilities": ["ideation", "brainstorming", "innovation"],
            "model_name": "gemma:270m"
        },
        {
            "agent_id": "critical_agent_001",
            "capabilities": ["evaluation", "risk_assessment", "quality_control"],
            "model_name": "phi:4b"
        }
    ]

    agent_ids = []
    for agent in agents:
        agent_id = await manager.create_agent_node(**agent)
        agent_ids.append(agent_id)
        print(f"Created agent node: {agent_id}")

    # Build task-agent graph
    graph_result = await manager.build_agent_task_graph(task_description, agents)
    print(f"Built task-agent graph with {graph_result['total_relationships']} relationships")

    # Find similar nodes
    similar_nodes = await manager.find_similar_nodes(
        "sustainable energy research",
        node_types=["task", "knowledge"],
        limit=5
    )

    print(f"Found {len(similar_nodes)} similar nodes")
    for node in similar_nodes:
        print(f"  {node['node_id']}: {node['similarity']:.3f} - {node['node_type']}")

    # Get statistics
    stats = manager.get_graph_statistics()
    print(f"Graph statistics: {stats}")

    print("Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_embedding_integration())
