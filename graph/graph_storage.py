"""Graph storage system with multiple backends for embeddings and vectors.

This module provides a unified interface for storing graph data using:
- Parquet files for efficient vector storage
- JSON for easy inspection and portability
- LanceDB for advanced vector operations
- File history tracking for all operations
"""

import os
import json
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    node_type: str  # 'agent', 'task', 'knowledge', 'concept'
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class GraphEdge:
    """Represents an edge/relationship in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str  # 'communicates_with', 'depends_on', 'related_to', 'contains'
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FileHistoryEntry:
    """Tracks history of file operations."""
    file_path: str
    operation: str  # 'created', 'modified', 'deleted'
    timestamp: datetime
    content_hash: str
    metadata: Optional[Dict[str, Any]] = None
    backup_path: Optional[str] = None

class FileHistoryTracker:
    """Tracks all file operations with history."""

    def __init__(self, history_file: str = "graph/file_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[FileHistoryEntry] = []
        self._load_history()

    def _load_history(self):
        """Load existing history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history = [
                        FileHistoryEntry(
                            file_path=entry['file_path'],
                            operation=entry['operation'],
                            timestamp=datetime.fromisoformat(entry['timestamp']),
                            content_hash=entry['content_hash'],
                            metadata=entry.get('metadata', {}),
                            backup_path=entry.get('backup_path')
                        )
                        for entry in data.get('history', [])
                    ]
            except Exception as e:
                logger.error(f"Failed to load file history: {e}")
                self.history = []

    def _save_history(self):
        """Save history to file."""
        try:
            data = {
                'history': [
                    {
                        'file_path': entry.file_path,
                        'operation': entry.operation,
                        'timestamp': entry.timestamp.isoformat(),
                        'content_hash': entry.content_hash,
                        'metadata': entry.metadata,
                        'backup_path': entry.backup_path
                    }
                    for entry in self.history
                ]
            }

            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file history: {e}")

    def add_entry(self, file_path: str, operation: str, content: str = "",
                  metadata: Dict[str, Any] = None, backup_path: str = None):
        """Add a file operation to history."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        entry = FileHistoryEntry(
            file_path=file_path,
            operation=operation,
            timestamp=datetime.now(),
            content_hash=content_hash,
            metadata=metadata or {},
            backup_path=backup_path
        )

        self.history.append(entry)
        self._save_history()

        logger.info(f"File history: {operation} {file_path}")

    def get_file_history(self, file_path: str) -> List[FileHistoryEntry]:
        """Get history for a specific file."""
        return [entry for entry in self.history if entry.file_path == file_path]

    def get_recent_operations(self, limit: int = 10) -> List[FileHistoryEntry]:
        """Get recent file operations."""
        return sorted(self.history, key=lambda x: x.timestamp, reverse=True)[:limit]

class ParquetGraphStorage:
    """Parquet-based storage for graph data with vector support."""

    def __init__(self, base_path: str = "graph/data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.nodes_file = self.base_path / "nodes.parquet"
        self.edges_file = self.base_path / "edges.parquet"
        self.vectors_file = self.base_path / "vectors.parquet"

        self.nodes_data = []
        self.edges_data = []
        self.vectors_data = []

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing data from Parquet files."""
        try:
            if self.nodes_file.exists():
                df = pq.read_table(self.nodes_file).to_pandas()
                self.nodes_data = df.to_dict('records')

            if self.edges_file.exists():
                df = pq.read_table(self.edges_file).to_pandas()
                self.edges_data = df.to_dict('records')

            if self.vectors_file.exists():
                df = pq.read_table(self.vectors_file).to_pandas()
                self.vectors_data = df.to_dict('records')

        except Exception as e:
            logger.error(f"Failed to load existing graph data: {e}")

    def _save_to_parquet(self):
        """Save data to Parquet files."""
        try:
            # Save nodes
            if self.nodes_data:
                nodes_table = pa.Table.from_pylist(self.nodes_data)
                pq.write_table(nodes_table, self.nodes_file)

            # Save edges
            if self.edges_data:
                edges_table = pa.Table.from_pylist(self.edges_data)
                pq.write_table(edges_table, self.edges_file)

            # Save vectors
            if self.vectors_data:
                vectors_table = pa.Table.from_pylist(self.vectors_data)
                pq.write_table(vectors_table, self.vectors_file)

        except Exception as e:
            logger.error(f"Failed to save graph data to Parquet: {e}")

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph."""
        node_dict = {
            'node_id': node.node_id,
            'node_type': node.node_type,
            'content': node.content,
            'embedding': node.embedding,
            'metadata': json.dumps(node.metadata),
            'created_at': node.created_at.isoformat(),
            'updated_at': node.updated_at.isoformat()
        }

        # Check if node exists, update if so
        existing_idx = None
        for i, existing in enumerate(self.nodes_data):
            if existing['node_id'] == node.node_id:
                existing_idx = i
                break

        if existing_idx is not None:
            self.nodes_data[existing_idx] = node_dict
        else:
            self.nodes_data.append(node_dict)

        # Add/update vector data
        vector_dict = {
            'node_id': node.node_id,
            'embedding': node.embedding or [],
            'content': node.content,
            'node_type': node.node_type
        }

        vector_idx = None
        for i, existing in enumerate(self.vectors_data):
            if existing['node_id'] == node.node_id:
                vector_idx = i
                break

        if vector_idx is not None:
            self.vectors_data[vector_idx] = vector_dict
        else:
            self.vectors_data.append(vector_dict)

        self._save_to_parquet()
        return node.node_id

    def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the graph."""
        edge_dict = {
            'edge_id': edge.edge_id,
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'edge_type': edge.edge_type,
            'weight': edge.weight,
            'metadata': json.dumps(edge.metadata),
            'created_at': edge.created_at.isoformat()
        }

        # Check if edge exists
        existing_idx = None
        for i, existing in enumerate(self.edges_data):
            if existing['edge_id'] == edge.edge_id:
                existing_idx = i
                break

        if existing_idx is not None:
            self.edges_data[existing_idx] = edge_dict
        else:
            self.edges_data.append(edge_dict)

        self._save_to_parquet()
        return edge.edge_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        for node_data in self.nodes_data:
            if node_data['node_id'] == node_id:
                return GraphNode(
                    node_id=node_data['node_id'],
                    node_type=node_data['node_type'],
                    content=node_data['content'],
                    embedding=node_data.get('embedding'),
                    metadata=json.loads(node_data.get('metadata', '{}')),
                    created_at=datetime.fromisoformat(node_data['created_at']),
                    updated_at=datetime.fromisoformat(node_data['updated_at'])
                )
        return None

    def get_edges(self, node_id: str = None, edge_type: str = None) -> List[GraphEdge]:
        """Get edges, optionally filtered by node or type."""
        edges = []
        for edge_data in self.edges_data:
            if node_id and edge_data['source_id'] != node_id and edge_data['target_id'] != node_id:
                continue
            if edge_type and edge_data['edge_type'] != edge_type:
                continue

            edges.append(GraphEdge(
                edge_id=edge_data['edge_id'],
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                edge_type=edge_data['edge_type'],
                weight=edge_data['weight'],
                metadata=json.loads(edge_data.get('metadata', '{}')),
                created_at=datetime.fromisoformat(edge_data['created_at'])
            ))
        return edges

    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar nodes using vector similarity."""
        if not self.vectors_data:
            return []

        # Simple cosine similarity (in production, use FAISS or similar)
        similarities = []
        query_vector = np.array(query_embedding)

        for vector_data in self.vectors_data:
            if not vector_data.get('embedding'):
                continue

            node_vector = np.array(vector_data['embedding'])
            if len(node_vector) == 0:
                continue

            # Cosine similarity
            similarity = np.dot(query_vector, node_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
            )

            similarities.append({
                'node_id': vector_data['node_id'],
                'similarity': float(similarity),
                'content': vector_data['content'],
                'node_type': vector_data['node_type']
            })

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]

class JSONGraphStorage:
    """JSON-based storage for easy inspection and portability."""

    def __init__(self, base_path: str = "graph/json_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.nodes_file = self.base_path / "nodes.json"
        self.edges_file = self.base_path / "edges.json"
        self.vectors_file = self.base_path / "vectors.json"

        self.nodes_data = []
        self.edges_data = []
        self.vectors_data = []

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing JSON data."""
        try:
            if self.nodes_file.exists():
                with open(self.nodes_file, 'r') as f:
                    self.nodes_data = json.load(f)

            if self.edges_file.exists():
                with open(self.edges_file, 'r') as f:
                    self.edges_data = json.load(f)

            if self.vectors_file.exists():
                with open(self.vectors_file, 'r') as f:
                    self.vectors_data = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load JSON graph data: {e}")

    def _save_data(self):
        """Save data to JSON files."""
        try:
            with open(self.nodes_file, 'w') as f:
                json.dump(self.nodes_data, f, indent=2)

            with open(self.edges_file, 'w') as f:
                json.dump(self.edges_data, f, indent=2)

            with open(self.vectors_file, 'w') as f:
                json.dump(self.vectors_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save JSON graph data: {e}")

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the JSON storage."""
        node_dict = asdict(node)
        node_dict['embedding'] = node.embedding
        node_dict['metadata'] = node.metadata

        # Update or add
        for i, existing in enumerate(self.nodes_data):
            if existing['node_id'] == node.node_id:
                self.nodes_data[i] = node_dict
                break
        else:
            self.nodes_data.append(node_dict)

        # Update vectors
        vector_dict = {
            'node_id': node.node_id,
            'embedding': node.embedding or [],
            'content': node.content,
            'node_type': node.node_type
        }

        for i, existing in enumerate(self.vectors_data):
            if existing['node_id'] == node.node_id:
                self.vectors_data[i] = vector_dict
                break
        else:
            self.vectors_data.append(vector_dict)

        self._save_data()
        return node.node_id

    def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the JSON storage."""
        edge_dict = asdict(edge)
        edge_dict['metadata'] = edge.metadata

        for i, existing in enumerate(self.edges_data):
            if existing['edge_id'] == edge.edge_id:
                self.edges_data[i] = edge_dict
                break
        else:
            self.edges_data.append(edge_dict)

        self._save_data()
        return edge.edge_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        for node_data in self.nodes_data:
            if node_data['node_id'] == node_id:
                return GraphNode(**node_data)
        return None

    def get_edges(self, node_id: str = None, edge_type: str = None) -> List[GraphEdge]:
        """Get edges, optionally filtered."""
        edges = []
        for edge_data in self.edges_data:
            if node_id and edge_data['source_id'] != node_id and edge_data['target_id'] != node_id:
                continue
            if edge_type and edge_data['edge_type'] != edge_type:
                continue

            edges.append(GraphEdge(**edge_data))
        return edges

    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar nodes using vector similarity."""
        # Simplified similarity search for JSON storage
        similarities = []

        for vector_data in self.vectors_data:
            if not vector_data.get('embedding'):
                continue

            # Simple similarity calculation
            similarity = self._calculate_similarity(query_embedding, vector_data['embedding'])
            similarities.append({
                'node_id': vector_data['node_id'],
                'similarity': similarity,
                'content': vector_data['content'],
                'node_type': vector_data['node_type']
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        if len(v1) == 0 or len(v2) == 0:
            return 0.0

        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

class GraphStorageManager:
    """Unified interface for graph storage with multiple backends."""

    def __init__(self, storage_type: str = "parquet", base_path: str = "graph/data"):
        self.storage_type = storage_type
        self.base_path = base_path

        if storage_type == "parquet":
            self.storage = ParquetGraphStorage(base_path)
        elif storage_type == "json":
            self.storage = JSONGraphStorage(base_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        self.file_tracker = FileHistoryTracker()

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph."""
        node_id = self.storage.add_node(node)
        self.file_tracker.add_entry(
            f"{self.base_path}/nodes",
            "node_added",
            json.dumps(asdict(node)),
            {"node_id": node_id, "node_type": node.node_type}
        )
        return node_id

    def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the graph."""
        edge_id = self.storage.add_edge(edge)
        self.file_tracker.add_entry(
            f"{self.base_path}/edges",
            "edge_added",
            json.dumps(asdict(edge)),
            {"edge_id": edge_id, "edge_type": edge.edge_type}
        )
        return edge_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.storage.get_node(node_id)

    def get_edges(self, node_id: str = None, edge_type: str = None) -> List[GraphEdge]:
        """Get edges with optional filtering."""
        return self.storage.get_edges(node_id, edge_type)

    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar nodes using embeddings."""
        return self.storage.search_similar(query_embedding, limit)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        return {
            "storage_type": self.storage_type,
            "base_path": str(self.base_path),
            "nodes_count": len(self.storage.nodes_data) if hasattr(self.storage, 'nodes_data') else 0,
            "edges_count": len(self.storage.edges_data) if hasattr(self.storage, 'edges_data') else 0,
            "vectors_count": len(self.storage.vectors_data) if hasattr(self.storage, 'vectors_data') else 0
        }

# Factory function for easy creation
def create_graph_storage(storage_type: str = "parquet", base_path: str = "graph/data") -> GraphStorageManager:
    """Create a graph storage manager with the specified backend."""
    return GraphStorageManager(storage_type, base_path)

# Example usage and testing
async def demo_graph_storage():
    """Demonstrate the graph storage system."""
    print("Graph Storage Demo")
    print("=" * 50)

    # Create storage
    storage = create_graph_storage("parquet", "graph/demo_data")

    # Create sample nodes
    agent_node = GraphNode(
        node_id="agent_001",
        node_type="agent",
        content="Research assistant with web search capabilities",
        embedding=[0.1, 0.2, 0.3] * 100,  # Mock embedding
        metadata={"model": "llama3.2:3b", "capabilities": ["research", "web_search"]}
    )

    task_node = GraphNode(
        node_id="task_001",
        node_type="task",
        content="Analyze renewable energy trends",
        embedding=[0.15, 0.25, 0.35] * 100,  # Mock embedding
        metadata={"priority": 8, "task_type": "research"}
    )

    # Add nodes
    storage.add_node(agent_node)
    storage.add_node(task_node)

    # Create edge
    edge = GraphEdge(
        edge_id="edge_001",
        source_id="task_001",
        target_id="agent_001",
        edge_type="assigned_to",
        weight=0.9,
        metadata={"assignment_time": datetime.now().isoformat()}
    )

    storage.add_edge(edge)

    # Search similar
    query_embedding = [0.12, 0.22, 0.32] * 100
    similar = storage.search_similar(query_embedding, limit=5)

    print(f"Storage info: {storage.get_storage_info()}")
    print(f"Similar nodes found: {len(similar)}")

    for result in similar:
        print(f"  {result['node_id']}: {result['similarity']:.3f}")

    print("Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_graph_storage())
