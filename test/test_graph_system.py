"""Comprehensive tests for the graph system components.

This module tests all graph functionality including:
- Graph storage (Parquet and JSON)
- Embedding integration
- Enhanced memory graph with swarm integration
- File history tracking
- Vector similarity search
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import graph components
from graph.graph_storage import (
    GraphNode, GraphEdge, FileHistoryTracker,
    ParquetGraphStorage, JSONGraphStorage, GraphStorageManager,
    create_graph_storage, demo_graph_storage
)
from graph.embedding_integration import (
    GraphEmbeddingManager, create_graph_embedding_manager, demo_embedding_integration
)
from graph.enhanced_memory_graph import (
    EnhancedMemoryGraph, create_enhanced_memory_graph, demo_enhanced_memory_graph
)


class TestGraphNode(unittest.TestCase):
    """Test GraphNode functionality."""

    def test_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(
            node_id="test_node",
            node_type="test",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"}
        )

        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.node_type, "test")
        self.assertEqual(node.content, "Test content")
        self.assertEqual(node.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(node.metadata["key"], "value")
        self.assertIsNotNone(node.created_at)
        self.assertIsNotNone(node.updated_at)

    def test_node_auto_timestamps(self):
        """Test automatic timestamp assignment."""
        node = GraphNode(
            node_id="test_node",
            node_type="test",
            content="Test content"
        )

        self.assertIsNotNone(node.created_at)
        self.assertIsNotNone(node.updated_at)
        self.assertEqual(node.created_at, node.updated_at)


class TestGraphEdge(unittest.TestCase):
    """Test GraphEdge functionality."""

    def test_edge_creation(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            edge_id="test_edge",
            source_id="source_node",
            target_id="target_node",
            edge_type="test_relationship",
            weight=0.8,
            metadata={"test": "data"}
        )

        self.assertEqual(edge.edge_id, "test_edge")
        self.assertEqual(edge.source_id, "source_node")
        self.assertEqual(edge.target_id, "target_node")
        self.assertEqual(edge.edge_type, "test_relationship")
        self.assertEqual(edge.weight, 0.8)
        self.assertEqual(edge.metadata["test"], "data")
        self.assertIsNotNone(edge.created_at)


class TestFileHistoryTracker(unittest.TestCase):
    """Test file history tracking functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.temp_dir, "test_history.json")

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_history_tracking(self):
        """Test file operation history tracking."""
        tracker = FileHistoryTracker(self.history_file)

        # Add some history entries
        tracker.add_entry("test_file.txt", "created", "test content", {"type": "test"})
        tracker.add_entry("test_file.txt", "modified", "modified content", {"type": "update"})

        # Check history
        history = tracker.get_file_history("test_file.txt")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].operation, "created")
        self.assertEqual(history[1].operation, "modified")

        # Check recent operations
        recent = tracker.get_recent_operations(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].operation, "modified")


class TestParquetGraphStorage(unittest.TestCase):
    """Test Parquet-based graph storage."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ParquetGraphStorage(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_add_and_get_node(self):
        """Test adding and retrieving nodes."""
        node = GraphNode(
            node_id="test_node",
            node_type="test",
            content="Test content",
            embedding=[0.1, 0.2, 0.3]
        )

        # Add node
        node_id = self.storage.add_node(node)
        self.assertEqual(node_id, "test_node")

        # Retrieve node
        retrieved = self.storage.get_node("test_node")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.node_id, "test_node")
        self.assertEqual(retrieved.content, "Test content")

    def test_add_and_get_edge(self):
        """Test adding and retrieving edges."""
        edge = GraphEdge(
            edge_id="test_edge",
            source_id="source_node",
            target_id="target_node",
            edge_type="test_relationship"
        )

        # Add edge
        edge_id = self.storage.add_edge(edge)
        self.assertEqual(edge_id, "test_edge")

        # Get edges for source node
        edges = self.storage.get_edges("source_node")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].edge_id, "test_edge")

    def test_vector_similarity_search(self):
        """Test vector similarity search."""
        # Create nodes with different embeddings
        node1 = GraphNode(
            node_id="node1",
            node_type="test",
            content="Similar content",
            embedding=[1.0, 0.0, 0.0]
        )
        node2 = GraphNode(
            node_id="node2",
            node_type="test",
            content="Different content",
            embedding=[0.0, 1.0, 0.0]
        )

        self.storage.add_node(node1)
        self.storage.add_node(node2)

        # Search for similar vectors
        query_embedding = [0.9, 0.1, 0.0]
        similar = self.storage.search_similar(query_embedding, limit=5)

        self.assertGreater(len(similar), 0)
        # Node1 should be more similar than node2
        self.assertEqual(similar[0]['node_id'], "node1")


class TestJSONGraphStorage(unittest.TestCase):
    """Test JSON-based graph storage."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = JSONGraphStorage(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_json_storage_persistence(self):
        """Test that JSON storage persists data correctly."""
        node = GraphNode(
            node_id="json_test",
            node_type="test",
            content="JSON test content"
        )

        # Add node
        self.storage.add_node(node)

        # Create new storage instance to test persistence
        new_storage = JSONGraphStorage(self.temp_dir)
        retrieved = new_storage.get_node("json_test")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "JSON test content")


class TestGraphStorageManager(unittest.TestCase):
    """Test the unified graph storage manager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_storage_manager_parquet(self):
        """Test storage manager with Parquet backend."""
        manager = GraphStorageManager("parquet", self.temp_dir)

        node = GraphNode(
            node_id="manager_test",
            node_type="test",
            content="Manager test content"
        )

        node_id = manager.add_node(node)
        retrieved = manager.get_node(node_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Manager test content")

    def test_storage_manager_json(self):
        """Test storage manager with JSON backend."""
        manager = GraphStorageManager("json", self.temp_dir)

        node = GraphNode(
            node_id="json_manager_test",
            node_type="test",
            content="JSON manager test content"
        )

        node_id = manager.add_node(node)
        retrieved = manager.get_node(node_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "JSON manager test content")

    def test_invalid_storage_type(self):
        """Test error handling for invalid storage type."""
        with self.assertRaises(ValueError):
            GraphStorageManager("invalid_type", self.temp_dir)


class TestGraphEmbeddingManager(unittest.TestCase):
    """Test the graph embedding manager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch('graph.embedding_integration.EmbeddingAssistant')
    def test_create_node_with_embedding(self, mock_embedding_assistant):
        """Test creating nodes with embeddings."""
        # Mock embedding assistant
        mock_assistant = Mock()
        mock_result = Mock()
        mock_result.vector = [0.1, 0.2, 0.3]
        mock_assistant.embed_text.return_value = mock_result
        mock_embedding_assistant.return_value = mock_assistant

        async def run_test():
            manager = GraphEmbeddingManager("json", self.temp_dir)
            manager.embedding_assistant = mock_assistant

            node_id = await manager.create_node_with_embedding(
                content="Test content",
                node_type="test",
                metadata={"test": "data"}
            )

            self.assertIsNotNone(node_id)
            self.assertTrue(node_id.startswith("test_"))

            # Verify embedding was requested
            mock_assistant.embed_text.assert_called_once_with("Test content")

        asyncio.run(run_test())

    @patch('graph.embedding_integration.EmbeddingAssistant')
    def test_create_task_node(self, mock_embedding_assistant):
        """Test creating task nodes."""
        mock_assistant = Mock()
        mock_result = Mock()
        mock_result.vector = [0.1, 0.2, 0.3]
        mock_assistant.embed_text.return_value = mock_result
        mock_embedding_assistant.return_value = mock_assistant

        async def run_test():
            manager = GraphEmbeddingManager("json", self.temp_dir)
            manager.embedding_assistant = mock_assistant

            task_id = await manager.create_task_node(
                "Test task description",
                priority=8,
                metadata={"domain": "test"}
            )

            self.assertIsNotNone(task_id)

            # Check that metadata was enhanced
            node = manager.storage.get_node(task_id)
            self.assertEqual(node.metadata["priority"], 8)
            self.assertEqual(node.metadata["domain"], "test")
            self.assertEqual(node.metadata["task_type"], "swarm_task")

        asyncio.run(run_test())

    @patch('graph.embedding_integration.EmbeddingAssistant')
    def test_find_similar_nodes(self, mock_embedding_assistant):
        """Test finding similar nodes."""
        mock_assistant = Mock()
        mock_result = Mock()
        mock_result.vector = [0.1, 0.2, 0.3]
        mock_assistant.embed_text.return_value = mock_result
        mock_embedding_assistant.return_value = mock_assistant

        async def run_test():
            manager = GraphEmbeddingManager("json", self.temp_dir)
            manager.embedding_assistant = mock_assistant

            # Create some test nodes
            await manager.create_task_node("Renewable energy analysis")
            await manager.create_task_node("Solar panel research")

            # Search for similar nodes
            similar = await manager.find_similar_nodes(
                "energy research",
                node_types=["task"],
                limit=5
            )

            # Should find some results (exact behavior depends on mock)
            self.assertIsInstance(similar, list)

        asyncio.run(run_test())


class TestEnhancedMemoryGraph(unittest.TestCase):
    """Test the enhanced memory graph with swarm integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch('graph.enhanced_memory_graph.SwarmMCPClient')
    @patch('graph.enhanced_memory_graph.SwarmMCPServer')
    def test_enhanced_graph_initialization(self, mock_server, mock_client):
        """Test enhanced graph initialization."""
        async def run_test():
            graph = EnhancedMemoryGraph("json", self.temp_dir)

            # Mock MCP components
            mock_client_instance = AsyncMock()
            mock_server_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_server.return_value = mock_server_instance

            await graph.initialize()

            self.assertTrue(graph.initialized)
            mock_client_instance.connect.assert_called_once()
            mock_server_instance.start_server.assert_called_once()

        asyncio.run(run_test())

    def test_memory_importance_calculation(self):
        """Test memory importance calculation."""
        graph = EnhancedMemoryGraph("json", self.temp_dir)

        # Test different memory types
        importance = graph._calculate_memory_importance("Short content", "task_result")
        self.assertEqual(importance, 0.8)  # task_result = 0.9, but short content reduces it

        importance = graph._calculate_memory_importance("Long content " * 200, "context")
        self.assertEqual(importance, 0.7)  # context = 0.6, but long content increases it

    @patch('graph.enhanced_memory_graph.SwarmMCPClient')
    @patch('graph.enhanced_memory_graph.SwarmMCPServer')
    def test_agent_registration(self, mock_server, mock_client):
        """Test swarm agent registration."""
        async def run_test():
            graph = EnhancedMemoryGraph("json", self.temp_dir)

            # Mock MCP components
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_server.return_value = AsyncMock()
            await graph.initialize()

            # Register agent
            node_id = await graph.register_swarm_agent(
                "test_agent",
                ["research", "analysis"],
                "llama3.2:3b",
                {"status": "active"}
            )

            self.assertIsNotNone(node_id)
            self.assertIn("test_agent", graph.agent_nodes)
            self.assertIn("test_agent", graph.connected_agents)

        asyncio.run(run_test())

    @patch('graph.enhanced_memory_graph.SwarmMCPClient')
    @patch('graph.enhanced_memory_graph.SwarmMCPServer')
    def test_memory_storage(self, mock_server, mock_client):
        """Test swarm memory storage."""
        async def run_test():
            graph = EnhancedMemoryGraph("json", self.temp_dir)

            # Mock MCP components
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_server.return_value = AsyncMock()
            await graph.initialize()

            # Store memory
            node_id = await graph.store_swarm_memory(
                "Test memory content",
                "task_result",
                "test_agent",
                {"importance": 0.8}
            )

            self.assertIsNotNone(node_id)
            self.assertIn("Test memory content", graph.memory_nodes)

        asyncio.run(run_test())

    @patch('graph.enhanced_memory_graph.SwarmMCPClient')
    @patch('graph.enhanced_memory_graph.SwarmMCPServer')
    def test_task_lifecycle_tracking(self, mock_server, mock_client):
        """Test task lifecycle tracking."""
        async def run_test():
            graph = EnhancedMemoryGraph("json", self.temp_dir)

            # Mock MCP components
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_server.return_value = AsyncMock()
            await graph.initialize()

            # Track task lifecycle
            node_id = await graph.track_task_lifecycle(
                "task_001",
                "Test task description",
                "test_agent",
                "in_progress",
                {"priority": 8}
            )

            self.assertIsNotNone(node_id)
            self.assertIn("task_001", graph.task_nodes)

        asyncio.run(run_test())


class TestGraphIntegration(unittest.TestCase):
    """Test integration between graph components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_factory_functions(self):
        """Test factory functions for creating graph components."""
        # Test storage creation
        storage = create_graph_storage("json", self.temp_dir)
        self.assertIsInstance(storage, GraphStorageManager)

        # Test embedding manager creation
        embedding_manager = create_graph_embedding_manager("json", self.temp_dir)
        self.assertIsInstance(embedding_manager, GraphEmbeddingManager)

        # Test enhanced graph creation
        enhanced_graph = create_enhanced_memory_graph("json", self.temp_dir)
        self.assertIsInstance(enhanced_graph, EnhancedMemoryGraph)

    def test_storage_type_validation(self):
        """Test validation of storage types."""
        with self.assertRaises(ValueError):
            create_graph_storage("invalid_type", self.temp_dir)


class TestGraphPerformance(unittest.TestCase):
    """Test performance characteristics of graph operations."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_large_graph_operations(self):
        """Test operations on larger graphs."""
        storage = GraphStorageManager("json", self.temp_dir)

        # Create multiple nodes
        node_ids = []
        for i in range(100):
            node = GraphNode(
                node_id=f"node_{i}",
                node_type="test",
                content=f"Content for node {i}",
                embedding=[float(i)] * 10  # Simple embedding
            )
            node_id = storage.add_node(node)
            node_ids.append(node_id)

        # Verify all nodes were created
        self.assertEqual(len(node_ids), 100)

        # Test retrieval
        for node_id in node_ids[:10]:  # Test first 10
            node = storage.get_node(node_id)
            self.assertIsNotNone(node)
            self.assertEqual(node.node_id, node_id)


def run_graph_storage_demo():
    """Run the graph storage demo for manual testing."""
    print("Running Graph Storage Demo...")
    try:
        asyncio.run(demo_graph_storage())
        print("✅ Graph Storage Demo completed successfully!")
    except Exception as e:
        print(f"❌ Graph Storage Demo failed: {e}")


def run_embedding_integration_demo():
    """Run the embedding integration demo for manual testing."""
    print("Running Embedding Integration Demo...")
    try:
        asyncio.run(demo_embedding_integration())
        print("✅ Embedding Integration Demo completed successfully!")
    except Exception as e:
        print(f"❌ Embedding Integration Demo failed: {e}")


def run_enhanced_memory_demo():
    """Run the enhanced memory graph demo for manual testing."""
    print("Running Enhanced Memory Graph Demo...")
    try:
        asyncio.run(demo_enhanced_memory_graph())
        print("✅ Enhanced Memory Graph Demo completed successfully!")
    except Exception as e:
        print(f"❌ Enhanced Memory Graph Demo failed: {e}")


if __name__ == "__main__":
    # Run all tests
    print("Running Graph System Tests...")
    print("=" * 50)

    # Create test suite
    test_classes = [
        TestGraphNode,
        TestGraphEdge,
        TestFileHistoryTracker,
        TestParquetGraphStorage,
        TestJSONGraphStorage,
        TestGraphStorageManager,
        TestGraphEmbeddingManager,
        TestEnhancedMemoryGraph,
        TestGraphIntegration,
        TestGraphPerformance
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split()[-1]}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split()[-1]}")

    # Run demos if tests pass
    if result.wasSuccessful():
        print("\n" + "=" * 50)
        print("Running Demos...")

        try:
            run_graph_storage_demo()
            run_embedding_integration_demo()
            run_enhanced_memory_demo()

            print("\n🎉 All tests and demos completed successfully!")
        except Exception as e:
            print(f"\n⚠️  Demos completed with some issues: {e}")
    else:
        print("\n❌ Tests failed. Skipping demos.")
