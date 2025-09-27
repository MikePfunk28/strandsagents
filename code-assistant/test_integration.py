"""Integration tests for the coding assistant system."""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

import pytest

from simple_coding_agent import create_simple_coding_agent
from coding_assistant import CodingAssistant
from database_manager import DatabaseManager
from ollama_model import create_ollama_model

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_ollama_available():
    """Check if Ollama is available for testing."""
    try:
        model = create_ollama_model("llama3.2", "http://localhost:11434")
        if model.health_check():
            return True
    except Exception:
        pass
    return False


class TestDatabaseManager:
    """Test database management functionality."""

    def test_database_initialization(self, temp_db_dir):
        """Test database initialization."""
        db_manager = DatabaseManager(temp_db_dir)

        # Check that database files are created
        assert (temp_db_dir / "cache.db").exists()
        assert (temp_db_dir / "knowledge.db").exists()
        assert (temp_db_dir / "memory.db").exists()

    def test_cache_operations(self, temp_db_dir):
        """Test cache operations."""
        db_manager = DatabaseManager(temp_db_dir)

        # Test set and get
        db_manager.cache_set("test_key", "test_value", ttl_seconds=60)
        value = db_manager.cache_get("test_key")
        assert value == "test_value"

        # Test expiration
        db_manager.cache_set("expire_key", "expire_value", ttl_seconds=0)
        time.sleep(0.1)
        value = db_manager.cache_get("expire_key")
        assert value is None

    def test_knowledge_operations(self, temp_db_dir):
        """Test knowledge database operations."""
        db_manager = DatabaseManager(temp_db_dir)

        # Add knowledge entry
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        entry_id = db_manager.knowledge_add(
            content="Test knowledge content",
            embedding=embedding,
            source="test_source",
            metadata={"category": "test"}
        )

        assert entry_id is not None

        # Search knowledge
        results = db_manager.knowledge_search(embedding, limit=5)
        assert len(results) == 1
        assert results[0].content == "Test knowledge content"

    def test_memory_operations(self, temp_db_dir):
        """Test memory database operations."""
        db_manager = DatabaseManager(temp_db_dir)

        # Add memory entry
        entry_id = db_manager.memory_add(
            session_id="test_session",
            content="Test memory content",
            memory_type="test",
            metadata={"importance": 0.8}
        )

        assert entry_id is not None

        # Get session memories
        memories = db_manager.memory_get_session("test_session", limit=10)
        assert len(memories) == 1
        assert memories[0].content == "Test memory content"


class TestOllamaModel:
    """Test Ollama model integration."""

    def test_model_creation(self):
        """Test model creation."""
        model = create_ollama_model("llama3.2", "http://localhost:11434")
        assert model.model_name == "llama3.2"
        assert model.host == "http://localhost:11434"

    def test_model_info(self):
        """Test model information."""
        model = create_ollama_model("llama3.2", "http://localhost:11434")
        info = model.model_info
        assert "name" in info
        assert "host" in info
        assert "available" in info
        assert "server_healthy" in info

    @pytest.mark.skipif(not pytest.importorskip("requests"), reason="requests not available")
    def test_health_check(self):
        """Test health check (requires Ollama server)."""
        model = create_ollama_model("llama3.2", "http://localhost:11434")
        # This will return False if Ollama is not running, which is expected in CI
        health = model.health_check()
        assert isinstance(health, bool)


class TestCodingAssistant:
    """Test coding assistant functionality."""

    def test_assistant_initialization(self, temp_db_dir):
        """Test assistant initialization."""
        assistant = CodingAssistant(
            db_dir=temp_db_dir,
            model_name="llama3.2",
            session_id="test_session"
        )

        assert assistant.session_id == "test_session"
        assert assistant.db_dir == temp_db_dir
        assert len(assistant.tools) > 0

    def test_context_classification(self, temp_db_dir):
        """Test content classification."""
        assistant = CodingAssistant(db_dir=temp_db_dir)

        # Test coding classification
        classification = assistant._classify_query("How do I write a Python function?")
        assert classification["primary_category"] in assistant.context_categories

        # Test documentation classification
        classification = assistant._classify_query("Explain this code documentation")
        assert isinstance(classification["scores"], dict)

    def test_cache_key_generation(self, temp_db_dir):
        """Test cache key generation."""
        assistant = CodingAssistant(db_dir=temp_db_dir)

        key1 = assistant._generate_cache_key("test query", "test context")
        key2 = assistant._generate_cache_key("test query", "test context")
        key3 = assistant._generate_cache_key("different query", "test context")

        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys

    def test_stats(self, temp_db_dir):
        """Test statistics generation."""
        assistant = CodingAssistant(db_dir=temp_db_dir)
        stats = assistant.get_stats()

        assert "session_id" in stats
        assert "memory_stats" in stats
        assert "model_info" in stats
        assert "tools_available" in stats
        assert "context_categories" in stats


class TestCodingAgent:
    """Test coding agent with workflow orchestration."""

    def test_agent_creation(self, temp_db_dir):
        """Test agent creation."""
        agent = create_simple_coding_agent(
            db_dir=str(temp_db_dir),
            model_name="llama3.2"
        )

        assert agent.assistant is not None
        assert hasattr(agent, 'current_task')
        assert hasattr(agent, 'task_history')

    def test_chat_interface(self, temp_db_dir):
        """Test simple chat interface."""
        agent = create_simple_coding_agent(db_dir=str(temp_db_dir))

        # This will work even without Ollama running (will use fallback)
        response = agent.chat("Hello, can you help me with Python?")
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_workflow_planning(self, temp_db_dir):
        """Test workflow planning."""
        agent = create_simple_coding_agent(db_dir=str(temp_db_dir))

        # Test planning for different task types
        plan = await agent.orchestrator.plan_workflow({
            "type": "debug",
            "description": "Fix a Python error"
        })

        assert isinstance(plan, list)
        assert len(plan) > 0
        assert all("name" in step for step in plan)

    @pytest.mark.asyncio
    async def test_step_execution(self, temp_db_dir):
        """Test workflow step execution."""
        agent = create_simple_coding_agent(db_dir=str(temp_db_dir))

        # Test analysis step
        result = await agent.executor.execute_step(
            "analysis",
            {"description": "Analyze code"},
            {"task": {"description": "Test analysis task"}}
        )

        assert isinstance(result, dict)
        assert "success" in result
        assert "step" in result

    def test_agent_stats(self, temp_db_dir):
        """Test agent statistics."""
        agent = create_simple_coding_agent(db_dir=str(temp_db_dir))
        stats = agent.get_stats()

        assert "session_id" in stats
        assert "memory_stats" in stats


class TestIntegrationFlow:
    """Test complete integration flows."""

    def test_memory_integration(self, temp_db_dir):
        """Test memory system integration."""
        assistant = CodingAssistant(db_dir=temp_db_dir)

        # Add memory
        memory_id = assistant.memory_manager.add_memory(
            content="Python is a high-level programming language",
            session_id=assistant.session_id,
            memory_type="knowledge",
            importance=0.8
        )

        assert memory_id is not None

        # Retrieve memory
        memories = assistant.memory_manager.retrieve_memory(
            query="What is Python?",
            session_id=assistant.session_id,
            limit=5
        )

        # Should find relevant memory even without embeddings working
        assert isinstance(memories, list)

    def test_conversation_flow(self, temp_db_dir):
        """Test conversation memory flow."""
        assistant = CodingAssistant(db_dir=temp_db_dir)

        # Update conversation
        assistant.memory_manager.update_conversation(
            assistant.session_id,
            {"role": "user", "content": "Hello", "timestamp": time.time()}
        )

        assistant.memory_manager.update_conversation(
            assistant.session_id,
            {"role": "assistant", "content": "Hi there!", "timestamp": time.time()}
        )

        # Get conversation context
        context = assistant.memory_manager.get_conversation_context(
            assistant.session_id,
            limit=10
        )

        assert "recent_messages" in context
        assert len(context["recent_messages"]) == 2

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_db_dir):
        """Test a complete workflow execution."""
        agent = create_simple_coding_agent(db_dir=str(temp_db_dir))

        # Execute a simple task
        result = await agent.execute_task(
            "Analyze a simple Python script",
            task_type="analysis"
        )

        assert isinstance(result, dict)
        assert "success" in result
        assert "task" in result
        assert "session_id" in result

        # Test cleanup
        agent.cleanup()


def run_manual_tests():
    """Run manual tests that require Ollama to be running."""
    print("🧪 Running manual integration tests...")

    # Test 1: Basic agent creation and health check
    print("\n1. Testing agent creation...")
    try:
        agent = create_simple_coding_agent()
        stats = agent.get_stats()
        print(f"✅ Agent created successfully")
        print(f"   Model: {stats['model_info']['name']}")
        print(f"   Server healthy: {stats['model_info']['server_healthy']}")
        print(f"   Tools available: {stats['tools_available']}")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")

    # Test 2: Simple chat
    print("\n2. Testing simple chat...")
    try:
        response = agent.chat("What is Python?")
        print(f"✅ Chat response received: {response[:100]}...")
    except Exception as e:
        print(f"❌ Chat failed: {e}")

    # Test 3: Memory system
    print("\n3. Testing memory system...")
    try:
        memory_id = agent.assistant.memory_manager.add_memory(
            content="This is a test memory entry about Python programming",
            session_id=agent.assistant.session_id,
            memory_type="test",
            importance=0.8
        )
        print(f"✅ Memory added: {memory_id}")

        memories = agent.assistant.memory_manager.retrieve_memory(
            query="Python programming",
            session_id=agent.assistant.session_id,
            limit=5
        )
        print(f"✅ Memory retrieved: {len(memories)} results")
    except Exception as e:
        print(f"❌ Memory system failed: {e}")

    # Test 4: Workflow execution
    print("\n4. Testing workflow execution...")
    try:
        async def test_workflow():
            result = await agent.execute_task(
                "Analyze this simple task",
                task_type="analysis"
            )
            return result

        result = asyncio.run(test_workflow())
        print(f"✅ Workflow executed: {result.get('success', False)}")
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")

    # Cleanup
    try:
        agent.cleanup()
        print("\n✅ Cleanup completed")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

    print("\n🎉 Manual tests completed!")


if __name__ == "__main__":
    # Run manual tests if called directly
    run_manual_tests()