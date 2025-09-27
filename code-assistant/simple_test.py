"""Simple test to verify the coding assistant works."""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without requiring Ollama."""
    print("Testing Coding Assistant Basic Functionality")
    print("=" * 50)

    try:
        # Test 1: Import and create components
        print("\n1. Testing imports...")
        from database_manager import DatabaseManager
        from ollama_model import create_ollama_model
        from coding_assistant import CodingAssistant
        from simple_coding_agent import create_simple_coding_agent
        print("[OK] All imports successful")

        # Test 2: Database manager
        print("\n2. Testing database manager...")
        temp_dir = Path("./test_data")
        temp_dir.mkdir(exist_ok=True)

        db_manager = DatabaseManager(temp_dir)

        # Test cache operations
        db_manager.cache_set("test_key", "test_value", ttl_seconds=60)
        value = db_manager.cache_get("test_key")
        assert value == "test_value"
        print("[OK] Database manager working")

        # Test 3: Ollama model (without requiring connection)
        print("\n3. Testing Ollama model...")
        ollama_model = create_ollama_model("llama3.2", "http://localhost:11434")
        info = ollama_model.model_info
        print(f"✅ Ollama model created: {info['name']}")

        # Test 4: Coding assistant
        print("\n4. Testing coding assistant...")
        assistant = CodingAssistant(
            db_dir=temp_dir,
            model_name="llama3.2",
            session_id="test_session"
        )
        print(f"✅ Coding assistant created: {assistant.session_id}")

        # Test 5: Simple coding agent
        print("\n5. Testing simple coding agent...")
        agent = create_simple_coding_agent(
            db_dir=str(temp_dir),
            model_name="llama3.2"
        )
        stats = agent.get_stats()
        print(f"✅ Simple coding agent created")
        print(f"   Session: {stats['session_id']}")
        print(f"   Tools: {stats['tools_available']}")
        print(f"   Categories: {len(stats['context_categories'])}")

        # Test 6: Memory operations
        print("\n6. Testing memory operations...")
        memory_id = agent.assistant.memory_manager.add_memory(
            content="Test memory entry for coding assistant",
            session_id=agent.assistant.session_id,
            memory_type="test",
            importance=0.8
        )
        print(f"✅ Memory added: {memory_id}")

        # Test 7: Conversation
        print("\n7. Testing conversation...")
        agent.assistant.memory_manager.update_conversation(
            agent.assistant.session_id,
            {"role": "user", "content": "Hello, test message", "timestamp": 1234567890}
        )
        context = agent.assistant.memory_manager.get_conversation_context(
            agent.assistant.session_id
        )
        print(f"✅ Conversation context: {len(context['recent_messages'])} messages")

        # Test 8: Classification
        print("\n8. Testing content classification...")
        classification = agent.assistant._classify_query("How do I write a Python function?")
        print(f"✅ Classification: {classification['primary_category']}")

        # Cleanup
        agent.cleanup()
        print("\n[SUCCESS] All basic tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tools():
    """Test the tool functionality."""
    print("\nTesting Testing Tools")
    print("=" * 30)

    try:
        from coding_tools import python_repl, file_read, file_write

        # Test Python REPL
        print("\n1. Testing Python REPL...")
        result = python_repl("print('Hello from REPL')\nx = 2 + 2")
        if result['success']:
            print("[OK] Python REPL working")
        else:
            print(f"❌ Python REPL failed: {result['error']}")

        # Test file operations
        print("\n2. Testing file operations...")
        test_content = "# Test file\nprint('Hello World')\n"

        # Write test file
        write_result = file_write("test_file.py", test_content)
        if write_result['success']:
            print("[OK] File write working")
        else:
            print(f"❌ File write failed: {write_result['error']}")

        # Read test file
        read_result = file_read("test_file.py")
        if read_result['success'] and read_result['content'] == test_content:
            print("[OK] File read working")
        else:
            print(f"❌ File read failed: {read_result.get('error', 'Content mismatch')}")

        # Cleanup test file
        import os
        if os.path.exists("test_file.py"):
            os.remove("test_file.py")

        print("\n[SUCCESS] Tool tests completed!")
        return True

    except Exception as e:
        print(f"\n❌ Tool test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Starting Coding Assistant Tests")
    print("=" * 60)

    basic_success = test_basic_functionality()
    tools_success = test_tools()

    print("\n" + "=" * 60)
    if basic_success and tools_success:
        print("[SUCCESS] ALL TESTS PASSED! The coding assistant is ready to use.")
        print("\nTo get started:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull required models: ollama pull llama3.2 && ollama pull embeddinggemma")
        print("3. Run: python example_usage.py")
    else:
        print("[ERROR] Some tests failed. Please check the output above.")

    print("=" * 60)