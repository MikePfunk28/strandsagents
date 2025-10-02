#!/usr/bin/env python3
"""
Simple test script for swarm system components
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database():
    """Test database functionality"""
    try:
        from swarm_system.utils.database_manager import db_manager
        print("✅ Database manager imported successfully")

        # Test basic operations
        knowledge_id = db_manager.store_knowledge(
            topic="test",
            content="Test knowledge entry",
            source="test_script",
            confidence=0.8
        )
        print(f"✅ Knowledge stored with ID: {knowledge_id}")

        results = db_manager.search_knowledge("test")
        print(f"✅ Knowledge search returned {len(results)} results")

        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_assistants():
    """Test assistant functionality"""
    try:
        from swarm_system.assistants.registry import global_registry
        from swarm_system.assistants.core.text_processor import TextProcessorAssistant
        from swarm_system.assistants.core.calculator_assistant import CalculatorAssistant
        from swarm_system.assistants.base_assistant import AssistantConfig
        print("✅ Assistant components imported successfully")

        # Test registry
        global_registry.register("text_processor", TextProcessorAssistant)
        print("✅ Text processor registered")

        global_registry.register("calculator", CalculatorAssistant)
        print("✅ Calculator registered")

        types = global_registry.list_available_types()
        print(f"✅ Available assistant types: {types}")

        return True
    except Exception as e:
        print(f"❌ Assistant test failed: {e}")
        return False

def test_tools():
    """Test tool functionality"""
    try:
        from swarm_system.utils.tools import (
            create_dynamic_tool, query_knowledge_base,
            store_learning, get_swarm_status
        )
        print("✅ Tool functions imported successfully")

        # Test knowledge base query
        results = query_knowledge_base("test", limit=5)
        print(f"✅ Knowledge base query working: {len(results)} chars")

        # Test learning storage
        result = store_learning("test_topic", "Test learning content")
        print(f"✅ Learning storage working: {result[:50]}...")

        # Test status
        status = get_swarm_status()
        print(f"✅ Status check working: {len(status)} chars")

        return True
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Swarm System Components")
    print("=" * 50)

    tests = [
        ("Database System", test_database),
        ("Assistant System", test_assistants),
        ("Tool System", test_tools),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(success for _, success in results)
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🚀 The swarm system is ready to use!")
        print("You can now run:")
        print("  python research_assistant.py")
        print("  python -m swarm_system.swarm_demo")
    else:
        print("\n🔧 Some components need fixing before full operation.")

if __name__ == "__main__":
    main()
