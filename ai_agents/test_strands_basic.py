#!/usr/bin/env python3
"""
Basic test to verify Strands Agents SDK functionality.
Tests the corrected import patterns and basic agent creation.
"""

import asyncio
import sys
from typing import Dict, Any

try:
    from strands import Agent
    from strands.tools.executors import SequentialToolExecutor, ConcurrentToolExecutor
    print("SUCCESS: Strands imports successful")
except ImportError as e:
    print(f"FAILED: Import failed: {e}")
    sys.exit(1)

async def test_basic_agent():
    """Test basic agent creation and simple interaction."""
    print("\nTesting Basic Agent Creation")
    print("=" * 40)

    try:
        # Create agent with default model (should work)
        agent = Agent(
            system_prompt="You are a helpful assistant. Respond briefly and clearly."
        )
        print("SUCCESS: Basic agent created successfully")

        # Test simple interaction
        response = await agent.invoke_async("What is 2 + 2?")
        print(f"SUCCESS: Agent response: {response}")

    except Exception as e:
        print(f"FAILED: Basic agent test failed: {e}")
        return False

    return True

async def test_agent_with_string_model():
    """Test agent with string model name."""
    print("\nTesting Agent with String Model")
    print("=" * 40)

    try:
        # Create agent with string model name
        agent = Agent(
            model="gpt-3.5-turbo",  # This should work with default provider
            system_prompt="You are a test assistant."
        )
        print("SUCCESS: Agent with string model created")

        response = await agent.invoke_async("Say hello")
        print(f"SUCCESS: Agent response: {response}")

    except Exception as e:
        print(f"FAILED: String model test failed: {e}")
        return False

    return True

async def test_tool_executors():
    """Test tool executor imports."""
    print("\nTesting Tool Executors")
    print("=" * 40)

    try:
        # Test executor creation
        seq_executor = SequentialToolExecutor()
        conc_executor = ConcurrentToolExecutor()
        print("SUCCESS: Tool executors created successfully")
        print(f"Sequential executor: {type(seq_executor)}")
        print(f"Concurrent executor: {type(conc_executor)}")

    except Exception as e:
        print(f"FAILED: Tool executor test failed: {e}")
        return False

    return True

def test_imports():
    """Test all necessary imports."""
    print("\nTesting All Imports")
    print("=" * 40)

    try:
        from strands import Agent
        print("SUCCESS: Agent import")

        from strands.tools.executors import SequentialToolExecutor, ConcurrentToolExecutor
        print("SUCCESS: Tool executors import")

        from strands.models import Model
        print("SUCCESS: Base Model import")

        # Check available models
        from strands.models import BedrockModel
        print("SUCCESS: BedrockModel import")

        return True

    except Exception as e:
        print(f"FAILED: Import test failed: {e}")
        return False

async def test_specialized_agent():
    """Test creating a specialized agent similar to our swarm patterns."""
    print("\nTesting Specialized Agent")
    print("=" * 40)

    try:
        # Create a research-like agent
        research_agent = Agent(
            system_prompt="""You are a research assistant. You help with:
- Information gathering and analysis
- Fact checking and verification
- Document search and synthesis
- Data extraction and insights

Provide accurate, well-sourced information and be transparent about limitations."""
        )
        print("SUCCESS: Research agent created")

        # Test research task
        response = await research_agent.invoke_async(
            "What are the main benefits of renewable energy?"
        )
        response_text = str(response)
        print(f"SUCCESS: Research response received: {len(response_text)} characters")
        print(f"Preview: {response_text[:150]}...")

        return True

    except Exception as e:
        print(f"FAILED: Specialized agent test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Strands Agents SDK - Basic Functionality Test")
    print("=" * 50)

    results = []

    # Test imports first
    results.append(("Imports", test_imports()))

    # Test basic functionality
    results.append(("Basic Agent", await test_basic_agent()))
    results.append(("String Model", await test_agent_with_string_model()))
    results.append(("Tool Executors", await test_tool_executors()))
    results.append(("Specialized Agent", await test_specialized_agent()))

    # Summary
    print("\nTest Results Summary")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("All tests passed! Strands SDK is working correctly.")
        return True
    else:
        print("Some tests failed. Check configuration.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)