#!/usr/bin/env python3
"""
Test script for the web interface of docker_test_agent
Tests the API endpoints to ensure the agent is working correctly
"""

import requests
import json
import time
import sys

def test_health_endpoint(base_url="http://localhost:8001"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Agent: {data.get('agent_name')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint(base_url="http://localhost:8001"):
    """Test the root endpoint"""
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working")
            print(f"   Message: {data.get('message')}")
            print(f"   Agent: {data.get('agent')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_query_endpoint(base_url="http://localhost:8001"):
    """Test the query endpoint"""
    try:
        test_query = "Hello! Can you tell me about Docker containers?"
        payload = {
            "query": test_query,
            "max_tokens": 100
        }

        response = requests.post(
            f"{base_url}/query",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ Query endpoint working")
            print(f"   Query: {test_query}")
            print(f"   Response: {data.get('response')[:100]}...")
            print(f"   Agent: {data.get('agent_name')}")
            return True
        else:
            print(f"❌ Query endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
        return False

def test_ollama_connection(base_url="http://localhost:11434"):
    """Test if Ollama is accessible"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Ollama connection working ({len(models)} models available)")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.get('name')}")
            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more")
            return True
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Docker Test Agent Web Interface")
    print("=" * 50)

    # Wait a moment for services to start
    print("Waiting for services to start...")
    time.sleep(5)

    base_url = "http://localhost:8001"
    ollama_url = "http://localhost:11434"

    tests = [
        ("Web Interface Health", lambda: test_health_endpoint(base_url)),
        ("Web Interface Root", lambda: test_root_endpoint(base_url)),
        ("Ollama Connection", lambda: test_ollama_connection(ollama_url)),
        ("Agent Query", lambda: test_query_endpoint(base_url)),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The agent is working correctly.")
        print("\nYou can now interact with the agent at:")
        print(f"  - Web interface: {base_url}")
        print(f"  - Health check: {base_url}/health")
        print(f"  - API queries: {base_url}/query")
        print(f"  - Ollama API: {ollama_url}")
        return 0
    else:
        print("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
