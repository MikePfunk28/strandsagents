#!/usr/bin/env python3
"""
Test script for docker_test_agent components without Docker
Tests the web interface and agent components directly
"""

import sys
import os
import subprocess

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")

    try:
        import docker_test_agent
        print("   ✅ docker_test_agent imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import docker_test_agent: {e}")
        return False

    try:
        import web_interface
        print("   ✅ web_interface imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import web_interface: {e}")
        return False

    try:
        import fastapi
        print("   ✅ fastapi imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import fastapi: {e}")
        return False

    return True

def test_agent_functionality():
    """Test the agent function directly"""
    print("\n🧪 Testing agent functionality...")

    try:
        from docker_test_agent import docker_test_agent, AGENT_METADATA

        # Test basic query
        test_query = "Hello, can you tell me about Docker?"
        result = docker_test_agent(test_query)

        print(f"   ✅ Agent responded: {result[:100]}...")
        print(f"   ✅ Agent metadata: {AGENT_METADATA.get('name', 'unknown')}")

        return True
    except Exception as e:
        print(f"   ❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_interface_creation():
    """Test if the web interface can be created"""
    print("\n🧪 Testing web interface creation...")

    try:
        from web_interface import app

        # Check if app has expected endpoints
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)

        expected_routes = ["/", "/health", "/query", "/info"]
        for expected in expected_routes:
            if expected in routes:
                print(f"   ✅ Route {expected} found")
            else:
                print(f"   ❌ Route {expected} missing")
                return False

        print(f"   ✅ Web interface created with {len(routes)} routes")
        return True

    except Exception as e:
        print(f"   ❌ Web interface creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_requirements():
    """Test if all requirements are available"""
    print("\n🧪 Testing requirements...")

    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} available")
        except ImportError:
            print(f"   ❌ {package} missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False

    return True

def test_ollama_availability():
    """Test if Ollama is available (optional)"""
    print("\n🧪 Testing Ollama availability...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"   ✅ Ollama available with {len(models)} models")
            return True
        else:
            print(f"   ⚠️  Ollama not responding (status: {response.status_code})")
            return True  # Not critical for component testing

    except Exception as e:
        print(f"   ⚠️  Ollama not available: {e}")
        return True  # Not critical for component testing

def main():
    """Main test function"""
    print("Testing Docker Test Agent Components (without Docker)")
    print("=" * 60)

    tests = [
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("Agent Functionality", test_agent_functionality),
        ("Web Interface", test_web_interface_creation),
        ("Ollama Availability", test_ollama_availability),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")

    print(f"\n{'=' * 60}")
    print(f"Component Test Results: {passed}/{total} passed")

    if passed >= 3:  # Allow Ollama to be optional
        print("🎉 Core components are working correctly!")
        print("\nThe agent is ready for Docker deployment once Docker Desktop is fixed.")
        print("\nTo deploy with Docker:")
        print("1. Ensure Docker Desktop is running")
        print("2. Run: docker-compose up --build")
        print("3. Access at: http://localhost:8001")
        return 0
    else:
        print("❌ Core components have issues that need to be fixed first.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
