#!/usr/bin/env python3
"""Test script for working_agent agent"""

import sys
import os
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("Testing working_agent in Docker environment...")

try:
    # Load the agent code
    with open("working_agent.py", encoding='utf-8') as f:
        agent_code = f.read()

    # Execute the agent code to define the function in globals
    exec(agent_code, globals())

    print("Testing working_agent agent...")
    print("Waiting for Ollama to be ready...")

    # Wait for Ollama to be available
    max_retries = 30
    for i in range(max_retries):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Ollama ready after {i+1} seconds!")
                break
        except:
            print(f"Waiting for Ollama... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("WARNING: Ollama not available, testing without it...")

    # Test the agent
    result = working_agent("Hello from Docker test script!")
    print(f"Result: {result}")
    print("SUCCESS: Agent test successful!")

except Exception as e:
    print(f"ERROR: Agent test failed: {e}")
    import traceback
    traceback.print_exc()
