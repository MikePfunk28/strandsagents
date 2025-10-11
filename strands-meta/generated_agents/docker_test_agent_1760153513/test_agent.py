#!/usr/bin/env python3
"""Test script for docker_test_agent agent"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Load and execute agent code
    with open("docker_test_agent.py", encoding='utf-8') as f:
        agent_code = f.read()
    exec(agent_code, globals())

    print("Testing docker_test_agent agent...")
    result = docker_test_agent("Hello from test script!")
    print(f"Result: {result}")
    print("SUCCESS: Agent test successful!")
except Exception as e:
    print(f"ERROR: Agent test failed: {e}")
    import traceback
    traceback.print_exc()
