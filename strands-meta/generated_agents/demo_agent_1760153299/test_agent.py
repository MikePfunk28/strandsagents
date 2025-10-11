#!/usr/bin/env python3
"""Test script for demo_agent agent"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    exec(open("demo_agent.py").read())
    print("Testing demo_agent agent...")
    result = demo_agent("Hello from test script!")
    print(f"Result: {result}")
    print("SUCCESS: Agent test successful!")
except Exception as e:
    print(f"ERROR: Agent test failed: {e}")
    import traceback
    traceback.print_exc()
