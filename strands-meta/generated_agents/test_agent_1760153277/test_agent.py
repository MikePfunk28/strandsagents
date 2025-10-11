#!/usr/bin/env python3
"""Test script for test_agent agent"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from test_agent import test_agent
    print("Testing test_agent agent...")
    result = test_agent("Hello from test script!")
    print(f"Result: {result}")
    print("SUCCESS: Agent test successful!")
except Exception as e:
    print(f"ERROR: Agent test failed: {e}")
    import traceback
    traceback.print_exc()
