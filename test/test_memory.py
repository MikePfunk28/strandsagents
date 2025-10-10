#!/usr/bin/env python3
"""
Simple test to verify knowledgebase.txt creation works
"""

import os
import sys
from datetime import datetime

# Add current directory to path so we can import agent.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_knowledge_base_creation():
    """Test if knowledgebase.txt gets created properly"""
    print("🧪 Testing knowledgebase.txt creation...")

    # Check if file exists
    if os.path.exists("knowledgebase.txt"):
        print(" knowledgebase.txt already exists")
        with open("knowledgebase.txt", "r", encoding="utf-8") as f:
            content = f.read()
            print(f"📄 Current content length: {len(content)} characters")
    else:
        print("❌ knowledgebase.txt does not exist")

    # Test the function from agent.py
    try:
        from agent import knowledge_base_memory
        print("\n🔧 Testing knowledge_base_memory() function...")

        result = knowledge_base_memory()
        print(f" Function returned content with {len(result)} characters")

        # Check if file was created
        if os.path.exists("knowledgebase.txt"):
            print(" knowledgebase.txt was created successfully!")
            with open("knowledgebase.txt", "r", encoding="utf-8") as f:
                final_content = f.read()
                print(
                    f"📄 Final content length: {len(final_content)} characters")
                print("📄 First 200 characters:")
                print(final_content[:200])
        else:
            print("❌ knowledgebase.txt was NOT created")

    except Exception as e:
        print(f"❌ Error testing function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_knowledge_base_creation()
