#!/usr/bin/env python3
"""
Native Windows deployment for docker_test_agent
Runs the agent without Docker for testing and development
"""

import uvicorn
import os
import sys
from web_interface import app

def main():
    """Run the agent natively on Windows"""
    print("🚀 Starting Docker Test Agent (Native Windows Mode)")
    print("=" * 55)
    print("Web interface will be available at: http://localhost:8001")
    print("Press Ctrl+C to stop")
    print()

    # Set environment variables for local development
    os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

    try:
        uvicorn.run(
            "web_interface:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
