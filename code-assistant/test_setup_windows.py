"""Windows-compatible test script for the adversarial coding system."""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import strands
        print("PASS - strands imported successfully")
    except ImportError as e:
        print(f"FAIL - strands import failed: {e}")
        return False

    try:
        import strands_tools
        print("PASS - strands_tools imported successfully")
    except ImportError as e:
        print(f"FAIL - strands_tools import failed: {e}")
        return False

    try:
        import mcp
        print("PASS - mcp imported successfully")
    except ImportError as e:
        print(f"FAIL - mcp import failed: {e}")
        return False

    try:
        from strands import Agent, tool
        from strands.models import OllamaModel
        print("PASS - StrandsAgents core classes imported successfully")
    except ImportError as e:
        print(f"FAIL - StrandsAgents core import failed: {e}")
        return False

    return True

def main():
    """Run basic import test."""
    print("Adversarial Coding System - Quick Import Test")
    print("=" * 60)

    if test_imports():
        print("\nSUCCESS: All required packages are installed!")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull models: ollama pull llama3.2:3b")
        print("3. Run demo: python main.py")
    else:
        print("\nWARNING: Some packages are missing.")
        print("Run the setup script: .\\setup_windows.ps1")

if __name__ == "__main__":
    main()