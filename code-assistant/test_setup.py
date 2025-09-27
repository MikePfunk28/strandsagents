"""Test script to verify the adversarial coding system setup."""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import strands
        print("✓ strands imported successfully")
    except ImportError as e:
        print(f"✗ strands import failed: {e}")
        return False

    try:
        import strands_tools
        print("✓ strands_tools imported successfully")
    except ImportError as e:
        print(f"✗ strands_tools import failed: {e}")
        return False

    try:
        import mcp
        print("✓ mcp imported successfully")
    except ImportError as e:
        print(f"✗ mcp import failed: {e}")
        return False

    try:
        from strands import Agent, tool
        from strands.models import OllamaModel
        print("✓ StrandsAgents core classes imported successfully")
    except ImportError as e:
        print(f"✗ StrandsAgents core import failed: {e}")
        return False

    return True

def test_local_modules():
    """Test that local modules can be imported."""
    print("\nTesting local modules...")

    try:
        from adversarial_agents import (
            AdversarialCodingCoordinator,
            ModelConfiguration,
            LanguageType
        )
        print("✓ Adversarial agents imported successfully")
    except ImportError as e:
        print(f"✗ Adversarial agents import failed: {e}")
        return False

    try:
        from prompts import CODE_GENERATOR_PROMPT
        print("✓ Prompts imported successfully")
    except ImportError as e:
        print(f"✗ Prompts import failed: {e}")
        return False

    try:
        from language_engine import LanguageEngine
        print("✓ Language engine imported successfully")
    except ImportError as e:
        print(f"✗ Language engine import failed: {e}")
        return False

    return True

def test_model_config():
    """Test model configuration."""
    print("\nTesting model configuration...")

    try:
        from adversarial_agents import ModelConfiguration

        config = ModelConfiguration()
        print(f"✓ Available models: {len(config.available_models)}")

        # Test different strategies
        for strategy in ["speed", "balanced", "quality"]:
            strategy_config = config.get_multi_model_config(strategy)
            print(f"✓ {strategy} strategy: {len(strategy_config)} agents configured")

        return True
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection (if available)."""
    print("\nTesting Ollama connection...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama connected, {len(models)} models available")

            # Check for required models
            model_names = [m["name"] for m in models]
            required = ["llama3.2", "gemma"]

            for req in required:
                found = any(req in name for name in model_names)
                if found:
                    print(f"✓ Found model matching '{req}'")
                else:
                    print(f"⚠ No model found matching '{req}' - consider: ollama pull {req}")

            return True
        else:
            print(f"⚠ Ollama server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Ollama connection failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def main():
    """Run all tests."""
    print("Adversarial Coding System - Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_local_modules,
        test_model_config,
        test_ollama_connection
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Summary:")

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull required models: ollama pull llama3.2:3b")
        print("3. Run the demo: python main.py")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install StrandsAgents: pip install strands-agents")
        print("3. Start Ollama: ollama serve")

if __name__ == "__main__":
    main()