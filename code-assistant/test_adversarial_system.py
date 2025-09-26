"""Test the adversarial coding system with real examples."""

import asyncio
from adversarial_coding_system import AdversarialCodingSystem, LanguageType
from database_manager import DatabaseManager
from ollama_model import OllamaModel

async def test_simple_python_function():
    """Test generating a simple Python function."""
    print("Testing Python Function Generation")
    print("=" * 50)

    # Initialize system
    db_manager = DatabaseManager()
    model = OllamaModel()
    system = AdversarialCodingSystem(db_manager, model)

    requirements = """
    Create a Python function that calculates the factorial of a number.
    Requirements:
    - Handle edge cases (0, 1, negative numbers)
    - Include proper error handling
    - Add docstring with examples
    - Use recursive approach
    """

    try:
        result = await system.generate_and_validate_code(
            requirements=requirements,
            language=LanguageType.PYTHON,
            max_iterations=2
        )

        print(f"Final code quality score: {result.final_score:.1f}/10")
        print(f"Iterations completed: {result.iterations}")
        print(f"Generator suggestions: {len(result.generator_suggestions)}")
        print(f"Discriminator issues: {len(result.discriminator_issues)}")

        print("\nFinal Generated Code:")
        print("-" * 30)
        print(result.final_code)

        if result.discriminator_issues:
            print("\nRemaining Issues:")
            for issue in result.discriminator_issues[-3:]:  # Last 3 issues
                print(f"- {issue}")

        return result

    except Exception as e:
        print(f"Error during generation: {e}")
        return None

async def test_javascript_class():
    """Test generating a JavaScript class."""
    print("\n\nTesting JavaScript Class Generation")
    print("=" * 50)

    db_manager = DatabaseManager()
    model = OllamaModel()
    system = AdversarialCodingSystem(db_manager, model)

    requirements = """
    Create a JavaScript class for a simple bank account.
    Requirements:
    - Constructor takes initial balance
    - Methods: deposit, withdraw, getBalance
    - Validate withdrawal amounts (no overdraft)
    - Throw errors for invalid operations
    - Include JSDoc comments
    """

    try:
        result = await system.generate_and_validate_code(
            requirements=requirements,
            language=LanguageType.JAVASCRIPT,
            max_iterations=2
        )

        print(f"Final code quality score: {result.final_score:.1f}/10")
        print(f"Iterations completed: {result.iterations}")

        print("\nFinal Generated Code:")
        print("-" * 30)
        print(result.final_code)

        return result

    except Exception as e:
        print(f"Error during generation: {e}")
        return None

async def test_rust_function():
    """Test generating a Rust function."""
    print("\n\nTesting Rust Function Generation")
    print("=" * 50)

    db_manager = DatabaseManager()
    model = OllamaModel()
    system = AdversarialCodingSystem(db_manager, model)

    requirements = """
    Create a Rust function that finds the largest element in a vector.
    Requirements:
    - Generic function that works with any comparable type
    - Return Option<T> to handle empty vectors
    - Include comprehensive documentation
    - Use idiomatic Rust patterns
    - Handle edge cases properly
    """

    try:
        result = await system.generate_and_validate_code(
            requirements=requirements,
            language=LanguageType.RUST,
            max_iterations=2
        )

        print(f"Final code quality score: {result.final_score:.1f}/10")
        print(f"Iterations completed: {result.iterations}")

        print("\nFinal Generated Code:")
        print("-" * 30)
        print(result.final_code)

        return result

    except Exception as e:
        print(f"Error during generation: {e}")
        return None

async def demonstrate_improvement_process():
    """Demonstrate the iterative improvement process."""
    print("\n\nDemonstrating Iterative Improvement")
    print("=" * 50)

    db_manager = DatabaseManager()
    model = OllamaModel()
    system = AdversarialCodingSystem(db_manager, model)

    # Intentionally vague requirements to trigger multiple iterations
    requirements = """
    Create a function that sorts a list of numbers.
    Make it efficient and handle edge cases.
    """

    try:
        result = await system.generate_and_validate_code(
            requirements=requirements,
            language=LanguageType.PYTHON,
            max_iterations=3
        )

        print(f"Improvement over {result.iterations} iterations:")
        print(f"Final score: {result.final_score:.1f}/10")

        print("\nIterative Improvements:")
        for i, suggestion in enumerate(result.generator_suggestions):
            print(f"Iteration {i+1}: {suggestion[:80]}...")

        return result

    except Exception as e:
        print(f"Error during demonstration: {e}")
        return None

async def main():
    """Run all adversarial coding tests."""
    print("Adversarial Coding System Test Suite")
    print("=" * 60)

    try:
        # Test different languages
        python_result = await test_simple_python_function()
        js_result = await test_javascript_class()
        rust_result = await test_rust_function()

        # Demonstrate improvement process
        improvement_result = await demonstrate_improvement_process()

        print("\n\n" + "=" * 60)
        print("Test Summary:")
        print(f"Python function: {'PASS' if python_result else 'FAIL'}")
        print(f"JavaScript class: {'PASS' if js_result else 'FAIL'}")
        print(f"Rust function: {'PASS' if rust_result else 'FAIL'}")
        print(f"Improvement demo: {'PASS' if improvement_result else 'FAIL'}")

        if all([python_result, js_result, rust_result, improvement_result]):
            print("\nAll tests completed successfully!")
            print("The adversarial coding system demonstrates:")
            print("- Multi-language code generation")
            print("- Iterative quality improvement")
            print("- GAN-like adversarial validation")
            print("- Comprehensive code analysis")
        else:
            print("\nSome tests failed - system needs refinement")

    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())