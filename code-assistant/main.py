"""Main entry point for the adversarial coding system."""

import asyncio
import logging
from typing import Optional

from adversarial_agents import (
    AdversarialCodingCoordinator,
    CodeGenerationRequest,
    LanguageType,
    ModelConfiguration
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_adversarial_coding():
    """Demonstrate the adversarial coding system."""
    print("🤖 Adversarial Coding System Demo")
    print("=" * 50)

    # Initialize coordinator
    coordinator = AdversarialCodingCoordinator(
        model_name="llama3.2:3b",  # Main coordinator model
        host="localhost:11434"
    )

    # Show available model configurations
    model_config = ModelConfiguration()
    print("\n📊 Available Model Strategies:")
    for strategy in ["speed", "balanced", "quality"]:
        config = model_config.get_multi_model_config(strategy)
        print(f"\n{strategy.upper()} Strategy:")
        for role, model in config.items():
            print(f"  {role}: {model}")

    # Initialize agents with balanced strategy
    print("\n🚀 Initializing agents with balanced strategy...")
    await coordinator.initialize_agents(strategy="balanced")

    # Create code generation request
    request = CodeGenerationRequest(
        requirements="Create a Python function that calculates the factorial of a number with proper error handling",
        language=LanguageType.PYTHON,
        context="This function will be used in a math library",
        constraints=["Must handle negative numbers", "Must include docstring", "Must be efficient"]
    )

    print(f"\n📝 Generating code for: {request.requirements}")
    print(f"Language: {request.language.value}")

    # Generate code using adversarial process
    result = await coordinator.generate_code_adversarially(request, max_iterations=2)

    print(f"\n✅ Code Generation Complete!")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Score: {result['final_score']:.1f}/10")

    print(f"\n📄 Generated Code:")
    print("-" * 40)
    print(result['final_code'])
    print("-" * 40)

    print(f"\n📊 Review Summary:")
    for i, review in enumerate(result['review_history']):
        print(f"Review {i+1}: Score {review.score:.1f}/10, Approved: {review.approved}")
        if review.issues:
            print(f"  Issues: {review.issues[:2]}")  # Show first 2 issues

def select_models_interactive():
    """Interactive model selection."""
    print("\n🎛️  Model Selection")
    print("Available models for different roles:")

    model_config = ModelConfiguration()

    print("\nAvailable models:")
    for model, info in model_config.available_models.items():
        print(f"  {model} - {info['size']} ({info['speed']} speed)")

    print("\nPre-configured strategies:")
    print("1. Speed (fast models, quick feedback)")
    print("2. Balanced (mix of quality and speed)")
    print("3. Quality (larger models, better results)")
    print("4. Custom (select individual models)")

    choice = input("\nSelect strategy (1-4): ").strip()

    if choice == "1":
        return "speed"
    elif choice == "2":
        return "balanced"
    elif choice == "3":
        return "quality"
    elif choice == "4":
        return custom_model_selection(model_config)
    else:
        print("Invalid choice, using balanced strategy")
        return "balanced"

def custom_model_selection(model_config: ModelConfiguration) -> str:
    """Allow custom model selection for each agent."""
    print("\n🔧 Custom Model Configuration")
    print("Select model for each agent:")

    agents = ["generator", "discriminator", "optimizer", "security", "tester", "reviewer"]
    custom_config = {}

    for agent in agents:
        print(f"\n{agent.capitalize()} Agent:")
        for i, (model, info) in enumerate(model_config.available_models.items(), 1):
            print(f"  {i}. {model} - {info['size']} ({info['speed']})")

        while True:
            try:
                choice = int(input(f"Select model for {agent} (1-{len(model_config.available_models)}): "))
                if 1 <= choice <= len(model_config.available_models):
                    selected_model = list(model_config.available_models.keys())[choice - 1]
                    custom_config[agent] = selected_model
                    print(f"✓ Selected {selected_model} for {agent}")
                    break
                else:
                    print("Invalid choice, please try again")
            except ValueError:
                print("Please enter a number")

    # Save custom configuration (simplified - in real app would persist this)
    print("\n📋 Custom Configuration:")
    for agent, model in custom_config.items():
        print(f"  {agent}: {model}")

    return "balanced"  # For now, return balanced since we don't implement custom config storage

async def main():
    """Main function."""
    print("🎯 Adversarial Coding System")
    print("Building code through collaborative AI agents")
    print()

    try:
        # Check if user wants interactive mode
        mode = input("Select mode:\n1. Demo (automatic)\n2. Interactive\nChoice (1-2): ").strip()

        if mode == "2":
            strategy = select_models_interactive()

            # Get requirements from user
            requirements = input("\nEnter code requirements: ").strip()
            if not requirements:
                requirements = "Create a Python function that calculates the factorial of a number"

            # Select language
            print("\nSupported languages:")
            for i, lang in enumerate(LanguageType, 1):
                print(f"  {i}. {lang.value}")

            lang_choice = input("Select language (1-7): ").strip()
            try:
                language = list(LanguageType)[int(lang_choice) - 1]
            except (ValueError, IndexError):
                language = LanguageType.PYTHON
                print("Invalid choice, using Python")

            # Create custom request
            request = CodeGenerationRequest(
                requirements=requirements,
                language=language
            )

            coordinator = AdversarialCodingCoordinator()
            await coordinator.initialize_agents(strategy=strategy)

            result = await coordinator.generate_code_adversarially(request)

            print(f"\n✅ Code Generation Complete!")
            print(result['final_code'])

        else:
            # Run demo
            await demo_adversarial_coding()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Required models are pulled (ollama pull llama3.2:3b)")
        print("3. All dependencies are installed")

if __name__ == "__main__":
    asyncio.run(main())