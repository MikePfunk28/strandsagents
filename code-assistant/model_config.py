"""Model configuration for the adversarial coding system with Gemma models."""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    size: str
    memory_usage: str
    speed: str
    quality: str
    use_case: str

class GemmaModelManager:
    """Manager for Gemma model configurations and assignments."""

    def __init__(self):
        self.available_models = {
            # Gemma 2 models (latest)
            "gemma2:2b-instruct": ModelConfig(
                name="gemma2:2b-instruct",
                size="2B",
                memory_usage="~3GB",
                speed="medium",
                quality="high",
                use_case="Primary generation and complex analysis"
            ),
            "gemma2:2b-instruct-q4_0": ModelConfig(
                name="gemma2:2b-instruct-q4_0",
                size="2B quantized",
                memory_usage="~1.5GB",
                speed="fast",
                quality="good",
                use_case="Efficient processing and validation"
            ),
            "gemma2:2b-instruct-q8_0": ModelConfig(
                name="gemma2:2b-instruct-q8_0",
                size="2B quantized",
                memory_usage="~2.2GB",
                speed="medium-fast",
                quality="very good",
                use_case="Balanced performance and quality"
            ),

            # Gemma 1 models
            "gemma:2b-instruct": ModelConfig(
                name="gemma:2b-instruct",
                size="2B",
                memory_usage="~3GB",
                speed="medium",
                quality="good",
                use_case="General purpose and fallback"
            ),
            "gemma:2b-instruct-q4_0": ModelConfig(
                name="gemma:2b-instruct-q4_0",
                size="2B quantized",
                memory_usage="~1.5GB",
                speed="fast",
                quality="decent",
                use_case="Fast validation and checking"
            ),

            # Smaller models for efficiency
            "gemma:7b-instruct-q4_0": ModelConfig(
                name="gemma:7b-instruct-q4_0",
                size="7B quantized",
                memory_usage="~4GB",
                speed="slow",
                quality="excellent",
                use_case="High-quality analysis when needed"
            ),

            # Hypothetical 270M model for GAN-coder
            "gemma:270m-instruct": ModelConfig(
                name="gemma:270m-instruct",
                size="270M",
                memory_usage="~500MB",
                speed="very fast",
                quality="basic",
                use_case="Rapid iteration and lightweight processing"
            )
        }

    def get_gan_coder_config(self) -> Dict[str, str]:
        """Get optimized model configuration for GAN-style adversarial coding."""
        return {
            # Generator: Needs creativity and code generation capability
            "generator": "gemma2:2b-instruct",

            # Discriminator: Fast analysis and issue detection
            "discriminator": "gemma:270m-instruct",  # Ultra-fast for rapid iteration

            # Optimizer: Performance analysis and improvement
            "optimizer": "gemma2:2b-instruct-q4_0",

            # Security: Thorough security analysis
            "security": "gemma2:2b-instruct",

            # Tester: Test case generation
            "tester": "gemma:2b-instruct-q4_0",

            # Code Reviewer: General code quality
            "reviewer": "gemma2:2b-instruct-q8_0"
        }

    def get_efficiency_config(self) -> Dict[str, str]:
        """Get memory-efficient configuration for resource-constrained environments."""
        return {
            "generator": "gemma2:2b-instruct-q4_0",
            "discriminator": "gemma:270m-instruct",
            "optimizer": "gemma:270m-instruct",
            "security": "gemma2:2b-instruct-q4_0",
            "tester": "gemma:270m-instruct",
            "reviewer": "gemma:2b-instruct-q4_0"
        }

    def get_quality_config(self) -> Dict[str, str]:
        """Get high-quality configuration when resources are available."""
        return {
            "generator": "gemma:7b-instruct-q4_0",
            "discriminator": "gemma2:2b-instruct",
            "optimizer": "gemma2:2b-instruct-q8_0",
            "security": "gemma:7b-instruct-q4_0",
            "tester": "gemma2:2b-instruct",
            "reviewer": "gemma:7b-instruct-q4_0"
        }

    def get_balanced_config(self) -> Dict[str, str]:
        """Get balanced configuration for general use."""
        return {
            "generator": "gemma2:2b-instruct",
            "discriminator": "gemma2:2b-instruct-q4_0",
            "optimizer": "gemma2:2b-instruct-q4_0",
            "security": "gemma2:2b-instruct-q8_0",
            "tester": "gemma:2b-instruct-q4_0",
            "reviewer": "gemma2:2b-instruct-q8_0"
        }

    def get_config_by_strategy(self, strategy: str = "balanced") -> Dict[str, str]:
        """Get model configuration by strategy."""
        strategies = {
            "gan": self.get_gan_coder_config(),
            "efficiency": self.get_efficiency_config(),
            "quality": self.get_quality_config(),
            "balanced": self.get_balanced_config()
        }
        return strategies.get(strategy, self.get_balanced_config())

    def estimate_memory_usage(self, config: Dict[str, str]) -> float:
        """Estimate total memory usage for a configuration."""
        total_memory = 0.0
        memory_map = {
            "gemma2:2b-instruct": 3.0,
            "gemma2:2b-instruct-q4_0": 1.5,
            "gemma2:2b-instruct-q8_0": 2.2,
            "gemma:2b-instruct": 3.0,
            "gemma:2b-instruct-q4_0": 1.5,
            "gemma:7b-instruct-q4_0": 4.0,
            "gemma:270m-instruct": 0.5
        }

        unique_models = set(config.values())
        for model in unique_models:
            total_memory += memory_map.get(model, 2.0)  # Default 2GB if unknown

        return total_memory

    def print_configuration_comparison(self):
        """Print comparison of different configurations."""
        strategies = ["gan", "efficiency", "quality", "balanced"]

        print("Model Configuration Comparison")
        print("=" * 60)

        for strategy in strategies:
            config = self.get_config_by_strategy(strategy)
            memory = self.estimate_memory_usage(config)

            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Estimated Memory: ~{memory:.1f}GB")
            for role, model in config.items():
                model_info = self.available_models.get(model)
                if model_info:
                    print(f"  {role.capitalize()}: {model} ({model_info.size}, {model_info.speed})")
                else:
                    print(f"  {role.capitalize()}: {model}")

    def get_installation_commands(self) -> List[str]:
        """Get Ollama installation commands for all required models."""
        all_models = set()
        for strategy in ["gan", "efficiency", "quality", "balanced"]:
            config = self.get_config_by_strategy(strategy)
            all_models.update(config.values())

        commands = []
        for model in sorted(all_models):
            if model != "gemma:270m-instruct":  # Skip hypothetical model
                commands.append(f"ollama pull {model}")

        return commands

def main():
    """Demonstrate model configuration options."""
    manager = GemmaModelManager()

    print("Gemma Model Manager for Adversarial Coding System")
    print("=" * 60)

    # Show configuration comparison
    manager.print_configuration_comparison()

    # Show installation commands
    print("\n\nInstallation Commands:")
    print("=" * 30)
    commands = manager.get_installation_commands()
    for cmd in commands:
        print(f"  {cmd}")

    # Show GAN-specific configuration details
    print("\n\nGAN-Coder Configuration Details:")
    print("=" * 40)
    gan_config = manager.get_gan_coder_config()
    print("This configuration uses the 270M model for the discriminator")
    print("to enable rapid iteration and fast adversarial feedback.")
    print(f"Total estimated memory: ~{manager.estimate_memory_usage(gan_config):.1f}GB")

    print("\nKey Benefits of 270M Discriminator:")
    print("- Ultra-fast feedback during code iteration")
    print("- Low memory footprint allows running multiple models")
    print("- Enables true GAN-like rapid improvement cycles")
    print("- Leaves more resources for the generator model")

if __name__ == "__main__":
    main()