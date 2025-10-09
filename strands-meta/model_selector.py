"""
Model Selector for StrandsAgents @agent Package

Automatically selects the best available Ollama model based on:
- Model capabilities and performance
- User's available models
- Task requirements
- System resources
"""

import subprocess
import json
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("model_selector")

# Remove emoji characters that cause encoding issues on Windows
def safe_print(text):
    """Print text with emoji characters replaced for Windows compatibility"""
    emoji_replacements = {
        '🔍': '[SEARCH]',
        '🔧': '[TOOL]',
        '🎯': '[TARGET]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '💡': '[IDEA]',
        '🚀': '[LAUNCH]',
        '🔄': '[SYNC]',
        '📦': '[PACKAGE]',
        '🔍': '[FIND]'
    }

    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)

    print(text)

# Add parent directory to path for imports
try:
    # When run as module
    pass
except:
    # When run as script, add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    size: str
    family: str
    capabilities: List[str]
    performance_score: int
    recommended_for: List[str]

class ModelSelector:
    """Intelligent model selection for optimal performance"""

    # Model capability matrix - AWS Bedrock compatible models
    MODEL_CAPABILITIES = {
        "gemma3:270m": ModelInfo(
            name="gemma3:270m",
            size="270M",
            family="gemma",
            capabilities=["basic_reasoning", "simple_tasks", "light_analysis"],
            performance_score=45,
            recommended_for=["simple_tasks", "basic_qa", "light_analysis"]
        ),
        "gemma3:1b": ModelInfo(
            name="gemma3:1b",
            size="1B",
            family="gemma",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=65,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "llama3.2:1b": ModelInfo(
            name="llama3.2:1b",
            size="1B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=60,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "llama3.2:3b": ModelInfo(
            name="llama3.2:3b",
            size="3B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=70,
            recommended_for=["general_tasks", "coding", "analysis", "creative_tasks"]
        ),
        "llama3.2": ModelInfo(
            name="llama3.2",
            size="7B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis", "creative", "research"],
            performance_score=75,
            recommended_for=["general_tasks", "coding", "analysis", "research"]
        ),
        "codellama:7b": ModelInfo(
            name="codellama:7b",
            size="7B",
            family="codellama",
            capabilities=["coding", "code_completion", "debugging", "mathematics"],
            performance_score=80,
            recommended_for=["coding", "debugging", "code_review", "mathematics"]
        ),
        "deepseek-coder:6.7b": ModelInfo(
            name="deepseek-coder:6.7b",
            size="6.7B",
            family="deepseek",
            capabilities=["coding", "mathematics", "problem_solving", "analysis"],
            performance_score=82,
            recommended_for=["coding", "mathematics", "algorithms", "analysis"]
        )
    }

    def __init__(self):
        self.available_models = []
        self._detect_available_models()

    def _detect_available_models(self):
        """Detect which models are available on the system"""
        try:
            # Check if Ollama is running and get available models
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        if model_name in self.MODEL_CAPABILITIES:
                            self.available_models.append(model_name)
                            logger.info(f"🔍 Found available model: {model_name}")

                if not self.available_models:
                    logger.warning("No recognized models found in Ollama")
                    # Add some fallback models
                    self.available_models = ["llama3.2", "gemma3:1b"]
            else:
                logger.warning("Could not connect to Ollama, using fallback models")
                self.available_models = ["llama3.2", "gemma3:1b"]

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Ollama not available, using fallback models")
            self.available_models = ["llama3.2", "gemma3:1b"]

        logger.info(f"🔍 Available models: {self.available_models}")

    def select_best_model(self, task_type: str = "general",
                         require_code_execution: bool = False,
                         require_reasoning: bool = False) -> str:
        """
        Select the best model for a given task

        Args:
            task_type: Type of task (coding, research, analysis, creative, meta, general)
            require_code_execution: Whether code execution is needed
            require_reasoning: Whether advanced reasoning is needed

        Returns:
            Best model name for the task
        """

        # Filter models based on availability
        candidate_models = [
            model for model in self.available_models
            if model in self.MODEL_CAPABILITIES
        ]

        if not candidate_models:
            logger.warning("No suitable models found, using fallback")
            return "llama3.2"

        # Score models based on task requirements
        model_scores = []

        for model_name in candidate_models:
            model_info = self.MODEL_CAPABILITIES[model_name]
            score = model_info.performance_score

            # Adjust score based on task requirements
            if require_code_execution and "coding" in model_info.capabilities:
                score += 10
            if require_reasoning and "reasoning" in model_info.capabilities:
                score += 5

            # Task-specific adjustments
            if task_type == "coding" and "coding" in model_info.capabilities:
                score += 15
            elif task_type == "research" and "research" in model_info.capabilities:
                score += 10
            elif task_type == "creative" and "creative" in model_info.capabilities:
                score += 10
            elif task_type == "analysis" and "analysis" in model_info.capabilities:
                score += 10
            elif task_type == "meta" and "reasoning" in model_info.capabilities:
                score += 12  # Meta agents need good reasoning

            model_scores.append((model_name, score))

        # Select best model
        best_model = max(model_scores, key=lambda x: x[1])[0]

        logger.info(f"Selected model '{best_model}' for task type '{task_type}'")
        return best_model

    def get_model_recommendations(self) -> Dict[str, str]:
        """Get model recommendations for different task types"""
        return {
            "general": self.select_best_model("general"),
            "coding": self.select_best_model("coding", require_code_execution=True),
            "research": self.select_best_model("research", require_reasoning=True),
            "analysis": self.select_best_model("analysis", require_reasoning=True),
            "creative": self.select_best_model("creative"),
            "simple": self.select_best_model("general")
        }

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their capabilities"""
        models_info = []

        for model_name in self.available_models:
            if model_name in self.MODEL_CAPABILITIES:
                info = self.MODEL_CAPABILITIES[model_name]
                models_info.append({
                    "name": info.name,
                    "size": info.size,
                    "family": info.family,
                    "capabilities": info.capabilities,
                    "performance_score": info.performance_score,
                    "recommended_for": info.recommended_for
                })

        return models_info

# Global model selector instance
model_selector = ModelSelector()

def get_best_model_for_task(task_type: str = "general",
                           require_code_execution: bool = False,
                           require_reasoning: bool = False) -> str:
    """Get the best model for a specific task"""
    return model_selector.select_best_model(task_type, require_code_execution, require_reasoning)

def list_available_models() -> List[Dict[str, Any]]:
    """List all available models with capabilities"""
    return model_selector.list_available_models()

def get_model_recommendations() -> Dict[str, str]:
    """Get model recommendations for different task types"""
    return model_selector.get_model_recommendations()

if __name__ == "__main__":
    print("🤖 Model Selector Demo")
    print("=" * 50)

    # Show available models
    available = list_available_models()
    print(f"Available models: {len(available)}")

    for model in available:
        print(f"\\n🔧 {model['name']} ({model['size']})")
        print(f"   Family: {model['family']}")
        print(f"   Capabilities: {', '.join(model['capabilities'])}")
        print(f"   Performance Score: {model['performance_score']}")
        print(f"   Recommended for: {', '.join(model['recommended_for'])}")

    # Show recommendations
    print("\\n🎯 Model Recommendations:")
    recommendations = get_model_recommendations()

    for task_type, model in recommendations.items():
        print(f"   {task_type}: {model}")

    print("\\n✅ Model selector ready!")
