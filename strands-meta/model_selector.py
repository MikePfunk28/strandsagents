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
        "qwen3:0.6b": ModelInfo(
            name="qwen3:0.6b",
            size="0.6b",
            family="qwen",
            capabilities=["basic_reasoning", "simple_tasks"],
            performance_score=40,
            recommended_for=["simple_tasks", "light_analysis"]
        ),
        "qwen3:1.7b": ModelInfo(
            name="qwen3:1.7b",
            size="1.7B",
            family="qwen",
            capabilities=["basic_reasoning", "simple_tasks"],
            performance_score=50,
            recommended_for=["simple_tasks", "light_analysis"]
        ),
        "qwen3-coder:30b": ModelInfo(
            name="qwen3-coder:30b",
            size="30B",
            family="qwen",
            capabilities=["basic_reasoning", "simple_tasks"],
            performance_score=40,
            recommended_for=["simple_tasks", "light_analysis"]
        ),
        "qwen3:30b": ModelInfo(
            name="qwen3:30b",
            size="30B",
            family="qwen",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=95,
            recommended_for=["complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "qwen3-coder:480b-cloud": ModelInfo(
            name="qwen3-coder:480b-cloud",
            size="480B",
            family="qwen",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=85,
            recommended_for=["analysis", "coding",
                             "research", "creative_tasks"]
        ),
        "qwen3:14b": ModelInfo(
            name="qwen3:14b",
            size="14B",
            family="qwen",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=85,
            recommended_for=["analysis", "coding",
                             "research", "creative_tasks"]
        ),
        "qwen3:8b": ModelInfo(
            name="qwen3:8b",
            size="8B",
            family="qwen",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=75,
            recommended_for=["coding", "analysis", "general_tasks"]
        ),
        "qwen3:4b": ModelInfo(
            name="qwen3:4b",
            size="4B",
            family="qwen",
            capabilities=["basic_reasoning", "simple_tasks"],
            performance_score=60,
            recommended_for=["simple_tasks", "light_analysis"]
        ),
        "llama3.2:3b": ModelInfo(
            name="llama3.2:3b",
            size="3B",
            family="llama",
            capabilities=["basic_reasoning", "simple_tasks"],
            performance_score=55,
            recommended_for=["simple_tasks", "basic_qa"]
        ),
        "gemma3:1b": ModelInfo(
            name="gemma3:1b",
            size="1B",
            family="gemma",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=65,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "gemma3:4b": ModelInfo(
            name="gemma3:4b",
            size="4B",
            family="gemma",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=72,
            recommended_for=["general_tasks",
                             "coding", "analysis", "creative_tasks"]
        ),
        "gemma3:12b": ModelInfo(
            name="gemma3:12b",
            size="12B",
            family="gemma",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=85,
            recommended_for=["complex_tasks", "research",
                             "analysis", "coding", "creative_tasks"]
        ),
        "gemma3:270m": ModelInfo(
            name="gemma3:270m",
            size="270m",
            family="gemma",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=50,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "gemma3:27b": ModelInfo(
            name="gemma3:27b",
            size="27b",
            family="gemma",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=50,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "phi4:14b": ModelInfo(
            name="phi4:14b",
            size="14B",
            family="phi",
            capabilities=["reasoning", "coding", "analysis", "mathematics"],
            performance_score=88,
            recommended_for=["complex_tasks", "research",
                             "mathematics", "coding", "analysis"]
        ),
        "phi4-mini-reasoning:3.8b": ModelInfo(
            name="phi4-mini-reasoning:3.8b",
            size="3.8B",
            family="phi",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=75,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "phi4-mini:3.8b": ModelInfo(
            name="phi4-mini:3.8b",
            size="3.8B",
            family="phi",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=75,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "qwen3-embedding:8b": ModelInfo(
            name="qwen3-embedding:8b",
            size="8B",
            family="qwen",
            capabilities=["embedding", "coding", "analysis"],
            performance_score=80,
            recommended_for=["embeddings", "coding", "analysis"]
        ),
        "qwen3-embedding:4b": ModelInfo(
            name="qwen3-embedding:4b",
            size="4B",
            family="qwen",
            capabilities=["embedding", "coding", "analysis"],
            performance_score=80,
            recommended_for=["embeddings", "coding", "analysis"]
        ),
        "qwen3-embedding:0.6b": ModelInfo(
            name="qwen3-embedding:0.6b",
            size="0.6B",
            family="qwen",
            capabilities=["embedding", "coding", "analysis"],
            performance_score=80,
            recommended_for=["embeddings", "coding", "analysis"]
        ),
        "embeddinggemma:300m": ModelInfo(
            name="embeddinggemma:300m",
            size="300m",
            family="gemma",
            capabilities=["embedding", "coding", "analysis"],
            performance_score=65,
            recommended_for=["embeddings", "coding", "analysis"]
        ),
        "gpt-oss:20b": ModelInfo(
            name="gpt-oss:20b",
            size="20B",
            family="gpt",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=80,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "llama3.2:1b": ModelInfo(
            name="llama3.2:1b",
            size="1B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=70,
            recommended_for=["general_tasks", "coding", "analysis"]
        ),
        "llama3.1:8b": ModelInfo(
            name="llama3.1:8b",
            size="8B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=75,
            recommended_for=["coding", "analysis", "general_tasks"]
        ),
        "devstral:24b": ModelInfo(
            name="devstral:24b",
            size="24B",
            family="devstral",
            capabilities=["coding", "mathematics", "problem_solving"],
            performance_score=80,
            recommended_for=["coding", "mathematics", "algorithms"]
        ),
        "codegemma:2b": ModelInfo(
            name="codegemma:2b",
            size="2B",
            family="gemma",
            capabilities=["coding", "mathematics", "problem_solving"],
            performance_score=80,
            recommended_for=["coding", "mathematics", "algorithms"]
        ),
        "codegemma:7b": ModelInfo(
            name="codegemma:7b",
            size="7B",
            family="gemma",
            capabilities=["coding", "mathematics", "problem_solving"],
            performance_score=80,
            recommended_for=["coding", "mathematics", "algorithms"]
        ),
        "deepseek-r1:8b": ModelInfo(
            name="deepseek-r1:8b",
            size="8B",
            family="deepseek",
            capabilities=["coding", "mathematics", "problem_solving"],
            performance_score=80,
            recommended_for=["coding", "mathematics", "algorithms"]
        ),
        "gpt-5": ModelInfo(
            name="gpt-5",
            size="175B",
            family="gpt",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=98,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "gpt5-mini": ModelInfo(
            name="gpt5-mini",
            size="175B",
            family="gpt",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=95,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "gpt5-nano": ModelInfo(
            name="gpt5-nano",
            size="175B",
            family="gpt",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=85,
            recommended_for=["general_tasks",
                             "coding", "analysis", "creative_tasks"]
        ),
        "claude-4.5-sonnet": ModelInfo(
            name="claude-4.5-sonnet",
            size="70B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=96,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "claude-3.7-sonnet": ModelInfo(
            name="claude-3.7-sonnet",
            size="70B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=96,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "claude-3.5-haiku": ModelInfo(
            name="claude-3.5-haiku",
            size="15B",
            family="claude",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=88,
            recommended_for=["general_tasks",
                             "coding", "analysis", "creative_tasks"]
        ),
        "deepseek-v3": ModelInfo(
            name="deepseek-v3",
            size="671B",
            family="deepseek",
            capabilities=["reasoning", "coding",
                          "mathematics", "analysis", "research"],
            performance_score=99,
            recommended_for=["maximum_performance",
                             "enterprise_tasks", "mathematics", "research", "coding"]
        ),
        "claude-4.0-sonnet": ModelInfo(
            name="claude-4.0-sonnet",
            size="400B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "creative", "analysis", "research"],
            performance_score=94,
            recommended_for=["enterprise_tasks",
                             "creative_tasks", "research", "coding", "analysis"]
        ),
        "claude-4.0-opus": ModelInfo(
            name="claude-4.0-opus",
            size="400B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "creative", "analysis", "research"],
            performance_score=94,
            recommended_for=["enterprise_tasks",
                             "creative_tasks", "research", "coding", "analysis"]
        ),
        "mistral-nemo": ModelInfo(
            name="mistral-nemo",
            size="12B",
            family="mistral",
            capabilities=["reasoning", "coding", "creative", "analysis"],
            performance_score=82,
            recommended_for=["general_tasks",
                             "creative_tasks", "coding", "analysis"]
        ),
        "mistral:7b": ModelInfo(
            name="mistral:7b",
            size="7B",
            family="mistral",
            capabilities=["reasoning", "coding", "creative", "analysis"],
            performance_score=74,
            recommended_for=["creative_tasks",
                             "coding", "general_tasks", "analysis"]
        ),
        "mistral:8x7b": ModelInfo(
            name="mistral:8x7b",
            size="56B",
            family="mistral",
            capabilities=["reasoning", "coding",
                          "creative", "analysis", "research"],
            performance_score=88,
            recommended_for=["complex_tasks", "creative_tasks",
                             "research", "coding", "analysis"]
        ),
        "mistral:8x22b": ModelInfo(
            name="mistral:8x22b",
            size="176B",
            family="mistral",
            capabilities=["reasoning", "coding",
                          "creative", "analysis", "research"],
            performance_score=97,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "codellama:7b": ModelInfo(
            name="codellama:7b",
            size="7B",
            family="codellama",
            capabilities=["coding", "code_completion",
                          "debugging", "mathematics"],
            performance_score=80,
            recommended_for=["coding", "debugging",
                             "code_review", "mathematics"]
        ),
        "codellama:13b": ModelInfo(
            name="codellama:13b",
            size="13B",
            family="codellama",
            capabilities=["coding", "code_completion",
                          "debugging", "mathematics"],
            performance_score=85,
            recommended_for=["coding", "debugging",
                             "code_review", "mathematics", "analysis"]
        ),
        "codellama:34b": ModelInfo(
            name="codellama:34b",
            size="34B",
            family="codellama",
            capabilities=["coding", "code_completion",
                          "debugging", "mathematics"],
            performance_score=92,
            recommended_for=["enterprise_coding", "debugging",
                             "code_review", "mathematics", "analysis"]
        ),
        "deepseek-coder:6.7b": ModelInfo(
            name="deepseek-coder:6.7b",
            size="6.7B",
            family="deepseek",
            capabilities=["coding", "mathematics",
                          "problem_solving", "analysis"],
            performance_score=82,
            recommended_for=["coding", "mathematics", "algorithms", "analysis"]
        ),
        "deepseek-coder:33b": ModelInfo(
            name="deepseek-coder:33b",
            size="33B",
            family="deepseek",
            capabilities=["coding", "mathematics",
                          "problem_solving", "analysis"],
            performance_score=90,
            recommended_for=["enterprise_coding", "mathematics",
                             "algorithms", "analysis", "research"]
        ),
        "claude-3-haiku": ModelInfo(
            name="claude-3-haiku",
            size="15B",
            family="claude",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=88,
            recommended_for=["general_tasks",
                             "coding", "analysis", "creative_tasks"]
        ),
        "claude-3.5-sonnet": ModelInfo(
            name="claude-3.5-sonnet",
            size="70B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=95,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "claude-4.1-opus": ModelInfo(
            name="claude-4.1-opus",
            size="200B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=99,
            recommended_for=["maximum_performance", "enterprise_tasks",
                             "complex_analysis", "research", "creative_tasks"]
        ),
        "gpt-4o": ModelInfo(
            name="gpt-4",
            size="175B",
            family="gpt",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=94,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "gpt-4-turbo": ModelInfo(
            name="gpt-4-turbo",
            size="175B",
            family="gpt",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=96,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "gpt-oss:120b-cloud": ModelInfo(
            name="gpt-oss:120b-cloud",
            size="120B",
            family="gpt-oss",
            capabilities=["reasoning", "coding",
                          "creative", "analysis", "research"],
            performance_score=97,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"]
        ),
        "llama3.2-vision:11b": ModelInfo(
            name="llama3.2-vision:11b",
            size="11B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=80,
            recommended_for=["general_tasks",
                             "coding", "analysis", "creative_tasks"]
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
                            logger.info(f"[SEARCH] Found available model: {model_name}")

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

        logger.info(f"[SEARCH] Available models: {self.available_models}")

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

        logger.info(f"[AGENT_BUILDER] Selected model '{best_model}' for task type '{task_type}'")
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
    print("[AGENT] Model Selector Demo")
    print("=" * 50)

    # Show available models
    available = list_available_models()
    print(f"Available models: {len(available)}")

    for model in available:
        print(f"\n[TOOL] {model['name']} ({model['size']})")
        print(f"   Family: {model['family']}")
        print(f"   Capabilities: {', '.join(model['capabilities'])}")
        print(f"   Performance Score: {model['performance_score']}")
        print(f"   Recommended for: {', '.join(model['recommended_for'])}")

    # Show recommendations
    print("\n[TARGET] Model Recommendations:")
    recommendations = get_model_recommendations()

    for task_type, model in recommendations.items():
        print(f"   {task_type}: {model}")

    print("\n[OK] Model selector ready!")
