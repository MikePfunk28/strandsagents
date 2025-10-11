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
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger("model_selector")


@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    size: str
    family: str
    capabilities: List[str]
    performance_score: int
    recommended_for: List[str]
    provider: str = "ollama"
    context_window: Optional[int] = None
    memory_requirement_gb: Optional[float] = None


class ModelSelector:
    """Intelligent model selection for optimal performance"""

    # Model capability matrix
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
        # AWS Bedrock models
        "anthropic.claude-3-sonnet-20240229-v1:0": ModelInfo(
            name="anthropic.claude-3-sonnet-20240229-v1:0",
            size="70B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=95,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"],
            provider="bedrock"
        ),
        "anthropic.claude-3-haiku-20240307-v1:0": ModelInfo(
            name="anthropic.claude-3-haiku-20240307-v1:0",
            size="15B",
            family="claude",
            capabilities=["reasoning", "coding", "analysis", "creative"],
            performance_score=88,
            recommended_for=["general_tasks", "coding",
                             "analysis", "creative_tasks"],
            provider="bedrock"
        ),
        "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelInfo(
            name="anthropic.claude-3-5-sonnet-20241022-v2:0",
            size="70B",
            family="claude",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=96,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"],
            provider="bedrock"
        ),
        "meta.llama3-1-8b-instruct-v1:0": ModelInfo(
            name="meta.llama3-1-8b-instruct-v1:0",
            size="8B",
            family="llama",
            capabilities=["reasoning", "coding", "analysis"],
            performance_score=75,
            recommended_for=["coding", "analysis", "general_tasks"],
            provider="bedrock"
        ),
        "meta.llama3-1-70b-instruct-v1:0": ModelInfo(
            name="meta.llama3-1-70b-instruct-v1:0",
            size="70B",
            family="llama",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=90,
            recommended_for=["enterprise_tasks", "complex_analysis",
                             "research", "creative_tasks", "coding"],
            provider="bedrock"
        ),
        "meta.llama3-1-405b-instruct-v1:0": ModelInfo(
            name="meta.llama3-1-405b-instruct-v1:0",
            size="405B",
            family="llama",
            capabilities=["reasoning", "coding",
                          "analysis", "creative", "research"],
            performance_score=98,
            recommended_for=["maximum_performance", "enterprise_tasks",
                             "complex_analysis", "research", "creative_tasks"],
            provider="bedrock"
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
        self.available_models: List[str] = []
        self.provider_filter: Optional[str] = None
        self._detect_available_models()

    # ------------------------------------------------------------------
    # Provider handling
    # ------------------------------------------------------------------
    def set_provider(self, provider: Optional[str]) -> None:
        """Restrict selection to a specific provider."""
        if provider:
            provider = provider.lower()
        self.provider_filter = provider
        logger.info("[MODEL_SELECTOR] Provider filter set to %s", provider or "auto")

    def list_providers(self) -> List[str]:
        """Return the list of known providers."""
        providers = {info.provider for info in self.MODEL_CAPABILITIES.values()}
        return sorted(providers)

    def _detect_available_models(self) -> None:
        """Detect available models across supported providers."""
        detected: List[str] = []

        detected.extend(self._detect_ollama_models())
        detected.extend(self._detect_bedrock_models())
        detected.extend(self._detect_llamacpp_models())

        if not detected:
            detected = ["llama3.2", "qwen3:4b"]

        seen = set()
        unique: List[str] = []
        for name in detected:
            if name in self.MODEL_CAPABILITIES and name not in seen:
                unique.append(name)
                seen.add(name)

        self.available_models = unique

        logger.info("[MODEL_SELECTOR] Available models: %s", self.available_models)

    def _detect_ollama_models(self) -> List[str]:
        """Detect available Ollama models."""
        models: List[str] = []
        try:
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
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            info = self.MODEL_CAPABILITIES.get(model_name)
                            if info and info.provider == "ollama":
                                models.append(model_name)
                                logger.info("[MODEL_SELECTOR] Found Ollama model: %s", model_name)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.warning("[MODEL_SELECTOR] Ollama detection failed: %s", exc)

        if not models:
            models = [name for name, info in self.MODEL_CAPABILITIES.items() if info.provider == "ollama"]

        return models

    def _detect_bedrock_models(self) -> List[str]:
        """Detect available AWS Bedrock models."""
        models = [name for name, info in self.MODEL_CAPABILITIES.items() if info.provider == "bedrock"]
        if not models:
            return []

        if boto3 is None:
            logger.info("[MODEL_SELECTOR] boto3 not installed; Bedrock models available for manual selection.")
            return models

        if os.getenv("AWS_REGION") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("BEDROCK_ASSUME_ROLE"):
            logger.info("[MODEL_SELECTOR] Bedrock environment detected.")
            return models

        logger.info("[MODEL_SELECTOR] Bedrock credentials not found; exposing models for manual selection.")
        return models

    def _detect_llamacpp_models(self) -> List[str]:
        """Return llama.cpp models (assumed locally available)."""
        return [name for name, info in self.MODEL_CAPABILITIES.items() if info.provider == "llama.cpp"]

    def _provider_matches(self, model_name: str) -> bool:
        if self.provider_filter is None:
            return True
        info = self.MODEL_CAPABILITIES.get(model_name)
        return bool(info and info.provider == self.provider_filter)

    def select_best_model(self, task_type: str = "general",
                          require_code_execution: bool = False,
                          require_reasoning: bool = False) -> str:
        """
        Select the best model for a given task

        Args:
            task_type: Type of task (coding, research, analysis, creative, general)
            require_code_execution: Whether code execution is needed
            require_reasoning: Whether advanced reasoning is needed

        Returns:
            Best model name for the task
        """

        # Filter models based on availability
        candidate_models = [
            model for model in self.available_models
            if model in self.MODEL_CAPABILITIES
            and self._provider_matches(model)
        ]

        if not candidate_models:
            logger.warning(" No suitable models found, using fallback")
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

            model_scores.append((model_name, score))

        # Select best model
        best_model = max(model_scores, key=lambda x: x[1])[0]

        logger.info(
            " Selected model %s for task type %s", best_model, task_type)
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
            if model_name in self.MODEL_CAPABILITIES and self._provider_matches(model_name):
                info = self.MODEL_CAPABILITIES[model_name]
                models_info.append({
                    "name": info.name,
                    "size": info.size,
                    "family": info.family,
                    "capabilities": info.capabilities,
                    "performance_score": info.performance_score,
                    "recommended_for": info.recommended_for,
                    "provider": info.provider,
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
    print(" Model Selector Demo")
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

    print("\\n Model selector ready!")
