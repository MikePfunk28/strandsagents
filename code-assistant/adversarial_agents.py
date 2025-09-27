"""Adversarial coding agents using StrandsAgents architecture."""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from strands import Agent, tool
from strands.models import OllamaModel
from strands_tools import editor, file_read, file_write, python_repl, shell

from .prompts import (
    CODE_GENERATOR_PROMPT,
    CODE_DISCRIMINATOR_PROMPT,
    CODE_OPTIMIZER_PROMPT,
    CODE_SECURITY_PROMPT,
    CODE_TESTER_PROMPT,
    CODE_REVIEWER_PROMPT,
    COORDINATOR_PROMPT
)

logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"

@dataclass
class CodeGenerationRequest:
    """Request for code generation."""
    requirements: str
    language: LanguageType
    context: Optional[str] = None
    constraints: Optional[List[str]] = None

@dataclass
class CodeReviewResult:
    """Result of code review by an agent."""
    agent_type: str
    score: float
    issues: List[str]
    suggestions: List[str]
    approved: bool

@tool
def analyze_code_structure(code: str, language: str) -> Dict[str, Any]:
    """Analyze code structure and extract metadata."""
    # Use the existing language engine
    from .language_engine import LanguageEngine

    engine = LanguageEngine()
    lang_info, code_structure = engine.analyze_content(code)

    return {
        "language": lang_info.language.value,
        "confidence": lang_info.confidence,
        "functions": [f.name for f in code_structure.functions],
        "classes": [c.name for c in code_structure.classes],
        "complexity": code_structure.complexity_score,
        "lines_of_code": code_structure.lines_of_code,
        "imports": code_structure.imports
    }

@tool
def execute_python_code(code: str) -> Dict[str, Any]:
    """Execute Python code safely and return results."""
    try:
        # Use python_repl tool from strands_tools
        result = python_repl(code)
        return {
            "success": True,
            "output": result,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": str(e)
        }

@tool
def validate_syntax(code: str, language: str) -> Dict[str, Any]:
    """Validate code syntax for supported languages."""
    try:
        if language.lower() == "python":
            compile(code, '<string>', 'exec')
        # Add other language validators as needed

        return {
            "valid": True,
            "errors": []
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [f"Syntax error: {e}"]
        }

@tool
def get_code_metrics(code: str) -> Dict[str, Any]:
    """Get code quality metrics."""
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    return {
        "total_lines": len(lines),
        "code_lines": len(non_empty_lines),
        "blank_lines": len(lines) - len(non_empty_lines),
        "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
        "max_line_length": max(len(line) for line in lines) if lines else 0
    }

class ModelConfiguration:
    """Configuration for different model types and sizes."""

    def __init__(self):
        self.available_models = {
            # Llama 3.2 models (as requested)
            "llama3.2:3b": {"size": "3B", "type": "general", "speed": "medium"},
            "llama3.2:1b": {"size": "1B", "type": "fast", "speed": "fast"},

            # Gemma models for specialized tasks
            "gemma2:2b": {"size": "2B", "type": "balanced", "speed": "medium"},
            "gemma:270m": {"size": "270M", "type": "ultrafast", "speed": "ultrafast"},

            # Additional models
            "qwen2.5:4b": {"size": "4B", "type": "quality", "speed": "slow"},
            "qwen2.5:1.5b": {"size": "1.5B", "type": "efficient", "speed": "fast"}
        }

    def get_model_for_task(self, task_type: str, priority: str = "balanced") -> str:
        """Get optimal model for specific task type."""
        if task_type == "generation" and priority == "quality":
            return "llama3.2:3b"
        elif task_type == "discriminator" and priority == "speed":
            return "gemma:270m"
        elif task_type == "optimization":
            return "gemma2:2b"
        elif task_type == "security":
            return "llama3.2:3b"
        else:
            return "llama3.2:1b"  # Default

    def get_multi_model_config(self, strategy: str = "balanced") -> Dict[str, str]:
        """Get configuration for multiple models running in parallel."""
        if strategy == "speed":
            return {
                "generator": "llama3.2:1b",
                "discriminator": "gemma:270m",
                "optimizer": "gemma:270m",
                "security": "llama3.2:1b",
                "tester": "gemma:270m",
                "reviewer": "llama3.2:1b"
            }
        elif strategy == "quality":
            return {
                "generator": "qwen2.5:4b",
                "discriminator": "llama3.2:3b",
                "optimizer": "gemma2:2b",
                "security": "qwen2.5:4b",
                "tester": "llama3.2:1b",
                "reviewer": "llama3.2:3b"
            }
        else:  # balanced
            return {
                "generator": "llama3.2:3b",
                "discriminator": "llama3.2:1b",
                "optimizer": "gemma2:2b",
                "security": "llama3.2:3b",
                "tester": "llama3.2:1b",
                "reviewer": "gemma2:2b"
            }

class CodeGeneratorAgent(Agent):
    """Agent that generates code solutions."""

    def __init__(self, model_name: str = "llama3.2:3b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_GENERATOR_PROMPT
        )

    async def generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate code based on requirements."""
        prompt = f"""
        Generate {request.language.value} code for the following requirements:

        Requirements: {request.requirements}

        {f"Context: {request.context}" if request.context else ""}
        {f"Constraints: {', '.join(request.constraints)}" if request.constraints else ""}

        Please provide clean, well-documented code that follows best practices.
        """

        response = await self.run(prompt)
        return response

class CodeDiscriminatorAgent(Agent):
    """Agent that finds issues and suggests improvements."""

    def __init__(self, model_name: str = "gemma:270m", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_DISCRIMINATOR_PROMPT
        )

    async def review_code(self, code: str, language: str, requirements: str) -> CodeReviewResult:
        """Review code and provide feedback."""
        prompt = f"""
        Review this {language} code against the requirements:

        Requirements: {requirements}

        Code:
        ```{language}
        {code}
        ```

        Provide a detailed analysis with:
        1. Score (0-10)
        2. Issues found
        3. Suggestions for improvement
        4. Whether you approve this code

        Format as JSON.
        """

        response = await self.run(prompt)

        # Parse response (simplified - in real implementation would be more robust)
        try:
            result_data = json.loads(response)
            return CodeReviewResult(
                agent_type="discriminator",
                score=result_data.get("score", 5.0),
                issues=result_data.get("issues", []),
                suggestions=result_data.get("suggestions", []),
                approved=result_data.get("approved", False)
            )
        except:
            # Fallback if JSON parsing fails
            return CodeReviewResult(
                agent_type="discriminator",
                score=6.0,
                issues=["Could not parse review format"],
                suggestions=["Improve response formatting"],
                approved=False
            )

class CodeOptimizerAgent(Agent):
    """Agent that optimizes code for performance."""

    def __init__(self, model_name: str = "gemma2:2b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_OPTIMIZER_PROMPT
        )

class CodeSecurityAgent(Agent):
    """Agent that analyzes code for security issues."""

    def __init__(self, model_name: str = "llama3.2:3b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_SECURITY_PROMPT
        )

class CodeTesterAgent(Agent):
    """Agent that generates test cases."""

    def __init__(self, model_name: str = "llama3.2:1b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_TESTER_PROMPT
        )

class CodeReviewerAgent(Agent):
    """Agent that provides overall code review."""

    def __init__(self, model_name: str = "gemma2:2b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=CODE_REVIEWER_PROMPT
        )

class AdversarialCodingCoordinator(Agent):
    """Coordinator that manages multiple agents in adversarial process."""

    def __init__(self, model_name: str = "llama3.2:3b", host: str = "localhost:11434"):
        super().__init__(
            model=OllamaModel(model=model_name, host=host),
            system_prompt=COORDINATOR_PROMPT
        )

        self.model_config = ModelConfiguration()
        self.agents = {}

    async def initialize_agents(self, strategy: str = "balanced", host: str = "localhost:11434"):
        """Initialize all agents with selected models."""
        config = self.model_config.get_multi_model_config(strategy)

        self.agents = {
            "generator": CodeGeneratorAgent(config["generator"], host),
            "discriminator": CodeDiscriminatorAgent(config["discriminator"], host),
            "optimizer": CodeOptimizerAgent(config["optimizer"], host),
            "security": CodeSecurityAgent(config["security"], host),
            "tester": CodeTesterAgent(config["tester"], host),
            "reviewer": CodeReviewerAgent(config["reviewer"], host)
        }

        logger.info(f"Initialized adversarial coding system with {strategy} strategy")
        logger.info(f"Model configuration: {config}")

    async def generate_code_adversarially(
        self,
        request: CodeGenerationRequest,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Generate code using adversarial process."""

        iteration = 0
        current_code = ""
        review_history = []

        while iteration < max_iterations:
            logger.info(f"Adversarial iteration {iteration + 1}")

            # Generate code
            if iteration == 0:
                current_code = await self.agents["generator"].generate_code(request)
            else:
                # Improve code based on previous feedback
                improvement_prompt = f"""
                Improve this code based on the following feedback:
                {review_history[-1]}

                Current code:
                ```{request.language.value}
                {current_code}
                ```
                """
                current_code = await self.agents["generator"].run(improvement_prompt)

            # Get reviews from multiple agents in parallel
            review_tasks = [
                self.agents["discriminator"].review_code(
                    current_code, request.language.value, request.requirements
                ),
                # Add other agents as needed
            ]

            reviews = await asyncio.gather(*review_tasks)
            review_history.extend(reviews)

            # Check if code is acceptable
            avg_score = sum(r.score for r in reviews) / len(reviews)
            all_approved = all(r.approved for r in reviews)

            if all_approved and avg_score >= 8.0:
                logger.info(f"Code approved after {iteration + 1} iterations")
                break

            iteration += 1

        return {
            "final_code": current_code,
            "iterations": iteration + 1,
            "final_score": avg_score,
            "review_history": review_history,
            "language": request.language.value
        }