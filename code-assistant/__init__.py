"""Adversarial Coding System using StrandsAgents.

A GAN-style coding assistant that uses multiple AI agents to iteratively
improve code quality through adversarial feedback.

Key Features:
- Multi-language support (Python, JavaScript, Rust, Go, Java, etc.)
- Model selection with Llama 3.2, Gemma, and other local models
- Agent2agent communication for parallel processing
- MCP integration for external coordination
- GAN-like adversarial improvement process

Example Usage:
    >>> from adversarial_agents import AdversarialCodingCoordinator
    >>> coordinator = AdversarialCodingCoordinator()
    >>> await coordinator.initialize_agents(strategy="balanced")
"""

__version__ = "1.0.0"
__author__ = "Adversarial Coding Team"

from .adversarial_agents import (
    AdversarialCodingCoordinator,
    CodeGeneratorAgent,
    CodeDiscriminatorAgent,
    CodeOptimizerAgent,
    CodeSecurityAgent,
    CodeTesterAgent,
    CodeReviewerAgent,
    ModelConfiguration,
    CodeGenerationRequest,
    CodeReviewResult,
    LanguageType
)

from .prompts import (
    CODE_GENERATOR_PROMPT,
    CODE_DISCRIMINATOR_PROMPT,
    CODE_OPTIMIZER_PROMPT,
    CODE_SECURITY_PROMPT,
    CODE_TESTER_PROMPT,
    CODE_REVIEWER_PROMPT,
    COORDINATOR_PROMPT
)

# Legacy imports (kept for compatibility)
from .database_manager import DatabaseManager
from .memory_manager import MemoryManager
from .language_engine import LanguageEngine

__all__ = [
    "AdversarialCodingCoordinator",
    "CodeGeneratorAgent",
    "CodeDiscriminatorAgent",
    "CodeOptimizerAgent",
    "CodeSecurityAgent",
    "CodeTesterAgent",
    "CodeReviewerAgent",
    "ModelConfiguration",
    "CodeGenerationRequest",
    "CodeReviewResult",
    "LanguageType",
    "CODE_GENERATOR_PROMPT",
    "CODE_DISCRIMINATOR_PROMPT",
    "CODE_OPTIMIZER_PROMPT",
    "CODE_SECURITY_PROMPT",
    "CODE_TESTER_PROMPT",
    "CODE_REVIEWER_PROMPT",
    "COORDINATOR_PROMPT",
    "DatabaseManager",
    "MemoryManager",
    "LanguageEngine"
]