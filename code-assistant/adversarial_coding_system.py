"""
Adversarial Coding System - GAN-inspired approach for code generation and validation.

This system uses multiple specialized assistants that work together:
- Generator Assistant: Creates code solutions
- Discriminator Assistant: Validates and critiques code
- Optimizer Assistant: Suggests improvements
- Security Assistant: Checks for vulnerabilities
- Performance Assistant: Analyzes efficiency

Like a GAN, the Generator tries to create perfect code while the Discriminator
tries to find flaws, leading to continuous improvement.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from language_engine import LanguageEngine, LanguageType, CodeStructure
from ollama_model import OllamaModel

logger = logging.getLogger(__name__)


class AssistantRole(Enum):
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"
    OPTIMIZER = "optimizer"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTER = "tester"


class CodeQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class CodeSolution:
    """Represents a code solution with metadata."""
    id: str
    code: str
    language: LanguageType
    description: str
    generator_confidence: float
    quality_scores: Dict[AssistantRole, float]
    issues: List[Dict[str, Any]]
    improvements: List[str]
    test_results: Optional[Dict[str, Any]] = None
    iteration: int = 1


@dataclass
class ValidationResult:
    """Result of code validation by an assistant."""
    role: AssistantRole
    score: float  # 0.0 to 1.0
    quality: CodeQuality
    issues_found: List[Dict[str, Any]]
    suggestions: List[str]
    confidence: float


class BaseAssistant(ABC):
    """Base class for all coding assistants."""

    def __init__(self, role: AssistantRole, ollama_model: OllamaModel, language_engine: LanguageEngine):
        self.role = role
        self.ollama = ollama_model
        self.language_engine = language_engine
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Process a code solution and return validation result."""
        pass

    def _create_prompt(self, base_prompt: str, solution: CodeSolution, context: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for the assistant."""
        return f"""
{base_prompt}

Code to analyze:
```{solution.language.value}
{solution.code}
```

Description: {solution.description}
Language: {solution.language.value}
Current iteration: {solution.iteration}

Previous issues identified: {json.dumps(solution.issues, indent=2)}
Current quality scores: {json.dumps({k.value: v for k, v in solution.quality_scores.items()}, indent=2)}

Context: {json.dumps(context, indent=2)}

Please provide your analysis in the following JSON format:
{{
    "score": <float 0.0-1.0>,
    "quality": "<excellent|good|fair|poor|critical>",
    "issues": [
        {{
            "type": "<issue_type>",
            "severity": "<low|medium|high|critical>",
            "description": "<description>",
            "line": <line_number>,
            "suggestion": "<how_to_fix>"
        }}
    ],
    "suggestions": ["<improvement_suggestion_1>", "<improvement_suggestion_2>"],
    "confidence": <float 0.0-1.0>
}}
"""


class GeneratorAssistant(BaseAssistant):
    """Generates code solutions based on requirements."""

    def __init__(self, ollama_model: OllamaModel, language_engine: LanguageEngine):
        super().__init__(AssistantRole.GENERATOR, ollama_model, language_engine)

    async def generate_solution(self, requirements: str, language: LanguageType, context: Dict[str, Any]) -> CodeSolution:
        """Generate a new code solution."""
        prompt = f"""
You are an expert code generator. Create high-quality, production-ready code based on the requirements.

Requirements: {requirements}
Language: {language.value}
Context: {json.dumps(context, indent=2)}

Guidelines:
1. Write clean, readable, and maintainable code
2. Follow language-specific best practices
3. Include proper error handling
4. Add comments for complex logic
5. Consider performance and security
6. Make the code testable

Generate the code and provide a confidence score (0.0-1.0) for your solution.

Response format:
```{language.value}
<your_code_here>
```

Confidence: <0.0-1.0>
Explanation: <brief_explanation_of_approach>
"""

        try:
            response = self.ollama.generate(prompt, max_tokens=2000)

            # Extract code from response
            code_start = response.find(f"```{language.value}")
            code_end = response.find("```", code_start + 1)

            if code_start != -1 and code_end != -1:
                code = response[code_start + len(f"```{language.value}"):code_end].strip()
            else:
                # Fallback: use entire response as code
                code = response.strip()

            # Extract confidence (simplified)
            confidence = 0.7  # Default confidence
            if "Confidence:" in response:
                try:
                    conf_text = response.split("Confidence:")[1].split("\n")[0].strip()
                    confidence = float(conf_text)
                except:
                    pass

            return CodeSolution(
                id=str(uuid4()),
                code=code,
                language=language,
                description=requirements,
                generator_confidence=confidence,
                quality_scores={},
                issues=[],
                improvements=[]
            )

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            # Return a basic fallback solution
            return CodeSolution(
                id=str(uuid4()),
                code=f"# TODO: Implement {requirements}",
                language=language,
                description=requirements,
                generator_confidence=0.1,
                quality_scores={},
                issues=[],
                improvements=[]
            )

    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Process existing solution (for consistency with base class)."""
        return ValidationResult(
            role=self.role,
            score=solution.generator_confidence,
            quality=CodeQuality.GOOD,
            issues_found=[],
            suggestions=[],
            confidence=solution.generator_confidence
        )


class DiscriminatorAssistant(BaseAssistant):
    """Validates code quality and finds issues."""

    def __init__(self, ollama_model: OllamaModel, language_engine: LanguageEngine):
        super().__init__(AssistantRole.DISCRIMINATOR, ollama_model, language_engine)

    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Validate code quality and identify issues."""
        prompt = self._create_prompt(
            f"""You are an expert code reviewer and discriminator. Your job is to critically analyze code and find any issues, bugs, or areas for improvement.

Analyze the code for:
1. Syntax errors and bugs
2. Logic errors and edge cases
3. Code quality and readability
4. Best practices compliance
5. Potential runtime errors
6. Design patterns usage
7. Maintainability issues

Be thorough and critical - your job is to find problems.""",
            solution,
            context
        )

        try:
            response = self.ollama.generate(prompt, max_tokens=1500)
            result = self._parse_response(response)
            return result

        except Exception as e:
            logger.error(f"Discriminator error: {e}")
            return ValidationResult(
                role=self.role,
                score=0.5,
                quality=CodeQuality.FAIR,
                issues_found=[],
                suggestions=["Unable to analyze due to error"],
                confidence=0.3
            )


class OptimizerAssistant(BaseAssistant):
    """Suggests performance and code optimizations."""

    def __init__(self, ollama_model: OllamaModel, language_engine: LanguageEngine):
        super().__init__(AssistantRole.OPTIMIZER, ollama_model, language_engine)

    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Analyze code for optimization opportunities."""
        prompt = self._create_prompt(
            f"""You are a performance optimization expert. Analyze the code for:

1. Time complexity improvements
2. Space complexity optimizations
3. Algorithm efficiency
4. Memory usage
5. Language-specific optimizations
6. Caching opportunities
7. Parallelization potential
8. Database query optimization (if applicable)

Focus on practical optimizations that significantly improve performance.""",
            solution,
            context
        )

        try:
            response = self.ollama.generate(prompt, max_tokens=1500)
            result = self._parse_response(response)
            return result

        except Exception as e:
            logger.error(f"Optimizer error: {e}")
            return ValidationResult(
                role=self.role,
                score=0.6,
                quality=CodeQuality.GOOD,
                issues_found=[],
                suggestions=["Unable to analyze for optimizations"],
                confidence=0.3
            )


class SecurityAssistant(BaseAssistant):
    """Analyzes code for security vulnerabilities."""

    def __init__(self, ollama_model: OllamaModel, language_engine: LanguageEngine):
        super().__init__(AssistantRole.SECURITY, ollama_model, language_engine)

    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Analyze code for security vulnerabilities."""
        prompt = self._create_prompt(
            f"""You are a cybersecurity expert. Analyze the code for security vulnerabilities:

1. Injection attacks (SQL, XSS, command injection)
2. Authentication and authorization issues
3. Data validation and sanitization
4. Cryptographic weaknesses
5. Information disclosure
6. Buffer overflows (for lower-level languages)
7. Insecure dependencies
8. Configuration issues
9. Race conditions
10. Input validation

Be very thorough - security issues can be critical.""",
            solution,
            context
        )

        try:
            response = self.ollama.generate(prompt, max_tokens=1500)
            result = self._parse_response(response)
            return result

        except Exception as e:
            logger.error(f"Security analysis error: {e}")
            return ValidationResult(
                role=self.role,
                score=0.7,  # Assume reasonably secure if can't analyze
                quality=CodeQuality.GOOD,
                issues_found=[],
                suggestions=["Unable to perform security analysis"],
                confidence=0.3
            )


class TesterAssistant(BaseAssistant):
    """Generates and validates test cases."""

    def __init__(self, ollama_model: OllamaModel, language_engine: LanguageEngine):
        super().__init__(AssistantRole.TESTER, ollama_model, language_engine)

    async def process(self, solution: CodeSolution, context: Dict[str, Any]) -> ValidationResult:
        """Analyze testability and suggest test cases."""
        prompt = self._create_prompt(
            f"""You are a testing expert. Analyze the code for testability and suggest comprehensive tests:

1. Unit test coverage potential
2. Integration test requirements
3. Edge cases to test
4. Error conditions to validate
5. Performance test scenarios
6. Mock requirements
7. Test data needs
8. Testability improvements

Suggest specific test cases and testing strategies.""",
            solution,
            context
        )

        try:
            response = self.ollama.generate(prompt, max_tokens=1500)
            result = self._parse_response(response)
            return result

        except Exception as e:
            logger.error(f"Testing analysis error: {e}")
            return ValidationResult(
                role=self.role,
                score=0.6,
                quality=CodeQuality.GOOD,
                issues_found=[],
                suggestions=["Unable to analyze testability"],
                confidence=0.3
            )

    def _parse_response(self, response: str) -> ValidationResult:
        """Parse LLM response into ValidationResult."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                quality_map = {
                    "excellent": CodeQuality.EXCELLENT,
                    "good": CodeQuality.GOOD,
                    "fair": CodeQuality.FAIR,
                    "poor": CodeQuality.POOR,
                    "critical": CodeQuality.CRITICAL
                }

                return ValidationResult(
                    role=self.role,
                    score=float(data.get('score', 0.5)),
                    quality=quality_map.get(data.get('quality', 'fair'), CodeQuality.FAIR),
                    issues_found=data.get('issues', []),
                    suggestions=data.get('suggestions', []),
                    confidence=float(data.get('confidence', 0.5))
                )

        except Exception as e:
            logger.warning(f"Failed to parse response as JSON: {e}")

        # Fallback parsing
        score = 0.5
        quality = CodeQuality.FAIR
        issues = []
        suggestions = []

        # Simple heuristics
        if "excellent" in response.lower():
            quality = CodeQuality.EXCELLENT
            score = 0.9
        elif "good" in response.lower():
            quality = CodeQuality.GOOD
            score = 0.7
        elif "poor" in response.lower() or "critical" in response.lower():
            quality = CodeQuality.POOR
            score = 0.3

        return ValidationResult(
            role=self.role,
            score=score,
            quality=quality,
            issues_found=issues,
            suggestions=suggestions.split('\n') if isinstance(suggestions, str) else [],
            confidence=0.5
        )


class AdversarialCodingSystem:
    """Main orchestrator for the adversarial coding system."""

    def __init__(self, ollama_model: OllamaModel):
        self.ollama = ollama_model
        self.language_engine = LanguageEngine()

        # Initialize all assistants
        self.generator = GeneratorAssistant(ollama_model, self.language_engine)
        self.discriminator = DiscriminatorAssistant(ollama_model, self.language_engine)
        self.optimizer = OptimizerAssistant(ollama_model, self.language_engine)
        self.security = SecurityAssistant(ollama_model, self.language_engine)
        self.tester = TesterAssistant(ollama_model, self.language_engine)

        self.assistants = [
            self.discriminator,
            self.optimizer,
            self.security,
            self.tester
        ]

        self.solutions_history: List[CodeSolution] = []

    async def generate_and_validate_code(
        self,
        requirements: str,
        language: LanguageType,
        max_iterations: int = 3,
        target_quality: float = 0.8
    ) -> CodeSolution:
        """Generate code and iteratively improve it using adversarial validation."""

        logger.info(f"Starting adversarial code generation for: {requirements}")

        # Initial generation
        solution = await self.generator.generate_solution(
            requirements, language, {"max_iterations": max_iterations}
        )

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # Run all validation assistants in parallel
            validation_tasks = [
                assistant.process(solution, {"iteration": iteration + 1})
                for assistant in self.assistants
            ]

            validation_results = await asyncio.gather(*validation_tasks)

            # Update solution with validation results
            solution.quality_scores = {
                result.role: result.score for result in validation_results
            }

            # Collect all issues and suggestions
            all_issues = []
            all_suggestions = []

            for result in validation_results:
                all_issues.extend(result.issues_found)
                all_suggestions.extend(result.suggestions)

            solution.issues = all_issues
            solution.improvements = all_suggestions
            solution.iteration = iteration + 1

            # Calculate overall quality score
            overall_score = sum(solution.quality_scores.values()) / len(solution.quality_scores)

            logger.info(f"Overall quality score: {overall_score:.2f}")
            logger.info(f"Issues found: {len(all_issues)}")
            logger.info(f"Suggestions: {len(all_suggestions)}")

            # Check if we've reached target quality
            if overall_score >= target_quality and len(all_issues) == 0:
                logger.info("Target quality reached!")
                break

            # If not the last iteration, improve the solution
            if iteration < max_iterations - 1:
                solution = await self._improve_solution(solution, validation_results)

        self.solutions_history.append(solution)
        return solution

    async def _improve_solution(self, solution: CodeSolution, validation_results: List[ValidationResult]) -> CodeSolution:
        """Improve the solution based on validation feedback."""

        # Collect all feedback
        issues_summary = []
        suggestions_summary = []

        for result in validation_results:
            for issue in result.issues_found:
                issues_summary.append(f"[{result.role.value}] {issue.get('description', 'Issue found')}")

            for suggestion in result.suggestions:
                suggestions_summary.append(f"[{result.role.value}] {suggestion}")

        improvement_prompt = f"""
Improve the following code based on the feedback from multiple expert reviewers:

Original Requirements: {solution.description}
Language: {solution.language.value}

Current Code:
```{solution.language.value}
{solution.code}
```

Issues Found:
{chr(10).join(issues_summary[:10])}  # Limit to first 10 issues

Suggestions for Improvement:
{chr(10).join(suggestions_summary[:10])}  # Limit to first 10 suggestions

Please provide an improved version of the code that addresses these issues and incorporates the suggestions.
Focus on the most critical issues first.

Improved Code:
```{solution.language.value}
<improved_code_here>
```
"""

        try:
            response = self.ollama.generate(improvement_prompt, max_tokens=2000)

            # Extract improved code
            code_start = response.find(f"```{solution.language.value}")
            code_end = response.find("```", code_start + 1)

            if code_start != -1 and code_end != -1:
                improved_code = response[code_start + len(f"```{solution.language.value}"):code_end].strip()

                # Create new solution with improved code
                improved_solution = CodeSolution(
                    id=str(uuid4()),
                    code=improved_code,
                    language=solution.language,
                    description=solution.description,
                    generator_confidence=min(1.0, solution.generator_confidence + 0.1),
                    quality_scores={},
                    issues=[],
                    improvements=[],
                    iteration=solution.iteration
                )

                return improved_solution

        except Exception as e:
            logger.error(f"Error improving solution: {e}")

        # Return original solution if improvement fails
        return solution

    def get_solution_report(self, solution: CodeSolution) -> Dict[str, Any]:
        """Generate a comprehensive report for a solution."""
        overall_score = 0
        if solution.quality_scores:
            overall_score = sum(solution.quality_scores.values()) / len(solution.quality_scores)

        return {
            "solution_id": solution.id,
            "language": solution.language.value,
            "description": solution.description,
            "iteration": solution.iteration,
            "overall_quality_score": overall_score,
            "generator_confidence": solution.generator_confidence,
            "quality_scores_by_role": {role.value: score for role, score in solution.quality_scores.items()},
            "total_issues": len(solution.issues),
            "critical_issues": len([i for i in solution.issues if i.get('severity') == 'critical']),
            "improvements_suggested": len(solution.improvements),
            "code_length": len(solution.code.splitlines()),
            "issues_by_type": self._group_issues_by_type(solution.issues),
            "top_suggestions": solution.improvements[:5]
        }

    def _group_issues_by_type(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group issues by type for reporting."""
        issue_counts = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        return issue_counts

    async def compare_solutions(self, solution1: CodeSolution, solution2: CodeSolution) -> Dict[str, Any]:
        """Compare two solutions and recommend the better one."""
        score1 = sum(solution1.quality_scores.values()) / len(solution1.quality_scores) if solution1.quality_scores else 0
        score2 = sum(solution2.quality_scores.values()) / len(solution2.quality_scores) if solution2.quality_scores else 0

        return {
            "solution1_score": score1,
            "solution2_score": score2,
            "recommended": solution1.id if score1 > score2 else solution2.id,
            "score_difference": abs(score1 - score2),
            "solution1_issues": len(solution1.issues),
            "solution2_issues": len(solution2.issues),
            "comparison_details": {
                "quality_by_role": {
                    role.value: {
                        "solution1": solution1.quality_scores.get(role, 0),
                        "solution2": solution2.quality_scores.get(role, 0)
                    }
                    for role in AssistantRole
                }
            }
        }