"""Answer validation framework for cross-checking agent responses.

Provides comprehensive validation of agent answers using multiple validation
strategies including cross-validation, consistency checking, and confidence scoring.
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationMethod(Enum):
    """Available validation methods."""
    CROSS_VALIDATION = "cross_validation"
    CONSISTENCY_CHECK = "consistency_check"
    CRITICAL_ANALYSIS = "critical_analysis"
    CONFIDENCE_SCORING = "confidence_scoring"
    FACTUAL_VERIFICATION = "factual_verification"

@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_valid: bool
    confidence_score: float
    trust_level: float
    validation_methods: List[str]
    issues: List[str]
    metadata: Dict[str, Any]
    validation_time: float
    cross_check_results: Dict[str, Any]

@dataclass
class AnswerContext:
    """Context for answer validation."""
    original_question: str
    agent_id: str
    agent_type: str
    answer: str
    task_context: Dict[str, Any]
    timestamp: datetime

class AnswerValidator:
    """Comprehensive answer validation system."""

    def __init__(self, critical_agent=None):
        """Initialize answer validator.

        Args:
            critical_agent: Critical thinking agent for validation (optional)
        """
        self.critical_agent = critical_agent
        self.validation_history: List[ValidationResult] = []
        self.agent_reliability_scores: Dict[str, float] = {}

        # Validation metrics
        self.validations_performed = 0
        self.answers_approved = 0
        self.answers_rejected = 0
        self.cross_validations_performed = 0

        # Suspicious patterns
        self.suspicious_patterns = [
            "I cannot", "I don't know", "I'm not sure",
            "cannot be determined", "insufficient information",
            "unclear", "ambiguous", "uncertain"
        ]

        # Quality indicators
        self.quality_indicators = [
            "specifically", "according to", "research shows",
            "studies indicate", "data suggests", "evidence demonstrates",
            "analysis reveals", "findings show"
        ]

    async def validate_answer(self, answer_context: AnswerContext,
                            validation_methods: List[ValidationMethod] = None,
                            cross_check_agents: List[Any] = None) -> ValidationResult:
        """Validate an agent's answer using multiple methods.

        Args:
            answer_context: Context and answer to validate
            validation_methods: Specific validation methods to use
            cross_check_agents: Other agents for cross-validation

        Returns:
            ValidationResult with comprehensive validation details
        """
        start_time = asyncio.get_event_loop().time()
        self.validations_performed += 1

        if validation_methods is None:
            validation_methods = [
                ValidationMethod.CONSISTENCY_CHECK,
                ValidationMethod.CONFIDENCE_SCORING,
                ValidationMethod.CRITICAL_ANALYSIS
            ]

        issues = []
        confidence_scores = []
        cross_check_results = {}
        metadata = {
            "agent_id": answer_context.agent_id,
            "agent_type": answer_context.agent_type,
            "answer_length": len(answer_context.answer),
            "timestamp": answer_context.timestamp.isoformat()
        }

        try:
            # Consistency checking
            if ValidationMethod.CONSISTENCY_CHECK in validation_methods:
                consistency_result = await self._check_consistency(answer_context)
                confidence_scores.append(consistency_result["confidence"])
                cross_check_results["consistency"] = consistency_result
                if consistency_result["issues"]:
                    issues.extend(consistency_result["issues"])

            # Confidence scoring
            if ValidationMethod.CONFIDENCE_SCORING in validation_methods:
                confidence_result = await self._score_confidence(answer_context)
                confidence_scores.append(confidence_result["confidence"])
                cross_check_results["confidence"] = confidence_result
                if confidence_result["issues"]:
                    issues.extend(confidence_result["issues"])

            # Critical analysis
            if ValidationMethod.CRITICAL_ANALYSIS in validation_methods and self.critical_agent:
                critical_result = await self._critical_analysis(answer_context)
                confidence_scores.append(critical_result["confidence"])
                cross_check_results["critical_analysis"] = critical_result
                if critical_result["issues"]:
                    issues.extend(critical_result["issues"])

            # Cross-validation with other agents
            if (ValidationMethod.CROSS_VALIDATION in validation_methods and
                cross_check_agents and len(cross_check_agents) > 0):
                cross_val_result = await self._cross_validate(answer_context, cross_check_agents)
                confidence_scores.append(cross_val_result["confidence"])
                cross_check_results["cross_validation"] = cross_val_result
                if cross_val_result["issues"]:
                    issues.extend(cross_val_result["issues"])

            # Factual verification (if applicable)
            if ValidationMethod.FACTUAL_VERIFICATION in validation_methods:
                factual_result = await self._verify_facts(answer_context)
                confidence_scores.append(factual_result["confidence"])
                cross_check_results["factual_verification"] = factual_result
                if factual_result["issues"]:
                    issues.extend(factual_result["issues"])

            # Calculate overall confidence
            overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5

            # Determine trust level based on agent history
            trust_level = self._calculate_trust_level(answer_context.agent_id, overall_confidence)

            # Determine if answer is valid
            is_valid = overall_confidence > 0.6 and trust_level > 0.5 and len(issues) < 3

            validation_result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_confidence,
                trust_level=trust_level,
                validation_methods=[method.value for method in validation_methods],
                issues=issues,
                metadata=metadata,
                validation_time=asyncio.get_event_loop().time() - start_time,
                cross_check_results=cross_check_results
            )

            # Update statistics
            if is_valid:
                self.answers_approved += 1
            else:
                self.answers_rejected += 1

            # Update agent reliability
            self._update_agent_reliability(answer_context.agent_id, overall_confidence)

            # Store validation history
            self.validation_history.append(validation_result)

            # Limit history size
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-500:]

            logger.info(f"Validated answer from {answer_context.agent_id}: "
                       f"confidence={overall_confidence:.2f}, valid={is_valid}")

            return validation_result

        except Exception as e:
            logger.error(f"Answer validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                trust_level=0.0,
                validation_methods=[method.value for method in validation_methods],
                issues=[f"Validation error: {str(e)}"],
                metadata=metadata,
                validation_time=asyncio.get_event_loop().time() - start_time,
                cross_check_results={}
            )

    async def _check_consistency(self, answer_context: AnswerContext) -> Dict[str, Any]:
        """Check answer consistency with question and context."""
        issues = []
        confidence = 1.0

        answer = answer_context.answer.lower()
        question = answer_context.original_question.lower()

        # Check for suspicious patterns
        suspicious_count = sum(1 for pattern in self.suspicious_patterns if pattern in answer)
        if suspicious_count > 2:
            issues.append(f"Multiple uncertain expressions detected ({suspicious_count})")
            confidence *= 0.7

        # Check for quality indicators
        quality_count = sum(1 for indicator in self.quality_indicators if indicator in answer)
        if quality_count == 0 and len(answer) > 100:
            issues.append("No quality indicators found in substantial answer")
            confidence *= 0.8

        # Check answer length appropriateness
        if len(answer) < 20:
            issues.append("Answer suspiciously short")
            confidence *= 0.6
        elif len(answer) > 5000:
            issues.append("Answer suspiciously long")
            confidence *= 0.9

        # Check for question-answer relevance (basic keyword matching)
        question_words = set(question.split())
        answer_words = set(answer.split())
        relevance_score = len(question_words & answer_words) / max(len(question_words), 1)

        if relevance_score < 0.1:
            issues.append("Low relevance between question and answer")
            confidence *= 0.5

        return {
            "confidence": confidence,
            "issues": issues,
            "relevance_score": relevance_score,
            "suspicious_count": suspicious_count,
            "quality_count": quality_count
        }

    async def _score_confidence(self, answer_context: AnswerContext) -> Dict[str, Any]:
        """Score confidence based on answer characteristics."""
        issues = []
        confidence = 1.0

        answer = answer_context.answer

        # Check for uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "possibly", "might", "could be", "seems like"]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer.lower())

        if uncertainty_count > 0:
            confidence -= uncertainty_count * 0.1
            issues.append(f"Uncertainty markers detected ({uncertainty_count})")

        # Check for definitiveness
        definitive_markers = ["definitely", "certainly", "clearly", "obviously", "without doubt"]
        definitive_count = sum(1 for marker in definitive_markers if marker in answer.lower())

        if definitive_count > 0:
            confidence += definitive_count * 0.05

        # Check for factual backing
        factual_markers = ["according to", "research shows", "studies indicate", "data reveals"]
        factual_count = sum(1 for marker in factual_markers if marker in answer.lower())

        if factual_count > 0:
            confidence += factual_count * 0.1

        # Ensure confidence stays in valid range
        confidence = max(0.0, min(1.0, confidence))

        return {
            "confidence": confidence,
            "issues": issues,
            "uncertainty_count": uncertainty_count,
            "definitive_count": definitive_count,
            "factual_count": factual_count
        }

    async def _critical_analysis(self, answer_context: AnswerContext) -> Dict[str, Any]:
        """Use critical agent to analyze the answer."""
        if not self.critical_agent:
            return {"confidence": 0.5, "issues": ["No critical agent available"]}

        try:
            critical_prompt = f"""Critically analyze this answer for accuracy, logic, and completeness:

Question: {answer_context.original_question}
Answer: {answer_context.answer}
Context: {answer_context.task_context}

Evaluate:
1. Logical consistency
2. Factual accuracy (if verifiable)
3. Completeness of response
4. Potential biases or errors
5. Overall quality

Provide a confidence score (0-1) and list any issues or concerns.
Format your response as: CONFIDENCE: [score] | ISSUES: [comma-separated list]"""

            critical_response = await self.critical_agent.run_async(critical_prompt)

            # Parse critical agent response
            confidence, issues = self._parse_critical_response(critical_response)

            return {
                "confidence": confidence,
                "issues": issues,
                "critical_response": critical_response
            }

        except Exception as e:
            logger.error(f"Critical analysis failed: {e}")
            return {
                "confidence": 0.5,
                "issues": [f"Critical analysis error: {str(e)}"],
                "critical_response": None
            }

    def _parse_critical_response(self, response: str) -> Tuple[float, List[str]]:
        """Parse the critical agent's response."""
        try:
            # Look for confidence score
            if "CONFIDENCE:" in response:
                conf_part = response.split("CONFIDENCE:")[1].split("|")[0].strip()
                confidence = float(conf_part)
            else:
                confidence = 0.5

            # Look for issues
            issues = []
            if "ISSUES:" in response:
                issues_part = response.split("ISSUES:")[1].strip()
                if issues_part and issues_part.lower() != "none":
                    issues = [issue.strip() for issue in issues_part.split(",")]

            return confidence, issues

        except Exception as e:
            logger.error(f"Failed to parse critical response: {e}")
            return 0.5, ["Failed to parse critical analysis"]

    async def _cross_validate(self, answer_context: AnswerContext,
                            cross_check_agents: List[Any]) -> Dict[str, Any]:
        """Cross-validate answer with other agents."""
        self.cross_validations_performed += 1

        if not cross_check_agents:
            return {"confidence": 0.5, "issues": ["No cross-check agents available"]}

        try:
            # Ask other agents the same question
            cross_check_answers = []

            for agent in cross_check_agents[:3]:  # Limit to 3 agents for efficiency
                try:
                    cross_answer = await agent.run_async(answer_context.original_question)
                    cross_check_answers.append(cross_answer)
                except Exception as e:
                    logger.warning(f"Cross-check agent failed: {e}")

            if not cross_check_answers:
                return {"confidence": 0.5, "issues": ["All cross-check agents failed"]}

            # Compare answers
            similarity_scores = []
            issues = []

            original_answer = answer_context.answer.lower()

            for i, cross_answer in enumerate(cross_check_answers):
                similarity = self._calculate_answer_similarity(original_answer, cross_answer.lower())
                similarity_scores.append(similarity)

                if similarity < 0.3:
                    issues.append(f"Low similarity with cross-check agent {i+1}")

            # Calculate overall cross-validation confidence
            avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0.0
            confidence = avg_similarity

            # If answers are too dissimilar, flag as potential issue
            if avg_similarity < 0.4:
                issues.append("Low consensus among agents")

            return {
                "confidence": confidence,
                "issues": issues,
                "similarity_scores": similarity_scores,
                "cross_check_count": len(cross_check_answers),
                "average_similarity": avg_similarity
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {
                "confidence": 0.5,
                "issues": [f"Cross-validation error: {str(e)}"],
                "cross_check_count": 0
            }

    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers (simplified)."""
        words1 = set(answer1.split())
        words2 = set(answer2.split())

        if not words1 and not words2:
            return 1.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    async def _verify_facts(self, answer_context: AnswerContext) -> Dict[str, Any]:
        """Verify factual claims in the answer."""
        # This would typically involve external fact-checking APIs
        # For now, we'll do basic heuristic checking

        issues = []
        confidence = 0.7  # Default confidence for factual verification

        answer = answer_context.answer.lower()

        # Check for specific claims that might be verifiable
        specific_numbers = len([word for word in answer.split() if word.isdigit()])
        specific_dates = len([word for word in answer.split() if any(year in word for year in ["2020", "2021", "2022", "2023", "2024"])])

        if specific_numbers > 0 or specific_dates > 0:
            # Answer contains specific claims - higher scrutiny needed
            if "according to" not in answer and "source" not in answer:
                issues.append("Specific claims without cited sources")
                confidence *= 0.8

        return {
            "confidence": confidence,
            "issues": issues,
            "specific_numbers": specific_numbers,
            "specific_dates": specific_dates
        }

    def _calculate_trust_level(self, agent_id: str, current_confidence: float) -> float:
        """Calculate trust level for an agent based on history."""
        if agent_id not in self.agent_reliability_scores:
            return 0.5  # Default trust for new agents

        historical_reliability = self.agent_reliability_scores[agent_id]

        # Combine historical reliability with current confidence
        trust_level = (historical_reliability * 0.7) + (current_confidence * 0.3)

        return max(0.0, min(1.0, trust_level))

    def _update_agent_reliability(self, agent_id: str, confidence: float):
        """Update agent reliability score based on validation results."""
        if agent_id not in self.agent_reliability_scores:
            self.agent_reliability_scores[agent_id] = confidence
        else:
            # Exponential moving average
            current_score = self.agent_reliability_scores[agent_id]
            self.agent_reliability_scores[agent_id] = (current_score * 0.8) + (confidence * 0.2)

    def get_agent_reliability(self, agent_id: str) -> Optional[float]:
        """Get reliability score for an agent."""
        return self.agent_reliability_scores.get(agent_id)

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics and statistics."""
        approval_rate = (self.answers_approved / max(1, self.validations_performed))

        return {
            "validations_performed": self.validations_performed,
            "answers_approved": self.answers_approved,
            "answers_rejected": self.answers_rejected,
            "approval_rate": approval_rate,
            "cross_validations_performed": self.cross_validations_performed,
            "tracked_agents": len(self.agent_reliability_scores),
            "average_agent_reliability": statistics.mean(self.agent_reliability_scores.values()) if self.agent_reliability_scores else 0.0,
            "validation_history_size": len(self.validation_history)
        }

    def get_agent_rankings(self) -> List[Tuple[str, float]]:
        """Get agents ranked by reliability."""
        return sorted(self.agent_reliability_scores.items(), key=lambda x: x[1], reverse=True)