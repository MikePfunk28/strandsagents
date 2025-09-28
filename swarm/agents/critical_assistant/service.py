"""Critical Assistant microservice implementation."""

import asyncio
import logging
from typing import List, Any, Dict

from ...base_assistant import BaseAssistant
from .prompts import CRITICAL_ASSISTANT_PROMPT, RISK_ASSESSMENT_PROMPT, ASSUMPTION_TESTING_PROMPT
from .tools import get_critical_tools

logger = logging.getLogger(__name__)

class CriticalAssistant(BaseAssistant):
    """Critical Assistant specializing in analysis and evaluation.

    Uses 270M model for fast critical thinking tasks:
    - Risk assessment and mitigation
    - Assumption challenging
    - Quality evaluation
    - Logic analysis
    """

    def __init__(self, assistant_id: str, model_name: str = "gemma:270m",
                 host: str = "localhost:11434"):
        super().__init__(
            assistant_id=assistant_id,
            assistant_type="critical",
            capabilities=[
                "risk_assessment",
                "assumption_analysis",
                "critical_evaluation",
                "logic_checking",
                "bias_detection",
                "quality_assessment"
            ],
            model_name=model_name,
            host=host
        )

        # Critical thinking specific state
        self.evaluation_history = []
        self.risk_assessments = []
        self.assumption_database = []
        self.critical_frameworks = ["SWOT", "Risk-Benefit", "Assumption Testing", "Logic Chain"]

    def get_system_prompt(self) -> str:
        """Get critical assistant system prompt."""
        return CRITICAL_ASSISTANT_PROMPT

    def get_tools(self) -> List[Any]:
        """Get critical analysis tools."""
        return get_critical_tools()

    async def evaluate_critically(self, subject: str, context: dict = None,
                                framework: str = "comprehensive") -> dict:
        """Perform critical evaluation of ideas or proposals."""
        try:
            # Select evaluation approach based on framework
            if framework == "risk_focused":
                system_prompt = RISK_ASSESSMENT_PROMPT
                evaluation_prompt = f"Risk Assessment Focus: {subject}"
            elif framework == "assumption_testing":
                system_prompt = ASSUMPTION_TESTING_PROMPT
                evaluation_prompt = f"Assumption Testing Focus: {subject}"
            else:
                system_prompt = self.get_system_prompt()
                evaluation_prompt = self._build_critical_prompt(subject, context, framework)

            # Temporarily switch system prompt for focused analysis
            original_prompt = self.agent.system_prompt
            self.agent.system_prompt = system_prompt

            result = await self.agent.run_async(evaluation_prompt)

            # Restore original system prompt
            self.agent.system_prompt = original_prompt

            # Generate critical metrics
            critical_metrics = self._generate_critical_metrics(result, subject)

            # Store evaluation in history
            evaluation_entry = {
                "subject": subject,
                "framework": framework,
                "evaluation": result,
                "metrics": critical_metrics,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context
            }
            self.evaluation_history.append(evaluation_entry)

            # Limit history size
            if len(self.evaluation_history) > 50:
                self.evaluation_history = self.evaluation_history[-50:]

            return {
                "subject": subject,
                "framework": framework,
                "evaluation": result,
                "critical_score": critical_metrics["critical_score"],
                "risk_level": critical_metrics["risk_level"],
                "assumptions_identified": critical_metrics["assumptions_count"],
                "assistant_type": "critical"
            }

        except Exception as e:
            logger.error(f"Critical evaluation failed: {e}")
            return {
                "subject": subject,
                "error": str(e),
                "assistant_type": "critical"
            }

    def _build_critical_prompt(self, subject: str, context: dict, framework: str) -> str:
        """Build enhanced critical evaluation prompt."""
        base_prompt = f"Subject for Critical Analysis: {subject}\n\n"

        if context:
            base_prompt += f"Context: {context}\n\n"

        framework_instructions = {
            "comprehensive": "Provide comprehensive critical analysis covering strengths, weaknesses, risks, and assumptions",
            "swot": "Analyze using SWOT framework: Strengths, Weaknesses, Opportunities, Threats",
            "risk_benefit": "Focus on risk-benefit analysis with detailed trade-offs",
            "logic_chain": "Evaluate logical consistency and identify any logical fallacies",
            "stakeholder": "Analyze from multiple stakeholder perspectives"
        }

        instruction = framework_instructions.get(framework, framework_instructions["comprehensive"])
        base_prompt += f"Analysis Framework: {instruction}\n\n"

        # Add critical thinking guidelines
        base_prompt += """Critical Analysis Guidelines:
1. Question assumptions explicitly
2. Identify potential biases
3. Consider alternative perspectives
4. Assess evidence quality
5. Evaluate logical consistency
6. Identify risks and limitations
7. Suggest improvements

Provide structured, evidence-based analysis."""

        return base_prompt

    def _generate_critical_metrics(self, evaluation: str, subject: str) -> dict:
        """Generate metrics for critical evaluation quality."""
        evaluation_lower = evaluation.lower()

        # Critical thinking indicators
        critical_indicators = [
            "however", "but", "although", "despite", "nevertheless",
            "assumption", "risk", "limitation", "weakness", "problem",
            "alternative", "consider", "question", "challenge", "evidence"
        ]

        # Quality indicators
        quality_indicators = [
            "analysis", "evaluation", "assessment", "examination",
            "systematic", "structured", "evidence-based", "logical"
        ]

        # Count indicators
        critical_count = sum(1 for indicator in critical_indicators if indicator in evaluation_lower)
        quality_count = sum(1 for indicator in quality_indicators if indicator in evaluation_lower)

        # Assumption detection
        assumption_patterns = ["assume", "assumption", "take for granted", "given that", "presume"]
        assumptions_count = sum(evaluation_lower.count(pattern) for pattern in assumption_patterns)

        # Risk assessment
        risk_patterns = ["risk", "danger", "threat", "problem", "issue", "concern", "vulnerability"]
        risk_count = sum(evaluation_lower.count(pattern) for pattern in risk_patterns)

        # Calculate scores
        critical_score = min(1.0, (critical_count + quality_count) / 20)  # Normalize to 0-1
        risk_level = "HIGH" if risk_count >= 5 else "MEDIUM" if risk_count >= 2 else "LOW"

        return {
            "critical_score": critical_score,
            "risk_level": risk_level,
            "assumptions_count": assumptions_count,
            "critical_indicators": critical_count,
            "quality_indicators": quality_count,
            "risk_indicators": risk_count
        }

    async def assess_collaboration_risks(self, collaboration_plan: str,
                                       participating_agents: List[str]) -> dict:
        """Assess risks in agent collaboration plans."""
        risk_prompt = f"""Collaboration Risk Assessment:

Plan: {collaboration_plan}
Participating Agents: {', '.join(participating_agents)}

Assess risks including:
- Communication breakdowns
- Task dependencies and bottlenecks
- Conflicting objectives
- Resource contention
- Quality control issues
- Timeline risks

Provide structured risk assessment with mitigation strategies."""

        result = await self.evaluate_critically(
            risk_prompt,
            {"type": "collaboration_assessment", "agents": participating_agents},
            "risk_focused"
        )

        return {
            "collaboration_plan": collaboration_plan,
            "participating_agents": participating_agents,
            "risk_assessment": result,
            "risk_level": result.get("risk_level", "UNKNOWN"),
            "mitigation_needed": result.get("risk_level") in ["HIGH", "MEDIUM"]
        }

    async def challenge_assumptions(self, statement: str, domain: str = "general") -> dict:
        """Challenge assumptions in statements or proposals."""
        challenge_prompt = f"""Assumption Challenge:

Statement: {statement}
Domain: {domain}

Systematically identify and challenge ALL underlying assumptions:
1. Explicit assumptions (clearly stated)
2. Implicit assumptions (hidden/unstated)
3. Meta-assumptions (about context/environment)
4. Domain-specific assumptions

For each assumption:
- State the assumption clearly
- Question its validity
- Identify what evidence would be needed
- Consider alternative possibilities"""

        result = await self.evaluate_critically(
            challenge_prompt,
            {"type": "assumption_challenge", "domain": domain},
            "assumption_testing"
        )

        # Extract assumptions for database
        assumptions_found = result.get("assumptions_identified", 0)
        if assumptions_found > 0:
            self.assumption_database.append({
                "statement": statement,
                "domain": domain,
                "assumptions_count": assumptions_found,
                "timestamp": asyncio.get_event_loop().time()
            })

        return {
            "statement": statement,
            "domain": domain,
            "assumption_analysis": result,
            "assumptions_found": assumptions_found,
            "validity_questions_raised": assumptions_found * 2  # Estimate
        }

    async def collaborate_critically(self, other_agents: List[str],
                                   evaluation_subject: str) -> dict:
        """Collaborate with other agents on critical evaluation."""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        collaboration_id = await self.mcp_client.request_collaboration(
            other_agents,
            f"Critical evaluation collaboration: {evaluation_subject}"
        )

        # Perform initial critical analysis
        initial_evaluation = await self.evaluate_critically(
            evaluation_subject,
            {"collaboration": True, "agents": other_agents},
            "comprehensive"
        )

        return {
            "collaboration_id": collaboration_id,
            "initial_evaluation": initial_evaluation,
            "evaluation_subject": evaluation_subject,
            "collaborating_agents": other_agents,
            "critical_focus": "risks, assumptions, and quality assessment"
        }

    def get_critical_stats(self) -> dict:
        """Get critical thinking performance statistics."""
        total_evaluations = len(self.evaluation_history)
        avg_critical_score = sum(
            eval_entry["metrics"]["critical_score"]
            for eval_entry in self.evaluation_history
        ) / max(1, total_evaluations)

        risk_levels = [eval_entry["metrics"]["risk_level"] for eval_entry in self.evaluation_history]
        high_risk_count = risk_levels.count("HIGH")

        return {
            "evaluations_performed": total_evaluations,
            "assumptions_challenged": len(self.assumption_database),
            "average_critical_score": avg_critical_score,
            "high_risk_evaluations": high_risk_count,
            "frameworks_used": list(set(eval_entry["framework"] for eval_entry in self.evaluation_history)),
            "capabilities": self.capabilities,
            "model": self.model_name
        }

# Factory function for creating critical assistant
def create_critical_assistant(assistant_id: str = None) -> CriticalAssistant:
    """Create a critical assistant instance."""
    if assistant_id is None:
        import uuid
        assistant_id = f"critical_{str(uuid.uuid4())[:8]}"

    return CriticalAssistant(assistant_id)

# Example usage and testing
async def demo_critical_assistant():
    """Demonstrate critical assistant functionality."""
    print("Critical Assistant Demo")
    print("=" * 30)

    assistant = create_critical_assistant("critical_demo_001")

    try:
        await assistant.start_service()

        # Test critical evaluation
        print("\n1. Critical Evaluation:")
        result = await assistant.evaluate_critically(
            "All employees should work from home permanently to increase productivity",
            {"domain": "workplace_policy"},
            "comprehensive"
        )
        print(f"Critical Score: {result.get('critical_score', 0):.2f}")
        print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")

        # Test assumption challenging
        print("\n2. Assumption Challenge:")
        assumption_result = await assistant.challenge_assumptions(
            "AI will replace most human jobs within 10 years",
            "technology"
        )
        print(f"Assumptions Found: {assumption_result.get('assumptions_found', 0)}")

        # Test collaboration risk assessment
        print("\n3. Collaboration Risk Assessment:")
        collab_risk = await assistant.assess_collaboration_risks(
            "Multiple AI agents working together on complex coding tasks",
            ["research_001", "creative_001", "critical_001"]
        )
        print(f"Risk Level: {collab_risk.get('risk_level', 'UNKNOWN')}")
        print(f"Mitigation Needed: {collab_risk.get('mitigation_needed', False)}")

        # Show critical stats
        stats = assistant.get_critical_stats()
        print(f"\nCritical Thinking Stats: {stats}")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await assistant.stop_service()

if __name__ == "__main__":
    asyncio.run(demo_critical_assistant())