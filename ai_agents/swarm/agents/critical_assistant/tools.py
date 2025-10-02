"""Specialized tools for the Critical Assistant."""

from typing import Any, List, Dict
from strands.types.tool_types import ToolUse, ToolResult
import re

# Risk assessment tool
RISK_ASSESSMENT_SPEC = {
    "name": "risk_assessment",
    "description": "Assess risks and potential problems in ideas or plans",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "proposal": {
                    "type": "string",
                    "description": "Proposal, idea, or plan to assess"
                },
                "context": {
                    "type": "string",
                    "description": "Context or domain for the assessment"
                },
                "risk_categories": {
                    "type": "string",
                    "description": "Specific risk categories to focus on",
                    "default": "all"
                }
            },
            "required": ["proposal"]
        }
    }
}

def risk_assessment(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Assess risks in proposals or plans."""
    tool_use_id = tool_use["toolUseId"]
    proposal = tool_use["input"]["proposal"]
    context = tool_use["input"].get("context", "")
    risk_categories = tool_use["input"].get("risk_categories", "all")

    try:
        # Risk category frameworks
        risk_types = {
            "technical": ["implementation complexity", "scalability issues", "technology dependencies"],
            "operational": ["resource requirements", "workflow disruption", "maintenance overhead"],
            "financial": ["cost overruns", "ROI uncertainty", "budget constraints"],
            "timeline": ["delayed delivery", "scope creep", "dependency bottlenecks"],
            "user": ["adoption resistance", "usability issues", "training requirements"],
            "security": ["data breaches", "access control", "compliance violations"],
            "strategic": ["misalignment", "opportunity cost", "competitive disadvantage"]
        }

        # Risk indicators in text
        high_risk_indicators = [
            "completely new", "never done before", "complex integration",
            "tight deadline", "limited resources", "unproven technology",
            "major change", "all or nothing", "dependencies"
        ]

        medium_risk_indicators = [
            "some experience", "partial solution", "workaround needed",
            "requires training", "moderate change", "external dependency"
        ]

        assessed_risks = []
        proposal_lower = proposal.lower()

        # Assess each risk category
        categories_to_check = list(risk_types.keys()) if risk_categories == "all" else [risk_categories]

        for category in categories_to_check:
            if category in risk_types:
                category_risks = []

                for risk in risk_types[category]:
                    # Simple risk detection based on keywords and patterns
                    risk_level = "LOW"

                    # Check for high-risk indicators
                    for indicator in high_risk_indicators:
                        if indicator in proposal_lower:
                            risk_level = "HIGH"
                            break

                    # Check for medium-risk indicators if not high
                    if risk_level == "LOW":
                        for indicator in medium_risk_indicators:
                            if indicator in proposal_lower:
                                risk_level = "MEDIUM"
                                break

                    category_risks.append({
                        "risk": risk,
                        "level": risk_level,
                        "category": category
                    })

                assessed_risks.extend(category_risks)

        # Generate risk assessment report
        result_text = f"Risk Assessment for: {proposal}\n\n"

        if context:
            result_text += f"Context: {context}\n\n"

        # Group by risk level
        high_risks = [r for r in assessed_risks if r["level"] == "HIGH"]
        medium_risks = [r for r in assessed_risks if r["level"] == "MEDIUM"]
        low_risks = [r for r in assessed_risks if r["level"] == "LOW"]

        if high_risks:
            result_text += "HIGH RISK AREAS:\n"
            for risk in high_risks[:5]:  # Limit to top 5
                result_text += f"• {risk['risk'].title()} ({risk['category']})\n"
            result_text += "\n"

        if medium_risks:
            result_text += "MEDIUM RISK AREAS:\n"
            for risk in medium_risks[:3]:  # Limit to top 3
                result_text += f"• {risk['risk'].title()} ({risk['category']})\n"
            result_text += "\n"

        # Overall risk score
        total_risks = len(assessed_risks)
        high_count = len(high_risks)
        medium_count = len(medium_risks)

        risk_score = (high_count * 3 + medium_count * 2) / (total_risks * 3) if total_risks > 0 else 0

        result_text += f"OVERALL RISK LEVEL: {('LOW' if risk_score < 0.3 else 'MEDIUM' if risk_score < 0.7 else 'HIGH')}\n"
        result_text += f"Risk Score: {risk_score:.2f} (0=low, 1=high)\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Risk assessment error: {str(e)}"}]
        }

# Assumption analysis tool
ASSUMPTION_ANALYSIS_SPEC = {
    "name": "assumption_analysis",
    "description": "Identify and analyze underlying assumptions in statements or plans",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Statement, plan, or argument to analyze"
                },
                "focus_area": {
                    "type": "string",
                    "description": "Specific focus area for assumption analysis",
                    "default": "all"
                }
            },
            "required": ["statement"]
        }
    }
}

def assumption_analysis(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Analyze assumptions in statements or plans."""
    tool_use_id = tool_use["toolUseId"]
    statement = tool_use["input"]["statement"]
    focus_area = tool_use["input"].get("focus_area", "all")

    try:
        # Common assumption patterns
        assumption_patterns = {
            "causal": r"(will cause|leads to|results in|because of)",
            "universal": r"(all|every|always|never|everyone|no one)",
            "predictive": r"(will be|going to|expect|predict|forecast)",
            "comparative": r"(better than|worse than|more|less|superior|inferior)",
            "necessity": r"(must|have to|need to|required|essential|necessary)",
            "capability": r"(can|able to|possible|feasible|easy|simple)"
        }

        # Assumption indicators
        assumption_keywords = [
            "obviously", "clearly", "naturally", "of course", "certainly",
            "should", "would", "could", "might", "probably", "likely"
        ]

        identified_assumptions = []
        statement_lower = statement.lower()

        # Pattern-based assumption detection
        for pattern_type, pattern in assumption_patterns.items():
            matches = re.findall(pattern, statement_lower)
            if matches:
                for match in matches:
                    identified_assumptions.append({
                        "type": pattern_type,
                        "text": match,
                        "assumption": f"Assumes {pattern_type} relationship is valid",
                        "question": f"What if the {pattern_type} assumption doesn't hold?"
                    })

        # Keyword-based assumption detection
        for keyword in assumption_keywords:
            if keyword in statement_lower:
                identified_assumptions.append({
                    "type": "implicit",
                    "text": keyword,
                    "assumption": f"Contains implicit certainty: '{keyword}'",
                    "question": f"What evidence supports this level of certainty?"
                })

        # Logical structure analysis
        if "if" in statement_lower and "then" in statement_lower:
            identified_assumptions.append({
                "type": "conditional",
                "text": "if-then logic",
                "assumption": "Assumes conditional relationship is complete",
                "question": "Are there other conditions or exceptions not considered?"
            })

        # Generate assumption analysis report
        result_text = f"Assumption Analysis for: {statement}\n\n"

        if identified_assumptions:
            result_text += "IDENTIFIED ASSUMPTIONS:\n"
            for i, assumption in enumerate(identified_assumptions[:8], 1):
                result_text += f"{i}. Type: {assumption['type'].title()}\n"
                result_text += f"   Text: '{assumption['text']}'\n"
                result_text += f"   Assumption: {assumption['assumption']}\n"
                result_text += f"   Critical Question: {assumption['question']}\n\n"
        else:
            result_text += "No explicit assumptions detected. Consider:\n"
            result_text += "• What background knowledge is assumed?\n"
            result_text += "• What conditions are taken for granted?\n"
            result_text += "• What alternatives are not considered?\n\n"

        # Meta-assumptions
        result_text += "POTENTIAL META-ASSUMPTIONS:\n"
        result_text += "• Current context will remain stable\n"
        result_text += "• All stakeholders share same priorities\n"
        result_text += "• Past patterns will continue\n"
        result_text += "• Resources and constraints are fixed\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Assumption analysis error: {str(e)}"}]
        }

# Critical evaluation tool
CRITICAL_EVALUATION_SPEC = {
    "name": "critical_evaluation",
    "description": "Provide comprehensive critical evaluation of ideas or solutions",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Subject to evaluate critically"
                },
                "criteria": {
                    "type": "string",
                    "description": "Evaluation criteria or standards",
                    "default": "effectiveness,feasibility,sustainability"
                },
                "perspective": {
                    "type": "string",
                    "description": "Evaluation perspective (stakeholder, technical, etc.)",
                    "default": "balanced"
                }
            },
            "required": ["subject"]
        }
    }
}

def critical_evaluation(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Provide critical evaluation of ideas or solutions."""
    tool_use_id = tool_use["toolUseId"]
    subject = tool_use["input"]["subject"]
    criteria = tool_use["input"].get("criteria", "effectiveness,feasibility,sustainability")
    perspective = tool_use["input"].get("perspective", "balanced")

    try:
        # Evaluation frameworks
        evaluation_criteria = {
            "effectiveness": "How well does this achieve stated goals?",
            "feasibility": "How realistic is implementation?",
            "sustainability": "Is this maintainable long-term?",
            "efficiency": "Does this optimize resource usage?",
            "scalability": "Can this grow and adapt?",
            "usability": "Is this user-friendly and accessible?",
            "reliability": "How dependable and consistent is this?",
            "security": "Are risks and vulnerabilities addressed?",
            "cost": "Is the cost-benefit ratio favorable?",
            "innovation": "How novel and differentiated is this?"
        }

        # Parse criteria
        criteria_list = [c.strip() for c in criteria.split(",")]

        # Evaluation structure
        result_text = f"Critical Evaluation: {subject}\n"
        result_text += f"Perspective: {perspective.title()}\n\n"

        # Detailed evaluation by criteria
        total_score = 0
        max_score = len(criteria_list) * 10

        for criterion in criteria_list:
            if criterion in evaluation_criteria:
                result_text += f"[{criterion.upper()}]\n"
                result_text += f"Question: {evaluation_criteria[criterion]}\n"

                # Simple scoring based on subject content analysis
                score = 5  # baseline neutral
                subject_lower = subject.lower()

                # Positive indicators
                positive_words = ["effective", "efficient", "sustainable", "reliable", "secure", "scalable"]
                for word in positive_words:
                    if word in subject_lower:
                        score += 1

                # Negative indicators
                negative_words = ["complex", "expensive", "risky", "difficult", "unstable", "limited"]
                for word in negative_words:
                    if word in subject_lower:
                        score -= 1

                score = max(1, min(10, score))  # Keep in 1-10 range
                total_score += score

                result_text += f"Score: {score}/10\n"
                result_text += f"Assessment: {_score_to_assessment(score)}\n\n"

        # Overall evaluation
        overall_score = (total_score / max_score) * 100
        result_text += f"OVERALL EVALUATION:\n"
        result_text += f"Score: {overall_score:.1f}%\n"
        result_text += f"Rating: {_score_to_rating(overall_score)}\n\n"

        # Strengths and weaknesses
        result_text += "KEY STRENGTHS:\n"
        result_text += "• [Identify based on high-scoring criteria]\n"
        result_text += "• [Look for positive indicators]\n\n"

        result_text += "KEY WEAKNESSES:\n"
        result_text += "• [Identify based on low-scoring criteria]\n"
        result_text += "• [Look for negative indicators]\n\n"

        result_text += "RECOMMENDATIONS:\n"
        result_text += "• Address identified weaknesses\n"
        result_text += "• Leverage strengths more effectively\n"
        result_text += "• Consider alternative approaches for weak areas\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Critical evaluation error: {str(e)}"}]
        }

def _score_to_assessment(score: int) -> str:
    """Convert numeric score to assessment text."""
    if score >= 8:
        return "Excellent - exceeds expectations"
    elif score >= 6:
        return "Good - meets expectations"
    elif score >= 4:
        return "Fair - some concerns"
    else:
        return "Poor - significant issues"

def _score_to_rating(score: float) -> str:
    """Convert percentage score to rating."""
    if score >= 80:
        return "Highly Recommended"
    elif score >= 60:
        return "Recommended with modifications"
    elif score >= 40:
        return "Needs significant improvement"
    else:
        return "Not recommended in current form"

# Get all critical tools
def get_critical_tools() -> List[Dict[str, Any]]:
    """Get all tools available to the critical assistant."""
    return [
        {
            "spec": RISK_ASSESSMENT_SPEC,
            "function": risk_assessment
        },
        {
            "spec": ASSUMPTION_ANALYSIS_SPEC,
            "function": assumption_analysis
        },
        {
            "spec": CRITICAL_EVALUATION_SPEC,
            "function": critical_evaluation
        }
    ]

# Tool registry for easy loading
CRITICAL_TOOLS = {
    "risk_assessment": risk_assessment,
    "assumption_analysis": assumption_analysis,
    "critical_evaluation": critical_evaluation
}