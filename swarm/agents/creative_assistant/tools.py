"""Specialized tools for the Creative Assistant."""

from typing import Any, List, Dict
from strands.types.tool_types import ToolUse, ToolResult
import json
import random
import re

# Idea generation tool
IDEA_GENERATION_SPEC = {
    "name": "idea_generation",
    "description": "Generate creative ideas based on prompts and constraints",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Creative prompt or challenge"
                },
                "constraints": {
                    "type": "string",
                    "description": "Optional constraints or parameters"
                },
                "idea_count": {
                    "type": "integer",
                    "description": "Number of ideas to generate",
                    "default": 5
                }
            },
            "required": ["prompt"]
        }
    }
}

def idea_generation(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Generate creative ideas based on prompts."""
    tool_use_id = tool_use["toolUseId"]
    prompt = tool_use["input"]["prompt"]
    constraints = tool_use["input"].get("constraints", "")
    idea_count = tool_use["input"].get("idea_count", 5)

    try:
        # Creative idea generation templates
        idea_starters = [
            "What if we combined",
            "Imagine a world where",
            "A revolutionary approach could be",
            "An unexpected solution might involve",
            "Drawing inspiration from nature",
            "Using technology in a new way",
            "Flipping the traditional approach",
            "A sustainable method could be"
        ]

        # Domain combinations for cross-pollination
        domains = [
            "nature", "technology", "art", "sports", "music", "architecture",
            "cooking", "gaming", "travel", "science", "history", "fashion"
        ]

        ideas = []
        for i in range(idea_count):
            starter = random.choice(idea_starters)
            domain = random.choice(domains)

            # Generate idea framework
            if i % 2 == 0:
                idea_template = f"{starter} {domain} concepts to address: {prompt}"
            else:
                idea_template = f"Inspired by {domain}: {starter.lower()} applied to {prompt}"

            ideas.append({
                "id": i + 1,
                "idea": idea_template,
                "domain_inspiration": domain,
                "approach": starter
            })

        # Add constraints consideration
        constraint_note = ""
        if constraints:
            constraint_note = f"\n\nConstraints to consider: {constraints}"

        result_text = f"Generated {len(ideas)} creative ideas for: {prompt}\n\n"
        for idea in ideas:
            result_text += f"{idea['id']}. {idea['idea']}\n"
            result_text += f"   (Inspired by: {idea['domain_inspiration']})\n\n"

        result_text += constraint_note

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Idea generation error: {str(e)}"}]
        }

# Creative combination tool
CREATIVE_COMBINATION_SPEC = {
    "name": "creative_combination",
    "description": "Combine different concepts creatively",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "concept_a": {
                    "type": "string",
                    "description": "First concept to combine"
                },
                "concept_b": {
                    "type": "string",
                    "description": "Second concept to combine"
                },
                "combination_style": {
                    "type": "string",
                    "description": "Style of combination (fusion, hybrid, inspired_by, etc.)",
                    "default": "fusion"
                }
            },
            "required": ["concept_a", "concept_b"]
        }
    }
}

def creative_combination(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Combine concepts creatively."""
    tool_use_id = tool_use["toolUseId"]
    concept_a = tool_use["input"]["concept_a"]
    concept_b = tool_use["input"]["concept_b"]
    style = tool_use["input"].get("combination_style", "fusion")

    try:
        combination_methods = {
            "fusion": f"A seamless blend of {concept_a} and {concept_b}",
            "hybrid": f"A hybrid that takes the best of {concept_a} and {concept_b}",
            "inspired_by": f"{concept_a} reimagined through the lens of {concept_b}",
            "opposite": f"{concept_a} designed to complement and contrast with {concept_b}",
            "evolution": f"{concept_a} evolved using principles from {concept_b}",
            "disruption": f"{concept_b} disrupting traditional {concept_a} approaches"
        }

        # Generate multiple combination ideas
        combinations = []

        for method, description in combination_methods.items():
            if method == style or style == "all":
                combinations.append({
                    "method": method,
                    "description": description,
                    "example": f"Example: {concept_a.title()}-{concept_b.title()} {method.title()}"
                })

        result_text = f"Creative combinations of '{concept_a}' and '{concept_b}':\n\n"

        for combo in combinations:
            result_text += f"• {combo['method'].title()}: {combo['description']}\n"
            result_text += f"  {combo['example']}\n\n"

        # Add innovation potential
        result_text += f"Innovation potential: High - combining {concept_a} with {concept_b} opens new possibilities"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Creative combination error: {str(e)}"}]
        }

# Problem reframing tool
PROBLEM_REFRAMING_SPEC = {
    "name": "problem_reframing",
    "description": "Reframe problems from different creative perspectives",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "Problem statement to reframe"
                },
                "perspective": {
                    "type": "string",
                    "description": "Perspective to use (user, system, future, past, etc.)",
                    "default": "multiple"
                }
            },
            "required": ["problem"]
        }
    }
}

def problem_reframing(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Reframe problems from different perspectives."""
    tool_use_id = tool_use["toolUseId"]
    problem = tool_use["input"]["problem"]
    perspective = tool_use["input"].get("perspective", "multiple")

    try:
        reframing_lenses = {
            "user": f"From user perspective: How does {problem} affect the end user experience?",
            "system": f"From system perspective: What systemic issues contribute to {problem}?",
            "opportunity": f"As opportunity: What opportunities does {problem} create?",
            "resource": f"As resource constraint: What if {problem} is about resource allocation?",
            "communication": f"As communication: What if {problem} is about information flow?",
            "time": f"From time perspective: How does {problem} change over different timeframes?",
            "scale": f"From scale perspective: How does {problem} look at micro vs macro levels?",
            "inversion": f"By inversion: What if the opposite of {problem} was the issue?"
        }

        reframes = []

        if perspective == "multiple":
            for lens, reframe in reframing_lenses.items():
                reframes.append({
                    "lens": lens,
                    "reframe": reframe,
                    "insight": f"This perspective might reveal new solution approaches"
                })
        else:
            if perspective in reframing_lenses:
                reframes.append({
                    "lens": perspective,
                    "reframe": reframing_lenses[perspective],
                    "insight": f"Focused {perspective} perspective on the problem"
                })

        result_text = f"Creative reframing of: {problem}\n\n"

        for reframe in reframes:
            result_text += f"• {reframe['lens'].title()} Lens:\n"
            result_text += f"  {reframe['reframe']}\n"
            result_text += f"  Insight: {reframe['insight']}\n\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Problem reframing error: {str(e)}"}]
        }

# Get all creative tools
def get_creative_tools() -> List[Dict[str, Any]]:
    """Get all tools available to the creative assistant."""
    return [
        {
            "spec": IDEA_GENERATION_SPEC,
            "function": idea_generation
        },
        {
            "spec": CREATIVE_COMBINATION_SPEC,
            "function": creative_combination
        },
        {
            "spec": PROBLEM_REFRAMING_SPEC,
            "function": problem_reframing
        }
    ]

# Tool registry for easy loading
CREATIVE_TOOLS = {
    "idea_generation": idea_generation,
    "creative_combination": creative_combination,
    "problem_reframing": problem_reframing
}