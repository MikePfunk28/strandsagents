"""Specialized tools for the Summarizer Assistant."""

from typing import Any, List, Dict
from strands.types.tool_types import ToolUse, ToolResult
import re

# Text summarization tool
TEXT_SUMMARIZATION_SPEC = {
    "name": "text_summarization",
    "description": "Create structured summaries of text content",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to summarize"
                },
                "summary_type": {
                    "type": "string",
                    "description": "Type of summary (brief, detailed, bullet_points, executive)",
                    "default": "brief"
                },
                "focus_areas": {
                    "type": "string",
                    "description": "Specific areas to focus on in summary"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of summary in words",
                    "default": 200
                }
            },
            "required": ["content"]
        }
    }
}

def text_summarization(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Create structured summaries of text content."""
    tool_use_id = tool_use["toolUseId"]
    content = tool_use["input"]["content"]
    summary_type = tool_use["input"].get("summary_type", "brief")
    focus_areas = tool_use["input"].get("focus_areas", "")
    max_length = tool_use["input"].get("max_length", 200)

    try:
        # Content analysis
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        compression_ratio = max_length / word_count if word_count > 0 else 1

        # Extract key information patterns
        key_phrases = _extract_key_phrases(content)
        action_items = _extract_action_items(content)
        numbers_stats = _extract_numbers_and_stats(content)

        # Generate summary based on type
        if summary_type == "executive":
            summary = _create_executive_summary(content, key_phrases, max_length)
        elif summary_type == "bullet_points":
            summary = _create_bullet_summary(content, key_phrases, max_length)
        elif summary_type == "detailed":
            summary = _create_detailed_summary(content, key_phrases, numbers_stats, max_length)
        else:  # brief
            summary = _create_brief_summary(content, key_phrases, max_length)

        # Add focus areas if specified
        if focus_areas:
            focused_content = _extract_focused_content(content, focus_areas)
            summary += f"\n\nFOCUS: {focus_areas.title()}\n{focused_content}"

        # Add action items if found
        if action_items:
            summary += f"\n\nACTION ITEMS:\n"
            for item in action_items[:5]:  # Limit to 5 items
                summary += f"• {item}\n"

        # Summary metadata
        result_text = f"SUMMARY ({summary_type.upper()}):\n{summary}\n\n"
        result_text += f"METADATA:\n"
        result_text += f"• Original: {word_count} words, {sentence_count} sentences\n"
        result_text += f"• Summary: ~{len(summary.split())} words\n"
        result_text += f"• Compression: {compression_ratio:.1%}\n"
        result_text += f"• Key phrases: {len(key_phrases)}\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Summarization error: {str(e)}"}]
        }

def _extract_key_phrases(content: str) -> List[str]:
    """Extract key phrases from content."""
    # Simple key phrase extraction
    sentences = content.split('.')
    key_phrases = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Meaningful sentences
            # Look for important indicators
            if any(indicator in sentence.lower() for indicator in
                   ['important', 'key', 'significant', 'critical', 'main', 'primary']):
                key_phrases.append(sentence[:100])  # Truncate long sentences

    return key_phrases[:10]  # Limit to 10 key phrases

def _extract_action_items(content: str) -> List[str]:
    """Extract action items from content."""
    action_patterns = [
        r'(?:need to|must|should|will|action:)\s+([^.]+)',
        r'(?:todo|task|action item):\s*([^.]+)',
        r'(?:next steps?|follow up):\s*([^.]+)'
    ]

    action_items = []
    content_lower = content.lower()

    for pattern in action_patterns:
        matches = re.findall(pattern, content_lower)
        action_items.extend(matches)

    # Clean up action items
    cleaned_items = []
    for item in action_items:
        item = item.strip()
        if len(item) > 10 and len(item) < 150:  # Reasonable length
            cleaned_items.append(item)

    return cleaned_items[:10]  # Limit to 10 items

def _extract_numbers_and_stats(content: str) -> List[str]:
    """Extract numbers and statistics from content."""
    number_patterns = [
        r'\b\d+(?:\.\d+)?%\b',  # Percentages
        r'\$\d+(?:,\d+)*(?:\.\d+)?[KMB]?\b',  # Money
        r'\b\d+(?:,\d+)*\s+(?:users|customers|people|items)\b',  # Counts
        r'\b\d+(?:\.\d+)?\s*(?:hours?|days?|weeks?|months?|years?)\b'  # Time
    ]

    stats = []
    for pattern in number_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        stats.extend(matches)

    return list(set(stats))[:10]  # Unique stats, limit to 10

def _create_brief_summary(content: str, key_phrases: List[str], max_length: int) -> str:
    """Create brief summary."""
    # Take first few key phrases and combine
    summary_parts = key_phrases[:3]
    summary = ". ".join(summary_parts)

    # Truncate if too long
    words = summary.split()
    if len(words) > max_length:
        summary = " ".join(words[:max_length]) + "..."

    return summary

def _create_bullet_summary(content: str, key_phrases: List[str], max_length: int) -> str:
    """Create bullet point summary."""
    bullet_summary = ""
    word_count = 0

    for phrase in key_phrases:
        phrase_words = len(phrase.split())
        if word_count + phrase_words <= max_length:
            bullet_summary += f"• {phrase}\n"
            word_count += phrase_words
        else:
            break

    return bullet_summary.strip()

def _create_detailed_summary(content: str, key_phrases: List[str],
                           stats: List[str], max_length: int) -> str:
    """Create detailed summary with sections."""
    summary = "OVERVIEW:\n"
    summary += _create_brief_summary(content, key_phrases, max_length // 2)

    if stats:
        summary += "\n\nKEY METRICS:\n"
        for stat in stats[:5]:
            summary += f"• {stat}\n"

    if len(key_phrases) > 3:
        summary += "\n\nADDITIONAL DETAILS:\n"
        remaining_phrases = key_phrases[3:6]  # Next 3 phrases
        for phrase in remaining_phrases:
            summary += f"• {phrase}\n"

    return summary

def _create_executive_summary(content: str, key_phrases: List[str], max_length: int) -> str:
    """Create executive summary."""
    summary = "EXECUTIVE SUMMARY:\n"

    # Focus on business impact and decisions
    business_keywords = ['revenue', 'cost', 'efficiency', 'growth', 'decision', 'strategy']
    business_phrases = [phrase for phrase in key_phrases
                       if any(keyword in phrase.lower() for keyword in business_keywords)]

    if business_phrases:
        summary += "\n".join(business_phrases[:3])
    else:
        summary += _create_brief_summary(content, key_phrases, max_length)

    return summary

def _extract_focused_content(content: str, focus_areas: str) -> str:
    """Extract content related to focus areas."""
    focus_keywords = focus_areas.lower().split(',')
    focus_keywords = [keyword.strip() for keyword in focus_keywords]

    sentences = content.split('.')
    focused_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in focus_keywords):
            focused_sentences.append(sentence.strip())

    return ". ".join(focused_sentences[:3])  # Limit to 3 sentences

# Key extraction tool
KEY_EXTRACTION_SPEC = {
    "name": "key_extraction",
    "description": "Extract key points, decisions, and action items from content",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to extract key information from"
                },
                "extraction_type": {
                    "type": "string",
                    "description": "Type of extraction (decisions, actions, insights, metrics)",
                    "default": "all"
                }
            },
            "required": ["content"]
        }
    }
}

def key_extraction(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Extract key information from content."""
    tool_use_id = tool_use["toolUseId"]
    content = tool_use["input"]["content"]
    extraction_type = tool_use["input"].get("extraction_type", "all")

    try:
        extractions = {}

        if extraction_type in ["decisions", "all"]:
            decisions = _extract_decisions(content)
            extractions["decisions"] = decisions

        if extraction_type in ["actions", "all"]:
            actions = _extract_action_items(content)
            extractions["actions"] = actions

        if extraction_type in ["insights", "all"]:
            insights = _extract_insights(content)
            extractions["insights"] = insights

        if extraction_type in ["metrics", "all"]:
            metrics = _extract_numbers_and_stats(content)
            extractions["metrics"] = metrics

        # Format results
        result_text = f"KEY EXTRACTION ({extraction_type.upper()}):\n\n"

        for category, items in extractions.items():
            if items:
                result_text += f"{category.upper()}:\n"
                for item in items[:5]:  # Limit to 5 items per category
                    result_text += f"• {item}\n"
                result_text += "\n"

        if not any(extractions.values()):
            result_text += "No key items found in the specified categories.\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Key extraction error: {str(e)}"}]
        }

def _extract_decisions(content: str) -> List[str]:
    """Extract decisions from content."""
    decision_patterns = [
        r'(?:decided?|decision|determined|concluded|agreed)\s+(?:to\s+)?([^.]+)',
        r'(?:we will|will be|plan to|going to)\s+([^.]+)',
        r'(?:approved|rejected|selected|chosen)\s+([^.]+)'
    ]

    decisions = []
    content_lower = content.lower()

    for pattern in decision_patterns:
        matches = re.findall(pattern, content_lower)
        decisions.extend(matches)

    # Clean up decisions
    cleaned_decisions = []
    for decision in decisions:
        decision = decision.strip()
        if len(decision) > 10 and len(decision) < 200:
            cleaned_decisions.append(decision)

    return cleaned_decisions[:10]

def _extract_insights(content: str) -> List[str]:
    """Extract insights from content."""
    insight_patterns = [
        r'(?:insight|learning|discovery|finding|observation):\s*([^.]+)',
        r'(?:this shows|this indicates|this suggests)\s+([^.]+)',
        r'(?:interesting|notable|significant)\s+(?:that\s+)?([^.]+)'
    ]

    insights = []
    content_lower = content.lower()

    for pattern in insight_patterns:
        matches = re.findall(pattern, content_lower)
        insights.extend(matches)

    # Clean up insights
    cleaned_insights = []
    for insight in insights:
        insight = insight.strip()
        if len(insight) > 15 and len(insight) < 200:
            cleaned_insights.append(insight)

    return cleaned_insights[:10]

# Multi-source synthesis tool
SYNTHESIS_SPEC = {
    "name": "multi_source_synthesis",
    "description": "Synthesize information from multiple sources into coherent summary",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    },
                    "description": "Multiple sources to synthesize"
                },
                "synthesis_focus": {
                    "type": "string",
                    "description": "Focus area for synthesis",
                    "default": "comprehensive"
                }
            },
            "required": ["sources"]
        }
    }
}

def multi_source_synthesis(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Synthesize information from multiple sources."""
    tool_use_id = tool_use["toolUseId"]
    sources = tool_use["input"]["sources"]
    synthesis_focus = tool_use["input"].get("synthesis_focus", "comprehensive")

    try:
        if not sources or len(sources) < 2:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "At least 2 sources required for synthesis"}]
            }

        # Extract key information from each source
        source_summaries = []
        all_key_phrases = []
        all_metrics = []

        for source in sources:
            name = source.get("name", "Unknown")
            content = source.get("content", "")

            key_phrases = _extract_key_phrases(content)
            metrics = _extract_numbers_and_stats(content)

            source_summaries.append({
                "name": name,
                "key_phrases": key_phrases[:3],  # Top 3 phrases
                "metrics": metrics[:3]  # Top 3 metrics
            })

            all_key_phrases.extend(key_phrases)
            all_metrics.extend(metrics)

        # Find common themes
        common_themes = _find_common_themes(all_key_phrases)

        # Create synthesis
        result_text = f"MULTI-SOURCE SYNTHESIS ({synthesis_focus.upper()}):\n\n"

        # Common themes section
        if common_themes:
            result_text += "COMMON THEMES:\n"
            for theme in common_themes[:5]:
                result_text += f"• {theme}\n"
            result_text += "\n"

        # Source-specific insights
        result_text += "SOURCE INSIGHTS:\n"
        for summary in source_summaries:
            result_text += f"\n{summary['name']}:\n"
            for phrase in summary['key_phrases']:
                result_text += f"  • {phrase}\n"

        # Combined metrics
        if all_metrics:
            unique_metrics = list(set(all_metrics))
            result_text += f"\nCOMBINED METRICS:\n"
            for metric in unique_metrics[:8]:
                result_text += f"• {metric}\n"

        # Synthesis conclusion
        result_text += f"\nSYNTHESIS CONCLUSION:\n"
        result_text += f"Analyzed {len(sources)} sources with {len(common_themes)} common themes identified.\n"
        result_text += f"Key focus: {synthesis_focus}\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Synthesis error: {str(e)}"}]
        }

def _find_common_themes(phrases: List[str]) -> List[str]:
    """Find common themes across phrases."""
    # Simple word frequency analysis
    word_counts = {}

    for phrase in phrases:
        words = phrase.lower().split()
        for word in words:
            if len(word) > 4:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1

    # Find words that appear multiple times
    common_words = [word for word, count in word_counts.items() if count >= 2]

    # Create themes from common words
    themes = []
    for word in common_words[:10]:  # Top 10 common words
        # Find phrases containing this word
        related_phrases = [phrase for phrase in phrases if word in phrase.lower()]
        if len(related_phrases) >= 2:
            themes.append(f"{word.title()}: mentioned in {len(related_phrases)} sources")

    return themes

# Get all summarizer tools
def get_summarizer_tools() -> List[Dict[str, Any]]:
    """Get all tools available to the summarizer assistant."""
    return [
        {
            "spec": TEXT_SUMMARIZATION_SPEC,
            "function": text_summarization
        },
        {
            "spec": KEY_EXTRACTION_SPEC,
            "function": key_extraction
        },
        {
            "spec": SYNTHESIS_SPEC,
            "function": multi_source_synthesis
        }
    ]

# Tool registry for easy loading
SUMMARIZER_TOOLS = {
    "text_summarization": text_summarization,
    "key_extraction": key_extraction,
    "multi_source_synthesis": multi_source_synthesis
}