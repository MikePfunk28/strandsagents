"""Specialized tools for the Research Assistant."""

from typing import Any, List, Dict
from strands.types.tool_types import ToolUse, ToolResult
import os
import json
import re

# Document search tool
DOCUMENT_SEARCH_SPEC = {
    "name": "document_search",
    "description": "Search through local documents for specific information",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query terms"
                },
                "document_path": {
                    "type": "string",
                    "description": "Optional path to specific document"
                }
            },
            "required": ["query"]
        }
    }
}

def document_search(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Search through local documents for information."""
    tool_use_id = tool_use["toolUseId"]
    query = tool_use["input"]["query"]
    document_path = tool_use["input"].get("document_path", "")

    try:
        results = []
        search_terms = query.lower().split()

        # Search in specified document or current directory
        search_dir = document_path if document_path and os.path.exists(document_path) else "."

        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith(('.txt', '.md', '.py', '.json')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            if any(term in content for term in search_terms):
                                # Extract relevant snippet
                                lines = content.split('\n')
                                relevant_lines = []
                                for i, line in enumerate(lines):
                                    if any(term in line for term in search_terms):
                                        # Get context around match
                                        start = max(0, i-1)
                                        end = min(len(lines), i+2)
                                        relevant_lines.extend(lines[start:end])

                                results.append({
                                    "file": file_path,
                                    "snippet": "\n".join(relevant_lines[:5])  # Limit snippet size
                                })
                    except Exception:
                        continue  # Skip files that can't be read

        result_text = f"Found {len(results)} matches for '{query}':\n"
        for i, result in enumerate(results[:5]):  # Limit to 5 results
            result_text += f"{i+1}. {result['file']}: {result['snippet'][:100]}...\n"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Search error: {str(e)}"}]
        }

# Fact extraction tool
FACT_EXTRACTION_SPEC = {
    "name": "fact_extraction",
    "description": "Extract factual information from text",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to extract facts from"
                },
                "fact_types": {
                    "type": "string",
                    "description": "Types of facts to extract (numbers, dates, names, etc.)"
                }
            },
            "required": ["text"]
        }
    }
}

def fact_extraction(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Extract factual information from text."""
    tool_use_id = tool_use["toolUseId"]
    text = tool_use["input"]["text"]
    fact_types = tool_use["input"].get("fact_types", "all")

    try:
        facts = []

        # Extract numbers and percentages
        if "numbers" in fact_types or fact_types == "all":
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
            if numbers:
                facts.append(f"Numbers: {', '.join(numbers)}")

        # Extract dates
        if "dates" in fact_types or fact_types == "all":
            dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
            if dates:
                facts.append(f"Dates: {', '.join(dates)}")

        # Extract names (capitalized words)
        if "names" in fact_types or fact_types == "all":
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            unique_names = list(set(names))[:10]  # Limit to 10 unique names
            if unique_names:
                facts.append(f"Names: {', '.join(unique_names)}")

        # Extract sentences with factual indicators
        factual_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'contains', 'includes']
        sentences = text.split('.')
        factual_sentences = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in factual_indicators):
                factual_sentences.append(sentence.strip())

        if factual_sentences:
            facts.append(f"Key facts: {' | '.join(factual_sentences[:3])}")

        result_text = "Extracted facts:\n" + "\n".join(facts) if facts else "No clear facts found in text."

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Fact extraction error: {str(e)}"}]
        }

# Data verification tool
DATA_VERIFICATION_SPEC = {
    "name": "data_verification",
    "description": "Verify and cross-check data points",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "Claim or data point to verify"
                },
                "sources": {
                    "type": "string",
                    "description": "Additional sources or context for verification"
                }
            },
            "required": ["claim"]
        }
    }
}

def data_verification(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """Verify data points and claims."""
    tool_use_id = tool_use["toolUseId"]
    claim = tool_use["input"]["claim"]
    sources = tool_use["input"].get("sources", "")

    try:
        # Simple verification logic (would be more sophisticated in production)
        verification_result = {
            "claim": claim,
            "verification_status": "needs_verification",
            "confidence": "low",
            "notes": []
        }

        # Check for specific patterns that might indicate verifiable data
        if re.search(r'\d+%', claim):
            verification_result["notes"].append("Contains percentage - verify source")

        if re.search(r'\b\d{4}\b', claim):
            verification_result["notes"].append("Contains year - check if current")

        if any(word in claim.lower() for word in ['according to', 'study shows', 'research indicates']):
            verification_result["confidence"] = "medium"
            verification_result["notes"].append("References study - verify source credibility")

        if sources:
            verification_result["notes"].append(f"Additional context: {sources[:100]}")

        # Format result
        result_text = f"Verification of: {claim}\n"
        result_text += f"Status: {verification_result['verification_status']}\n"
        result_text += f"Confidence: {verification_result['confidence']}\n"
        if verification_result["notes"]:
            result_text += f"Notes: {'; '.join(verification_result['notes'])}"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result_text}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Verification error: {str(e)}"}]
        }

# Get all research tools
def get_research_tools() -> List[Dict[str, Any]]:
    """Get all tools available to the research assistant."""
    return [
        {
            "spec": DOCUMENT_SEARCH_SPEC,
            "function": document_search
        },
        {
            "spec": FACT_EXTRACTION_SPEC,
            "function": fact_extraction
        },
        {
            "spec": DATA_VERIFICATION_SPEC,
            "function": data_verification
        }
    ]

# Tool registry for easy loading
RESEARCH_TOOLS = {
    "document_search": document_search,
    "fact_extraction": fact_extraction,
    "data_verification": data_verification
}