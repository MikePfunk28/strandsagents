"""
Text Processor Assistant

A focused assistant for basic text processing operations.
Demonstrates the assistant pattern with simple, specific functionality.
"""

import time
from typing import Any, Dict
from ..base_assistant import BaseAssistant, AssistantConfig


class TextProcessorAssistant(BaseAssistant):
    """
    Assistant specialized in text processing operations.

    Capabilities:
    - Text analysis and statistics
    - Basic text transformations
    - Content summarization
    - Text validation
    """

    def get_default_prompt(self) -> str:
        return """You are a Text Processing Assistant specialized in analyzing and manipulating text content.

Your capabilities include:
- Text analysis (length, word count, character analysis)
- Text transformations (uppercase, lowercase, capitalize)
- Content summarization (key points extraction)
- Text validation (format checking, content verification)

Always provide clear, structured responses with relevant statistics and insights."""

    async def execute_async(self, input_data: Any, **kwargs) -> Any:
        """
        Execute text processing operations.

        Args:
            input_data: Text to process or operation specification
            **kwargs: Additional parameters (operation, format, etc.)

        Returns:
            Processed text results with analysis
        """
        start_time = time.time()

        try:
            # Handle different input formats
            if isinstance(input_data, str):
                text = input_data
                operation = kwargs.get('operation', 'analyze')
            elif isinstance(input_data, dict):
                text = input_data.get('text', '')
                operation = input_data.get('operation', 'analyze')
            else:
                text = str(input_data)
                operation = kwargs.get('operation', 'analyze')

            # Perform the requested operation
            if operation == 'analyze':
                result = self._analyze_text(text)
            elif operation == 'summarize':
                result = self._summarize_text(text)
            elif operation == 'transform':
                transform_type = kwargs.get('transform_type', 'uppercase')
                result = self._transform_text(text, transform_type)
            elif operation == 'validate':
                result = self._validate_text(text)
            else:
                result = self._analyze_text(text)  # Default to analysis

            execution_time = time.time() - start_time
            self._record_execution(input_data, result, execution_time)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "error": str(e),
                "operation": operation,
                "execution_time": execution_time
            }
            self._record_execution(input_data, error_result, execution_time)
            return error_result

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return statistics."""
        lines = text.split('\n')
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])

        return {
            "operation": "analyze",
            "statistics": {
                "total_characters": len(text),
                "total_words": len(words),
                "total_lines": len(lines),
                "total_sentences": sentences,
                "average_words_per_sentence": sentences and len(words) / sentences,
                "average_characters_per_word": len(text) / len(words) if words else 0
            },
            "sample": text[:200] + "..." if len(text) > 200 else text
        }

    def _summarize_text(self, text: str) -> Dict[str, Any]:
        """Create a summary of the text."""
        words = text.split()
        total_words = len(words)

        # Simple extractive summarization - take first and last parts
        if total_words <= 50:
            summary = text
        else:
            first_part = ' '.join(words[:25])
            last_part = ' '.join(words[-25:]) if total_words > 50 else ''
            summary = f"{first_part} ... {last_part}" if last_part else first_part

        return {
            "operation": "summarize",
            "original_length": total_words,
            "summary_length": len(summary.split()),
            "summary": summary,
            "compression_ratio": len(summary.split()) / total_words if total_words > 0 else 0
        }

    def _transform_text(self, text: str, transform_type: str) -> Dict[str, Any]:
        """Transform text according to specified type."""
        if transform_type == 'uppercase':
            transformed = text.upper()
        elif transform_type == 'lowercase':
            transformed = text.lower()
        elif transform_type == 'capitalize':
            transformed = text.capitalize()
        elif transform_type == 'title':
            transformed = text.title()
        else:
            transformed = text  # No transformation

        return {
            "operation": "transform",
            "transform_type": transform_type,
            "original_text": text[:100] + "..." if len(text) > 100 else text,
            "transformed_text": transformed[:100] + "..." if len(transformed) > 100 else transformed
        }

    def _validate_text(self, text: str) -> Dict[str, Any]:
        """Validate text content and format."""
        validation_results = {
            "operation": "validate",
            "is_empty": len(text.strip()) == 0,
            "has_special_characters": any(ord(char) > 127 for char in text),
            "has_numbers": any(char.isdigit() for char in text),
            "has_punctuation": any(char in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for char in text),
            "estimated_language": "unknown"  # Could be enhanced with language detection
        }

        return validation_results


def create_text_processor_assistant(name: str = "text_processor") -> TextProcessorAssistant:
    """Factory function to create a text processor assistant."""
    config = AssistantConfig(
        name=name,
        description="Specialized assistant for text processing and analysis operations",
        model_id="llama3.2",  # Lightweight model for text processing
        system_prompt=None,  # Will use get_default_prompt()
        tools=[],  # No external tools needed for basic text processing
        metadata={
            "version": "1.0.0",
            "capabilities": ["analyze", "summarize", "transform", "validate"],
            "model_size": "lightweight"
        }
    )

    return TextProcessorAssistant(config)
