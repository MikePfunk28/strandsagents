"""Critical Assistant microservice package."""

from .service import CriticalAssistant, create_critical_assistant
from .prompts import CRITICAL_ASSISTANT_PROMPT
from .tools import get_critical_tools

__all__ = ["CriticalAssistant", "create_critical_assistant", "CRITICAL_ASSISTANT_PROMPT", "get_critical_tools"]