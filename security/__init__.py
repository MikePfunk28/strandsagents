"""Security and verification layer for swarm and graph systems.

This module provides comprehensive security mechanisms including:
- Agent authentication and authorization
- Message integrity verification
- Answer validation and cross-checking
- Central reporting with audit trails
- Anomaly detection for rogue agents

Example:
    from security import SecurityManager, AgentAuthenticator, AnswerValidator

    security = SecurityManager()
    await security.initialize()

    # Authenticate agent
    token = await security.authenticate_agent(agent_id, agent_type)

    # Validate message
    is_valid = await security.verify_message(message, token)

    # Validate answer
    confidence = await security.validate_answer(answer, context)
"""

from .security_manager import SecurityManager
from .agent_authenticator import AgentAuthenticator
from .message_verifier import MessageVerifier
from .answer_validator import AnswerValidator
from .secure_reporter import SecureReporter

__version__ = "1.0.0"
__all__ = [
    "SecurityManager",
    "AgentAuthenticator",
    "MessageVerifier",
    "AnswerValidator",
    "SecureReporter"
]