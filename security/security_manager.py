"""Central security manager for coordinating all security components.

Provides unified security management for the swarm system including
authentication, message verification, answer validation, and secure reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .agent_authenticator import AgentAuthenticator, AgentCredentials
from .message_verifier import MessageVerifier, MessageSignature, VerificationResult
from .answer_validator import AnswerValidator, AnswerContext, ValidationResult, ValidationMethod
from .secure_reporter import SecureReporter, SecurityReport, AnomalyAlert, ReportType, Severity

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for monitoring and alerting."""
    event_id: str
    event_type: str
    severity: Severity
    agent_id: str
    timestamp: datetime
    description: str
    metadata: Dict[str, Any]

class SecurityManager:
    """Central security manager coordinating all security components."""

    def __init__(self, orchestrator_id: str = "orchestrator",
                 master_secret: Optional[str] = None,
                 critical_agent: Optional[Any] = None):
        """Initialize security manager.

        Args:
            orchestrator_id: ID of the central orchestrator
            master_secret: Master secret for cryptographic operations
            critical_agent: Critical thinking agent for answer validation
        """
        self.orchestrator_id = orchestrator_id

        # Initialize security components
        self.authenticator = AgentAuthenticator(master_secret)
        self.verifier = MessageVerifier(master_secret)
        self.validator = AnswerValidator(critical_agent)
        self.reporter = SecureReporter(orchestrator_id)

        # Security state
        self.security_events: List[SecurityEvent] = []
        self.threat_level = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
        self.active_threats: Dict[str, List[str]] = {}

        # Security metrics
        self.security_checks_performed = 0
        self.security_violations_detected = 0
        self.blocked_operations = 0

        # Agent whitelist for high-security mode
        self.trusted_agents: set = set()
        self.high_security_mode = False

        # Anomaly detection threshold
        self.anomaly_threshold = 0.8

    async def initialize_security(self) -> bool:
        """Initialize the security system.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing security manager...")

            # Register orchestrator as trusted agent
            await self.register_trusted_agent(
                self.orchestrator_id,
                "orchestrator",
                ["system_admin", "monitoring", "coordination"],
                trust_level=1.0
            )

            # Start background security tasks
            asyncio.create_task(self._security_monitor_loop())
            asyncio.create_task(self._cleanup_expired_data())

            logger.info("Security manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize security manager: {e}")
            return False

    async def register_trusted_agent(self, agent_id: str, agent_type: str,
                                   capabilities: List[str], trust_level: float = 1.0) -> bool:
        """Register a trusted agent in the system.

        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            capabilities: Agent capabilities
            trust_level: Initial trust level (0.0-1.0)

        Returns:
            True if registration successful
        """
        try:
            # Register with authenticator
            success = await self.authenticator.register_agent(
                agent_id, agent_type, capabilities, trust_level
            )

            if success:
                # Register signing key with verifier
                import secrets
                agent_key = secrets.token_hex(32)
                await self.verifier.register_agent_key(agent_id, agent_key)

                # Add to trusted agents if high trust
                if trust_level >= 0.8:
                    self.trusted_agents.add(agent_id)

                # Report registration
                await self.reporter.report_agent_status(
                    agent_id,
                    {
                        "status": "registered",
                        "trust_level": trust_level,
                        "capabilities": capabilities
                    },
                    Severity.LOW
                )

                logger.info(f"Registered trusted agent {agent_id} with trust level {trust_level}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            await self._log_security_event(
                "agent_registration_failed",
                Severity.MEDIUM,
                agent_id,
                f"Failed to register agent: {str(e)}"
            )
            return False

    async def register_agent(self, agent_id: str, agent_type: str,
                           capabilities: List[str], trust_level: float = 1.0) -> AgentCredentials:
        """Register agent and return credentials (test-compatible alias).

        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            capabilities: Agent capabilities
            trust_level: Initial trust level (0.0-1.0)

        Returns:
            Agent credentials
        """
        success = await self.register_trusted_agent(agent_id, agent_type, capabilities, trust_level)
        if success:
            # Return credentials from authenticator
            return await self.authenticator.authenticate_agent(agent_id, agent_type)
        else:
            raise Exception(f"Failed to register agent {agent_id}")

    async def authenticate_and_authorize(self, agent_id: str, agent_type: str,
                                       operation: str, context: Dict[str, Any] = None) -> Tuple[bool, Optional[AgentCredentials]]:
        """Authenticate agent and authorize operation.

        Args:
            agent_id: Agent identifier
            agent_type: Expected agent type
            operation: Operation being attempted
            context: Additional context for authorization

        Returns:
            Tuple of (authorized, credentials)
        """
        self.security_checks_performed += 1

        try:
            # Authenticate agent
            credentials = await self.authenticator.authenticate_agent(agent_id, agent_type)
            if not credentials:
                await self._log_security_event(
                    "authentication_failed",
                    Severity.HIGH,
                    agent_id,
                    f"Authentication failed for operation: {operation}"
                )
                return False, None

            # Check authorization
            authorized = await self._authorize_operation(credentials, operation, context)
            if not authorized:
                self.blocked_operations += 1
                await self._log_security_event(
                    "authorization_failed",
                    Severity.HIGH,
                    agent_id,
                    f"Authorization failed for operation: {operation}"
                )
                return False, credentials

            logger.debug(f"Agent {agent_id} authenticated and authorized for {operation}")
            return True, credentials

        except Exception as e:
            logger.error(f"Security check failed for {agent_id}: {e}")
            await self._log_security_event(
                "security_check_error",
                Severity.CRITICAL,
                agent_id,
                f"Security check error: {str(e)}"
            )
            return False, None

    async def verify_secure_message(self, message: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], VerificationResult]:
        """Verify a secure message with embedded signature.

        Args:
            message: Message with embedded signature

        Returns:
            Tuple of (verified, original_message, verification_result)
        """
        try:
            original_message, verification_result = await self.verifier.extract_and_verify_async(message)

            if not verification_result.is_valid:
                self.security_violations_detected += 1
                await self._log_security_event(
                    "message_verification_failed",
                    Severity.HIGH,
                    message.get("signature", {}).get("agent_id", "unknown"),
                    f"Message verification failed: {', '.join(verification_result.issues)}"
                )

            return verification_result.is_valid, original_message, verification_result

        except Exception as e:
            logger.error(f"Message verification error: {e}")
            await self._log_security_event(
                "message_verification_error",
                Severity.CRITICAL,
                "unknown",
                f"Message verification error: {str(e)}"
            )
            return False, None, VerificationResult(
                is_valid=False,
                trust_score=0.0,
                verification_time=0.0,
                issues=[f"Verification error: {str(e)}"],
                metadata={}
            )

    async def validate_agent_answer(self, question: str, answer: str, agent_id: str,
                                  agent_type: str, task_context: Dict[str, Any] = None,
                                  cross_check_agents: List[Any] = None) -> ValidationResult:
        """Validate an agent's answer using comprehensive validation.

        Args:
            question: Original question
            answer: Agent's answer
            agent_id: Agent identifier
            agent_type: Agent type
            task_context: Task context
            cross_check_agents: Other agents for cross-validation

        Returns:
            ValidationResult with comprehensive validation details
        """
        try:
            answer_context = AnswerContext(
                original_question=question,
                agent_id=agent_id,
                agent_type=agent_type,
                answer=answer,
                task_context=task_context or {},
                timestamp=datetime.now()
            )

            validation_methods = [
                ValidationMethod.CONSISTENCY_CHECK,
                ValidationMethod.CONFIDENCE_SCORING,
                ValidationMethod.CRITICAL_ANALYSIS
            ]

            # Add cross-validation if agents available
            if cross_check_agents:
                validation_methods.append(ValidationMethod.CROSS_VALIDATION)

            # Add factual verification for specific answer types
            if any(keyword in answer.lower() for keyword in ['according to', 'research shows', 'data indicates']):
                validation_methods.append(ValidationMethod.FACTUAL_VERIFICATION)

            validation_result = await self.validator.validate_answer(
                answer_context, validation_methods, cross_check_agents
            )

            # Report validation results
            if not validation_result.is_valid:
                await self.reporter.report_security_alert(
                    agent_id,
                    "invalid_answer",
                    {
                        "question": question,
                        "answer_excerpt": answer[:200] + "..." if len(answer) > 200 else answer,
                        "validation_issues": validation_result.issues,
                        "confidence_score": validation_result.confidence_score
                    },
                    Severity.MEDIUM
                )

            return validation_result

        except Exception as e:
            logger.error(f"Answer validation error for {agent_id}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                trust_level=0.0,
                validation_methods=[],
                issues=[f"Validation error: {str(e)}"],
                metadata={},
                validation_time=0.0,
                cross_check_results={}
            )

    async def report_security_incident(self, incident_type: str, agent_id: str,
                                     severity: Severity, details: Dict[str, Any]) -> bool:
        """Report a security incident.

        Args:
            incident_type: Type of security incident
            agent_id: Agent involved in incident
            severity: Incident severity
            details: Incident details

        Returns:
            True if report sent successfully
        """
        try:
            success = await self.reporter.report_security_alert(
                agent_id, incident_type, details, severity
            )

            if success:
                await self._log_security_event(
                    incident_type,
                    severity,
                    agent_id,
                    f"Security incident reported: {details.get('description', 'No description')}"
                )

                # Update threat level if severe
                if severity in [Severity.HIGH, Severity.CRITICAL]:
                    await self._update_threat_level(severity, incident_type)

            return success

        except Exception as e:
            logger.error(f"Failed to report security incident: {e}")
            return False

    async def _authorize_operation(self, credentials: AgentCredentials,
                                 operation: str, context: Dict[str, Any] = None) -> bool:
        """Check if agent is authorized for operation."""
        # High security mode - only trusted agents
        if self.high_security_mode and credentials.agent_id not in self.trusted_agents:
            return False

        # Check trust level
        if credentials.trust_level < 0.1:
            return False

        # Check capabilities
        required_capabilities = self._get_required_capabilities(operation)
        if required_capabilities and not any(cap in credentials.capabilities for cap in required_capabilities):
            return False

        # Check if agent is revoked
        if credentials.is_revoked:
            return False

        # Operation-specific checks
        if operation == "system_admin" and credentials.trust_level < 0.9:
            return False

        if operation == "sensitive_data" and credentials.trust_level < 0.7:
            return False

        return True

    def _get_required_capabilities(self, operation: str) -> List[str]:
        """Get required capabilities for an operation."""
        capability_map = {
            "task_execution": ["execution", "processing"],
            "data_access": ["data_read", "data_analysis"],
            "system_admin": ["system_admin", "configuration"],
            "monitoring": ["monitoring", "logging"],
            "coordination": ["coordination", "communication"],
            "sensitive_data": ["data_read", "security_clearance"]
        }

        return capability_map.get(operation, [])

    async def _log_security_event(self, event_type: str, severity: Severity,
                                agent_id: str, description: str, metadata: Dict[str, Any] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_id=f"{event_type}_{agent_id}_{int(datetime.now().timestamp())}",
            event_type=event_type,
            severity=severity,
            agent_id=agent_id,
            timestamp=datetime.now(),
            description=description,
            metadata=metadata or {}
        )

        self.security_events.append(event)

        # Limit event history
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]

        logger.info(f"Security event logged: {event_type} for {agent_id} ({severity.value})")

    async def _update_threat_level(self, severity: Severity, incident_type: str):
        """Update system threat level based on incidents."""
        current_time = datetime.now()
        recent_incidents = [
            event for event in self.security_events
            if current_time - event.timestamp < timedelta(minutes=30)
            and event.severity in [Severity.HIGH, Severity.CRITICAL]
        ]

        if len(recent_incidents) >= 5:
            self.threat_level = "CRITICAL"
            self.high_security_mode = True
        elif len(recent_incidents) >= 3:
            self.threat_level = "HIGH"
        elif len(recent_incidents) >= 1:
            self.threat_level = "MEDIUM"
        else:
            self.threat_level = "LOW"
            self.high_security_mode = False

        logger.warning(f"Threat level updated to {self.threat_level}")

    async def _security_monitor_loop(self):
        """Background security monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Cleanup expired tokens
                await self.authenticator.cleanup_expired_tokens()

                # Cleanup old nonces
                await self.verifier.cleanup_old_nonces()

                # Check for anomalies in recent activity
                await self._check_system_anomalies()

            except Exception as e:
                logger.error(f"Security monitor error: {e}")

    async def _check_system_anomalies(self):
        """Check for system-wide security anomalies."""
        current_time = datetime.now()

        # Check for unusual authentication patterns
        recent_auths = self.authenticator.get_security_metrics()
        if recent_auths["failed_auths"] > 10:  # High failure rate
            await self._log_security_event(
                "high_auth_failure_rate",
                Severity.HIGH,
                "system",
                f"High authentication failure rate: {recent_auths['failed_auths']} failures"
            )

        # Check verification metrics
        verification_metrics = self.verifier.get_verification_metrics()
        if verification_metrics["tampering_detected"] > 0:
            await self._log_security_event(
                "message_tampering_detected",
                Severity.CRITICAL,
                "system",
                f"Message tampering detected: {verification_metrics['tampering_detected']} instances"
            )

    async def _cleanup_expired_data(self):
        """Cleanup expired security data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour

                # Clean old security events
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.security_events = [
                    event for event in self.security_events
                    if event.timestamp > cutoff_time
                ]

                logger.debug("Cleaned up expired security data")

            except Exception as e:
                logger.error(f"Security cleanup error: {e}")

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "orchestrator_id": self.orchestrator_id,
            "threat_level": self.threat_level,
            "high_security_mode": self.high_security_mode,
            "trusted_agents": len(self.trusted_agents),
            "authenticated_agents": len(self.authenticator.authenticated_agents),
            "security_checks_performed": self.security_checks_performed,
            "security_violations_detected": self.security_violations_detected,
            "blocked_operations": self.blocked_operations,
            "security_events_count": len(self.security_events),
            "recent_events": len([
                e for e in self.security_events
                if datetime.now() - e.timestamp < timedelta(hours=1)
            ]),
            "authenticator_metrics": self.authenticator.get_security_metrics(),
            "verifier_metrics": self.verifier.get_verification_metrics(),
            "validator_metrics": self.validator.get_validation_metrics(),
            "reporter_summary": self.reporter.get_audit_summary(24)
        }

    def get_agent_security_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get security profile for a specific agent."""
        auth_status = self.authenticator.get_agent_status(agent_id)
        if not auth_status:
            return None

        reliability = self.validator.get_agent_reliability(agent_id)
        anomaly_history = self.reporter.get_agent_anomaly_history(agent_id)

        return {
            "agent_id": agent_id,
            "authentication": auth_status,
            "reliability_score": reliability,
            "anomaly_count": len(anomaly_history),
            "is_trusted": agent_id in self.trusted_agents,
            "recent_events": [
                event for event in self.security_events
                if event.agent_id == agent_id and
                datetime.now() - event.timestamp < timedelta(hours=24)
            ]
        }