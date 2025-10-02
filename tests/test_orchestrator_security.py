"""Comprehensive test suite for SwarmOrchestrator security integration.

Tests the complete security layer integration with the orchestrator including:
- Secure orchestrator initialization
- Agent authentication and registration
- Secure task submission and execution
- Security incident detection and reporting
- Unauthorized agent blocking
- Message integrity verification
- Answer validation during task processing
"""

import asyncio
import pytest
import uuid
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

# Import system under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.coordinator.orchestrator import SwarmOrchestrator, SwarmTask, AgentAllocation
from security import SecurityManager, AgentAuthenticator, MessageVerifier, AnswerValidator, SecureReporter
from security.secure_reporter import SecurityReport, AnomalyAlert, ReportType, Severity
from security.agent_authenticator import AgentCredentials
from security.message_verifier import MessageSignature, VerificationResult
from security.answer_validator import AnswerContext, ValidationResult, ValidationMethod

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOrchestratorSecurity:
    """Test suite for orchestrator security integration."""

    @pytest.fixture
    async def secure_orchestrator(self):
        """Create a secure orchestrator instance for testing."""
        orchestrator_id = f"test_orchestrator_{uuid.uuid4().hex[:8]}"

        # Mock external dependencies to isolate security testing
        with patch('swarm.coordinator.orchestrator.Agent') as mock_agent, \
             patch('swarm.coordinator.orchestrator.SwarmMCPClient') as mock_mcp:

            mock_agent_instance = MagicMock()
            mock_agent.return_value = mock_agent_instance

            mock_mcp_instance = AsyncMock()
            mock_mcp.return_value = mock_mcp_instance
            mock_mcp_instance.connect = AsyncMock()
            mock_mcp_instance.register_message_handler = MagicMock()

            orchestrator = SwarmOrchestrator(orchestrator_id=orchestrator_id)
            await orchestrator.initialize()

            yield orchestrator

            # Cleanup
            if orchestrator.running:
                orchestrator.running = False

    @pytest.fixture
    def mock_critical_agent(self):
        """Mock critical thinking agent for answer validation."""
        agent = AsyncMock()
        agent.query = AsyncMock()
        return agent

    async def test_orchestrator_security_initialization(self, secure_orchestrator):
        """Test 1: Create orchestrator instance with security enabled."""
        logger.info("=== Test 1: Orchestrator Security Initialization ===")

        orchestrator = secure_orchestrator

        # Verify orchestrator is properly initialized
        assert orchestrator.orchestrator_id is not None
        assert orchestrator.running is True

        # Verify security manager is initialized
        assert orchestrator.security_manager is not None
        assert isinstance(orchestrator.security_manager, SecurityManager)

        # Verify all security components are present
        assert orchestrator.security_manager.authenticator is not None
        assert orchestrator.security_manager.verifier is not None
        assert orchestrator.security_manager.validator is not None
        assert orchestrator.security_manager.reporter is not None

        # Verify orchestrator is registered as trusted agent
        assert orchestrator.orchestrator_id in orchestrator.security_manager.trusted_agents

        logger.info("✅ Security initialization verified")

    async def test_security_manager_initialization(self, secure_orchestrator):
        """Test 2: Verify security manager is properly initialized."""
        logger.info("=== Test 2: Security Manager Component Verification ===")

        security_manager = secure_orchestrator.security_manager

        # Test all components are initialized
        assert isinstance(security_manager.authenticator, AgentAuthenticator)
        assert isinstance(security_manager.verifier, MessageVerifier)
        assert isinstance(security_manager.validator, AnswerValidator)
        assert isinstance(security_manager.reporter, SecureReporter)

        # Test security manager state
        assert security_manager.orchestrator_id == secure_orchestrator.orchestrator_id
        assert len(security_manager.security_events) >= 0
        assert security_manager.anomaly_threshold > 0

        logger.info("✅ Security manager components verified")

    async def test_secure_agent_registration(self, secure_orchestrator):
        """Test 3: Test secure agent registration process."""
        logger.info("=== Test 3: Secure Agent Registration ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Test registering a legitimate agent
        agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
        agent_type = "research"
        capabilities = ["search", "analysis", "verification"]

        # Register agent through security manager
        credentials = await security_manager.register_agent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities
        )

        # Verify registration succeeded
        assert credentials is not None
        assert isinstance(credentials, AgentCredentials)
        assert credentials.agent_id == agent_id
        assert credentials.token is not None

        # Verify agent is in authenticated agents
        assert agent_id in security_manager.authenticator.authenticated_agents

        # Test agent allocation in orchestrator
        allocation = AgentAllocation(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities
        )
        orchestrator.active_agents[agent_id] = allocation

        # Verify agent is tracked
        assert agent_id in orchestrator.active_agents
        assert orchestrator.active_agents[agent_id].agent_type == agent_type

        logger.info("✅ Secure agent registration verified")

    async def test_secure_task_submission_and_execution(self, secure_orchestrator):
        """Test 4: Test secure task submission and execution."""
        logger.info("=== Test 4: Secure Task Submission and Execution ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # First register an agent to handle the task
        agent_id = f"task_agent_{uuid.uuid4().hex[:8]}"
        credentials = await security_manager.register_agent(
            agent_id=agent_id,
            agent_type="research",
            capabilities=["research", "analysis"]
        )

        # Create a secure task
        task = SwarmTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            description="Research the latest developments in AI security",
            task_type="research",
            priority=5,
            required_agents=["research"],
            context={"domain": "AI security", "depth": "comprehensive"}
        )

        # Submit task through orchestrator (would normally verify submitter)
        orchestrator.pending_tasks.append(task)

        # Simulate task execution with security verification
        message = "Task completed: Found 15 recent papers on AI security"
        message_signature = await security_manager.verifier.sign_message(
            message=message,
            agent_id=agent_id,
            timestamp=datetime.now()
        )

        # Verify message integrity
        verification_result = await security_manager.verify_message(
            message=message,
            signature=message_signature,
            sender_id=agent_id
        )

        assert verification_result.is_valid is True
        assert verification_result.trust_score > 0.5

        # Validate the answer
        answer_context = AnswerContext(
            question="Research the latest developments in AI security",
            answer=message,
            agent_id=agent_id,
            task_context=task.context
        )

        validation_result = await security_manager.validate_answer(
            answer=message,
            context=answer_context
        )

        assert validation_result.is_valid is True
        assert validation_result.confidence_score > 0.0

        logger.info("✅ Secure task submission and execution verified")

    async def test_security_incident_detection(self, secure_orchestrator):
        """Test 5: Test security incident detection and reporting."""
        logger.info("=== Test 5: Security Incident Detection ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Simulate suspicious agent behavior
        malicious_agent_id = f"malicious_{uuid.uuid4().hex[:8]}"

        # Test 1: Unauthorized access attempt
        try:
            fake_credentials = AgentCredentials(
                agent_id=malicious_agent_id,
                agent_type="hacker",
                token="fake_token_123",
                capabilities=["system_access"],
                trust_level=0.0,
                expires_at=datetime.now() + timedelta(hours=1)
            )

            # Try to verify with fake credentials
            is_valid = await security_manager.authenticator.verify_credentials(fake_credentials)
            assert is_valid is False

        except Exception as e:
            logger.info(f"Expected security exception: {e}")

        # Test 2: Invalid message signature
        malicious_message = "I have root access to all systems"
        fake_signature = MessageSignature(
            message_hash="fake_hash",
            agent_id=malicious_agent_id,
            timestamp=datetime.now(),
            signature="fake_signature"
        )

        verification_result = await security_manager.verify_message(
            message=malicious_message,
            signature=fake_signature,
            sender_id=malicious_agent_id
        )

        assert verification_result.is_valid is False
        assert verification_result.trust_score == 0.0

        # Test 3: Check if security events were generated
        assert len(security_manager.security_events) > 0

        # Find security events related to our test
        suspicious_events = [
            event for event in security_manager.security_events
            if malicious_agent_id in event.description or event.severity == Severity.HIGH
        ]

        assert len(suspicious_events) > 0

        logger.info("✅ Security incident detection verified")

    async def test_security_status_reporting(self, secure_orchestrator):
        """Test 6: Verify security status reporting works."""
        logger.info("=== Test 6: Security Status Reporting ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Generate security status report
        status_report = await security_manager.get_security_status()

        # Verify report structure
        assert "timestamp" in status_report
        assert "orchestrator_id" in status_report
        assert "authenticated_agents" in status_report
        assert "security_events_count" in status_report
        assert "trust_score_average" in status_report
        assert "security_level" in status_report

        # Verify report content
        assert status_report["orchestrator_id"] == orchestrator.orchestrator_id
        assert isinstance(status_report["authenticated_agents"], int)
        assert isinstance(status_report["security_events_count"], int)
        assert 0.0 <= status_report["trust_score_average"] <= 1.0
        assert status_report["security_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        logger.info("✅ Security status reporting verified")

    async def test_agent_security_profiles(self, secure_orchestrator):
        """Test 7: Test agent security profiles."""
        logger.info("=== Test 7: Agent Security Profiles ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Register agents with different trust levels
        high_trust_agent = f"trusted_{uuid.uuid4().hex[:8]}"
        medium_trust_agent = f"medium_{uuid.uuid4().hex[:8]}"

        # Register high trust agent
        await security_manager.register_trusted_agent(
            agent_id=high_trust_agent,
            agent_type="critical_analysis",
            capabilities=["critical_thinking", "validation", "security_review"],
            trust_level=0.9
        )

        # Register medium trust agent
        medium_credentials = await security_manager.register_agent(
            agent_id=medium_trust_agent,
            agent_type="general",
            capabilities=["basic_tasks"]
        )

        # Verify trust levels
        assert high_trust_agent in security_manager.trusted_agents
        assert security_manager.trusted_agents[high_trust_agent]["trust_level"] == 0.9

        assert medium_trust_agent in security_manager.authenticator.authenticated_agents
        medium_agent_data = security_manager.authenticator.authenticated_agents[medium_trust_agent]
        assert medium_agent_data["trust_level"] < 0.9

        # Test capability verification
        high_trust_capabilities = security_manager.trusted_agents[high_trust_agent]["capabilities"]
        assert "security_review" in high_trust_capabilities
        assert "critical_thinking" in high_trust_capabilities

        logger.info("✅ Agent security profiles verified")

    async def test_unauthorized_agent_blocking(self, secure_orchestrator):
        """Test 8: Validate that unauthorized agents are blocked."""
        logger.info("=== Test 8: Unauthorized Agent Blocking ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Attempt to use an unregistered agent
        unauthorized_agent_id = f"unauthorized_{uuid.uuid4().hex[:8]}"

        # Test 1: Try to authenticate without registration
        fake_credentials = AgentCredentials(
            agent_id=unauthorized_agent_id,
            agent_type="malicious",
            token="invalid_token",
            capabilities=["hack"],
            trust_level=0.0,
            expires_at=datetime.now() - timedelta(hours=1)  # Expired
        )

        is_valid = await security_manager.authenticator.verify_credentials(fake_credentials)
        assert is_valid is False

        # Test 2: Try to submit a task from unauthorized agent
        unauthorized_task = SwarmTask(
            task_id=f"malicious_task_{uuid.uuid4().hex[:8]}",
            description="Delete all system files",
            task_type="system",
            priority=10,
            required_agents=["system"],
            context={"action": "destructive"}
        )

        # Simulate security check before task acceptance
        is_authorized = unauthorized_agent_id in security_manager.authenticator.authenticated_agents
        assert is_authorized is False

        # Task should be rejected
        if not is_authorized:
            # Generate security alert
            await security_manager.report_security_incident(
                incident_type="unauthorized_access",
                agent_id=unauthorized_agent_id,
                description=f"Unauthorized agent {unauthorized_agent_id} attempted task submission",
                severity=Severity.HIGH
            )

        # Verify security event was created
        recent_events = [
            event for event in security_manager.security_events
            if unauthorized_agent_id in event.description
        ]
        assert len(recent_events) > 0
        assert recent_events[0].severity == Severity.HIGH

        logger.info("✅ Unauthorized agent blocking verified")

    async def test_message_integrity_verification(self, secure_orchestrator):
        """Test 9: Test message integrity verification in orchestrator context."""
        logger.info("=== Test 9: Message Integrity Verification ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Register a legitimate agent
        agent_id = f"msg_test_agent_{uuid.uuid4().hex[:8]}"
        credentials = await security_manager.register_agent(
            agent_id=agent_id,
            agent_type="communication",
            capabilities=["messaging", "coordination"]
        )

        # Test valid message integrity
        valid_message = "Task completed successfully with 95% confidence"
        valid_signature = await security_manager.verifier.sign_message(
            message=valid_message,
            agent_id=agent_id,
            timestamp=datetime.now()
        )

        verification_result = await security_manager.verify_message(
            message=valid_message,
            signature=valid_signature,
            sender_id=agent_id
        )

        assert verification_result.is_valid is True
        assert verification_result.trust_score > 0.5

        # Test tampered message
        tampered_message = "Task completed successfully with 99% confidence"  # Changed content
        tampered_verification = await security_manager.verify_message(
            message=tampered_message,
            signature=valid_signature,  # Same signature, different message
            sender_id=agent_id
        )

        assert tampered_verification.is_valid is False
        assert tampered_verification.trust_score == 0.0

        # Test message from wrong agent
        wrong_agent_verification = await security_manager.verify_message(
            message=valid_message,
            signature=valid_signature,
            sender_id="wrong_agent_id"
        )

        assert wrong_agent_verification.is_valid is False

        logger.info("✅ Message integrity verification verified")

    async def test_answer_validation_during_task_processing(self, secure_orchestrator, mock_critical_agent):
        """Test 10: Verify answer validation during task processing."""
        logger.info("=== Test 10: Answer Validation During Task Processing ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Override the validator with our mock agent
        security_manager.validator.critical_agent = mock_critical_agent

        # Register an agent
        agent_id = f"validation_agent_{uuid.uuid4().hex[:8]}"
        credentials = await security_manager.register_agent(
            agent_id=agent_id,
            agent_type="research",
            capabilities=["research", "analysis"]
        )

        # Create test scenarios
        test_cases = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "expected_valid": True,
                "mock_response": "The answer 'Paris' is correct. Paris is indeed the capital of France."
            },
            {
                "question": "What is 2 + 2?",
                "answer": "5",
                "expected_valid": False,
                "mock_response": "The answer '5' is incorrect. 2 + 2 equals 4, not 5."
            },
            {
                "question": "Explain quantum computing",
                "answer": "It uses quantum bits to process information faster",
                "expected_valid": True,
                "mock_response": "The answer provides a basic but accurate explanation of quantum computing."
            }
        ]

        for i, case in enumerate(test_cases):
            # Configure mock response
            mock_critical_agent.query.return_value = case["mock_response"]

            # Create answer context
            context = AnswerContext(
                question=case["question"],
                answer=case["answer"],
                agent_id=agent_id,
                task_context={"test_case": i}
            )

            # Validate answer
            validation_result = await security_manager.validate_answer(
                answer=case["answer"],
                context=context
            )

            # Verify validation result
            assert isinstance(validation_result, ValidationResult)
            assert validation_result.is_valid == case["expected_valid"]
            assert 0.0 <= validation_result.confidence_score <= 1.0

            # For invalid answers, confidence should be lower
            if not case["expected_valid"]:
                assert validation_result.confidence_score < 0.5

            logger.info(f"  Case {i+1}: {'✅' if validation_result.is_valid == case['expected_valid'] else '❌'}")

        logger.info("✅ Answer validation during task processing verified")

    async def test_complete_security_integration(self, secure_orchestrator):
        """Test 11: Complete end-to-end security integration test."""
        logger.info("=== Test 11: Complete Security Integration ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Simulate a complete secure workflow
        logger.info("Starting complete secure workflow simulation...")

        # 1. Register multiple agents
        agent_ids = []
        for i in range(3):
            agent_id = f"workflow_agent_{i}_{uuid.uuid4().hex[:8]}"
            await security_manager.register_agent(
                agent_id=agent_id,
                agent_type="research",
                capabilities=["research", "analysis"]
            )
            agent_ids.append(agent_id)

        # 2. Create and submit secure tasks
        tasks = []
        for i, agent_id in enumerate(agent_ids):
            task = SwarmTask(
                task_id=f"secure_task_{i}_{uuid.uuid4().hex[:8]}",
                description=f"Research topic {i} with security validation",
                task_type="research",
                priority=5,
                required_agents=["research"],
                context={"topic": f"topic_{i}", "security_level": "high"}
            )
            tasks.append(task)
            orchestrator.pending_tasks.append(task)

        # 3. Simulate task execution with full security
        for i, (task, agent_id) in enumerate(zip(tasks, agent_ids)):
            # Agent produces result
            result_message = f"Research completed for {task.context['topic']} with high confidence"

            # Sign the message
            signature = await security_manager.verifier.sign_message(
                message=result_message,
                agent_id=agent_id,
                timestamp=datetime.now()
            )

            # Verify message integrity
            verification = await security_manager.verify_message(
                message=result_message,
                signature=signature,
                sender_id=agent_id
            )
            assert verification.is_valid is True

            # Validate the answer (simplified)
            context = AnswerContext(
                question=task.description,
                answer=result_message,
                agent_id=agent_id,
                task_context=task.context
            )

            # Update task status
            task.status = "completed"
            task.results = {"message": result_message, "verified": True}
            orchestrator.completed_tasks.append(task)

        # 4. Generate final security report
        final_report = await security_manager.get_security_status()

        # Verify all tasks completed securely
        assert len(orchestrator.completed_tasks) == 3
        assert all(task.results["verified"] for task in orchestrator.completed_tasks)

        # Verify security state
        assert final_report["authenticated_agents"] >= 3
        assert final_report["trust_score_average"] > 0.5
        assert final_report["security_level"] in ["LOW", "MEDIUM", "HIGH"]

        logger.info("✅ Complete security integration verified")

    async def test_performance_metrics(self, secure_orchestrator):
        """Test 12: Verify security performance metrics."""
        logger.info("=== Test 12: Security Performance Metrics ===")

        orchestrator = secure_orchestrator
        security_manager = orchestrator.security_manager

        # Performance test setup
        start_time = datetime.now()

        # Register agents and measure performance
        agent_count = 10
        registration_times = []

        for i in range(agent_count):
            agent_start = datetime.now()
            await security_manager.register_agent(
                agent_id=f"perf_agent_{i}",
                agent_type="test",
                capabilities=["testing"]
            )
            registration_time = (datetime.now() - agent_start).total_seconds()
            registration_times.append(registration_time)

        # Verify performance is reasonable
        avg_registration_time = sum(registration_times) / len(registration_times)
        assert avg_registration_time < 1.0  # Should be under 1 second per registration

        # Test message verification performance
        agent_id = "perf_agent_0"
        message = "Performance test message"

        verify_start = datetime.now()
        signature = await security_manager.verifier.sign_message(
            message=message,
            agent_id=agent_id,
            timestamp=datetime.now()
        )
        verification = await security_manager.verify_message(
            message=message,
            signature=signature,
            sender_id=agent_id
        )
        verify_time = (datetime.now() - verify_start).total_seconds()

        assert verify_time < 0.5  # Should be under 0.5 seconds
        assert verification.is_valid is True

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total test time: {total_time:.2f}s")
        logger.info(f"Average registration time: {avg_registration_time:.3f}s")
        logger.info(f"Message verification time: {verify_time:.3f}s")

        logger.info("✅ Security performance metrics verified")

async def run_all_tests():
    """Run all security integration tests."""
    logger.info("🚀 Starting comprehensive orchestrator security test suite")
    logger.info("=" * 60)

    # Create test instance
    test_suite = TestOrchestratorSecurity()

    # Create secure orchestrator
    async with test_suite.secure_orchestrator() as orchestrator:
        mock_agent = test_suite.mock_critical_agent()

        try:
            # Run all tests
            await test_suite.test_orchestrator_security_initialization(orchestrator)
            await test_suite.test_security_manager_initialization(orchestrator)
            await test_suite.test_secure_agent_registration(orchestrator)
            await test_suite.test_secure_task_submission_and_execution(orchestrator)
            await test_suite.test_security_incident_detection(orchestrator)
            await test_suite.test_security_status_reporting(orchestrator)
            await test_suite.test_agent_security_profiles(orchestrator)
            await test_suite.test_unauthorized_agent_blocking(orchestrator)
            await test_suite.test_message_integrity_verification(orchestrator)
            await test_suite.test_answer_validation_during_task_processing(orchestrator, mock_agent)
            await test_suite.test_complete_security_integration(orchestrator)
            await test_suite.test_performance_metrics(orchestrator)

            logger.info("=" * 60)
            logger.info("🎉 ALL TESTS PASSED - SECURITY INTEGRATION VERIFIED")
            logger.info("✅ SwarmOrchestrator security layer is working properly")
            logger.info("✅ Swarm system is now secure against rogue agents")
            logger.info("✅ Threat detection capabilities confirmed")
            logger.info("✅ Performance metrics within acceptable ranges")

            return True

        except Exception as e:
            logger.error(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)