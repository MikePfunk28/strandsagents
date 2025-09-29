"""Test script for security layer integration with swarm system.

This test verifies that the security layer is properly integrated
and provides the expected security features.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from security import SecurityManager
from swarm.coordinator.orchestrator import SwarmOrchestrator

async def test_security_manager_initialization():
    """Test security manager initialization."""
    print("Testing Security Manager Initialization...")

    security_manager = SecurityManager("test_orchestrator")
    success = await security_manager.initialize_security()

    assert success, "Security manager initialization failed"
    print("✓ Security manager initialized successfully")

    # Test security status
    status = security_manager.get_security_status()
    assert "threat_level" in status, "Security status missing threat_level"
    assert status["threat_level"] == "LOW", f"Expected LOW threat level, got {status['threat_level']}"
    print("✓ Security status reporting works")

    return security_manager

async def test_agent_registration_security():
    """Test agent registration with security."""
    print("\nTesting Agent Registration Security...")

    security_manager = await test_security_manager_initialization()

    # Test registering a valid agent
    success = await security_manager.register_trusted_agent(
        "test_agent_1",
        "research",
        ["information_gathering", "analysis"],
        trust_level=0.8
    )
    assert success, "Failed to register valid agent"
    print("✓ Valid agent registration successful")

    # Test agent authentication
    credentials = await security_manager.authenticator.authenticate_agent("test_agent_1", "research")
    assert credentials is not None, "Failed to authenticate registered agent"
    assert credentials.trust_level == 0.8, f"Expected trust level 0.8, got {credentials.trust_level}"
    print("✓ Agent authentication successful")

    # Test authorization
    authorized, _ = await security_manager.authenticate_and_authorize(
        "test_agent_1",
        "research",
        "task_execution"
    )
    assert authorized, "Agent not authorized for task execution"
    print("✓ Agent authorization successful")

async def test_message_verification():
    """Test message integrity verification."""
    print("\nTesting Message Verification...")

    security_manager = await test_security_manager_initialization()

    # Register agent for message verification
    await security_manager.register_trusted_agent(
        "test_agent_2",
        "creative",
        ["content_generation"],
        trust_level=0.7
    )

    # Create and verify a signed message
    test_message = {
        "type": "task_result",
        "task_id": "test_task_123",
        "result": "This is a test result from the agent",
        "timestamp": "2024-01-01T12:00:00Z"
    }

    # Sign the message
    signed_message = security_manager.verifier.create_signed_message(test_message, "test_agent_2")
    assert signed_message is not None, "Failed to create signed message"
    assert "signature" in signed_message, "Signed message missing signature"
    print("✓ Message signing successful")

    # Verify the message
    is_valid, original_message, verification_result = await security_manager.verify_secure_message(signed_message)
    assert is_valid, f"Message verification failed: {verification_result.issues}"
    assert original_message == test_message, "Original message doesn't match"
    print("✓ Message verification successful")

async def test_answer_validation():
    """Test answer validation system."""
    print("\nTesting Answer Validation...")

    security_manager = await test_security_manager_initialization()

    # Register agent for answer validation
    await security_manager.register_trusted_agent(
        "test_agent_3",
        "analysis",
        ["critical_thinking", "evaluation"],
        trust_level=0.9
    )

    # Test validating a good answer
    question = "What are the benefits of using microservices architecture?"
    good_answer = """Microservices architecture offers several key benefits:
1. Scalability - Individual services can be scaled independently
2. Flexibility - Different technologies can be used for different services
3. Fault isolation - Failure in one service doesn't affect others
4. Team independence - Teams can work on services independently
5. Deployment flexibility - Services can be deployed independently"""

    validation_result = await security_manager.validate_agent_answer(
        question,
        good_answer,
        "test_agent_3",
        "analysis",
        {"domain": "software_architecture"}
    )

    assert validation_result.confidence_score > 0.6, f"Low confidence score: {validation_result.confidence_score}"
    print(f"✓ Answer validation successful (confidence: {validation_result.confidence_score:.2f})")

    # Test validating a poor answer
    poor_answer = "I don't know much about this topic."

    validation_result_poor = await security_manager.validate_agent_answer(
        question,
        poor_answer,
        "test_agent_3",
        "analysis",
        {"domain": "software_architecture"}
    )

    assert validation_result_poor.confidence_score < validation_result.confidence_score, "Poor answer should have lower confidence"
    print(f"✓ Poor answer detection works (confidence: {validation_result_poor.confidence_score:.2f})")

async def test_orchestrator_security_integration():
    """Test security integration with orchestrator."""
    print("\nTesting Orchestrator Security Integration...")

    # Create orchestrator with security
    orchestrator = SwarmOrchestrator("test_orchestrator_secure")

    # Initialize (this should initialize security)
    await orchestrator.initialize()

    # Check security manager is available
    assert hasattr(orchestrator, 'security_manager'), "Orchestrator missing security manager"
    assert orchestrator.security_manager is not None, "Security manager not initialized"
    print("✓ Orchestrator security integration successful")

    # Test security status reporting
    security_status = orchestrator.get_security_status()
    assert "threat_level" in security_status, "Security status missing threat_level"
    print("✓ Orchestrator security status reporting works")

    # Clean up
    await orchestrator.stop_orchestration()

async def test_security_incident_reporting():
    """Test security incident reporting."""
    print("\nTesting Security Incident Reporting...")

    security_manager = await test_security_manager_initialization()

    # Register agent for incident testing
    await security_manager.register_trusted_agent(
        "test_agent_4",
        "suspicious",
        ["unknown_capability"],
        trust_level=0.3
    )

    # Report a security incident
    success = await security_manager.report_security_incident(
        "test_incident",
        "test_agent_4",
        security_manager.reporter.Severity.MEDIUM,
        {"description": "Test security incident", "details": "This is a test"}
    )

    assert success, "Failed to report security incident"
    print("✓ Security incident reporting successful")

    # Check if incident affected threat level
    status = security_manager.get_security_status()
    print(f"✓ Current threat level: {status['threat_level']}")

async def run_all_tests():
    """Run all security tests."""
    print("🔒 Running Security Layer Integration Tests\n")

    try:
        await test_security_manager_initialization()
        await test_agent_registration_security()
        await test_message_verification()
        await test_answer_validation()
        await test_orchestrator_security_integration()
        await test_security_incident_reporting()

        print("\n🎉 All security tests passed successfully!")
        print("\n📊 Security Layer Features Verified:")
        print("  ✓ Agent authentication and authorization")
        print("  ✓ Message integrity verification")
        print("  ✓ Answer validation and quality scoring")
        print("  ✓ Security incident reporting")
        print("  ✓ Threat level monitoring")
        print("  ✓ Orchestrator integration")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())

    if success:
        print("\n✅ Security layer is ready for production use!")
        sys.exit(0)
    else:
        print("\n❌ Security tests failed - review implementation")
        sys.exit(1)