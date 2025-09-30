"""Message integrity verification system.

Provides cryptographic verification of message integrity and authenticity
to prevent tampering and ensure secure communication between agents.
"""

import hashlib
import hmac
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MessageSignature:
    """Cryptographic signature for a message."""
    message_hash: str
    signature: str
    timestamp: float
    nonce: str
    agent_id: str

@dataclass
class VerificationResult:
    """Result of message verification."""
    is_valid: bool
    trust_score: float
    verification_time: float
    issues: list
    metadata: dict

class MessageVerifier:
    """Handles message integrity verification and signing."""

    def __init__(self, master_key: Optional[str] = None):
        """Initialize message verifier.

        Args:
            master_key: Master key for HMAC signing (generated if None)
        """
        self.master_key = master_key or self._generate_master_key()
        self.agent_keys: Dict[str, str] = {}
        self.message_cache: Dict[str, MessageSignature] = {}
        self.nonce_cache: set = set()

        # Verification metrics
        self.verifications_attempted = 0
        self.verifications_successful = 0
        self.verifications_failed = 0
        self.replay_attacks_detected = 0
        self.tampering_detected = 0

    def _generate_master_key(self) -> str:
        """Generate a cryptographically secure master key."""
        import secrets
        return secrets.token_hex(32)

    async def register_agent_key(self, agent_id: str, agent_key: str) -> bool:
        """Register an agent's signing key.

        Args:
            agent_id: Agent identifier
            agent_key: Agent's cryptographic key

        Returns:
            True if registration successful
        """
        if not agent_id or not agent_key:
            return False

        if len(agent_key) < 32:  # Minimum key length
            logger.warning(f"Agent key for {agent_id} is too short")
            return False

        self.agent_keys[agent_id] = agent_key
        logger.info(f"Registered signing key for agent {agent_id}")
        return True

    async def sign_message(self, message, agent_id: str, timestamp=None) -> Optional[MessageSignature]:
        """Sign a message for integrity verification.

        Args:
            message: Message to sign (Dict or str)
            agent_id: Agent sending the message
            timestamp: Optional timestamp (for test compatibility)

        Returns:
            MessageSignature if successful, None otherwise
        """
        if agent_id not in self.agent_keys:
            logger.error(f"No signing key registered for agent {agent_id}")
            return None

        try:
            # Handle both dict and string messages for test compatibility
            if isinstance(message, str):
                canonical_message = message
            else:
                # Prepare message for signing
                message_copy = message.copy()
                message_copy.pop('signature', None)  # Remove any existing signature
                # Canonicalize message (ensure consistent ordering)
                canonical_message = json.dumps(message_copy, sort_keys=True, separators=(',', ':'))

            # Generate nonce and timestamp (use provided timestamp if available)
            if timestamp is None:
                timestamp = time.time()
            elif hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()

            nonce = self._generate_nonce()

            # Create message hash
            message_hash = hashlib.sha256(
                f"{canonical_message}:{timestamp}:{nonce}".encode()
            ).hexdigest()

            # Create signature using agent key + master key
            agent_key = self.agent_keys[agent_id]
            combined_key = f"{agent_key}:{self.master_key}"

            signature = hmac.new(
                combined_key.encode(),
                f"{message_hash}:{agent_id}".encode(),
                hashlib.sha256
            ).hexdigest()

            message_signature = MessageSignature(
                message_hash=message_hash,
                signature=signature,
                timestamp=timestamp,
                nonce=nonce,
                agent_id=agent_id
            )

            # Cache for replay detection
            self.message_cache[message_hash] = message_signature
            self.nonce_cache.add(nonce)

            logger.debug(f"Signed message from agent {agent_id}")
            return message_signature

        except Exception as e:
            logger.error(f"Failed to sign message for agent {agent_id}: {e}")
            return None

    async def verify_message(self, message: Dict[str, Any], signature: MessageSignature) -> VerificationResult:
        """Verify message integrity and authenticity.

        Args:
            message: Message to verify
            signature: Message signature

        Returns:
            VerificationResult with verification details
        """
        start_time = time.time()
        self.verifications_attempted += 1

        issues = []
        trust_score = 1.0
        metadata = {}

        try:
            # Check if agent key is registered
            if signature.agent_id not in self.agent_keys:
                issues.append(f"Unknown agent: {signature.agent_id}")
                trust_score = 0.0
                self.verifications_failed += 1
                return self._create_verification_result(False, trust_score, start_time, issues, metadata)

            # Check message age (prevent replay attacks)
            message_age = time.time() - signature.timestamp
            if message_age > 300:  # 5 minutes max age
                issues.append(f"Message too old: {message_age:.0f} seconds")
                trust_score *= 0.1

            if message_age < -30:  # Allow 30 seconds clock skew
                issues.append(f"Message from future: {message_age:.0f} seconds")
                trust_score *= 0.1

            # Check for nonce reuse (replay attack detection)
            if signature.nonce in self.nonce_cache:
                # Check if it's the exact same message (legitimate reprocessing)
                if signature.message_hash not in self.message_cache:
                    issues.append("Nonce reuse detected - possible replay attack")
                    trust_score = 0.0
                    self.replay_attacks_detected += 1
                else:
                    # Same message, probably legitimate retry
                    issues.append("Duplicate message detected")
                    trust_score *= 0.8

            # Recreate message hash
            message_copy = message.copy()
            message_copy.pop('signature', None)  # Remove signature field

            canonical_message = json.dumps(message_copy, sort_keys=True, separators=(',', ':'))
            expected_hash = hashlib.sha256(
                f"{canonical_message}:{signature.timestamp}:{signature.nonce}".encode()
            ).hexdigest()

            # Verify hash matches
            if expected_hash != signature.message_hash:
                issues.append("Message hash mismatch - message has been tampered")
                trust_score = 0.0
                self.tampering_detected += 1
                self.verifications_failed += 1
                return self._create_verification_result(False, trust_score, start_time, issues, metadata)

            # Verify signature
            agent_key = self.agent_keys[signature.agent_id]
            combined_key = f"{agent_key}:{self.master_key}"

            expected_signature = hmac.new(
                combined_key.encode(),
                f"{signature.message_hash}:{signature.agent_id}".encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature.signature, expected_signature):
                issues.append("Invalid signature - authentication failed")
                trust_score = 0.0
                self.verifications_failed += 1
                return self._create_verification_result(False, trust_score, start_time, issues, metadata)

            # Additional trust factors
            metadata['message_size'] = len(canonical_message)
            metadata['agent_id'] = signature.agent_id
            metadata['message_age'] = message_age

            # Reduce trust for very large messages (potential DoS)
            if len(canonical_message) > 100000:  # 100KB
                issues.append("Very large message")
                trust_score *= 0.9

            # Successful verification
            if not issues:
                self.verifications_successful += 1
                logger.debug(f"Message from {signature.agent_id} verified successfully")
            else:
                # Partial success with warnings
                logger.warning(f"Message from {signature.agent_id} verified with issues: {issues}")

            return self._create_verification_result(
                trust_score > 0.5, trust_score, start_time, issues, metadata
            )

        except Exception as e:
            logger.error(f"Message verification error: {e}")
            issues.append(f"Verification error: {str(e)}")
            self.verifications_failed += 1
            return self._create_verification_result(False, 0.0, start_time, issues, metadata)

    def _create_verification_result(self, is_valid: bool, trust_score: float,
                                  start_time: float, issues: list, metadata: dict) -> VerificationResult:
        """Create a verification result."""
        return VerificationResult(
            is_valid=is_valid,
            trust_score=trust_score,
            verification_time=time.time() - start_time,
            issues=issues,
            metadata=metadata
        )

    def _generate_nonce(self) -> str:
        """Generate a cryptographic nonce."""
        import secrets
        return secrets.token_hex(16)

    async def cleanup_old_nonces(self, max_age_seconds: int = 3600):
        """Clean up old nonces to prevent memory bloat.

        Args:
            max_age_seconds: Maximum age of nonces to keep
        """
        # For a production system, we'd need to track nonce timestamps
        # For now, we'll just limit the cache size
        if len(self.nonce_cache) > 10000:
            # Keep only the most recent half
            nonce_list = list(self.nonce_cache)
            self.nonce_cache = set(nonce_list[-5000:])
            logger.info("Cleaned up old nonces")

    async def create_signed_message_async(self, message: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
        """Create a message with embedded signature (async version).

        Args:
            message: Original message
            agent_id: Agent creating the message

        Returns:
            Message with embedded signature, or None if signing fails
        """
        signature = await self.sign_message(message, agent_id)
        if not signature:
            return None

        # Embed signature in message
        signed_message = message.copy()
        signed_message['signature'] = {
            'message_hash': signature.message_hash,
            'signature': signature.signature,
            'timestamp': signature.timestamp,
            'nonce': signature.nonce,
            'agent_id': signature.agent_id
        }

        return signed_message

    def create_signed_message(self, message: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
        """Create a message with embedded signature (sync wrapper).

        Args:
            message: Original message
            agent_id: Agent creating the message

        Returns:
            Message with embedded signature, or None if signing fails
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run()
                # Return None and let caller use async version
                logger.warning("create_signed_message called from async context - use create_signed_message_async instead")
                return None
            else:
                return asyncio.run(self.create_signed_message_async(message, agent_id))
        except RuntimeError:
            # Event loop already running
            logger.warning("create_signed_message called from async context - use create_signed_message_async instead")
            return None

    async def extract_and_verify_async(self, signed_message: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], VerificationResult]:
        """Extract message and verify signature (async version).

        Args:
            signed_message: Message with embedded signature

        Returns:
            Tuple of (original_message, verification_result)
        """
        try:
            if 'signature' not in signed_message:
                return None, VerificationResult(
                    is_valid=False,
                    trust_score=0.0,
                    verification_time=0.0,
                    issues=["No signature found in message"],
                    metadata={}
                )

            signature_data = signed_message['signature']
            signature = MessageSignature(
                message_hash=signature_data['message_hash'],
                signature=signature_data['signature'],
                timestamp=signature_data['timestamp'],
                nonce=signature_data['nonce'],
                agent_id=signature_data['agent_id']
            )

            # Extract original message (without signature)
            original_message = signed_message.copy()
            del original_message['signature']

            # Verify
            verification_result = await self.verify_message(original_message, signature)

            return original_message, verification_result

        except Exception as e:
            return None, VerificationResult(
                is_valid=False,
                trust_score=0.0,
                verification_time=0.0,
                issues=[f"Signature extraction error: {str(e)}"],
                metadata={}
            )

    def extract_and_verify(self, signed_message: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], VerificationResult]:
        """Extract message and verify signature (sync wrapper).

        Args:
            signed_message: Message with embedded signature

        Returns:
            Tuple of (original_message, verification_result)
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run()
                logger.warning("extract_and_verify called from async context - use extract_and_verify_async instead")
                return None, VerificationResult(
                    is_valid=False,
                    trust_score=0.0,
                    verification_time=0.0,
                    issues=["Called from async context - use extract_and_verify_async"],
                    metadata={}
                )
            else:
                return asyncio.run(self.extract_and_verify_async(signed_message))
        except RuntimeError:
            # Event loop already running
            logger.warning("extract_and_verify called from async context - use extract_and_verify_async instead")
            return None, VerificationResult(
                is_valid=False,
                trust_score=0.0,
                verification_time=0.0,
                issues=["Called from async context - use extract_and_verify_async"],
                metadata={}
            )

    def get_verification_metrics(self) -> Dict[str, Any]:
        """Get verification metrics."""
        total_attempts = max(1, self.verifications_attempted)

        return {
            "verifications_attempted": self.verifications_attempted,
            "verifications_successful": self.verifications_successful,
            "verifications_failed": self.verifications_failed,
            "success_rate": self.verifications_successful / total_attempts,
            "replay_attacks_detected": self.replay_attacks_detected,
            "tampering_detected": self.tampering_detected,
            "registered_agents": len(self.agent_keys),
            "cached_nonces": len(self.nonce_cache),
            "cached_messages": len(self.message_cache)
        }