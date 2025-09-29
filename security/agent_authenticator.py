"""Agent authentication system with token-based security.

Provides secure authentication for agents communicating with the orchestrator
and validates agent identity on each interaction.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentCredentials:
    """Secure credentials for an authenticated agent."""
    agent_id: str
    agent_type: str
    token: str
    secret_key: str
    issued_at: datetime
    expires_at: datetime
    capabilities: list
    trust_level: float = 1.0
    is_revoked: bool = False

class AgentAuthenticator:
    """Handles agent authentication and token management."""

    def __init__(self, master_secret: Optional[str] = None, token_ttl: int = 3600):
        """Initialize authenticator.

        Args:
            master_secret: Master secret for signing (generated if None)
            token_ttl: Token time-to-live in seconds (default 1 hour)
        """
        self.master_secret = master_secret or secrets.token_hex(32)
        self.token_ttl = token_ttl
        self.active_tokens: Dict[str, AgentCredentials] = {}
        self.revoked_tokens: set = set()
        self.agent_registry: Dict[str, Dict[str, Any]] = {}

        # For test compatibility
        self.authenticated_agents: Dict[str, AgentCredentials] = {}

        # Security metrics
        self.auth_attempts = 0
        self.successful_auths = 0
        self.failed_auths = 0
        self.revocations = 0

    async def register_agent(self, agent_id: str, agent_type: str,
                           capabilities: list, trust_level: float = 1.0) -> bool:
        """Register a new agent in the system.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (research, creative, etc.)
            capabilities: List of agent capabilities
            trust_level: Initial trust level (0.0-1.0)

        Returns:
            True if registration successful
        """
        if agent_id in self.agent_registry:
            logger.warning(f"Agent {agent_id} already registered")
            return False

        self.agent_registry[agent_id] = {
            "agent_type": agent_type,
            "capabilities": capabilities,
            "trust_level": trust_level,
            "registered_at": datetime.now(),
            "last_auth": None,
            "auth_count": 0
        }

        logger.info(f"Registered agent {agent_id} ({agent_type}) with trust level {trust_level}")
        return True

    async def authenticate_agent(self, agent_id: str, agent_type: str) -> Optional[AgentCredentials]:
        """Authenticate an agent and issue credentials.

        Args:
            agent_id: Agent identifier
            agent_type: Expected agent type

        Returns:
            AgentCredentials if authentication successful, None otherwise
        """
        self.auth_attempts += 1

        # Check if agent is registered
        if agent_id not in self.agent_registry:
            logger.warning(f"Authentication failed: Agent {agent_id} not registered")
            self.failed_auths += 1
            return None

        agent_info = self.agent_registry[agent_id]

        # Verify agent type matches
        if agent_info["agent_type"] != agent_type:
            logger.warning(f"Authentication failed: Agent type mismatch for {agent_id}")
            self.failed_auths += 1
            return None

        # Check trust level
        if agent_info["trust_level"] < 0.1:  # Minimum trust threshold
            logger.warning(f"Authentication failed: Agent {agent_id} trust level too low")
            self.failed_auths += 1
            return None

        # Generate secure token and secret
        token = self._generate_token(agent_id, agent_type)
        secret_key = secrets.token_hex(16)

        # Create credentials
        now = datetime.now()
        credentials = AgentCredentials(
            agent_id=agent_id,
            agent_type=agent_type,
            token=token,
            secret_key=secret_key,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.token_ttl),
            capabilities=agent_info["capabilities"],
            trust_level=agent_info["trust_level"]
        )

        # Store active token
        self.active_tokens[token] = credentials

        # Store in authenticated agents for test compatibility
        self.authenticated_agents[agent_id] = credentials

        # Update registry
        agent_info["last_auth"] = now
        agent_info["auth_count"] += 1

        self.successful_auths += 1
        logger.info(f"Authenticated agent {agent_id} with token {token[:8]}...")

        return credentials

    def _generate_token(self, agent_id: str, agent_type: str) -> str:
        """Generate a secure token for an agent."""
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(8)

        # Create token payload
        payload = f"{agent_id}:{agent_type}:{timestamp}:{nonce}"

        # Sign with master secret
        signature = hmac.new(
            self.master_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}:{signature}"

    async def verify_token(self, token: str) -> Optional[AgentCredentials]:
        """Verify a token and return credentials if valid.

        Args:
            token: Token to verify

        Returns:
            AgentCredentials if valid, None otherwise
        """
        # Check if token is revoked
        if token in self.revoked_tokens:
            logger.warning(f"Token verification failed: Token {token[:8]}... is revoked")
            return None

        # Check if token exists
        if token not in self.active_tokens:
            logger.warning(f"Token verification failed: Token {token[:8]}... not found")
            return None

        credentials = self.active_tokens[token]

        # Check expiration
        if datetime.now() > credentials.expires_at:
            logger.warning(f"Token verification failed: Token {token[:8]}... expired")
            await self._remove_token(token)
            return None

        # Verify token signature
        if not self._verify_token_signature(token):
            logger.warning(f"Token verification failed: Invalid signature for {token[:8]}...")
            await self._remove_token(token)
            return None

        return credentials

    def _verify_token_signature(self, token: str) -> bool:
        """Verify the signature of a token."""
        try:
            parts = token.split(":")
            if len(parts) != 5:
                return False

            agent_id, agent_type, timestamp, nonce, signature = parts
            payload = f"{agent_id}:{agent_type}:{timestamp}:{nonce}"

            expected_signature = hmac.new(
                self.master_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.error(f"Error verifying token signature: {e}")
            return False

    async def revoke_token(self, token: str, reason: str = "Manual revocation") -> bool:
        """Revoke a token.

        Args:
            token: Token to revoke
            reason: Reason for revocation

        Returns:
            True if revocation successful
        """
        if token in self.active_tokens:
            credentials = self.active_tokens[token]
            await self._remove_token(token)
            self.revoked_tokens.add(token)
            self.revocations += 1

            logger.info(f"Revoked token for agent {credentials.agent_id}: {reason}")
            return True

        return False

    async def revoke_agent_tokens(self, agent_id: str, reason: str = "Agent revocation") -> int:
        """Revoke all tokens for an agent.

        Args:
            agent_id: Agent identifier
            reason: Reason for revocation

        Returns:
            Number of tokens revoked
        """
        revoked_count = 0
        tokens_to_revoke = []

        # Find all tokens for this agent
        for token, credentials in self.active_tokens.items():
            if credentials.agent_id == agent_id:
                tokens_to_revoke.append(token)

        # Revoke each token
        for token in tokens_to_revoke:
            if await self.revoke_token(token, reason):
                revoked_count += 1

        logger.info(f"Revoked {revoked_count} tokens for agent {agent_id}: {reason}")
        return revoked_count

    async def _remove_token(self, token: str):
        """Remove a token from active tokens."""
        if token in self.active_tokens:
            del self.active_tokens[token]

    async def cleanup_expired_tokens(self):
        """Remove expired tokens from active tokens."""
        now = datetime.now()
        expired_tokens = []

        for token, credentials in self.active_tokens.items():
            if now > credentials.expires_at:
                expired_tokens.append(token)

        for token in expired_tokens:
            await self._remove_token(token)

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")

    async def refresh_token(self, old_token: str) -> Optional[AgentCredentials]:
        """Refresh an existing token.

        Args:
            old_token: Current token to refresh

        Returns:
            New AgentCredentials if successful
        """
        credentials = await self.verify_token(old_token)
        if not credentials:
            return None

        # Revoke old token
        await self.revoke_token(old_token, "Token refresh")

        # Issue new token
        return await self.authenticate_agent(credentials.agent_id, credentials.agent_type)

    def update_agent_trust(self, agent_id: str, trust_delta: float, reason: str = ""):
        """Update agent trust level.

        Args:
            agent_id: Agent identifier
            trust_delta: Change in trust level (-1.0 to 1.0)
            reason: Reason for trust change
        """
        if agent_id not in self.agent_registry:
            return

        old_trust = self.agent_registry[agent_id]["trust_level"]
        new_trust = max(0.0, min(1.0, old_trust + trust_delta))

        self.agent_registry[agent_id]["trust_level"] = new_trust

        logger.info(f"Updated trust for agent {agent_id}: {old_trust:.2f} -> {new_trust:.2f} ({reason})")

        # Revoke tokens if trust falls too low
        if new_trust < 0.1:
            asyncio.create_task(self.revoke_agent_tokens(agent_id, f"Trust level too low: {new_trust:.2f}"))

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return {
            "auth_attempts": self.auth_attempts,
            "successful_auths": self.successful_auths,
            "failed_auths": self.failed_auths,
            "success_rate": self.successful_auths / max(1, self.auth_attempts),
            "active_tokens": len(self.active_tokens),
            "revoked_tokens": len(self.revoked_tokens),
            "revocations": self.revocations,
            "registered_agents": len(self.agent_registry)
        }

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific agent."""
        if agent_id not in self.agent_registry:
            return None

        agent_info = self.agent_registry[agent_id]
        active_tokens = [
            token for token, creds in self.active_tokens.items()
            if creds.agent_id == agent_id
        ]

        return {
            "agent_id": agent_id,
            "agent_type": agent_info["agent_type"],
            "trust_level": agent_info["trust_level"],
            "capabilities": agent_info["capabilities"],
            "registered_at": agent_info["registered_at"],
            "last_auth": agent_info["last_auth"],
            "auth_count": agent_info["auth_count"],
            "active_tokens": len(active_tokens)
        }