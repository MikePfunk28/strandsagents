"""Agent-to-agent feedback channel utilities."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from .inmemory_mcp import InMemoryMCPClient


class FeedbackAgentChannel:
    """Helper for requesting feedback results from a code feedback agent."""

    def __init__(self, client: InMemoryMCPClient, target_agent_id: str = "code_feedback") -> None:
        self.client = client
        self.target_agent_id = target_agent_id
        self._pending: Dict[str, asyncio.Future] = {}

    async def connect(self) -> None:
        await self.client.connect()
        self.client.register_message_handler("feedback_response", self._handle_response)

    async def request_feedback(
        self,
        *,
        file_path: str,
        code: str,
        iterations: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        correlation_id = str(uuid.uuid4())
        future: asyncio.Future = loop.create_future()
        self._pending[correlation_id] = future
        await self.client.send_message(
            recipient_id=self.target_agent_id,
            message_type="feedback_request",
            payload={
                "file_path": file_path,
                "code": code,
                "iterations": iterations,
                "metadata": metadata or {},
            },
            correlation_id=correlation_id,
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(correlation_id, None)

    async def _handle_response(self, message: Dict[str, Any]) -> None:
        correlation_id = message.get("correlation_id")
        future = self._pending.get(correlation_id)
        if future and not future.done():
            future.set_result(message.get("payload", {}))


async def register_feedback_responder(
    client: InMemoryMCPClient,
    handler,
) -> None:
    """Register a coroutine handler that processes feedback requests."""
    await client.connect()

    async def _handle_request(message: Dict[str, Any]) -> None:
        payload = message.get("payload", {})
        response = await handler(payload)
        await client.send_message(
            recipient_id=message.get("sender_id"),
            message_type="feedback_response",
            payload=response,
            correlation_id=message.get("correlation_id"),
        )

    client.register_message_handler("feedback_request", _handle_request)
