"""In-memory MCP-style broker for agent-to-agent communication."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Optional

MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class InMemoryMCPServer:
    """Minimal in-memory message router mimicking MCP semantics."""

    def __init__(self) -> None:
        self._handlers: Dict[str, MessageHandler] = {}
        self._subscriptions: Dict[str, set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register_agent(self, agent_id: str, handler: MessageHandler) -> None:
        async with self._lock:
            self._handlers[agent_id] = handler

    async def unregister_agent(self, agent_id: str) -> None:
        async with self._lock:
            self._handlers.pop(agent_id, None)
            self._subscriptions.pop(agent_id, None)

    async def subscribe(self, agent_id: str, message_type: str) -> None:
        async with self._lock:
            self._subscriptions[message_type].add(agent_id)

    async def send(self, sender_id: str, recipient_id: str, message_type: str,
                   payload: Dict[str, Any], correlation_id: Optional[str] = None) -> str:
        message = {
            "type": message_type,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "payload": payload,
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id or str(uuid.uuid4()),
        }
        if recipient_id == "*":
            await self._broadcast(sender_id, message)
        else:
            handler = self._handlers.get(recipient_id)
            if handler:
                await handler(message)
        return message["correlation_id"]

    async def _broadcast(self, sender_id: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            targets = [
                agent_id for agent_id, handler in self._handlers.items()
                if agent_id != sender_id and agent_id in self._subscriptions.get(message["type"], set())
            ]
        for agent_id in targets:
            handler = self._handlers.get(agent_id)
            if handler:
                await handler(dict(message, recipient_id=agent_id))


class InMemoryMCPClient:
    """Client interface compatible with the in-memory MCP server."""

    def __init__(self, agent_id: str, server: InMemoryMCPServer):
        self.agent_id = agent_id
        self.server = server
        self.handlers: Dict[str, MessageHandler] = {}
        self._inbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def connect(self) -> None:
        await self.server.register_agent(self.agent_id, self._handle_message)

    async def disconnect(self) -> None:
        await self.server.unregister_agent(self.agent_id)

    def register_message_handler(self, message_type: str, handler: MessageHandler) -> None:
        self.handlers[message_type] = handler

    async def subscribe_to_messages(self, message_types: list[str]) -> None:
        for message_type in message_types:
            await self.server.subscribe(self.agent_id, message_type)

    async def send_message(self, recipient_id: str, message_type: str,
                          payload: Dict[str, Any], correlation_id: Optional[str] = None) -> str:
        return await self.server.send(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )

    async def get_next_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        handler = self.handlers.get(message["type"])
        if handler:
            await handler(message)
        else:
            await self._inbox.put(message)
