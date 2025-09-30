"""Bridge adaptive benchmarking with agent-to-agent challenge requests."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional

from .adaptive_benchmark import AdaptiveFeedbackBenchmark


class AdaptiveChallengeManager:
    """Serve adaptive challenge sets to agents over an MCP-style channel."""

    def __init__(
        self,
        benchmark: AdaptiveFeedbackBenchmark,
        client: Any,
        *,
        manager_id: str = "adaptive_benchmark_manager",
        request_message_type: str = "challenge_request",
        response_message_type: str = "challenge_response",
    ) -> None:
        self.benchmark = benchmark
        self.client = client
        self.manager_id = manager_id
        self.request_message_type = request_message_type
        self.response_message_type = response_message_type
        self._connected = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Connect the underlying client and register handlers."""
        if hasattr(self.client, "connect"):
            await self.client.connect()

        if hasattr(self.client, "register_message_handler"):
            self.client.register_message_handler(
                self.request_message_type,
                self._handle_challenge_request,
            )

        # Subscribe when supported so broadcast requests are received
        if hasattr(self.client, "subscribe_to_messages"):
            await self.client.subscribe_to_messages([self.request_message_type])

        self._connected = True

    async def stop(self) -> None:
        """Disconnect the client when it exposes a disconnect coroutine."""
        if hasattr(self.client, "disconnect"):
            await self.client.disconnect()
        self._connected = False

    def ingest_run(self, run_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store a new workflow run and return the refreshed benchmark summary."""
        self.benchmark.update_from_run(run_payload)
        return self.benchmark.summarise()

    async def _handle_challenge_request(self, message: Dict[str, Any]) -> None:
        """Respond to `challenge_request` messages with adaptive challenges."""
        async with self._lock:
            payload = message.get("payload", {})
            top_n = int(payload.get("top_n", 5) or 5)
            file_filter = payload.get("file_filter")
            include_summary = payload.get("include_summary", True)

            challenges = self._filter_challenges(
                self.benchmark.build_challenge_set(top_n=top_n),
                file_filter,
            )

            response_payload: Dict[str, Any] = {
                "challenges": challenges,
                "generated_at": time.time(),
                "request": {
                    "top_n": top_n,
                    "file_filter": file_filter,
                },
            }

            if include_summary:
                response_payload["summary"] = self.benchmark.summarise()

            await self._dispatch_response(
                recipient_id=message.get("sender_id"),
                correlation_id=message.get("correlation_id"),
                payload=response_payload,
            )

    def _filter_challenges(
        self,
        challenges: List[Dict[str, Any]],
        file_filter: Optional[Iterable[str]],
    ) -> List[Dict[str, Any]]:
        if not file_filter:
            return challenges

        if isinstance(file_filter, str):
            filters = {file_filter}
        else:
            filters = {item for item in file_filter if isinstance(item, str)}

        return [
            challenge
            for challenge in challenges
            if challenge.get("file_path") in filters
        ]

    async def _dispatch_response(
        self,
        recipient_id: Optional[str],
        correlation_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        """Send a response message via the client."""
        if recipient_id is None:
            return

        if hasattr(self.client, "send_message"):
            await self.client.send_message(
                recipient_id=recipient_id,
                message_type=self.response_message_type,
                payload=payload,
                correlation_id=correlation_id,
            )
        elif hasattr(self.client, "send"):
            await self.client.send(
                sender_id=self.manager_id,
                recipient_id=recipient_id,
                message_type=self.response_message_type,
                payload=payload,
                correlation_id=correlation_id,
            )

