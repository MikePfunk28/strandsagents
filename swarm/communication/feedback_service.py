"""Service wrapper for handling feedback requests via workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from graph.feedback_workflow import FeedbackWorkflow


class FeedbackAgentService:
    """Handles incoming feedback requests using `FeedbackWorkflow`."""

    def __init__(self, workflow: Optional[FeedbackWorkflow] = None) -> None:
        self.workflow = workflow or FeedbackWorkflow()

    async def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        file_path = payload.get("file_path", "unknown.py")
        code = payload.get("code", "")
        iterations = int(payload.get("iterations", 1) or 1)
        metadata = payload.get("metadata", {})

        state = await self.workflow.run(
            file_path=file_path,
            code=code,
            iterations=iterations,
            metadata=metadata,
        )

        latest = (state.results.get("iterations") or [{}])[-1]
        score = latest.get("discriminator_score", {})
        return {
            "file_path": file_path,
            "iterations": state.results.get("iterations"),
            "reward": score.get("reward"),
            "summary": latest.get("generator_output", {}).get("overall_summary"),
            "run": state.results.get("run"),
        }
