"""Code feedback assistant microservice."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from swarm.agents.base_assistant import BaseAssistant
from .prompts import CODE_FEEDBACK_SYSTEM_PROMPT
from .tools import get_code_feedback_tools
from swarm.orchestration.feedback_graph import FeedbackGraph
from swarm_system.learning import CodeFeedbackLoop

logger = logging.getLogger(__name__)


class CodeFeedbackAssistant(BaseAssistant):
    """Assistant that runs the generator/discriminator/agitator feedback pipeline."""

    def __init__(
        self,
        assistant_id: str,
        model_name: str = "gemma:270m",
        host: str = "localhost:11434",
        feedback_loop: Optional[CodeFeedbackLoop] = None,
        feedback_graph: Optional[FeedbackGraph] = None,
    ):
        super().__init__(
            assistant_id=assistant_id,
            assistant_type="code_feedback",
            capabilities=[
                "code_commentary",
                "gan_feedback",
                "reward_analysis",
                "guidance_generation",
            ],
            model_name=model_name,
            host=host,
        )
        self.feedback_loop = feedback_loop or CodeFeedbackLoop()
        self.feedback_graph = feedback_graph or FeedbackGraph(self.feedback_loop)

    def get_system_prompt(self) -> str:
        return CODE_FEEDBACK_SYSTEM_PROMPT

    def get_tools(self) -> List[Any]:
        return get_code_feedback_tools()

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        context = task.get("context", {}) or {}
        code = context.get("code")
        file_path = context.get("file_path") or task.get("file_path")

        if code and file_path:
            iterations = int(context.get("iterations", 1) or 1)
            start_index = int(context.get("iteration_offset", 0) or 0)
            loop_metadata = context.get("metadata") or {}

            cache_key = f"{file_path}:{hash(code)}:{iterations}:{start_index}"
            if cache_key in self.results_cache:
                logger.info("Returning cached feedback results for %s", file_path)
                cached = self.results_cache[cache_key]
                cached["task_id"] = task.get("task_id")
                return cached

            if iterations <= 0:
                iterations = 1

            if iterations == 1:
                iteration_payload = await self.feedback_graph.run_iteration(
                    file_path=file_path,
                    code=code,
                    iteration_index=start_index,
                    metadata=loop_metadata,
                )
                iterations_payload = [iteration_payload]
            else:
                iterations_payload = await self.feedback_graph.run_iterations(
                    file_path=file_path,
                    code=code,
                    count=iterations,
                    start_index=start_index,
                    metadata=loop_metadata,
                )

            last_iteration = iterations_payload[-1]
            discriminator = last_iteration.get("discriminator_score", {})
            summary = last_iteration.get("generator_output", {}).get("overall_summary")

            response = {
                "task_id": task.get("task_id", "unknown"),
                "assistant_id": self.assistant_id,
                "status": "completed",
                "file_path": file_path,
                "iterations": iterations_payload,
                "reward": discriminator.get("reward"),
                "coverage": discriminator.get("coverage"),
                "accuracy": discriminator.get("accuracy"),
                "coherence": discriminator.get("coherence"),
                "formatting": discriminator.get("formatting"),
                "summary": summary,
                "graph": last_iteration.get("graph", self.feedback_graph.describe()),
            }

            self.results_cache[cache_key] = response
            if len(self.results_cache) > 100:
                oldest_key = next(iter(self.results_cache))
                self.results_cache.pop(oldest_key, None)

            return response

        return await super().process_task(task)