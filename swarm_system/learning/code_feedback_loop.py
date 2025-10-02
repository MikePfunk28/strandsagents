"""GAN-inspired feedback loop orchestrating generator, discriminator, and agitator."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from swarm_system.assistants.base_assistant import AssistantConfig
from swarm_system.learning.agitator_assistant import CodeCommentAgitatorAssistant
from swarm_system.learning.discriminator_assistant import CodeCommentDiscriminatorAssistant
from swarm_system.learning.generator_assistant import CodeCommentGeneratorAssistant
from swarm_system.learning.schemas import AgitatorFeedback, IterationRecord
from swarm_system.utils.database_manager import db_manager

logger = logging.getLogger(__name__)


class CodeFeedbackLoop:
    """Coordinates generator, discriminator, and agitator for continual learning."""

    def __init__(
        self,
        generator_config: Optional[AssistantConfig] = None,
        discriminator_config: Optional[AssistantConfig] = None,
        agitator_config: Optional[AssistantConfig] = None,
        max_guidance_history: int = 5,
    ) -> None:
        self.generator = CodeCommentGeneratorAssistant(
            generator_config
            or AssistantConfig(
                name="codegan_generator",
                description="Produces line-level commentary",
                model_id="llama3.2",
            )
        )
        self.discriminator = CodeCommentDiscriminatorAssistant(
            discriminator_config
            or AssistantConfig(
                name="codegan_discriminator",
                description="Evaluates generator commentary",
                model_id="llama3.2",
            )
        )
        self.agitator = CodeCommentAgitatorAssistant(
            agitator_config
            or AssistantConfig(
                name="codegan_agitator",
                description="Challenger that supplies improvement prompts",
                model_id="gemma2:27b",
                temperature=0.9,
            )
        )

        self.guidance_history: Dict[str, Deque[str]] = {}
        self.max_guidance_history = max_guidance_history

    async def run_iteration_async(
        self,
        file_path: str,
        code: str,
        iteration_index: int = 0,
    ) -> IterationRecord:
        logger.info("Starting feedback iteration %s for %s", iteration_index, file_path)
        guidance = self._compose_guidance(file_path)
        generator_output = await self.generator.execute_async(
            {"file_path": file_path, "code": code, "guidance": guidance}
        )
        discriminator_score = await self.discriminator.execute_async(
            {"generator_output": generator_output, "code": code}
        )
        agitator_feedback = await self.agitator.execute_async(
            {
                "generator_output": generator_output,
                "discriminator_score": discriminator_score,
                "code": code,
            }
        )

        self._update_guidance(file_path, agitator_feedback)

        record = IterationRecord(
            generator_output=generator_output,
            discriminator_score=discriminator_score,
            agitator_feedback=agitator_feedback,
            iteration_index=iteration_index,
            timestamp=time.time(),
            metadata={"guidance": guidance},
        )
        self._persist_iteration(record)
        return record

    def run_iteration(self, file_path: str, code: str, iteration_index: int = 0) -> IterationRecord:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = loop.create_task(self.run_iteration_async(file_path, code, iteration_index))
                return task
            return loop.run_until_complete(self.run_iteration_async(file_path, code, iteration_index))
        except RuntimeError:
            return asyncio.run(self.run_iteration_async(file_path, code, iteration_index))

    async def train_async(
        self,
        dataset: Iterable[Tuple[str, str]],
        epochs: int = 1,
    ) -> List[IterationRecord]:
        results: List[IterationRecord] = []
        for epoch in range(epochs):
            for index, (file_path, code) in enumerate(dataset):
                record = await self.run_iteration_async(file_path, code, iteration_index=epoch * 1000 + index)
                results.append(record)
        return results

    def train(self, dataset: Iterable[Tuple[str, str]], epochs: int = 1) -> List[IterationRecord]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = loop.create_task(self.train_async(dataset, epochs))
                return task
            return loop.run_until_complete(self.train_async(dataset, epochs))
        except RuntimeError:
            return asyncio.run(self.train_async(dataset, epochs))

    def _compose_guidance(self, file_path: str) -> Optional[str]:
        history = self.guidance_history.get(file_path)
        if not history:
            return None
        return " | ".join(list(history))

    def _update_guidance(self, file_path: str, feedback: AgitatorFeedback) -> None:
        if not feedback.targeted_prompts and not feedback.guidance:
            return
        history = self.guidance_history.setdefault(file_path, deque(maxlen=self.max_guidance_history))
        if feedback.guidance:
            history.append(feedback.guidance)
        for prompt in feedback.targeted_prompts:
            history.append(prompt)

    def _persist_iteration(self, record: IterationRecord) -> None:
        iteration_payload = {
            "iteration_index": record.iteration_index,
            "timestamp": record.timestamp,
            "metadata": record.metadata,
            "generator_output": asdict(record.generator_output),
            "discriminator_score": asdict(record.discriminator_score),
            "agitator_feedback": asdict(record.agitator_feedback),
        }

        db_manager.store_memory(
            session_id=f"codegan::{record.generator_output.file_path}",
            content="Iteration record stored",
            memory_type="feedback_iteration",
            importance_score=record.discriminator_score.reward,
            metadata={
                "iteration_index": record.iteration_index,
                "score": record.discriminator_score.reward,
                "timestamp": record.timestamp,
            },
        )
        db_manager.store_knowledge(
            topic="code_gan_iteration",
            subtopic=record.generator_output.file_path,
            content="Feedback loop iteration",
            source="code_feedback_loop",
            confidence=record.discriminator_score.reward,
            metadata=iteration_payload,
        )