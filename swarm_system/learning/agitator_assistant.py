"""Agitator assistant that pushes generator toward better outputs."""

import asyncio
import json
import logging
from typing import Any, Dict

from swarm_system.assistants.base_assistant import AssistantConfig, BaseAssistant
from swarm_system.learning.schemas import AgitatorFeedback, DiscriminatorScore, GeneratorOutput
from swarm_system.utils.database_manager import db_manager

logger = logging.getLogger(__name__)


class CodeCommentAgitatorAssistant(BaseAssistant):
    """Provides corrective coaching informed by discriminator results."""

    def __init__(self, config: AssistantConfig) -> None:
        config.temperature = max(config.temperature, 0.8)
        super().__init__(config)

    def get_default_prompt(self) -> str:
        return (
            "You are CodeGAN-Agitator. Your job is to critique the generator and craft"
            " actionable improvement prompts. Respond with JSON:\n"
            '{"guidance": str, "improvement_plan": [str], "targeted_prompts": [str]}'
        )

    async def execute_async(self, input_data: Dict[str, Any], **kwargs: Any) -> AgitatorFeedback:
        generator_output = input_data.get("generator_output")
        discriminator_score = input_data.get("discriminator_score")
        code = input_data.get("code")

        if not isinstance(generator_output, GeneratorOutput):
            raise TypeError("Agitator requires GeneratorOutput instance")
        if not isinstance(discriminator_score, DiscriminatorScore):
            raise TypeError("Agitator requires DiscriminatorScore instance")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Agitator requires original code string")

        prompt = self._build_prompt(code, generator_output, discriminator_score)
        response = await asyncio.to_thread(self.agent, prompt)
        raw_text = str(response)
        payload = self._parse_json(raw_text)
        feedback = self._to_dataclass(payload, raw_text)
        self._persist(feedback, generator_output.file_path, discriminator_score.reward)
        return feedback

    def _build_prompt(
        self,
        code: str,
        generator_output: GeneratorOutput,
        discriminator_score: DiscriminatorScore,
    ) -> str:
        numbered_code = "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(code.splitlines(), start=1))
        assessment = {
            "coverage": discriminator_score.coverage,
            "accuracy": discriminator_score.accuracy,
            "coherence": discriminator_score.coherence,
            "formatting": discriminator_score.formatting,
            "reward": discriminator_score.reward,
            "issues": discriminator_score.issues,
        }
        return (
            "Coach the generator based on the discriminator assessment.\n\n"
            "CODE:\n" + numbered_code + "\n\n"
            "GENERATOR OUTPUT:\n" + json.dumps(self._generator_to_json(generator_output), indent=2) + "\n\n"
            "DISCRIMINATOR SCORE:\n" + json.dumps(assessment, indent=2) + "\n"
        )

    def _generator_to_json(self, output: GeneratorOutput) -> Dict[str, Any]:
        return {
            "overall_summary": output.overall_summary,
            "confidence": output.confidence,
            "line_annotations": [
                {
                    "line_number": item.line_number,
                    "explanation": item.explanation,
                    "scope": item.scope,
                    "critique": item.critique,
                }
                for item in output.line_annotations
            ],
        }

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        try:
            cleaned = raw_text.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Agitator returned invalid JSON: %s", raw_text)
            raise ValueError("Agitator produced invalid JSON") from exc

    def _to_dataclass(self, payload: Dict[str, Any], raw_text: str) -> AgitatorFeedback:
        improvement_plan = [str(item).strip() for item in payload.get("improvement_plan", []) if str(item).strip()]
        targeted_prompts = [str(item).strip() for item in payload.get("targeted_prompts", []) if str(item).strip()]
        return AgitatorFeedback(
            guidance=str(payload.get("guidance", "")).strip(),
            improvement_plan=improvement_plan,
            targeted_prompts=targeted_prompts,
            raw_response=raw_text,
            metadata={k: v for k, v in payload.items() if k not in {"guidance", "improvement_plan", "targeted_prompts"}},
        )

    def _persist(self, feedback: AgitatorFeedback, file_path: str, reward: float) -> None:
        db_manager.store_knowledge(
            topic="generator_improvement",
            subtopic=file_path,
            content=feedback.guidance,
            source="agitator",
            confidence=reward,
            metadata={
                "improvement_plan": feedback.improvement_plan,
                "targeted_prompts": feedback.targeted_prompts,
            },
        )