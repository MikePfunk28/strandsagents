"""Discriminator assistant that evaluates generator outputs."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from swarm_system.assistants.base_assistant import AssistantConfig, BaseAssistant
from swarm_system.learning.schemas import DiscriminatorScore, GeneratorOutput
from swarm_system.utils.database_manager import db_manager

logger = logging.getLogger(__name__)


class CodeCommentDiscriminatorAssistant(BaseAssistant):
    """Scores generator annotations and surfaces issues."""

    def __init__(self, config: AssistantConfig) -> None:
        config.temperature = min(config.temperature, 0.3)
        super().__init__(config)

    def get_default_prompt(self) -> str:
        return (
            "You are CodeGAN-Discriminator. Assess JSON annotations for a source file.\n"
            "Respond with JSON:"
            '{"coverage": number (0-1), "accuracy": number (0-1), "coherence": number (0-1), '
            '"formatting": number (0-1), "issues": [str], "justification": str, '
            '"reward_breakdown": {"coverage": number, "accuracy": number, "coherence": number, "formatting": number}}\n'
            "Reward breakdown values must be 0-1 multipliers that sum to <= 1."
        )

    async def execute_async(self, input_data: Dict[str, Any], **kwargs: Any) -> DiscriminatorScore:
        generator_output = input_data.get("generator_output")
        code = input_data.get("code")
        if not isinstance(generator_output, GeneratorOutput):
            raise TypeError("Discriminator requires GeneratorOutput instance")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Discriminator requires original code string")

        prompt = self._build_prompt(code, generator_output)
        response = await asyncio.to_thread(self.agent, prompt)
        raw_text = str(response)
        payload = self._parse_json(raw_text)
        score = self._to_dataclass(payload, raw_text)
        self._persist(score, generator_output.file_path)
        return score

    def _build_prompt(self, code: str, generator_output: GeneratorOutput) -> str:
        numbered_lines = []
        for idx, line in enumerate(code.splitlines(), start=1):
            numbered_lines.append(f"{idx:04d}: {line}")
        numbered_code = "\n".join(numbered_lines)
        return (
            "Evaluate the following code annotations.\n\n"
            "CODE:\n" + numbered_code + "\n\n"
            "ANNOTATIONS:\n" + json.dumps(self._generator_to_json(generator_output), indent=2)
        )

    def _generator_to_json(self, output: GeneratorOutput) -> Dict[str, Any]:
        return {
            "file_path": output.file_path,
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
            "scope_annotations": [
                {
                    "name": scope.name,
                    "scope_type": scope.scope_type,
                    "start_line": scope.start_line,
                    "end_line": scope.end_line,
                    "summary": scope.summary,
                }
                for scope in output.scope_annotations
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
            logger.error("Discriminator returned invalid JSON: %s", raw_text)
            raise ValueError("Discriminator produced invalid JSON") from exc

    def _to_dataclass(self, payload: Dict[str, Any], raw_text: str) -> DiscriminatorScore:
        coverage = self._bounded_float(payload.get("coverage"))
        accuracy = self._bounded_float(payload.get("accuracy"))
        coherence = self._bounded_float(payload.get("coherence"))
        formatting = self._bounded_float(payload.get("formatting"))

        reward = self._compute_reward(payload.get("reward_breakdown"), coverage, accuracy, coherence, formatting)
        issues = [str(item) for item in payload.get("issues", [])]

        return DiscriminatorScore(
            coverage=coverage,
            accuracy=accuracy,
            coherence=coherence,
            formatting=formatting,
            reward=reward,
            issues=issues,
            raw_response=raw_text,
            metadata={k: v for k, v in payload.items() if k not in {"coverage", "accuracy", "coherence", "formatting", "issues", "reward_breakdown"}},
        )

    def _compute_reward(
        self,
        reward_breakdown: Optional[Dict[str, Any]],
        coverage: float,
        accuracy: float,
        coherence: float,
        formatting: float,
    ) -> float:
        if isinstance(reward_breakdown, dict):
            total = 0.0
            for key, metric in (
                ("coverage", coverage),
                ("accuracy", accuracy),
                ("coherence", coherence),
                ("formatting", formatting),
            ):
                weight = self._bounded_float(reward_breakdown.get(key), default=0.25)
                total += weight * metric
            return min(max(total, 0.0), 1.0)
        return min(max((coverage + accuracy * 1.5 + coherence + formatting) / 4.5, 0.0), 1.0)

    def _bounded_float(self, value: Any, default: float = 0.0) -> float:
        try:
            num = float(value)
            if num < 0.0:
                return 0.0
            if num > 1.0:
                return 1.0
            return num
        except (TypeError, ValueError):
            return default

    def _persist(self, score: DiscriminatorScore, file_path: str) -> None:
        db_manager.store_memory(
            session_id=f"discriminator::{file_path}",
            content=json.dumps(
                {
                    "coverage": score.coverage,
                    "accuracy": score.accuracy,
                    "coherence": score.coherence,
                    "formatting": score.formatting,
                    "reward": score.reward,
                    "issues": score.issues,
                }
            ),
            memory_type="discriminator_feedback",
            importance_score=score.reward,
            metadata={"file_path": file_path},
        )