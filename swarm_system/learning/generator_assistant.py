"""Generator assistant that produces line-by-line code explanations."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from swarm_system.assistants.base_assistant import AssistantConfig, BaseAssistant
from swarm_system.learning.schemas import GeneratorOutput, LineAnnotation, ScopeAnnotation
from swarm_system.utils.database_manager import db_manager

logger = logging.getLogger(__name__)


class CodeCommentGeneratorAssistant(BaseAssistant):
    """Creates structured annotations for source files."""

    def __init__(self, config: AssistantConfig) -> None:
        config.temperature = min(config.temperature, 0.4)
        super().__init__(config)

    def get_default_prompt(self) -> str:
        return (
            "You are CodeGAN-Generator. "
            "Produce strict JSON describing each line of the file and scope summaries. "
            "Always respond with JSON matching this schema:\n"
            '{"overall_summary": str, "confidence": number (0-1), '
            '"line_annotations": [{"line_number": int, "scope": str, "explanation": str, "critique": str|null}], '
            '"scope_annotations": [{"name": str, "scope_type": str, "start_line": int|null, "end_line": int|null, "summary": str}]}\n'
            "Cover every line. Keep explanations concise but precise."
        )

    async def execute_async(self, input_data: Dict[str, Any], **kwargs: Any) -> GeneratorOutput:
        file_path = input_data.get("file_path", "unknown_file")
        code = input_data.get("code", "")
        if not code.strip():
            raise ValueError("Generator requires non-empty code input")

        guidance = input_data.get("guidance")
        prompt = self._build_prompt(file_path, code, guidance)
        response = await asyncio.to_thread(self.agent, prompt)
        raw_text = str(response)
        payload = self._parse_json(raw_text)
        output = self._to_dataclass(payload, file_path, raw_text, code)
        self._persist_annotations(output, code)
        return output

    def _build_prompt(self, file_path: str, code: str, guidance: Optional[str] = None) -> str:
        numbered_lines = []
        for idx, line in enumerate(code.splitlines(), start=1):
            numbered_lines.append(f"{idx:04d}: {line}")
        numbered_code = "\n".join(numbered_lines)

        parts = [
            f"File path: {file_path}\n",
            "Provide exhaustive annotations for the code below.",
            " Use the enforced JSON schema only.",
        ]
        if guidance:
            parts.append(f" Guidance: {guidance}.")
        parts.append("\n\n" + numbered_code + "\n")
        return "".join(parts)

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        try:
            cleaned = raw_text.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Generator returned invalid JSON: %s", raw_text)
            raise ValueError("Generator produced invalid JSON") from exc

    def _to_dataclass(
        self,
        payload: Dict[str, Any],
        file_path: str,
        raw_text: str,
        code: str,
    ) -> GeneratorOutput:
        line_items = []
        for entry in payload.get("line_annotations", []):
            line_items.append(
                LineAnnotation(
                    line_number=int(entry.get("line_number", 0)),
                    content=self._safe_line(code, int(entry.get("line_number", 0))),
                    explanation=str(entry.get("explanation", "")).strip(),
                    scope=str(entry.get("scope", "global")).strip() or "global",
                    critique=(entry.get("critique") or None),
                    metadata={k: v for k, v in entry.items() if k not in {"line_number", "explanation", "scope", "critique"}},
                )
            )

        scope_items = []
        for entry in payload.get("scope_annotations", []):
            scope_items.append(
                ScopeAnnotation(
                    name=str(entry.get("name", "unknown")),
                    scope_type=str(entry.get("scope_type", "block")),
                    start_line=self._safe_int(entry.get("start_line")),
                    end_line=self._safe_int(entry.get("end_line")),
                    summary=str(entry.get("summary", "")).strip(),
                    metadata={k: v for k, v in entry.items() if k not in {"name", "scope_type", "start_line", "end_line", "summary"}},
                )
            )

        return GeneratorOutput(
            file_path=file_path,
            raw_response=raw_text,
            line_annotations=line_items,
            scope_annotations=scope_items,
            overall_summary=payload.get("overall_summary"),
            confidence=self._safe_float(payload.get("confidence")),
            metadata={k: v for k, v in payload.items() if k not in {"line_annotations", "scope_annotations", "overall_summary", "confidence"}},
        )

    def _persist_annotations(self, output: GeneratorOutput, code: str) -> None:
        session_id = f"generator::{output.file_path}"
        db_manager.store_memory(
            session_id=session_id,
            content=output.overall_summary or "",
            memory_type="generator_summary",
            importance_score=output.confidence or 0.5,
            metadata={"file_path": output.file_path},
        )

        for annotation in output.line_annotations:
            db_manager.store_code_explanation(
                file_path=output.file_path,
                line_number=annotation.line_number,
                line_content=annotation.content,
                explanation=annotation.explanation,
                scope=annotation.scope,
                metadata={"critique": annotation.critique},
            )

        for scope in output.scope_annotations:
            db_manager.store_file_scope(
                file_path=output.file_path,
                scope_name=scope.name,
                scope_type=scope.scope_type,
                start_line=scope.start_line,
                end_line=scope.end_line,
                content=self._scope_content(code, scope.start_line, scope.end_line),
            )

    def _scope_content(self, code: str, start: Optional[int], end: Optional[int]) -> str:
        if start is None or end is None or start <= 0 or end < start:
            return ""
        lines = code.splitlines()
        clip = lines[start - 1:end]
        return "\n".join(clip)

    def _safe_line(self, code: str, line_number: int) -> str:
        if line_number <= 0:
            return ""
        lines = code.splitlines()
        if line_number > len(lines):
            return ""
        return lines[line_number - 1]

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None