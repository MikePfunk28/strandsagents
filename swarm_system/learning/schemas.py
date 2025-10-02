"""Data structures supporting GAN-style learning pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LineAnnotation:
    line_number: int
    content: str
    explanation: str
    scope: str
    critique: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "LineAnnotation":
        return LineAnnotation(
            line_number=int(payload.get("line_number", 0)),
            content=str(payload.get("content", "")),
            explanation=str(payload.get("explanation", "")),
            scope=str(payload.get("scope", "global")),
            critique=payload.get("critique"),
            metadata={k: v for k, v in payload.items() if k not in {"line_number", "content", "explanation", "scope", "critique"}},
        )


@dataclass
class ScopeAnnotation:
    name: str
    scope_type: str
    start_line: Optional[int]
    end_line: Optional[int]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ScopeAnnotation":
        return ScopeAnnotation(
            name=str(payload.get("name", "")),
            scope_type=str(payload.get("scope_type", "block")),
            start_line=payload.get("start_line"),
            end_line=payload.get("end_line"),
            summary=str(payload.get("summary", "")),
            metadata={k: v for k, v in payload.items() if k not in {"name", "scope_type", "start_line", "end_line", "summary"}},
        )


@dataclass
class GeneratorOutput:
    file_path: str
    raw_response: str
    line_annotations: List[LineAnnotation] = field(default_factory=list)
    scope_annotations: List[ScopeAnnotation] = field(default_factory=list)
    overall_summary: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "GeneratorOutput":
        return GeneratorOutput(
            file_path=str(payload.get("file_path", "")),
            raw_response=str(payload.get("raw_response", "")),
            line_annotations=[LineAnnotation.from_dict(item) for item in payload.get("line_annotations", [])],
            scope_annotations=[ScopeAnnotation.from_dict(item) for item in payload.get("scope_annotations", [])],
            overall_summary=payload.get("overall_summary"),
            confidence=payload.get("confidence"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class DiscriminatorScore:
    coverage: float
    accuracy: float
    coherence: float
    formatting: float
    reward: float
    issues: List[str] = field(default_factory=list)
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "DiscriminatorScore":
        return DiscriminatorScore(
            coverage=float(payload.get("coverage", 0.0)),
            accuracy=float(payload.get("accuracy", 0.0)),
            coherence=float(payload.get("coherence", 0.0)),
            formatting=float(payload.get("formatting", 0.0)),
            reward=float(payload.get("reward", 0.0)),
            issues=[str(item) for item in payload.get("issues", [])],
            raw_response=str(payload.get("raw_response", "")),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class AgitatorFeedback:
    guidance: str
    improvement_plan: List[str]
    targeted_prompts: List[str]
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AgitatorFeedback":
        return AgitatorFeedback(
            guidance=str(payload.get("guidance", "")),
            improvement_plan=[str(item) for item in payload.get("improvement_plan", [])],
            targeted_prompts=[str(item) for item in payload.get("targeted_prompts", [])],
            raw_response=str(payload.get("raw_response", "")),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class IterationRecord:
    generator_output: GeneratorOutput
    discriminator_score: DiscriminatorScore
    agitator_feedback: AgitatorFeedback
    iteration_index: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "IterationRecord":
        return IterationRecord(
            generator_output=GeneratorOutput.from_dict(payload.get("generator_output", {})),
            discriminator_score=DiscriminatorScore.from_dict(payload.get("discriminator_score", {})),
            agitator_feedback=AgitatorFeedback.from_dict(payload.get("agitator_feedback", {})),
            iteration_index=int(payload.get("iteration_index", 0)),
            timestamp=float(payload.get("timestamp", 0.0)),
            metadata=payload.get("metadata", {}),
        )