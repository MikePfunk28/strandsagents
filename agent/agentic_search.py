#!/usr/bin/env python3
"""
Agentic search utilities for StrandsAgents.

Provides a lightweight multi-step search engine that can reason through a
query, plan actions, retrieve evidence from local knowledge or memory
managers, and synthesise the final answer along with confidence scoring.

This implementation is self-contained and does not require network access,
making it safe to run inside the workspace while still demonstrating the
agentic-search pattern.  External integrations (real web search, APIs, etc.)
can be wired in by extending the `_retrieve_evidence` method.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SearchEvidence:
    """Evidence retrieved during the search process."""

    source: str
    title: str
    content: str
    relevance: float
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["relevance"] = round(self.relevance, 3)
        return payload


@dataclass
class SearchStep:
    """A single reasoning step inside the agentic search flow."""

    description: str
    status: str
    findings: List[SearchEvidence] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "status": self.status,
            "notes": self.notes,
            "findings": [evidence.to_dict() for evidence in self.findings],
        }


@dataclass
class SearchOutcome:
    """Final outcome of the agentic search."""

    query: str
    answer: str
    confidence: float
    steps: List[SearchStep]
    evidence: List[SearchEvidence]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "steps": [step.to_dict() for step in self.steps],
            "evidence": [item.to_dict() for item in self.evidence],
            "generated_at": self.generated_at,
        }


class AgenticSearchEngine:
    """
    Simple agentic search engine that mirrors the multi-step reasoning pattern.

    Parameters
    ----------
    memory_manager:
        Optional memory manager that implements ``search_memory`` and
        ``store_memory`` coroutines similar to the ones in the project.
    knowledge_base:
        Optional iterable of evidence dicts that seed the local knowledge base.
    """

    def __init__(
        self,
        *,
        memory_manager: Any = None,
        knowledge_base: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.knowledge_base: Dict[str, SearchEvidence] = {}

        if knowledge_base:
            for entry in knowledge_base:
                evidence = SearchEvidence(
                    source=entry.get("source", "seed"),
                    title=entry.get("title", entry.get("id", "Untitled Source")),
                    content=entry.get("content", ""),
                    relevance=float(entry.get("relevance", 0.9)),
                    url=entry.get("url"),
                    metadata=entry.get("metadata", {}),
                )
                self.knowledge_base[evidence.title.lower()] = evidence

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def search(self, query: str) -> SearchOutcome:
        """
        Execute an agentic search cycle.

        The engine will:
            1. Analyse the query to understand intent and constraints.
            2. Plan a set of retrieval steps.
            3. Execute each step, retrieving evidence from the knowledge base
               or memory manager.
            4. Synthesise a final answer and estimate confidence.
        """

        analysis = self._analyse_query(query)
        plan = self._plan_steps(analysis)

        steps: List[SearchStep] = []
        collected_evidence: List[SearchEvidence] = []

        for description in plan:
            step = SearchStep(description=description, status="in_progress")
            evidence = await self._retrieve_evidence(query, description, analysis)
            step.findings.extend(evidence)
            step.status = "completed" if evidence else "no_data"
            steps.append(step)
            collected_evidence.extend(evidence)

        answer, confidence = self._synthesise_answer(query, analysis, collected_evidence)

        return SearchOutcome(
            query=query,
            answer=answer,
            confidence=confidence,
            steps=steps,
            evidence=collected_evidence,
        )

    # ------------------------------------------------------------------ #
    # Query Analysis and Planning
    # ------------------------------------------------------------------ #
    def _analyse_query(self, query: str) -> Dict[str, Any]:
        lowered = query.lower()

        intent = "research"
        if any(token in lowered for token in ("how do i", "build", "create")):
            intent = "build"
        elif "optimise" in lowered or "optimize" in lowered:
            intent = "optimise"
        elif "cost" in lowered or "price" in lowered:
            intent = "cost"

        platform = "general"
        if "aws" in lowered:
            platform = "aws"
        elif "azure" in lowered:
            platform = "azure"
        elif "google" in lowered or "gcp" in lowered:
            platform = "gcp"

        components: List[str] = []
        if "lambda" in lowered:
            components.append("lambda")
        if "bedrock" in lowered:
            components.append("bedrock")
        if "dynamodb" in lowered or "database" in lowered:
            components.append("database")
        if "chatbot" in lowered or "agent" in lowered:
            components.append("agent")

        return {
            "query": query,
            "intent": intent,
            "platform": platform,
            "components": components,
        }

    def _plan_steps(self, analysis: Dict[str, Any]) -> List[str]:
        steps = ["Clarify requirements", "Gather relevant references", "Validate constraints"]

        platform = analysis["platform"]
        intent = analysis["intent"]

        if platform == "aws":
            steps.append("Review AWS documentation")
        elif platform == "azure":
            steps.append("Review Azure documentation")
        elif platform == "gcp":
            steps.append("Review Google Cloud documentation")

        if intent == "cost":
            steps.append("Estimate cost implications")
        elif intent == "optimise":
            steps.append("Identify optimisation opportunities")
        elif intent == "build":
            steps.append("Outline implementation steps")

        steps.append("Cross-verify findings")
        steps.append("Prepare final synthesis")
        return steps

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    async def _retrieve_evidence(
        self,
        query: str,
        step_description: str,
        analysis: Dict[str, Any],
    ) -> List[SearchEvidence]:
        evidence: List[SearchEvidence] = []

        evidence.extend(self._search_local_knowledge(query, step_description))

        if self.memory_manager:
            try:
                results = await self.memory_manager.search_memory(
                    query=query,
                    limit=5,
                    metadata_filter={"platform": analysis["platform"]},
                )
                for result in results:
                    chunk = result.chunk
                    evidence.append(
                        SearchEvidence(
                            source=chunk.metadata.get("source", "memory"),
                            title=chunk.metadata.get("title", "Memory Chunk"),
                            content=chunk.content,
                            relevance=float(result.similarity_score),
                            url=chunk.metadata.get("url"),
                            metadata={"memory_type": chunk.memory_type.value},
                        )
                    )
            except Exception as exc:  # pragma: no cover - optional integration
                evidence.append(
                    SearchEvidence(
                        source="memory",
                        title="Memory retrieval error",
                        content=f"Failed to retrieve from memory: {exc}",
                        relevance=0.2,
                    )
                )

        return evidence

    def _search_local_knowledge(self, query: str, step: str) -> List[SearchEvidence]:
        lowered_query = query.lower()
        lowered_step = step.lower()

        matches: List[SearchEvidence] = []
        for evidence in self.knowledge_base.values():
            title_match = evidence.title.lower()
            if any(token in title_match for token in lowered_query.split()):
                matches.append(evidence)
            elif any(token in evidence.content.lower() for token in lowered_query.split()):
                matches.append(evidence)
            elif evidence.metadata and "tags" in evidence.metadata:
                tags = [t.lower() for t in evidence.metadata.get("tags", [])]
                if any(tag in lowered_step for tag in tags):
                    matches.append(evidence)

        matches.sort(key=lambda ev: ev.relevance, reverse=True)
        return matches[:3]

    # ------------------------------------------------------------------ #
    # Synthesis helpers
    # ------------------------------------------------------------------ #
    def _synthesise_answer(
        self,
        query: str,
        analysis: Dict[str, Any],
        evidence: List[SearchEvidence],
    ) -> tuple[str, float]:
        if not evidence:
            answer = (
                f"No relevant evidence found for '{query}'. "
                "Consider expanding the knowledge base or enabling external search."
            )
            return answer, 0.2

        summary_lines: List[str] = []
        citations: List[str] = []

        for ev in evidence:
            summary_lines.append(f"- {ev.content}")
            citation = ev.title
            if ev.url:
                citation += f" ({ev.url})"
            citations.append(citation)

        answer = f"""# Agentic Search Summary

**Query:** {analysis['query']}
**Intent:** {analysis['intent']}
**Platform:** {analysis['platform']}

## Key Findings
{chr(10).join(summary_lines)}

## Citations
- {chr(10).join(citations)}
"""
        relevance_scores = [ev.relevance for ev in evidence]
        confidence = sum(relevance_scores) / len(relevance_scores)
        confidence = min(max(confidence, 0.0), 1.0)

        if analysis["intent"] == "build" and any("step" in ev.content.lower() for ev in evidence):
            confidence = min(confidence + 0.1, 1.0)

        return answer, confidence


async def demo_agentic_search() -> None:
    """Simple demonstration helper for manual testing."""
    knowledge_base = [
        {
            "title": "AWS Lambda Overview",
            "content": "AWS Lambda provides serverless compute that integrates with Bedrock agents.",
            "relevance": 0.92,
            "url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            "metadata": {"tags": ["lambda", "aws", "serverless"], "source": "AWS Docs"},
        },
        {
            "title": "Bedrock Agents",
            "content": "Bedrock Agents can invoke Lambda functions to perform secure backend actions.",
            "relevance": 0.89,
            "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
            "metadata": {"tags": ["bedrock", "agent", "aws"], "source": "AWS Docs"},
        },
    ]

    engine = AgenticSearchEngine(knowledge_base=knowledge_base)
    outcome = await engine.search("How do I connect AWS Bedrock agents to Lambda functions?")

    print("-" * 80)
    print(outcome.answer)
    print(f"\nConfidence: {outcome.confidence:.2%}")


if __name__ == "__main__":
    asyncio.run(demo_agentic_search())
