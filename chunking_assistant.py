from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from strands import Agent, tool
from strands_tools import file_read

from logging_config import init_logging

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None

try:  # Optional RAGAS evaluation support
    from ragas import SingleTurnSample
    from ragas.metrics import LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    SingleTurnSample = None
    LLMContextPrecisionWithoutReference = None
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None
    ChatBedrockConverse = None
    BedrockEmbeddings = None


init_logging()
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Lightweight representation of a text chunk with hierarchy metadata."""

    id: str
    text: str
    start: int
    end: int
    metadata: Dict[str, Any]
    level: str = "chunk"
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class HierarchicalChunks:
    """Hierarchical view over a document."""

    document: Chunk
    sections: List[Chunk]
    chunks: List[Chunk]
    sentences: List[Chunk]
    summary: Optional[Chunk] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "summary": self.summary.to_dict() if self.summary else None,
            "sections": [chunk.to_dict() for chunk in self.sections],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "sentences": [chunk.to_dict() for chunk in self.sentences],
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "HierarchicalChunks":
        def build(item: Optional[Dict[str, Any]]) -> Optional[Chunk]:
            if not item:
                return None
            return Chunk(**item)

        return cls(
            document=Chunk(**payload["document"]),
            summary=build(payload.get("summary")),
            sections=[Chunk(**item) for item in payload.get("sections", [])],
            chunks=[Chunk(**item) for item in payload.get("chunks", [])],
            sentences=[Chunk(**item) for item in payload.get("sentences", [])],
        )


class ChunkingAssistant:
    """Utility class for producing text chunks, sections, and sentences."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators or ["\n\n", "\n", ".", "?", "!"])

        if RecursiveCharacterTextSplitter is not None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=list(self.separators),
                keep_separator=False,
            )
        else:
            self._splitter = None

        self._section_pattern = re.compile(r"(?m)^(?P<header>#{1,6}\s+.+?)\s*$")

        logger.debug(
            "ChunkingAssistant initialized | chunk_size=%s overlap=%s separators=%s",
            self.chunk_size,
            self.chunk_overlap,
            self.separators,
        )

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a plain text string into manageable pieces."""
        text = (text or "").strip()
        if not text:
            logger.warning("chunk_text called with empty text")
            return []

        if self._splitter is not None:
            splits = self._splitter.split_text(text)
            logger.debug("RecursiveCharacterTextSplitter produced %d chunks", len(splits))
        else:
            logger.debug("Falling back to naive splitter")
            splits = self._naive_split(text)

        chunks: List[Chunk] = []
        cursor = 0
        base_meta = dict(metadata or {})
        for index, split in enumerate(splits):
            split = split.strip()
            if not split:
                continue
            start = text.find(split, cursor)
            if start == -1:
                start = cursor
            end = start + len(split)
            cursor = end
            chunk_meta = dict(base_meta)
            chunk_meta.setdefault("level", "chunk")
            chunk_meta.setdefault("chunk_index", index)
            chunk_meta.setdefault("chunk_token_estimate", len(split) // 4)
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    text=split,
                    start=start,
                    end=end,
                    metadata=chunk_meta,
                    level="chunk",
                    parent_id=chunk_meta.get("parent_id"),
                )
            )
        logger.info("chunk_text generated %d chunks", len(chunks))
        return chunks

    def chunk_documents(
        self,
        documents: Dict[str, str],
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Chunk the contents of multiple documents."""
        results: Dict[str, List[Dict[str, Any]]] = {}
        for name, content in documents.items():
            metadata = dict(base_metadata or {})
            metadata.setdefault("source", name)
            metadata.setdefault("doc_id", name)
            results[name] = [chunk.to_dict() for chunk in self.chunk_text(content, metadata)]
            logger.debug("Chunked document '%s' into %d chunks", name, len(results[name]))
        return results

    def chunk_files(
        self,
        file_paths: Sequence[str],
        encoding: str = "utf-8",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Read files via strands file_read tool and chunk their contents."""
        reader = Agent(tools=[file_read])
        documents: Dict[str, str] = {}
        for path in file_paths:
            try:
                logger.debug("Reading file %s", path)
                response = reader.tool.file_read(path=path, mode="view", encoding=encoding)
                text = "".join(part.get("text", "") for part in response.get("content", []))
                documents[path] = text
            except Exception as exc:  # pragma: no cover - filesystem dependent
                logger.error("Failed to read %s: %s", path, exc)
        return self.chunk_documents(documents)

    def chunk_hierarchy(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_sentences: bool = True,
        include_summary: bool = True,
        summary_sentence_limit: int = 5,
    ) -> HierarchicalChunks:
        """Produce document, section, chunk, and sentence level splits."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot build hierarchy for empty text")

        document_id = doc_id or str(uuid4())
        base_metadata = dict(metadata or {})
        base_metadata.setdefault("doc_id", document_id)

        document_chunk = Chunk(
            id=document_id,
            text=text,
            start=0,
            end=len(text),
            metadata={**base_metadata, "level": "document"},
            level="document",
            parent_id=None,
        )

        summary_chunk: Optional[Chunk] = None
        if include_summary:
            summary_text = self._create_summary(text, max_sentences=summary_sentence_limit)
            if summary_text:
                summary_chunk = Chunk(
                    id=f"{document_id}::summary",
                    text=summary_text,
                    start=0,
                    end=0,
                    metadata={**base_metadata, "level": "summary"},
                    level="summary",
                    parent_id=document_id,
                )

        sections = self._build_sections(text, document_id, base_metadata)
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        chunks = self._build_chunks(sections, base_metadata, chunk_size, chunk_overlap)

        sentences: List[Chunk] = []
        if include_sentences:
            sentences = self._build_sentences(
                text,
                base_metadata,
                document_id,
                sections,
                chunks,
            )

        return HierarchicalChunks(
            document=document_chunk,
            summary=summary_chunk,
            sections=sections,
            chunks=chunks,
            sentences=sentences,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _naive_split(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        step = max(1, chunk_size - chunk_overlap)
        pieces: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(length, start + chunk_size)
            pieces.append(text[start:end])
            start += step
        return pieces

    def _create_summary(self, text: str, max_sentences: int = 5, max_chars: int = 1200) -> str:
        sentences = [sentence for sentence, _, _ in self._split_sentences_raw(text)]
        if not sentences:
            return text[:max_chars].strip()
        summary = " ".join(sentences[:max_sentences]).strip()
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
        return summary

    def _build_sections(
        self,
        text: str,
        doc_id: str,
        base_metadata: Dict[str, Any],
    ) -> List[Chunk]:
        matches = list(self._section_pattern.finditer(text))
        if not matches:
            clean_text, start, end = self._trimmed_span(text, 0, len(text))
            metadata = {**base_metadata, "level": "section", "section_index": 0}
            return [
                Chunk(
                    id=f"{doc_id}::section::0",
                    text=clean_text,
                    start=start,
                    end=end,
                    metadata=metadata,
                    level="section",
                    parent_id=doc_id,
                )
            ]

        sections: List[Chunk] = []
        for index, match in enumerate(matches):
            start = match.start()
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            segment, seg_start, seg_end = self._trimmed_span(text, start, next_start)
            if not segment:
                continue
            header = match.group("header").strip()
            metadata = {
                **base_metadata,
                "level": "section",
                "title": header,
                "section_index": index,
            }
            sections.append(
                Chunk(
                    id=f"{doc_id}::section::{index}",
                    text=segment,
                    start=seg_start,
                    end=seg_end,
                    metadata=metadata,
                    level="section",
                    parent_id=doc_id,
                )
            )
        if not sections:
            clean_text, start, end = self._trimmed_span(text, 0, len(text))
            metadata = {**base_metadata, "level": "section", "section_index": 0}
            sections.append(
                Chunk(
                    id=f"{doc_id}::section::0",
                    text=clean_text,
                    start=start,
                    end=end,
                    metadata=metadata,
                    level="section",
                    parent_id=doc_id,
                )
            )
        return sections

    def _build_chunks(
        self,
        sections: List[Chunk],
        base_metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        for section in sections:
            section_chunks = self._chunk_section(section, base_metadata, chunk_size, chunk_overlap)
            chunks.extend(section_chunks)
        return chunks

    def _chunk_section(
        self,
        section: Chunk,
        base_metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Chunk]:
        if not section.text:
            return []

        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=list(self.separators),
                keep_separator=False,
            )
            splits = splitter.split_text(section.text)
        else:
            splits = self._naive_split(section.text, chunk_size, chunk_overlap)

        chunks: List[Chunk] = []
        cursor = 0
        for index, split in enumerate(splits):
            split = split.strip()
            if not split:
                continue
            local_start = section.text.find(split, cursor)
            if local_start == -1:
                local_start = cursor
            local_end = local_start + len(split)
            cursor = local_end
            global_start = section.start + local_start
            global_end = section.start + local_end
            metadata = {
                **base_metadata,
                "level": "chunk",
                "doc_id": base_metadata.get("doc_id"),
                "section_id": section.id,
                "section_title": section.metadata.get("title"),
                "chunk_index": index,
            }
            chunk = Chunk(
                id=f"{section.id}::chunk::{index}",
                text=split,
                start=global_start,
                end=global_end,
                metadata=metadata,
                level="chunk",
                parent_id=section.id,
            )
            chunk.metadata.setdefault("chunk_token_estimate", len(split) // 4)
            chunks.append(chunk)
        return chunks

    def _build_sentences(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        doc_id: str,
        sections: Sequence[Chunk],
        chunks: Sequence[Chunk],
    ) -> List[Chunk]:
        sentences: List[Chunk] = []
        splits = self._split_sentences_raw(text)
        for index, (sentence, start, end) in enumerate(splits):
            metadata = {
                **base_metadata,
                "level": "sentence",
                "doc_id": doc_id,
                "sentence_index": index,
            }
            section_parent = self._locate_parent(sections, start, end)
            if section_parent:
                metadata["section_id"] = section_parent.id
                metadata["section_title"] = section_parent.metadata.get("title")
            chunk_parent = self._locate_parent(chunks, start, end)
            if chunk_parent:
                metadata["chunk_id"] = chunk_parent.id
            parent_id = (
                chunk_parent.id
                if chunk_parent
                else section_parent.id if section_parent else doc_id
            )
            sentences.append(
                Chunk(
                    id=f"{doc_id}::sentence::{index}",
                    text=sentence,
                    start=start,
                    end=end,
                    metadata=metadata,
                    level="sentence",
                    parent_id=parent_id,
                )
            )
        return sentences

    def _split_sentences_raw(self, text: str) -> List[Tuple[str, int, int]]:
        pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
        spans: List[Tuple[str, int, int]] = []
        start = 0
        for match in pattern.finditer(text):
            end = match.start()
            sentence, real_start, real_end = self._trimmed_span(text, start, end)
            if sentence:
                spans.append((sentence, real_start, real_end))
            start = match.end()
        sentence, real_start, real_end = self._trimmed_span(text, start, len(text))
        if sentence:
            spans.append((sentence, real_start, real_end))
        return spans

    def _locate_parent(
        self,
        items: Sequence[Chunk],
        start: int,
        end: int,
    ) -> Optional[Chunk]:
        for item in items:
            if item.start <= start and end <= item.end:
                return item
        return None

    def _trimmed_span(self, text: str, start: int, end: int) -> Tuple[str, int, int]:
        segment = text[start:end]
        stripped = segment.strip()
        if not stripped:
            return "", start, start
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment) - len(segment.rstrip())
        real_start = start + leading
        real_end = end - trailing
        return stripped, real_start, real_end


# ----------------------------------------------------------------------
# Optional chunk relevance scoring via RAGAS
# ----------------------------------------------------------------------


def _build_ragas_components():
    if not all(
        (
            SingleTurnSample,
            LLMContextPrecisionWithoutReference,
            LangchainLLMWrapper,
            LangchainEmbeddingsWrapper,
            ChatBedrockConverse,
            BedrockEmbeddings,
        )
    ):
        logger.debug("RAGAS dependencies not available; relevance scoring disabled")
        return None, None

    model_id = os.getenv(
        "RAGAS_BEDROCK_MODEL",
        "anthropic.claude-3-sonnet-20240229-v1:0",
    )
    region = os.getenv("AWS_REGION", "us-east-1")

    llm_wrapper = LangchainLLMWrapper(
        ChatBedrockConverse(model_id=model_id, region=region)
    )
    embedding_wrapper = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id=os.getenv("RAGAS_EMBED_MODEL", "cohere.embed-english-v3"))
    )
    return llm_wrapper, embedding_wrapper


_LL_M_WRAPPER, _EMB_WRAPPER = _build_ragas_components()


@tool
def chunk_text_tool(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    assistant = ChunkingAssistant(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.to_dict() for chunk in assistant.chunk_text(text)]


@tool
def chunk_files_tool(
    file_paths: Sequence[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    encoding: str = "utf-8",
) -> Dict[str, List[Dict[str, Any]]]:
    assistant = ChunkingAssistant(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return assistant.chunk_files(file_paths, encoding=encoding)


@tool
def chunk_hierarchy_tool(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    include_sentences: bool = True,
    include_summary: bool = True,
    summary_sentence_limit: int = 5,
    doc_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    assistant = ChunkingAssistant(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    hierarchy = assistant.chunk_hierarchy(
        text=text,
        metadata=metadata,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_sentences=include_sentences,
        include_summary=include_summary,
        summary_sentence_limit=summary_sentence_limit,
    )
    return hierarchy.to_dict()


@tool
def check_chunks_relevance(results: str, question: str) -> Dict[str, Any]:
    """Evaluate retrieved chunks against a question using RAGAS if available."""
    if not results or not isinstance(results, str):
        raise ValueError("'results' must be a non-empty string")
    if not question or not isinstance(question, str):
        raise ValueError("'question' must be a non-empty string")

    if _LL_M_WRAPPER is None or _EMB_WRAPPER is None:
        logger.warning("RAGAS dependencies missing; returning neutral relevance result")
        return {
            "chunk_relevance_score": "unknown",
            "chunk_relevance_value": None,
            "detail": "Install ragas[bedrock] dependencies to enable scoring.",
        }

    pattern = r"Score:.*?\nContent:\s*(.*?)(?=Score:|\Z)"
    docs = [chunk.strip() for chunk in re.findall(pattern, results, re.DOTALL)]
    if not docs:
        raise ValueError("No valid content chunks found in 'results'.")

    sample = SingleTurnSample(
        user_input=question,
        response="placeholder-response",
        retrieved_contexts=docs,
    )

    scorer = LLMContextPrecisionWithoutReference(llm=_LL_M_WRAPPER, embeddings=_EMB_WRAPPER)
    score = asyncio.run(scorer.single_turn_ascore(sample))

    logger.info("Chunk relevance score for question '%s': %.3f", question, score)

    return {
        "chunk_relevance_score": "yes" if score > 0.5 else "no",
        "chunk_relevance_value": score,
    }
