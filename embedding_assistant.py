from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

try:  # Optional dependency for direct Bedrock support
    from langchain_aws import BedrockEmbeddings  # type: ignore
except ImportError:  # pragma: no cover - optional path
    BedrockEmbeddings = None

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests is ubiquitous but optional
    requests = None

from strands import tool

from logging_config import init_logging
from chunking_assistant import Chunk, ChunkingAssistant, HierarchicalChunks


init_logging()
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_EMBED_TIMEOUT", "60"))


class OllamaEmbeddingError(RuntimeError):
    """Raised when the Ollama embedding endpoint returns an unexpected response."""


@dataclass
class EmbeddingResult:
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "vector": self.vector,
            "metadata": self.metadata,
        }


class OllamaEmbeddingClient:
    """Small helper for calling the local Ollama embeddings endpoint."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model_id: str = DEFAULT_OLLAMA_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if requests is None:
            raise ImportError("The 'requests' package is required for Ollama embeddings.")

        self.host = host.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.endpoint = f"{self.host}/api/embeddings"
        logger.debug(
            "Initialized OllamaEmbeddingClient | host=%s model=%s timeout=%ss",
            self.host,
            self.model_id,
            self.timeout,
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            payload = {"model": self.model_id, "prompt": text}
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if response.status_code >= 400:
                raise OllamaEmbeddingError(
                    f"Ollama embedding request failed: {response.status_code} {response.text}"
                )
            data = response.json()
            vector = data.get("embedding") or data.get("vector")
            if not isinstance(vector, list):
                raise OllamaEmbeddingError("No embedding vector returned from Ollama.")
            vectors.append(vector)
        return vectors


class EmbeddingAssistant:
    """Embedding helper that defaults to the local Ollama embeddinggemma model."""

    def __init__(
        self,
        provider: str = "ollama",
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        host: Optional[str] = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        provider = provider.lower()
        self.provider = provider
        self.model_id = model_id or DEFAULT_OLLAMA_EMBED_MODEL
        self.region = region
        self.host = host or DEFAULT_OLLAMA_HOST
        self.request_timeout = request_timeout

        if provider == "ollama":
            self.embedder = OllamaEmbeddingClient(
                host=self.host,
                model_id=self.model_id,
                timeout=self.request_timeout,
            )
        elif provider == "bedrock":
            if BedrockEmbeddings is None:
                raise ImportError(
                    "langchain-aws is required for Bedrock embeddings. Install with `pip install langchain-aws`."
                )
            self.embedder = BedrockEmbeddings(
                model_id=self.model_id or "cohere.embed-english-v3",
                region_name=self.region,
            )
        else:
            raise ValueError("Supported providers are 'ollama' and 'bedrock'.")

        logger.debug(
            "EmbeddingAssistant initialized | provider=%s model=%s host=%s region=%s",
            self.provider,
            self.model_id,
            self.host,
            self.region,
        )

    def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        vector = self._embed_single(text)
        logger.debug("Embedded single text into %d dimensions", len(vector))
        return EmbeddingResult(text=text, vector=vector, metadata=dict(metadata or {}))

    def embed_texts(
        self,
        texts: Iterable[str],
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> List[EmbeddingResult]:
        texts = list(texts)
        if not texts:
            return []

        vectors = self._embed_batch(texts)
        logger.info("Embedded %d texts", len(texts))

        metadata_iterable: List[Dict[str, Any]]
        if metadata is None:
            metadata_iterable = [{} for _ in texts]
        else:
            metadata_iterable = list(metadata)
            if len(metadata_iterable) != len(texts):
                metadata_iterable = (metadata_iterable + [{}] * len(texts))[: len(texts)]

        return [
            EmbeddingResult(text, vector, dict(meta))
            for text, vector, meta in zip(texts, vectors, metadata_iterable)
        ]

    def embed_hierarchy(
        self,
        hierarchy: Union[HierarchicalChunks, Dict[str, Any]],
        levels: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        parsed = self._ensure_hierarchy(hierarchy)
        available: Dict[str, List[Chunk]] = {
            "document": [parsed.document],
            "summary": [parsed.summary] if parsed.summary else [],
            "sections": parsed.sections,
            "chunks": parsed.chunks,
            "sentences": parsed.sentences,
        }
        if levels is not None:
            selected = {level: available.get(level, []) for level in levels}
        else:
            selected = available

        embeddings: Dict[str, List[Dict[str, Any]]] = {}
        for level, items in selected.items():
            if not items:
                continue
            texts = [item.text for item in items]
            metadata = [self._prepare_metadata(item, level) for item in items]
            results = self.embed_texts(texts, metadata=metadata)
            embeddings[level] = [result.to_dict() for result in results]
        return embeddings

    def embed_document_hierarchy(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk_size: int = 900,
        chunk_overlap: int = 180,
        include_sentences: bool = True,
        include_summary: bool = True,
        summary_sentence_limit: int = 5,
        levels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        chunker = ChunkingAssistant(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        hierarchy = chunker.chunk_hierarchy(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_sentences=include_sentences,
            include_summary=include_summary,
            summary_sentence_limit=summary_sentence_limit,
        )
        embeddings = self.embed_hierarchy(hierarchy, levels=levels)
        return {
            "hierarchy": hierarchy.to_dict(),
            "embeddings": embeddings,
        }

    def save_embeddings(self, results: List[EmbeddingResult], path: Path) -> None:
        payload = [result.to_dict() for result in results]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("Persisted %d embeddings to %s", len(results), path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed_single(self, text: str) -> List[float]:
        if self.provider == "ollama":
            return self.embedder.embed([text])[0]
        return self.embedder.embed_query(text)  # type: ignore[return-value]

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if self.provider == "ollama":
            return self.embedder.embed(texts)
        return self.embedder.embed_documents(list(texts))  # type: ignore[return-value]

    def _ensure_hierarchy(
        self,
        hierarchy: Union[HierarchicalChunks, Dict[str, Any]],
    ) -> HierarchicalChunks:
        if isinstance(hierarchy, HierarchicalChunks):
            return hierarchy
        if not isinstance(hierarchy, dict):
            raise TypeError("hierarchy must be HierarchicalChunks or dict payload")
        return HierarchicalChunks.from_payload(hierarchy)

    def _prepare_metadata(self, chunk: Chunk, level: str) -> Dict[str, Any]:
        metadata = dict(chunk.metadata)
        metadata.setdefault("level", level)
        metadata.setdefault("chunk_id", chunk.id)
        if chunk.parent_id is not None:
            metadata.setdefault("parent_id", chunk.parent_id)
        metadata.setdefault("start", chunk.start)
        metadata.setdefault("end", chunk.end)
        return metadata


@tool
def embed_text(
    text: str,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    provider: str = "ollama",
    host: Optional[str] = None,
) -> Dict[str, Any]:
    assistant = EmbeddingAssistant(provider=provider, model_id=model_id, region=region, host=host)
    return assistant.embed_text(text).to_dict()


@tool
def embed_texts(
    texts: List[str],
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    provider: str = "ollama",
    host: Optional[str] = None,
) -> Dict[str, Any]:
    assistant = EmbeddingAssistant(provider=provider, model_id=model_id, region=region, host=host)
    results = assistant.embed_texts(texts)
    return {
        "embeddings": [result.to_dict() for result in results],
    }


@tool
def embed_and_save(
    texts: List[str],
    output_path: str,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    provider: str = "ollama",
    host: Optional[str] = None,
) -> Dict[str, Any]:
    assistant = EmbeddingAssistant(provider=provider, model_id=model_id, region=region, host=host)
    results = assistant.embed_texts(texts)
    output = Path(output_path)
    assistant.save_embeddings(results, output)
    return {"saved_to": str(output), "count": len(results)}


@tool
def embed_document_hierarchy(
    text: str,
    model_id: Optional[str] = None,
    provider: str = "ollama",
    host: Optional[str] = None,
    doc_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    include_sentences: bool = True,
    include_summary: bool = True,
    summary_sentence_limit: int = 5,
    levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    assistant = EmbeddingAssistant(provider=provider, model_id=model_id, host=host)
    result = assistant.embed_document_hierarchy(
        text=text,
        metadata=metadata,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_sentences=include_sentences,
        include_summary=include_summary,
        summary_sentence_limit=summary_sentence_limit,
        levels=levels,
    )
    return result
