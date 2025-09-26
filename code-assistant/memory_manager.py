"""Memory management system with hierarchical embeddings and long/short term memory."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from database_manager import DatabaseManager
from ollama_model import OllamaModel

logger = logging.getLogger(__name__)


@dataclass
class MemoryFragment:
    """A memory fragment with hierarchical embedding levels."""
    id: str
    content: str
    level: str  # 'word', 'sentence', 'paragraph', 'document'
    embedding: List[float]
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
    timestamp: float
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class ConversationMemory:
    """Conversation context memory with sliding window."""
    session_id: str
    messages: List[Dict[str, Any]]
    summary: Optional[str]
    key_points: List[str]
    updated_at: float


class MemoryManager:
    """Manages hierarchical memory with embeddings for sentence, word, and document levels."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        ollama_model: OllamaModel,
        short_term_limit: int = 50,
        long_term_threshold: float = 0.7,
        summary_threshold: int = 20,
    ):
        self.db = db_manager
        self.ollama = ollama_model
        self.short_term_limit = short_term_limit
        self.long_term_threshold = long_term_threshold
        self.summary_threshold = summary_threshold

        # In-memory caches for performance
        self.short_term_cache: Dict[str, List[MemoryFragment]] = {}
        self.conversation_cache: Dict[str, ConversationMemory] = {}

        logger.info("MemoryManager initialized with short_term_limit=%d", short_term_limit)

    def add_memory(
        self,
        content: str,
        session_id: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new memory with hierarchical embeddings."""
        now = time.time()
        memory_id = str(uuid4())

        # Create hierarchical fragments
        fragments = self._create_hierarchical_fragments(
            content, memory_id, session_id, memory_type, importance, metadata or {}
        )

        # Store fragments in database
        for fragment in fragments:
            self._store_fragment(fragment, session_id)

        # Update short-term cache
        if session_id not in self.short_term_cache:
            self.short_term_cache[session_id] = []

        self.short_term_cache[session_id].extend(fragments)
        self._manage_short_term_memory(session_id)

        logger.debug("Added memory %s with %d fragments", memory_id, len(fragments))
        return memory_id

    def retrieve_memory(
        self,
        query: str,
        session_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        include_long_term: bool = True,
    ) -> List[MemoryFragment]:
        """Retrieve relevant memories based on query embedding."""
        query_embedding = self.ollama.embed(query)

        # Search short-term memory first
        short_term_results = self._search_short_term(query_embedding, session_id, limit // 2)

        # Search long-term memory if enabled
        long_term_results = []
        if include_long_term:
            long_term_results = self._search_long_term(
                query_embedding, session_id, memory_types, limit // 2
            )

        # Combine and rank results
        all_results = short_term_results + long_term_results
        ranked_results = self._rank_memories(all_results, query_embedding)

        # Update access statistics
        for fragment in ranked_results[:limit]:
            fragment.access_count += 1
            fragment.last_accessed = time.time()
            self._update_fragment_stats(fragment)

        return ranked_results[:limit]

    def update_conversation(self, session_id: str, message: Dict[str, Any]):
        """Update conversation memory."""
        if session_id not in self.conversation_cache:
            self.conversation_cache[session_id] = ConversationMemory(
                session_id=session_id,
                messages=[],
                summary=None,
                key_points=[],
                updated_at=time.time()
            )

        conv = self.conversation_cache[session_id]
        conv.messages.append(message)
        conv.updated_at = time.time()

        # Generate summary if conversation is getting long
        if len(conv.messages) > self.summary_threshold:
            self._update_conversation_summary(conv)

        # Store in database
        self.db.memory_add(
            session_id=session_id,
            content=json.dumps(message),
            memory_type="conversation",
            metadata={"message_index": len(conv.messages) - 1}
        )

    def get_conversation_context(self, session_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get conversation context for session."""
        if session_id in self.conversation_cache:
            conv = self.conversation_cache[session_id]
            return {
                "recent_messages": conv.messages[-limit:],
                "summary": conv.summary,
                "key_points": conv.key_points,
                "total_messages": len(conv.messages)
            }

        # Load from database if not in cache
        memory_entries = self.db.memory_get_session(session_id, "conversation", limit)
        messages = []
        for entry in reversed(memory_entries):  # Reverse to get chronological order
            try:
                message = json.loads(entry.content)
                messages.append(message)
            except json.JSONDecodeError:
                continue

        return {
            "recent_messages": messages,
            "summary": None,
            "key_points": [],
            "total_messages": len(memory_entries)
        }

    def consolidate_memories(self, session_id: str):
        """Move important short-term memories to long-term storage."""
        if session_id not in self.short_term_cache:
            return

        fragments = self.short_term_cache[session_id]
        for fragment in fragments:
            if fragment.importance_score >= self.long_term_threshold:
                # Store in knowledge database for long-term retrieval
                self.db.knowledge_add(
                    content=fragment.content,
                    embedding=fragment.embedding,
                    source=f"memory_{session_id}",
                    metadata={
                        "memory_id": fragment.id,
                        "level": fragment.level,
                        "importance": fragment.importance_score,
                        "access_count": fragment.access_count,
                        "session_id": session_id,
                        **fragment.metadata
                    }
                )
                logger.debug("Consolidated memory fragment %s to long-term", fragment.id)

    def cleanup_old_memories(self, days_old: int = 7):
        """Clean up old memories based on access patterns and importance."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        # Clean up short-term cache
        for session_id in list(self.short_term_cache.keys()):
            fragments = self.short_term_cache[session_id]
            self.short_term_cache[session_id] = [
                f for f in fragments
                if f.timestamp > cutoff_time or f.importance_score > 0.8 or f.access_count > 2
            ]

        # Clean up database
        self.db.memory_cleanup_old(days_old)

    def get_memory_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self.db.get_stats()

        if session_id:
            short_term_count = len(self.short_term_cache.get(session_id, []))
            conversation_length = 0
            if session_id in self.conversation_cache:
                conversation_length = len(self.conversation_cache[session_id].messages)

            stats.update({
                "session_short_term": short_term_count,
                "session_conversation_length": conversation_length,
            })

        return stats

    def _create_hierarchical_fragments(
        self,
        content: str,
        memory_id: str,
        session_id: str,
        memory_type: str,
        importance: float,
        metadata: Dict[str, Any],
    ) -> List[MemoryFragment]:
        """Create hierarchical memory fragments from content."""
        fragments = []
        now = time.time()

        # Document level
        doc_embedding = self.ollama.embed(content)
        doc_fragment = MemoryFragment(
            id=f"{memory_id}_doc",
            content=content,
            level="document",
            embedding=doc_embedding,
            parent_id=None,
            children_ids=[],
            metadata={**metadata, "memory_type": memory_type},
            timestamp=now,
            importance_score=importance,
        )
        fragments.append(doc_fragment)

        # Paragraph level (split by double newlines)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 20:  # Skip very short paragraphs
                continue

            para_embedding = self.ollama.embed(paragraph)
            para_id = f"{memory_id}_para_{i}"
            para_fragment = MemoryFragment(
                id=para_id,
                content=paragraph,
                level="paragraph",
                embedding=para_embedding,
                parent_id=doc_fragment.id,
                children_ids=[],
                metadata={**metadata, "memory_type": memory_type, "paragraph_index": i},
                timestamp=now,
                importance_score=importance,
            )
            fragments.append(para_fragment)
            doc_fragment.children_ids.append(para_id)

            # Sentence level
            sentences = self._split_sentences(paragraph)
            for j, sentence in enumerate(sentences):
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                sent_embedding = self.ollama.embed(sentence)
                sent_id = f"{memory_id}_sent_{i}_{j}"
                sent_fragment = MemoryFragment(
                    id=sent_id,
                    content=sentence,
                    level="sentence",
                    embedding=sent_embedding,
                    parent_id=para_id,
                    children_ids=[],
                    metadata={**metadata, "memory_type": memory_type, "sentence_index": j},
                    timestamp=now,
                    importance_score=importance,
                )
                fragments.append(sent_fragment)
                para_fragment.children_ids.append(sent_id)

        return fragments

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _store_fragment(self, fragment: MemoryFragment, session_id: str):
        """Store memory fragment in database."""
        self.db.memory_add(
            session_id=session_id,
            content=json.dumps({
                "id": fragment.id,
                "content": fragment.content,
                "level": fragment.level,
                "embedding": fragment.embedding,
                "parent_id": fragment.parent_id,
                "children_ids": fragment.children_ids,
                "importance_score": fragment.importance_score,
                "access_count": fragment.access_count,
                "last_accessed": fragment.last_accessed,
            }),
            memory_type=f"fragment_{fragment.level}",
            metadata=fragment.metadata
        )

    def _manage_short_term_memory(self, session_id: str):
        """Manage short-term memory cache size."""
        if session_id not in self.short_term_cache:
            return

        fragments = self.short_term_cache[session_id]
        if len(fragments) > self.short_term_limit:
            # Sort by importance and recency, keep the best ones
            fragments.sort(key=lambda f: (f.importance_score, f.timestamp), reverse=True)
            self.short_term_cache[session_id] = fragments[:self.short_term_limit]

    def _search_short_term(
        self,
        query_embedding: List[float],
        session_id: str,
        limit: int,
    ) -> List[MemoryFragment]:
        """Search short-term memory cache."""
        if session_id not in self.short_term_cache:
            return []

        fragments = self.short_term_cache[session_id]
        scored_fragments = [
            (self._cosine_similarity(query_embedding, f.embedding), f)
            for f in fragments
        ]
        scored_fragments.sort(key=lambda x: x[0], reverse=True)

        return [f for _, f in scored_fragments[:limit]]

    def _search_long_term(
        self,
        query_embedding: List[float],
        session_id: str,
        memory_types: Optional[List[str]],
        limit: int,
    ) -> List[MemoryFragment]:
        """Search long-term memory in knowledge database."""
        knowledge_entries = self.db.knowledge_search(query_embedding, limit * 2)

        fragments = []
        for entry in knowledge_entries:
            if session_id in entry.source or not session_id:
                # Convert knowledge entry back to memory fragment
                metadata = entry.metadata
                fragment = MemoryFragment(
                    id=metadata.get("memory_id", entry.id),
                    content=entry.content,
                    level=metadata.get("level", "document"),
                    embedding=entry.embedding,
                    parent_id=metadata.get("parent_id"),
                    children_ids=metadata.get("children_ids", []),
                    metadata=metadata,
                    timestamp=entry.created_at,
                    importance_score=metadata.get("importance", 0.5),
                    access_count=metadata.get("access_count", 0),
                    last_accessed=metadata.get("last_accessed", 0.0),
                )
                fragments.append(fragment)

        return fragments[:limit]

    def _rank_memories(
        self,
        fragments: List[MemoryFragment],
        query_embedding: List[float],
    ) -> List[MemoryFragment]:
        """Rank memory fragments by relevance, importance, and recency."""
        scored_fragments = []

        for fragment in fragments:
            # Semantic similarity
            similarity = self._cosine_similarity(query_embedding, fragment.embedding)

            # Recency factor (more recent = higher score)
            recency_factor = min(1.0, (time.time() - fragment.timestamp) / (24 * 60 * 60))  # Days
            recency_score = 1.0 - recency_factor

            # Access frequency factor
            access_score = min(1.0, fragment.access_count / 10.0)

            # Combined score
            total_score = (
                similarity * 0.5 +
                fragment.importance_score * 0.3 +
                recency_score * 0.1 +
                access_score * 0.1
            )

            scored_fragments.append((total_score, fragment))

        scored_fragments.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored_fragments]

    def _update_fragment_stats(self, fragment: MemoryFragment):
        """Update fragment access statistics."""
        # Update in database
        self.db.memory_add(
            session_id="stats_update",
            content=json.dumps({
                "fragment_id": fragment.id,
                "access_count": fragment.access_count,
                "last_accessed": fragment.last_accessed,
            }),
            memory_type="fragment_stats",
            metadata={"update_type": "access_stats"}
        )

    def _update_conversation_summary(self, conv: ConversationMemory):
        """Update conversation summary using LLM."""
        if len(conv.messages) < 5:
            return

        # Get recent messages for summary
        recent_messages = conv.messages[-20:]
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in recent_messages
        ])

        summary_prompt = f"""
        Please provide a concise summary of this conversation, highlighting key topics and important points:

        {conversation_text}

        Summary:"""

        try:
            summary = self.ollama.generate(summary_prompt, max_tokens=200)
            conv.summary = summary.strip()

            # Extract key points
            key_points_prompt = f"""
            Based on this conversation summary, list the 3-5 most important key points:

            {summary}

            Key points (one per line):"""

            key_points_response = self.ollama.generate(key_points_prompt, max_tokens=150)
            conv.key_points = [
                point.strip() for point in key_points_response.split('\n')
                if point.strip() and not point.strip().startswith('-')
            ][:5]

        except Exception as e:
            logger.error("Failed to update conversation summary: %s", e)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)