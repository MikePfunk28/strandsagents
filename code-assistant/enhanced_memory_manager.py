"""Enhanced memory manager that integrates with existing embedding and chunking assistants."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from database_manager import DatabaseManager
from ollama_model import OllamaModel

# Import existing assistants
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from embedding_assistant import EmbeddingAssistant
from chunking_assistant import ChunkingAssistant

logger = logging.getLogger(__name__)


class EnhancedMemoryManager:
    """Enhanced memory manager using existing embedding and chunking assistants."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        ollama_model: OllamaModel,
        embedding_assistant: EmbeddingAssistant,
        chunking_assistant: ChunkingAssistant,
        short_term_limit: int = 50,
        long_term_threshold: float = 0.7,
    ):
        self.db = db_manager
        self.ollama = ollama_model
        self.embedding_assistant = embedding_assistant
        self.chunking_assistant = chunking_assistant
        self.short_term_limit = short_term_limit
        self.long_term_threshold = long_term_threshold

        # In-memory caches
        self.short_term_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("EnhancedMemoryManager initialized")

    def add_memory(
        self,
        content: str,
        session_id: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add memory using hierarchical chunking and embeddings."""
        memory_id = str(uuid4())

        try:
            # Use the existing embed_document_hierarchy function
            result = self.embedding_assistant.embed_document_hierarchy(
                text=content,
                metadata=metadata or {},
                doc_id=memory_id,
                chunk_size=900,
                chunk_overlap=180,
                include_sentences=True,
                include_summary=True,
            )

            # Store the hierarchical embeddings
            hierarchy = result["hierarchy"]
            embeddings = result["embeddings"]

            # Store each level in knowledge database
            for level, level_embeddings in embeddings.items():
                for embedding_data in level_embeddings:
                    self.db.knowledge_add(
                        content=embedding_data["text"],
                        embedding=embedding_data["vector"],
                        source=f"memory_{session_id}_{level}",
                        metadata={
                            "memory_id": memory_id,
                            "session_id": session_id,
                            "memory_type": memory_type,
                            "level": level,
                            "importance": importance,
                            "timestamp": time.time(),
                            **embedding_data.get("metadata", {}),
                            **(metadata or {})
                        }
                    )

            # Add to short-term cache
            if session_id not in self.short_term_cache:
                self.short_term_cache[session_id] = []

            memory_entry = {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "timestamp": time.time(),
                "hierarchy": hierarchy,
                "embeddings_count": sum(len(embs) for embs in embeddings.values()),
            }

            self.short_term_cache[session_id].append(memory_entry)
            self._manage_short_term_memory(session_id)

            logger.debug("Added hierarchical memory %s with %d embedding levels",
                        memory_id, len(embeddings))
            return memory_id

        except Exception as e:
            logger.error("Error adding memory: %s", e)
            # Fallback to simple storage
            return self._add_simple_memory(content, session_id, memory_type, importance, metadata)

    def retrieve_memory(
        self,
        query: str,
        session_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        include_long_term: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using semantic search."""
        try:
            # Get query embedding
            query_result = self.embedding_assistant.embed_text(query)
            query_embedding = query_result.vector

            # Search knowledge database
            knowledge_entries = self.db.knowledge_search(query_embedding, limit * 2)

            # Filter and rank results
            relevant_memories = []
            seen_memory_ids = set()

            for entry in knowledge_entries:
                entry_metadata = entry.metadata

                # Filter by session and memory types
                if session_id not in entry.source and session_id != "global":
                    continue

                if memory_types and entry_metadata.get("memory_type") not in memory_types:
                    continue

                memory_id = entry_metadata.get("memory_id")
                if memory_id in seen_memory_ids:
                    continue
                seen_memory_ids.add(memory_id)

                # Calculate relevance score
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                importance = entry_metadata.get("importance", 0.5)
                recency = self._calculate_recency_score(entry_metadata.get("timestamp", 0))

                total_score = similarity * 0.6 + importance * 0.3 + recency * 0.1

                relevant_memories.append({
                    "memory_id": memory_id,
                    "content": entry.content,
                    "level": entry_metadata.get("level", "unknown"),
                    "memory_type": entry_metadata.get("memory_type", "general"),
                    "relevance_score": total_score,
                    "similarity": similarity,
                    "importance": importance,
                    "timestamp": entry_metadata.get("timestamp", 0),
                    "metadata": entry_metadata,
                })

            # Sort by relevance and return top results
            relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            return relevant_memories[:limit]

        except Exception as e:
            logger.error("Error retrieving memories: %s", e)
            return []

    def update_conversation(self, session_id: str, message: Dict[str, Any]):
        """Update conversation memory with summarization."""
        if session_id not in self.conversation_cache:
            self.conversation_cache[session_id] = {
                "messages": [],
                "summary": None,
                "key_points": [],
                "updated_at": time.time(),
            }

        conv = self.conversation_cache[session_id]
        conv["messages"].append(message)
        conv["updated_at"] = time.time()

        # Store in database
        self.db.memory_add(
            session_id=session_id,
            content=json.dumps(message),
            memory_type="conversation",
            metadata={"message_index": len(conv["messages"]) - 1}
        )

        # Generate summary if conversation is getting long
        if len(conv["messages"]) % 10 == 0:  # Every 10 messages
            self._update_conversation_summary(conv, session_id)

    def get_conversation_context(self, session_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get conversation context."""
        if session_id in self.conversation_cache:
            conv = self.conversation_cache[session_id]
            return {
                "recent_messages": conv["messages"][-limit:],
                "summary": conv["summary"],
                "key_points": conv["key_points"],
                "total_messages": len(conv["messages"])
            }

        # Load from database
        memory_entries = self.db.memory_get_session(session_id, "conversation", limit)
        messages = []
        for entry in reversed(memory_entries):
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

    def classify_content(self, content: str) -> Dict[str, Any]:
        """Classify content using semantic analysis."""
        try:
            # Use embedding for classification
            content_embedding = self.embedding_assistant.embed_text(content).vector

            # Define category embeddings
            categories = {
                "code": "programming code function class method variable algorithm implementation",
                "documentation": "documentation explanation description guide tutorial readme",
                "error": "error exception bug traceback debugging failure problem",
                "question": "question ask help how what why when where",
                "solution": "solution answer fix resolve implement create build",
                "analysis": "analysis review examine investigate study assess",
            }

            category_scores = {}
            for category, description in categories.items():
                cat_embedding = self.embedding_assistant.embed_text(description).vector
                similarity = self._cosine_similarity(content_embedding, cat_embedding)
                category_scores[category] = similarity

            # Get top category
            top_category = max(category_scores, key=category_scores.get)

            return {
                "primary_category": top_category,
                "scores": category_scores,
                "confidence": category_scores[top_category],
            }

        except Exception as e:
            logger.error("Error classifying content: %s", e)
            return {
                "primary_category": "general",
                "scores": {},
                "confidence": 0.0,
            }

    def consolidate_memories(self, session_id: str):
        """Consolidate important memories to long-term storage."""
        if session_id not in self.short_term_cache:
            return

        consolidated_count = 0
        for memory in self.short_term_cache[session_id]:
            if memory.get("importance", 0) >= self.long_term_threshold:
                # Already stored in knowledge database via add_memory
                consolidated_count += 1

        logger.info("Consolidated %d memories for session %s", consolidated_count, session_id)

    def _add_simple_memory(self, content: str, session_id: str, memory_type: str, importance: float, metadata: Optional[Dict[str, Any]]) -> str:
        """Fallback method for simple memory storage."""
        memory_id = str(uuid4())

        # Simple embedding without hierarchy
        try:
            embedding_result = self.embedding_assistant.embed_text(content, metadata)

            self.db.knowledge_add(
                content=content,
                embedding=embedding_result.vector,
                source=f"memory_{session_id}",
                metadata={
                    "memory_id": memory_id,
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "importance": importance,
                    "timestamp": time.time(),
                    **(metadata or {})
                }
            )

        except Exception as e:
            logger.error("Error creating simple memory: %s", e)

        return memory_id

    def _manage_short_term_memory(self, session_id: str):
        """Manage short-term memory cache size."""
        if session_id not in self.short_term_cache:
            return

        memories = self.short_term_cache[session_id]
        if len(memories) > self.short_term_limit:
            # Sort by importance and recency, keep the best ones
            memories.sort(key=lambda m: (m.get("importance", 0), m.get("timestamp", 0)), reverse=True)
            self.short_term_cache[session_id] = memories[:self.short_term_limit]

    def _update_conversation_summary(self, conv: Dict[str, Any], session_id: str):
        """Update conversation summary."""
        try:
            if len(conv["messages"]) < 5:
                return

            # Get recent messages for summary
            recent_messages = conv["messages"][-15:]
            conversation_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in recent_messages
            ])

            # Generate summary
            summary_prompt = f"""Summarize this conversation concisely, highlighting key topics and important information:

{conversation_text}

Summary:"""

            summary = self.ollama.generate(summary_prompt, max_tokens=200)
            conv["summary"] = summary.strip()

            # Store summary as memory
            self.add_memory(
                content=f"Conversation summary: {conv['summary']}",
                session_id=session_id,
                memory_type="summary",
                importance=0.8
            )

        except Exception as e:
            logger.error("Error updating conversation summary: %s", e)

    def _calculate_recency_score(self, timestamp: float) -> float:
        """Calculate recency score (0-1, higher = more recent)."""
        if timestamp == 0:
            return 0.0

        days_old = (time.time() - timestamp) / (24 * 60 * 60)
        return max(0.0, 1.0 - (days_old / 30.0))  # Linear decay over 30 days

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

    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self.db.get_stats()

        if session_id:
            short_term_count = len(self.short_term_cache.get(session_id, []))
            conversation_length = 0
            if session_id in self.conversation_cache:
                conversation_length = len(self.conversation_cache[session_id]["messages"])

            stats.update({
                "session_short_term": short_term_count,
                "session_conversation_length": conversation_length,
            })

        return stats