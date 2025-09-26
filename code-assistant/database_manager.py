"""Database management for coding assistant with sqlite backends."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for storing query results."""
    key: str
    value: str
    timestamp: float
    expires_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeEntry:
    """Knowledge base entry with embeddings."""
    id: str
    content: str
    embedding: List[float]
    source: str
    metadata: Dict[str, Any]
    created_at: float


@dataclass
class MemoryEntry:
    """Memory entry for context and conversation history."""
    id: str
    session_id: str
    content: str
    memory_type: str  # 'context', 'conversation', 'solution', 'code'
    timestamp: float
    metadata: Dict[str, Any]


class DatabaseManager:
    """Manages sqlite databases for cache, knowledge, and memory."""

    def __init__(self, db_dir: Path):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)

        # Initialize databases
        self.cache_db = self.db_dir / "cache.db"
        self.knowledge_db = self.db_dir / "knowledge.db"
        self.memory_db = self.db_dir / "memory.db"

        self._init_cache_db()
        self._init_knowledge_db()
        self._init_memory_db()

        logger.info("DatabaseManager initialized with DBs: %s", self.db_dir)

    def _init_cache_db(self):
        """Initialize cache database schema."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")

    def _init_knowledge_db(self):
        """Initialize knowledge database schema."""
        with sqlite3.connect(self.knowledge_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_entries(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_entries(created_at)")

    def _init_memory_db(self):
        """Initialize memory database schema."""
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON memory_entries(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)")

    # Cache operations
    def cache_get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        now = time.time()
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT value FROM cache_entries WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                (key, now)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def cache_set(self, key: str, value: str, ttl_seconds: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """Set value in cache."""
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries (key, value, timestamp, expires_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (key, value, now, expires_at, metadata_json)
            )

    def cache_delete(self, key: str):
        """Delete key from cache."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

    def cache_cleanup_expired(self):
        """Remove expired cache entries."""
        now = time.time()
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
            removed = cursor.rowcount
            logger.debug("Removed %d expired cache entries", removed)

    # Knowledge operations
    def knowledge_add(self, content: str, embedding: List[float], source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add knowledge entry."""
        entry_id = str(uuid4())
        now = time.time()
        embedding_blob = json.dumps(embedding).encode()
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.knowledge_db) as conn:
            conn.execute(
                "INSERT INTO knowledge_entries (id, content, embedding, source, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, content, embedding_blob, source, metadata_json, now)
            )

        logger.debug("Added knowledge entry: %s from %s", entry_id, source)
        return entry_id

    def knowledge_search(self, query_embedding: List[float], limit: int = 10) -> List[KnowledgeEntry]:
        """Search knowledge by embedding similarity (cosine similarity)."""
        with sqlite3.connect(self.knowledge_db) as conn:
            cursor = conn.execute("SELECT id, content, embedding, source, metadata, created_at FROM knowledge_entries")
            rows = cursor.fetchall()

        # Calculate cosine similarity
        results = []
        for row in rows:
            entry_id, content, embedding_blob, source, metadata_json, created_at = row
            embedding = json.loads(embedding_blob.decode())

            # Simple cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)

            results.append((
                similarity,
                KnowledgeEntry(
                    id=entry_id,
                    content=content,
                    embedding=embedding,
                    source=source,
                    metadata=json.loads(metadata_json),
                    created_at=created_at
                )
            ))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def knowledge_get_by_source(self, source: str) -> List[KnowledgeEntry]:
        """Get all knowledge entries from a specific source."""
        with sqlite3.connect(self.knowledge_db) as conn:
            cursor = conn.execute(
                "SELECT id, content, embedding, source, metadata, created_at FROM knowledge_entries WHERE source = ?",
                (source,)
            )
            rows = cursor.fetchall()

        return [
            KnowledgeEntry(
                id=row[0],
                content=row[1],
                embedding=json.loads(row[2].decode()),
                source=row[3],
                metadata=json.loads(row[4]),
                created_at=row[5]
            )
            for row in rows
        ]

    # Memory operations
    def memory_add(self, session_id: str, content: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add memory entry."""
        entry_id = str(uuid4())
        now = time.time()
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.memory_db) as conn:
            conn.execute(
                "INSERT INTO memory_entries (id, session_id, content, memory_type, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, session_id, content, memory_type, now, metadata_json)
            )

        logger.debug("Added memory entry: %s for session %s", entry_id, session_id)
        return entry_id

    def memory_get_session(self, session_id: str, memory_type: Optional[str] = None, limit: int = 100) -> List[MemoryEntry]:
        """Get memory entries for a session."""
        query = "SELECT id, session_id, content, memory_type, timestamp, metadata FROM memory_entries WHERE session_id = ?"
        params = [session_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            MemoryEntry(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                timestamp=row[4],
                metadata=json.loads(row[5])
            )
            for row in rows
        ]

    def memory_get_recent(self, memory_type: str, limit: int = 50) -> List[MemoryEntry]:
        """Get recent memory entries of a specific type."""
        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute(
                "SELECT id, session_id, content, memory_type, timestamp, metadata FROM memory_entries WHERE memory_type = ? ORDER BY timestamp DESC LIMIT ?",
                (memory_type, limit)
            )
            rows = cursor.fetchall()

        return [
            MemoryEntry(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                timestamp=row[4],
                metadata=json.loads(row[5])
            )
            for row in rows
        ]

    def memory_cleanup_old(self, days_old: int = 30):
        """Remove memory entries older than specified days."""
        cutoff = time.time() - (days_old * 24 * 60 * 60)
        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE timestamp < ?", (cutoff,))
            removed = cursor.rowcount
            logger.debug("Removed %d old memory entries", removed)

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

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        # Cache stats
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            stats['cache_entries'] = cursor.fetchone()[0]

        # Knowledge stats
        with sqlite3.connect(self.knowledge_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
            stats['knowledge_entries'] = cursor.fetchone()[0]

        # Memory stats
        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_entries")
            stats['memory_entries'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM memory_entries")
            stats['unique_sessions'] = cursor.fetchone()[0]

        return stats