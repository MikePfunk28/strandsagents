"""
Database Manager for Swarm System

Manages multiple SQLite databases for the swarm system:
- cache.db: Caching layer for frequently accessed data
- memory.db: Context and memory storage
- knowledge.db: External knowledge and learned information
- coderl.db: Code explanations with embeddings and similarity search
"""

import sqlite3
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages multiple SQLite databases for the swarm system.

    Provides unified interface for:
    - Cache operations (cache.db)
    - Memory storage (memory.db)
    - Knowledge base (knowledge.db)
    - Code explanations (coderl.db)
    """

    def __init__(self, db_directory: str = "swarm_system/databases"):
        """
        Initialize database manager.

        Args:
            db_directory: Directory to store database files
        """
        self.db_directory = Path(db_directory)
        self.db_directory.mkdir(parents=True, exist_ok=True)

        # Database file paths
        self.cache_db = self.db_directory / "cache.db"
        self.memory_db = self.db_directory / "memory.db"
        self.knowledge_db = self.db_directory / "knowledge.db"
        self.coderl_db = self.db_directory / "coderl.db"

        # Initialize all databases
        self._init_cache_db()
        self._init_memory_db()
        self._init_knowledge_db()
        self._init_coderl_db()

        logger.info(f"Database manager initialized with directory: {self.db_directory}")

    def _get_connection(self, db_path: Path) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_cache_db(self) -> None:
        """Initialize cache database."""
        with self._get_connection(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _init_memory_db(self) -> None:
        """Initialize memory database."""
        with self._get_connection(self.memory_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL, -- 'short_term', 'long_term', 'working'
                    content TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT -- JSON metadata
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_memory_id INTEGER,
                    target_memory_id INTEGER,
                    relationship_type TEXT, -- 'related', 'contradicts', 'supports', etc.
                    strength REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                    FOREIGN KEY (target_memory_id) REFERENCES memories (id)
                )
            """)
            conn.commit()

    def _init_knowledge_db(self) -> None:
        """Initialize knowledge database."""
        with self._get_connection(self.knowledge_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    subtopic TEXT,
                    content TEXT NOT NULL,
                    source TEXT, -- URL, file, user_input, etc.
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT -- JSON metadata
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_knowledge_id INTEGER,
                    target_knowledge_id INTEGER,
                    relationship_type TEXT, -- 'related', 'prerequisite', 'example_of', etc.
                    strength REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_knowledge_id) REFERENCES knowledge (id),
                    FOREIGN KEY (target_knowledge_id) REFERENCES knowledge (id)
                )
            """)
            conn.commit()

    def _init_coderl_db(self) -> None:
        """Initialize code explanation database with embeddings support."""
        with self._get_connection(self.coderl_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_explanations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    line_content TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    scope TEXT, -- 'global', 'class', 'function', 'local'
                    embedding_vector TEXT, -- JSON array of embedding values
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT -- JSON metadata
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_scopes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    scope_type TEXT NOT NULL, -- 'file', 'class', 'function', 'block'
                    start_line INTEGER,
                    end_line INTEGER,
                    content TEXT,
                    embedding_vector TEXT, -- JSON array for scope-level embeddings
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # Cache operations
    def set_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a cache entry."""
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now().timestamp() + ttl_seconds

        with self._get_connection(self.cache_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at, last_accessed)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, json.dumps(value), expires_at))
            conn.commit()

    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cache entry."""
        with self._get_connection(self.cache_db) as conn:
            row = conn.execute("""
                SELECT value, expires_at FROM cache WHERE key = ?
            """, (key,)).fetchone()

            if not row:
                return None

            # Check expiration
            if row['expires_at'] and datetime.now().timestamp() > row['expires_at']:
                self.delete_cache(key)
                return None

            # Update access statistics
            conn.execute("""
                UPDATE cache SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE key = ?
            """, (key,))
            conn.commit()

            return json.loads(row['value'])

    def delete_cache(self, key: str) -> None:
        """Delete a cache entry."""
        with self._get_connection(self.cache_db) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    # Memory operations
    def store_memory(
        self,
        session_id: str,
        content: str,
        memory_type: str = "short_term",
        importance_score: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> int:
        """Store a memory entry."""
        with self._get_connection(self.memory_db) as conn:
            cursor = conn.execute("""
                INSERT INTO memories (session_id, memory_type, content, importance_score, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, memory_type, content, importance_score, json.dumps(metadata or {})))
            conn.commit()
            return cursor.lastrowid

    def get_memories(
        self,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve memory entries."""
        with self._get_connection(self.memory_db) as conn:
            query = "SELECT * FROM memories WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type)

            query += " ORDER BY importance_score DESC, created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # Knowledge operations
    def store_knowledge(
        self,
        topic: str,
        content: str,
        subtopic: Optional[str] = None,
        source: Optional[str] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> int:
        """Store a knowledge entry."""
        with self._get_connection(self.knowledge_db) as conn:
            cursor = conn.execute("""
                INSERT INTO knowledge (topic, subtopic, content, source, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (topic, subtopic, content, source, confidence, json.dumps(metadata or {})))
            conn.commit()
            return cursor.lastrowid

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """Search knowledge base using simple text matching."""
        with self._get_connection(self.knowledge_db) as conn:
            # Simple LIKE-based search (can be enhanced with FTS)
            search_term = f"%{query}%"
            rows = conn.execute("""
                SELECT * FROM knowledge
                WHERE topic LIKE ? OR subtopic LIKE ? OR content LIKE ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """, (search_term, search_term, search_term, limit)).fetchall()

            return [dict(row) for row in rows]

    # Code explanation operations
    def store_code_explanation(
        self,
        file_path: str,
        line_number: int,
        line_content: str,
        explanation: str,
        scope: str = "global",
        embedding_vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Store a code explanation with optional embedding."""
        with self._get_connection(self.coderl_db) as conn:
            cursor = conn.execute("""
                INSERT INTO code_explanations
                (file_path, line_number, line_content, explanation, scope, embedding_vector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path, line_number, line_content, explanation, scope,
                json.dumps(embedding_vector) if embedding_vector else None,
                json.dumps(metadata or {})
            ))
            conn.commit()
            return cursor.lastrowid

    def search_similar_code(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict]:
        """Search for similar code explanations using embeddings."""
        # Note: This is a simplified implementation
        # In production, you'd use vector similarity search (e.g., cosine similarity)
        with self._get_connection(self.coderl_db) as conn:
            rows = conn.execute("""
                SELECT * FROM code_explanations
                WHERE embedding_vector IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

            results = []
            for row in rows:
                # Simple scoring based on text similarity as fallback
                similarity_score = self._calculate_text_similarity(
                    query_embedding, row['line_content']
                )

                if similarity_score >= threshold:
                    results.append({
                        **dict(row),
                        "similarity_score": similarity_score
                    })

            return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def _calculate_text_similarity(self, embedding: List[float], text: str) -> float:
        """Calculate simple text-based similarity score."""
        # This is a placeholder - in reality, you'd compute cosine similarity
        # between the query embedding and stored embeddings
        return 0.5  # Placeholder score

    def store_file_scope(
        self,
        file_path: str,
        scope_name: str,
        scope_type: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        content: Optional[str] = None,
        embedding_vector: Optional[List[float]] = None
    ) -> int:
        """Store file scope information."""
        with self._get_connection(self.coderl_db) as conn:
            cursor = conn.execute("""
                INSERT INTO file_scopes
                (file_path, scope_name, scope_type, start_line, end_line, content, embedding_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, scope_name, scope_type, start_line, end_line, content,
                  json.dumps(embedding_vector) if embedding_vector else None))
            conn.commit()
            return cursor.lastrowid

    # Utility methods
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all databases."""
        stats = {}

        for db_name, db_path in [
            ("cache", self.cache_db),
            ("memory", self.memory_db),
            ("knowledge", self.knowledge_db),
            ("coderl", self.coderl_db)
        ]:
            try:
                with self._get_connection(db_path) as conn:
                    # Get table count
                    cursor = conn.execute("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()['count']

                    # Get total records across all tables
                    total_records = 0
                    if table_count > 0:
                        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = cursor.fetchall()

                        for table in tables:
                            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table['name']}")
                            total_records += cursor.fetchone()['count']

                    stats[db_name] = {
                        "exists": True,
                        "size_mb": db_path.stat().st_size / (1024 * 1024),
                        "tables": table_count,
                        "records": total_records
                    }

            except Exception as e:
                stats[db_name] = {
                    "exists": False,
                    "error": str(e)
                }

        return stats

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        with self._get_connection(self.cache_db) as conn:
            cursor = conn.execute("""
                DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.now().timestamp(),))
            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")

            return deleted_count

    def vacuum_all(self) -> None:
        """Vacuum all databases to reclaim space."""
        for db_path in [self.cache_db, self.memory_db, self.knowledge_db, self.coderl_db]:
            try:
                with self._get_connection(db_path) as conn:
                    conn.execute("VACUUM")
                    conn.commit()
                logger.info(f"Vacuumed database: {db_path}")
            except Exception as e:
                logger.error(f"Failed to vacuum {db_path}: {e}")


# Global database manager instance
db_manager = DatabaseManager()
