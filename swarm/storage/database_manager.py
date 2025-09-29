"""Database management system for the swarm following DESIGN.md requirements."""

import asyncio
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Memory entry for context storage."""
    id: str
    content: str
    timestamp: datetime
    session_id: str
    agent_id: str
    memory_type: str  # 'conversation', 'task', 'knowledge', 'code'
    importance: float  # 0.0 to 1.0
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class KnowledgeEntry:
    """Knowledge entry for external information."""
    id: str
    content: str
    source: str
    timestamp: datetime
    category: str
    confidence: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class CodeEntry:
    """Code understanding entry with explanations."""
    id: str
    file_path: str
    line_start: int
    line_end: int
    code_content: str
    explanation: str
    scope: str  # 'function', 'class', 'module', 'block'
    complexity: float
    timestamp: datetime
    embedding: Optional[List[float]] = None

class DatabaseManager:
    """Manages all database operations for the swarm system."""

    def __init__(self, storage_dir: str = "./swarm/storage/data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Database paths
        self.cache_db = self.storage_dir / "cache.db"
        self.memory_db = self.storage_dir / "memory.db"
        self.knowledge_db = self.storage_dir / "knowledge.db"
        self.coderl_db = self.storage_dir / "coderl.db"

        # Initialize databases
        self._init_databases()

    def _init_databases(self):
        """Initialize all database schemas."""
        self._init_cache_db()
        self._init_memory_db()
        self._init_knowledge_db()
        self._init_coderl_db()

        logger.info("Database manager initialized with all schemas")

    def _init_cache_db(self):
        """Initialize cache database for response caching."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expiry_time REAL,
                    access_count INTEGER DEFAULT 1,
                    last_accessed REAL NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_agent
                ON cache_entries(agent_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_timestamp
                ON cache_entries(timestamp)
            """)

    def _init_memory_db(self):
        """Initialize memory database for context storage."""
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON memory_entries(session_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_agent
                ON memory_entries(agent_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type
                ON memory_entries(memory_type)
            """)

    def _init_knowledge_db(self):
        """Initialize knowledge database for external information."""
        with sqlite3.connect(self.knowledge_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_category
                ON knowledge_entries(category)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_source
                ON knowledge_entries(source)
            """)

    def _init_coderl_db(self):
        """Initialize code understanding database."""
        with sqlite3.connect(self.coderl_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_entries (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    code_content TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    complexity REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    embedding BLOB
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_file
                ON code_entries(file_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_scope
                ON code_entries(scope)
            """)

    # Cache operations
    async def store_cache(self, cache_key: str, response: str, agent_id: str,
                         agent_type: str, expiry_seconds: Optional[int] = None):
        """Store response in cache."""
        timestamp = datetime.now().timestamp()
        expiry_time = None
        if expiry_seconds:
            expiry_time = timestamp + expiry_seconds

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries
                (cache_key, response, agent_id, agent_type, timestamp, expiry_time, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, response, agent_id, agent_type, timestamp, expiry_time, timestamp))

    async def get_cache(self, cache_key: str) -> Optional[str]:
        """Retrieve response from cache."""
        current_time = datetime.now().timestamp()

        with sqlite3.connect(self.cache_db) as conn:
            # Check if entry exists and not expired
            cursor = conn.execute("""
                SELECT response FROM cache_entries
                WHERE cache_key = ? AND (expiry_time IS NULL OR expiry_time > ?)
            """, (cache_key, current_time))

            result = cursor.fetchone()
            if result:
                # Update access count and timestamp
                conn.execute("""
                    UPDATE cache_entries
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE cache_key = ?
                """, (current_time, cache_key))

                return result[0]

        return None

    async def clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = datetime.now().timestamp()

        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("""
                DELETE FROM cache_entries
                WHERE expiry_time IS NOT NULL AND expiry_time <= ?
            """, (current_time,))

            deleted_count = cursor.rowcount
            logger.info(f"Cleared {deleted_count} expired cache entries")

    # Memory operations
    async def store_memory(self, entry: MemoryEntry):
        """Store memory entry."""
        embedding_blob = None
        if entry.embedding:
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_entries
                (id, content, timestamp, session_id, agent_id, memory_type,
                 importance, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.content, entry.timestamp.timestamp(),
                entry.session_id, entry.agent_id, entry.memory_type,
                entry.importance, json.dumps(entry.metadata), embedding_blob
            ))

    async def get_memories(self, session_id: Optional[str] = None,
                          agent_id: Optional[str] = None,
                          memory_type: Optional[str] = None,
                          limit: int = 50) -> List[MemoryEntry]:
        """Retrieve memories with filters."""
        query = "SELECT * FROM memory_entries WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        memories = []
        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                embedding = None
                if row[8]:  # embedding blob
                    embedding = np.frombuffer(row[8], dtype=np.float32).tolist()

                memory = MemoryEntry(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromtimestamp(row[2]),
                    session_id=row[3],
                    agent_id=row[4],
                    memory_type=row[5],
                    importance=row[6],
                    metadata=json.loads(row[7]),
                    embedding=embedding
                )
                memories.append(memory)

        return memories

    # Knowledge operations
    async def store_knowledge(self, entry: KnowledgeEntry):
        """Store knowledge entry."""
        embedding_blob = None
        if entry.embedding:
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

        with sqlite3.connect(self.knowledge_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_entries
                (id, content, source, timestamp, category, confidence, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.content, entry.source, entry.timestamp.timestamp(),
                entry.category, entry.confidence, json.dumps(entry.metadata), embedding_blob
            ))

    async def search_knowledge(self, category: Optional[str] = None,
                              source: Optional[str] = None,
                              min_confidence: float = 0.0,
                              limit: int = 50) -> List[KnowledgeEntry]:
        """Search knowledge entries."""
        query = "SELECT * FROM knowledge_entries WHERE confidence >= ?"
        params = [min_confidence]

        if category:
            query += " AND category = ?"
            params.append(category)

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        knowledge_entries = []
        with sqlite3.connect(self.knowledge_db) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                embedding = None
                if row[7]:  # embedding blob
                    embedding = np.frombuffer(row[7], dtype=np.float32).tolist()

                entry = KnowledgeEntry(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    timestamp=datetime.fromtimestamp(row[3]),
                    category=row[4],
                    confidence=row[5],
                    metadata=json.loads(row[6]),
                    embedding=embedding
                )
                knowledge_entries.append(entry)

        return knowledge_entries

    # Code understanding operations
    async def store_code_understanding(self, entry: CodeEntry):
        """Store code understanding entry."""
        embedding_blob = None
        if entry.embedding:
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

        with sqlite3.connect(self.coderl_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO code_entries
                (id, file_path, line_start, line_end, code_content,
                 explanation, scope, complexity, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.file_path, entry.line_start, entry.line_end,
                entry.code_content, entry.explanation, entry.scope,
                entry.complexity, entry.timestamp.timestamp(), embedding_blob
            ))

    async def get_code_understanding(self, file_path: Optional[str] = None,
                                   scope: Optional[str] = None,
                                   limit: int = 50) -> List[CodeEntry]:
        """Get code understanding entries."""
        query = "SELECT * FROM code_entries WHERE 1=1"
        params = []

        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)

        if scope:
            query += " AND scope = ?"
            params.append(scope)

        query += " ORDER BY complexity DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        code_entries = []
        with sqlite3.connect(self.coderl_db) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                embedding = None
                if row[9]:  # embedding blob
                    embedding = np.frombuffer(row[9], dtype=np.float32).tolist()

                entry = CodeEntry(
                    id=row[0],
                    file_path=row[1],
                    line_start=row[2],
                    line_end=row[3],
                    code_content=row[4],
                    explanation=row[5],
                    scope=row[6],
                    complexity=row[7],
                    timestamp=datetime.fromtimestamp(row[8]),
                    embedding=embedding
                )
                code_entries.append(entry)

        return code_entries

    # Utility methods
    def create_cache_key(self, content: str, context: Dict[str, Any] = None) -> str:
        """Create a cache key from content and context."""
        key_content = content
        if context:
            key_content += json.dumps(context, sort_keys=True)

        return hashlib.md5(key_content.encode()).hexdigest()

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about all databases."""
        stats = {}

        # Cache stats
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("SELECT COUNT(*), AVG(access_count) FROM cache_entries")
            cache_count, avg_access = cursor.fetchone()
            stats['cache'] = {
                'total_entries': cache_count or 0,
                'average_access_count': avg_access or 0
            }

        # Memory stats
        with sqlite3.connect(self.memory_db) as conn:
            cursor = conn.execute("""
                SELECT memory_type, COUNT(*), AVG(importance)
                FROM memory_entries GROUP BY memory_type
            """)
            memory_by_type = {}
            for row in cursor.fetchall():
                memory_by_type[row[0]] = {
                    'count': row[1],
                    'avg_importance': row[2]
                }
            stats['memory'] = memory_by_type

        # Knowledge stats
        with sqlite3.connect(self.knowledge_db) as conn:
            cursor = conn.execute("""
                SELECT category, COUNT(*), AVG(confidence)
                FROM knowledge_entries GROUP BY category
            """)
            knowledge_by_category = {}
            for row in cursor.fetchall():
                knowledge_by_category[row[0]] = {
                    'count': row[1],
                    'avg_confidence': row[2]
                }
            stats['knowledge'] = knowledge_by_category

        # Code stats
        with sqlite3.connect(self.coderl_db) as conn:
            cursor = conn.execute("""
                SELECT scope, COUNT(*), AVG(complexity)
                FROM code_entries GROUP BY scope
            """)
            code_by_scope = {}
            for row in cursor.fetchall():
                code_by_scope[row[0]] = {
                    'count': row[1],
                    'avg_complexity': row[2]
                }
            stats['code'] = code_by_scope

        return stats

# Example usage and testing
async def demo_database_manager():
    """Demonstrate database manager functionality."""
    print("Database Manager Demo")
    print("=" * 30)

    db_manager = DatabaseManager("./demo_storage")

    # Test cache operations
    await db_manager.store_cache(
        "test_key",
        "This is a test response",
        "agent_001",
        "research",
        expiry_seconds=3600
    )

    cached_response = await db_manager.get_cache("test_key")
    print(f"Cached response: {cached_response}")

    # Test memory operations
    memory_entry = MemoryEntry(
        id="mem_001",
        content="User asked about renewable energy",
        timestamp=datetime.now(),
        session_id="session_001",
        agent_id="research_001",
        memory_type="conversation",
        importance=0.8,
        metadata={"topic": "renewable_energy"}
    )

    await db_manager.store_memory(memory_entry)

    memories = await db_manager.get_memories(session_id="session_001")
    print(f"Retrieved {len(memories)} memories")

    # Test knowledge operations
    knowledge_entry = KnowledgeEntry(
        id="know_001",
        content="Solar panel efficiency increased by 20% in 2024",
        source="research_paper",
        timestamp=datetime.now(),
        category="renewable_energy",
        confidence=0.9,
        metadata={"year": 2024, "technology": "solar"}
    )

    await db_manager.store_knowledge(knowledge_entry)

    knowledge = await db_manager.search_knowledge(category="renewable_energy")
    print(f"Retrieved {len(knowledge)} knowledge entries")

    # Test code understanding
    code_entry = CodeEntry(
        id="code_001",
        file_path="./swarm/agents/base_assistant.py",
        line_start=1,
        line_end=10,
        code_content="class BaseAssistant(ABC):",
        explanation="Base abstract class for all assistant microservices",
        scope="class",
        complexity=0.6,
        timestamp=datetime.now()
    )

    await db_manager.store_code_understanding(code_entry)

    code_entries = await db_manager.get_code_understanding(scope="class")
    print(f"Retrieved {len(code_entries)} code entries")

    # Get database stats
    stats = await db_manager.get_database_stats()
    print(f"Database stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(demo_database_manager())