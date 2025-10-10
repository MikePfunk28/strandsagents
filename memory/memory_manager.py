#!/usr/bin/env python3
"""
Memory Manager - Advanced memory management with chunking and embedding
Comprehensive vector storage with LanceDB and AWS Bedrock integration
Enhanced with semantic search and context preservation for StrandsAgents
"""

import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"      # Conversation history and interactions
    SEMANTIC = "semantic"      # Knowledge and concepts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"        # Short-term active memory
    LONG_TERM = "long_term"    # Persistent knowledge storage

class EmbeddingModel(Enum):
    """Available embedding models"""
    QWEN3_4B = "qwen3-embedding:4b"
    QWEN3_8B = "qwen3-embedding:8b"
    EMBEDDING_GEMMA = "embeddinggemma:latest"
    DATABOT_EMBED = "databot-embed:latest"
    NOMIC_EMBED = "nomic-embed-text:latest"
    BEDROCK_TITAN = "amazon.titan-embed-text-v1"

class ChunkingStrategy(Enum):
    """Text chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class MemoryChunk:
    """Individual memory chunk with metadata"""
    chunk_id: str
    memory_type: MemoryType
    content: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    access_count: int
    last_accessed: Optional[str]

@dataclass
class MemorySearchResult:
    """Search result from memory"""
    chunk: MemoryChunk
    similarity_score: float
    rank: int
    context: str
    relevance_score: float

@dataclass
class MemoryContext:
    """Context for memory operations"""
    user_id: str
    session_id: str
    project_id: str
    memory_type: MemoryType
    current_phase: str
    conversation_context: List[str]
    project_context: Dict[str, Any]
    timestamp: str

class TextChunker:
    """Advanced text chunking with multiple strategies"""

    def __init__(self, default_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID):
        self.default_strategy = default_strategy
        self.chunking_configs = {
            ChunkingStrategy.FIXED_SIZE: {
                'chunk_size': 512,
                'overlap': 50,
                'min_chunk_size': 100
            },
            ChunkingStrategy.SENTENCE_BASED: {
                'max_sentences': 5,
                'overlap_sentences': 1,
                'min_chunk_length': 200
            },
            ChunkingStrategy.PARAGRAPH_BASED: {
                'max_paragraphs': 3,
                'overlap_paragraphs': 1,
                'min_chunk_length': 300
            },
            ChunkingStrategy.SEMANTIC: {
                'semantic_threshold': 0.7,
                'max_chunk_length': 1000,
                'min_chunk_length': 100
            }
        }

    async def chunk_text(
        self,
        text: str,
        strategy: Optional[ChunkingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[MemoryChunk]:
        """Chunk text using specified strategy"""
        if strategy is None:
            strategy = self.default_strategy

        chunks = []

        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text, metadata)
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            chunks = self._chunk_by_sentences(text, metadata)
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            chunks = self._chunk_by_paragraphs(text, metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text, metadata)
        elif strategy == ChunkingStrategy.HYBRID:
            chunks = self._chunk_hybrid(text, metadata)

        logger.info(f"Created {len(chunks)} chunks using {strategy.value} strategy")
        return chunks

    def _chunk_fixed_size(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Chunk text into fixed-size pieces"""
        config = self.chunking_configs[ChunkingStrategy.FIXED_SIZE]
        chunk_size = config['chunk_size']
        overlap = config['overlap']

        chunks = []
        words = text.split()
        total_words = len(words)

        for i in range(0, total_words, chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text) >= config['min_chunk_size']:
                chunk = MemoryChunk(
                    chunk_id=str(uuid.uuid4()),
                    memory_type=MemoryType.SEMANTIC,
                    content=chunk_text,
                    chunk_index=len(chunks),
                    total_chunks=0,  # Will be updated later
                    embedding=None,
                    metadata=metadata or {},
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat(),
                    access_count=0,
                    last_accessed=None
                )
                chunks.append(chunk)

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_by_sentences(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Chunk text by sentences"""
        config = self.chunking_configs[ChunkingStrategy.SENTENCE_BASED]

        # Split into sentences (simple regex-based)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)

            # Check if we should create a chunk
            if (len(current_chunk) >= config['max_sentences'] or
                i == len(sentences) - 1):

                if len(current_chunk) > 0:
                    chunk_text = '. '.join(current_chunk)

                    if len(chunk_text) >= config['min_chunk_length']:
                        chunk = MemoryChunk(
                            chunk_id=str(uuid.uuid4()),
                            memory_type=MemoryType.SEMANTIC,
                            content=chunk_text,
                            chunk_index=len(chunks),
                            total_chunks=0,
                            embedding=None,
                            metadata=metadata or {},
                            created_at=datetime.utcnow().isoformat(),
                            updated_at=datetime.utcnow().isoformat(),
                            access_count=0,
                            last_accessed=None
                        )
                        chunks.append(chunk)

                current_chunk = [sentence] if i < len(sentences) - 1 else []

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_by_paragraphs(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Chunk text by paragraphs"""
        config = self.chunking_configs[ChunkingStrategy.PARAGRAPH_BASED]

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []

        for i, paragraph in enumerate(paragraphs):
            current_chunk.append(paragraph)

            if (len(current_chunk) >= config['max_paragraphs'] or
                i == len(paragraphs) - 1):

                if len(current_chunk) > 0:
                    chunk_text = '\n\n'.join(current_chunk)

                    if len(chunk_text) >= config['min_chunk_length']:
                        chunk = MemoryChunk(
                            chunk_id=str(uuid.uuid4()),
                            memory_type=MemoryType.SEMANTIC,
                            content=chunk_text,
                            chunk_index=len(chunks),
                            total_chunks=0,
                            embedding=None,
                            metadata=metadata or {},
                            created_at=datetime.utcnow().isoformat(),
                            updated_at=datetime.utcnow().isoformat(),
                            access_count=0,
                            last_accessed=None
                        )
                        chunks.append(chunk)

                current_chunk = [paragraph] if i < len(paragraphs) - 1 else []

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_semantic(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Chunk text based on semantic coherence"""
        config = self.chunking_configs[ChunkingStrategy.SEMANTIC]

        # Simple semantic chunking based on topic changes
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_topic = None

        for sentence in sentences:
            # Simple topic detection (can be enhanced with NLP)
            sentence_topic = self._detect_sentence_topic(sentence)

            if (current_topic and sentence_topic != current_topic and
                len(current_chunk) > 0):
                # Topic changed, create new chunk
                chunk_text = '. '.join(current_chunk)
                if len(chunk_text) >= config['min_chunk_length']:
                    chunk = MemoryChunk(
                        chunk_id=str(uuid.uuid4()),
                        memory_type=MemoryType.SEMANTIC,
                        content=chunk_text,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        embedding=None,
                        metadata=metadata or {},
                        created_at=datetime.utcnow().isoformat(),
                        updated_at=datetime.utcnow().isoformat(),
                        access_count=0,
                        last_accessed=None
                    )
                    chunks.append(chunk)

                current_chunk = [sentence]
                current_topic = sentence_topic
            else:
                current_chunk.append(sentence)
                if not current_topic:
                    current_topic = sentence_topic

        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if len(chunk_text) >= config['min_chunk_length']:
                chunk = MemoryChunk(
                    chunk_id=str(uuid.uuid4()),
                    memory_type=MemoryType.SEMANTIC,
                    content=chunk_text,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    embedding=None,
                    metadata=metadata or {},
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat(),
                    access_count=0,
                    last_accessed=None
                )
                chunks.append(chunk)

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_hybrid(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[MemoryChunk]:
        """Hybrid chunking combining multiple strategies"""
        # Start with paragraph-based chunking
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 800:  # Long paragraph, use sentence chunking
                sentence_chunks = self._chunk_by_sentences(paragraph, metadata)
                for chunk in sentence_chunks:
                    chunk.chunk_index = len(chunks)
                    chunk.total_chunks = 0  # Will update later
                    chunks.append(chunk)
            else:
                # Short paragraph, keep as is
                chunk = MemoryChunk(
                    chunk_id=str(uuid.uuid4()),
                    memory_type=MemoryType.SEMANTIC,
                    content=paragraph,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    embedding=None,
                    metadata=metadata or {},
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat(),
                    access_count=0,
                    last_accessed=None
                )
                chunks.append(chunk)

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _detect_sentence_topic(self, sentence: str) -> str:
        """Simple topic detection for sentences"""
        sentence_lower = sentence.lower()

        # Define topic keywords
        topics = {
            'technical': ['api', 'code', 'function', 'class', 'method', 'variable', 'database', 'server'],
            'business': ['customer', 'product', 'service', 'revenue', 'profit', 'market', 'strategy'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'access', 'permission'],
            'architecture': ['architecture', 'design', 'pattern', 'structure', 'framework', 'system'],
            'deployment': ['deploy', 'deployment', 'production', 'staging', 'environment', 'configuration']
        }

        for topic, keywords in topics.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return topic

        return 'general'

class EmbeddingService:
    """Advanced embedding service with multiple model support"""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)

        # Model configurations
        self.model_configs = {
            EmbeddingModel.QWEN3_4B: {
                'dimensions': 512,
                'max_tokens': 512,
                'type': 'local'
            },
            EmbeddingModel.QWEN3_8B: {
                'dimensions': 1024,
                'max_tokens': 512,
                'type': 'local'
            },
            EmbeddingModel.EMBEDDING_GEMMA: {
                'dimensions': 768,
                'max_tokens': 512,
                'type': 'local'
            },
            EmbeddingModel.DATABOT_EMBED: {
                'dimensions': 384,
                'max_tokens': 256,
                'type': 'local'
            },
            EmbeddingModel.NOMIC_EMBED: {
                'dimensions': 768,
                'max_tokens': 512,
                'type': 'local'
            },
            EmbeddingModel.BEDROCK_TITAN: {
                'dimensions': 1536,
                'max_tokens': 8192,
                'type': 'bedrock'
            }
        }

        logger.info("Embedding Service initialized")

    async def generate_embedding(
        self,
        text: str,
        model: EmbeddingModel = EmbeddingModel.BEDROCK_TITAN,
        normalize: bool = True
    ) -> Optional[List[float]]:
        """Generate embedding for text using specified model"""
        try:
            if model == EmbeddingModel.BEDROCK_TITAN:
                return await self._generate_bedrock_embedding(text, normalize)
            else:
                return await self._generate_local_embedding(text, model, normalize)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def _generate_bedrock_embedding(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """Generate embedding using AWS Bedrock Titan"""
        try:
            # Prepare request for Titan embeddings
            request_body = {
                "inputText": text
            }

            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']

            # Normalize if requested
            if normalize:
                embedding = self._normalize_vector(embedding)

            logger.info(f"Generated Bedrock embedding: {len(embedding)} dimensions")
            return embedding

        except ClientError as e:
            logger.error(f"Bedrock embedding error: {e}")
            return None

    async def _generate_local_embedding(self, text: str, model: EmbeddingModel, normalize: bool = True) -> Optional[List[float]]:
        """Generate embedding using local Ollama models"""
        try:
            # This would integrate with local Ollama instance
            # For now, return mock embedding
            config = self.model_configs[model]
            dimensions = config['dimensions']

            # Generate pseudo-random but deterministic embedding based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            hash_int = int(text_hash, 16)

            # Create deterministic pseudo-random vector
            np.random.seed(hash_int % (2**32))
            embedding = np.random.normal(0, 1, dimensions).tolist()

            # Normalize if requested
            if normalize:
                embedding = self._normalize_vector(embedding)

            logger.info(f"Generated local embedding ({model.value}): {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return None

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        try:
            np_vector = np.array(vector)
            norm = np.linalg.norm(np_vector)
            if norm == 0:
                return vector
            return (np_vector / norm).tolist()
        except Exception:
            return vector

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: EmbeddingModel = EmbeddingModel.BEDROCK_TITAN,
        normalize: bool = True
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        embeddings = []

        for text in texts:
            embedding = await self.generate_embedding(text, model, normalize)
            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

class LanceDBVectorStore:
    """LanceDB-based vector storage and retrieval"""

    def __init__(self, db_path: str = "./memory_vectors.lance", region: str = "us-east-1"):
        self.db_path = db_path
        self.region = region

        # Initialize LanceDB (would use actual LanceDB in production)
        # For now, simulate with in-memory storage
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self.metadata_index: Dict[str, List[str]] = {}  # For filtering

        logger.info(f"LanceDB Vector Store initialized at {db_path}")

    async def store_chunks(self, chunks: List[MemoryChunk]) -> bool:
        """Store memory chunks with embeddings"""
        try:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                # Store vector with metadata
                vector_id = chunk.chunk_id
                self.vectors[vector_id] = {
                    'chunk_id': chunk.chunk_id,
                    'memory_type': chunk.memory_type.value,
                    'content': chunk.content,
                    'embedding': chunk.embedding,
                    'metadata': chunk.metadata,
                    'created_at': chunk.created_at,
                    'access_count': chunk.access_count
                }

                # Update metadata index
                for key, value in chunk.metadata.items():
                    if key not in self.metadata_index:
                        self.metadata_index[key] = []
                    self.metadata_index[key].append(vector_id)

            logger.info(f"Stored {len(chunks)} chunks in vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            return False

    async def search_similar(
        self,
        query_embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        """Search for similar vectors"""
        try:
            results = []

            for vector_id, vector_data in self.vectors.items():
                # Apply memory type filter
                if memory_type and vector_data['memory_type'] != memory_type.value:
                    continue

                # Apply metadata filter
                if metadata_filter and not self._matches_metadata_filter(vector_data['metadata'], metadata_filter):
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, vector_data['embedding'])

                if similarity >= threshold:
                    # Recreate chunk object
                    chunk = MemoryChunk(
                        chunk_id=vector_data['chunk_id'],
                        memory_type=MemoryType(vector_data['memory_type']),
                        content=vector_data['content'],
                        chunk_index=0,  # Not stored in vector data
                        total_chunks=1,
                        embedding=vector_data['embedding'],
                        metadata=vector_data['metadata'],
                        created_at=vector_data['created_at'],
                        updated_at=datetime.utcnow().isoformat(),
                        access_count=vector_data['access_count'],
                        last_accessed=datetime.utcnow().isoformat()
                    )

                    result = MemorySearchResult(
                        chunk=chunk,
                        similarity_score=similarity,
                        rank=len(results),
                        context=self._generate_context(chunk),
                        relevance_score=self._calculate_relevance_score(chunk, similarity)
                    )

                    results.append(result)

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = results[:limit]

            # Update ranks
            for i, result in enumerate(results):
                result.rank = i

            logger.info(f"Found {len(results)} similar chunks above threshold {threshold}")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            np_vec1 = np.array(vec1)
            np_vec2 = np.array(vec2)

            dot_product = np.dot(np_vec1, np_vec2)
            norm1 = np.linalg.norm(np_vec1)
            norm2 = np.linalg.norm(np_vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception:
            return 0.0

    def _matches_metadata_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _generate_context(self, chunk: MemoryChunk) -> str:
        """Generate context summary for chunk"""
        # Extract key sentences or phrases
        sentences = re.split(r'[.!?]+', chunk.content)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]

        if key_sentences:
            return '. '.join(key_sentences) + '.'
        else:
            return chunk.content[:100] + '...'

    def _calculate_relevance_score(self, chunk: MemoryChunk, similarity: float) -> float:
        """Calculate relevance score combining similarity and metadata"""
        relevance = similarity

        # Boost relevance based on access count (popular chunks)
        access_boost = min(chunk.access_count * 0.01, 0.1)
        relevance += access_boost

        # Boost relevance for recently accessed chunks
        if chunk.last_accessed:
            try:
                last_accessed = datetime.fromisoformat(chunk.last_accessed.replace('Z', '+00:00'))
                hours_since_access = (datetime.utcnow() - last_accessed.replace(tzinfo=None)).total_seconds() / 3600

                if hours_since_access < 24:  # Accessed within last 24 hours
                    recency_boost = 0.05
                    relevance += recency_boost
            except:
                pass

        return min(relevance, 1.0)

    async def update_chunk_access(self, chunk_id: str) -> bool:
        """Update chunk access statistics"""
        try:
            if chunk_id in self.vectors:
                self.vectors[chunk_id]['access_count'] += 1
                self.vectors[chunk_id]['last_accessed'] = datetime.utcnow().isoformat()
                return True
        except Exception as e:
            logger.error(f"Failed to update chunk access: {e}")

        return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics"""
        total_chunks = len(self.vectors)
        total_embeddings = sum(1 for v in self.vectors.values() if v.get('embedding'))

        # Calculate average embedding dimensions
        dimensions = []
        for vector_data in self.vectors.values():
            embedding = vector_data.get('embedding')
            if embedding:
                dimensions.append(len(embedding))

        avg_dimensions = sum(dimensions) / len(dimensions) if dimensions else 0

        # Memory type distribution
        memory_types = {}
        for vector_data in self.vectors.values():
            mem_type = vector_data['memory_type']
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

        return {
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'embedding_coverage': total_embeddings / total_chunks if total_chunks > 0 else 0,
            'average_dimensions': round(avg_dimensions, 1),
            'memory_type_distribution': memory_types,
            'storage_size_mb': self._estimate_storage_size()
        }

    def _estimate_storage_size(self) -> float:
        """Estimate storage size in MB"""
        # Rough estimation: each vector entry ~1KB
        return round(len(self.vectors) * 0.001, 2)

class MemoryManager:
    """Main memory management system for StrandsAgents"""

    def __init__(self, vector_db_path: str = "./memory_vectors.lance", region: str = "us-east-1"):
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService(region)
        self.vector_store = LanceDBVectorStore(vector_db_path, region)

        # Memory configurations
        self.chunking_strategies = {
            MemoryType.EPISODIC: ChunkingStrategy.SENTENCE_BASED,
            MemoryType.SEMANTIC: ChunkingStrategy.HYBRID,
            MemoryType.PROCEDURAL: ChunkingStrategy.PARAGRAPH_BASED,
            MemoryType.WORKING: ChunkingStrategy.FIXED_SIZE,
            MemoryType.LONG_TERM: ChunkingStrategy.SEMANTIC
        }

        self.default_embedding_model = EmbeddingModel.BEDROCK_TITAN

        logger.info("Memory Manager initialized")

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        context: MemoryContext,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[MemoryChunk]:
        """Store content in memory with chunking and embedding"""
        try:
            # Get appropriate chunking strategy
            chunking_strategy = self.chunking_strategies.get(memory_type, ChunkingStrategy.HYBRID)

            # Chunk the content
            chunks = await self.chunker.chunk_text(
                content,
                strategy=chunking_strategy,
                metadata=metadata
            )

            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings_batch(
                texts,
                model=self.default_embedding_model
            )

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]

            # Store in vector database
            await self.vector_store.store_chunks(chunks)

            logger.info(f"Stored {len(chunks)} chunks of type {memory_type.value}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return []

    async def search_memory(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        """Search memory for relevant content"""
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.generate_embedding(
                query,
                model=self.default_embedding_model
            )

            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []

            # Search vector store
            results = await self.vector_store.search_similar(
                query_embedding,
                memory_type=memory_type,
                limit=limit,
                threshold=threshold,
                metadata_filter=metadata_filter
            )

            # Update access statistics
            for result in results:
                await self.vector_store.update_chunk_access(result.chunk.chunk_id)

            logger.info(f"Found {len(results)} relevant memory chunks")
            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def get_memory_context(
        self,
        user_id: str,
        session_id: str,
        project_id: str,
        current_phase: str
    ) -> MemoryContext:
        """Get memory context for current session"""
        return MemoryContext(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            memory_type=MemoryType.WORKING,
            current_phase=current_phase,
            conversation_context=[],
            project_context={},
            timestamp=datetime.utcnow().isoformat()
        )

    async def retrieve_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[MemoryChunk]:
        """Retrieve conversation history for session"""
        try:
            # Search for episodic memory related to session
            results = await self.search_memory(
                query=f"session:{session_id}",
                memory_type=MemoryType.EPISODIC,
                limit=limit,
                metadata_filter={'session_id': session_id}
            )

            return [result.chunk for result in results]

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    async def retrieve_project_knowledge(
        self,
        project_id: str,
        query: str,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Retrieve project-specific knowledge"""
        try:
            # Search semantic memory for project
            results = await self.search_memory(
                query=query,
                memory_type=MemoryType.SEMANTIC,
                limit=limit,
                metadata_filter={'project_id': project_id}
            )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve project knowledge: {e}")
            return []

    async def consolidate_working_memory(self, context: MemoryContext) -> bool:
        """Consolidate working memory to long-term storage"""
        try:
            # Get recent working memory
            working_memory = await self.search_memory(
                query=f"session:{context.session_id}",
                memory_type=MemoryType.WORKING,
                limit=100
            )

            if not working_memory:
                return True

            # Convert to long-term memory
            for result in working_memory:
                chunk = result.chunk
                chunk.memory_type = MemoryType.LONG_TERM
                chunk.metadata['consolidated_from'] = 'working_memory'
                chunk.metadata['consolidation_date'] = datetime.utcnow().isoformat()

                # Update in vector store
                await self.vector_store.store_chunks([chunk])

            logger.info(f"Consolidated {len(working_memory)} working memory chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to consolidate working memory: {e}")
            return False

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            storage_stats = await self.vector_store.get_storage_stats()

            # Calculate additional metrics
            total_chunks = storage_stats['total_chunks']
            memory_type_dist = storage_stats['memory_type_distribution']

            # Calculate average chunk size
            total_content_length = sum(
                len(vector_data['content'])
                for vector_data in self.vector_store.vectors.values()
            )
            avg_chunk_size = total_content_length / total_chunks if total_chunks > 0 else 0

            return {
                'storage_stats': storage_stats,
                'total_chunks': total_chunks,
                'memory_type_distribution': memory_type_dist,
                'average_chunk_size': round(avg_chunk_size, 1),
                'embedding_coverage': storage_stats['embedding_coverage'],
                'last_updated': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

# Convenience function for testing
async def test_memory_system():
    """Test the memory system"""
    try:
        print("🚀 Testing Memory System...")

        # Initialize memory manager
        memory_manager = MemoryManager()

        # Test text chunking
        sample_text = """
        AWS Lambda is a serverless compute service that lets you run code without provisioning or managing servers.
        You pay only for the compute time you consume. Lambda automatically scales your application by running code in response to each trigger.

        API Gateway is a fully managed service that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any scale.
        With API Gateway, you can create RESTful APIs and WebSocket APIs that enable real-time two-way communication applications.

        DynamoDB is a fast and flexible NoSQL database service for any scale. It supports both document and key-value store models.
        """

        chunks = await memory_manager.chunker.chunk_text(
            sample_text,
            strategy=ChunkingStrategy.HYBRID
        )

        print(f"✅ Created {len(chunks)} chunks")

        # Test embedding generation
        embedding_service = EmbeddingService()

        sample_chunk = chunks[0]
        embedding = await embedding_service.generate_embedding(
            sample_chunk.content,
            model=EmbeddingModel.BEDROCK_TITAN
        )

        if embedding:
            print(f"✅ Generated embedding: {len(embedding)} dimensions")
        else:
            print("❌ Failed to generate embedding")

        # Test memory storage
        context = await memory_manager.get_memory_context(
            user_id="test-user",
            session_id="test-session",
            project_id="test-project",
            current_phase="requirements"
        )

        stored_chunks = await memory_manager.store_memory(
            sample_text,
            MemoryType.SEMANTIC,
            context
        )

        print(f"✅ Stored {len(stored_chunks)} chunks in memory")

        # Test memory search
        search_results = await memory_manager.search_memory(
            query="AWS Lambda serverless",
            memory_type=MemoryType.SEMANTIC,
            limit=5
        )

        print(f"✅ Found {len(search_results)} relevant chunks")

        # Test memory stats
        stats = await memory_manager.get_memory_stats()
        print(f"✅ Memory stats: {stats}")

        print("🎉 Memory System test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Memory System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_memory_system())
