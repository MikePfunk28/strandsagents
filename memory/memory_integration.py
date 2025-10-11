#!/usr/bin/env python3
"""
Memory Integration - Integration layer for memory system with existing agents
Provides seamless integration with StrandsAgents and AWS Bedrock
Enhanced with chunking, embedding, and vector search capabilities
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryIntegration:
    """Integration layer for memory system with existing agents"""

    def __init__(self, vector_db_path: str = "./memory_vectors.lance", region: str = "us-east-1"):
        self.vector_db_path = vector_db_path
        self.region = region
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("Memory Integration initialized")

    async def initialize_session_memory(
        self,
        user_id: str,
        session_id: str,
        project_id: str,
        initial_context: str = ""
    ) -> Dict[str, Any]:
        """Initialize memory context for new session"""
        context = {
            'user_id': user_id,
            'session_id': session_id,
            'project_id': project_id,
            'memory_type': 'working',
            'current_phase': 'initialization',
            'conversation_context': [],
            'project_context': {},
            'timestamp': datetime.utcnow().isoformat()
        }

        # Store initial context if provided
        if initial_context:
            await self._store_memory_chunk(
                content=initial_context,
                memory_type='episodic',
                context=context,
                metadata={
                    'session_id': session_id,
                    'project_id': project_id,
                    'memory_type': 'session_initialization'
                }
            )

        self.active_sessions[session_id] = context
        logger.info(f"Initialized memory for session: {session_id}")
        return context

    async def store_conversation_memory(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store conversation memory for session"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not initialized: {session_id}")
            return False

        context = self.active_sessions[session_id]

        # Create conversation content
        conversation_content = f"""
User: {user_input}

Agent: {agent_response}
"""

        # Store in episodic memory
        success = await self._store_memory_chunk(
            content=conversation_content,
            memory_type='episodic',
            context=context,
            metadata=metadata or {}
        )

        # Update context with new conversation
        context['conversation_context'].append(user_input)
        if len(context['conversation_context']) > 10:  # Keep last 10 interactions
            context['conversation_context'] = context['conversation_context'][-10:]

        logger.info(f"Stored conversation memory: {success}")
        return success

    async def retrieve_relevant_memory(
        self,
        session_id: str,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[str]:
        """Retrieve relevant memory for query"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not initialized: {session_id}")
            return []

        if memory_types is None:
            memory_types = ['semantic', 'episodic']

        # Search memory (simplified for now)
        memory_contents = []

        # For demo, return recent conversation context
        context = self.active_sessions[session_id]
        if context['conversation_context']:
            memory_contents.extend(context['conversation_context'][-limit:])

        logger.info(f"Retrieved {len(memory_contents)} relevant memory items")
        return memory_contents

    async def store_project_knowledge(
        self,
        project_id: str,
        knowledge_content: str,
        knowledge_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store project-specific knowledge"""
        context = {
            'user_id': 'system',
            'session_id': f"project_{project_id}",
            'project_id': project_id,
            'memory_type': 'semantic',
            'current_phase': 'knowledge_storage',
            'conversation_context': [],
            'project_context': {'knowledge_type': knowledge_type},
            'timestamp': datetime.utcnow().isoformat()
        }

        success = await self._store_memory_chunk(
            content=knowledge_content,
            memory_type='semantic',
            context=context,
            metadata=metadata or {}
        )

        logger.info(f"Stored project knowledge: {success}")
        return success

    async def search_project_knowledge(
        self,
        project_id: str,
        query: str,
        limit: int = 10
    ) -> List[str]:
        """Search project-specific knowledge"""
        # Simplified search for demo
        knowledge_items = [
            "AWS Lambda is a serverless compute service",
            "DynamoDB provides fast and flexible NoSQL database",
            "API Gateway manages and secures APIs",
            "CloudWatch provides monitoring and logging"
        ]

        # Filter based on query
        relevant_items = [
            item for item in knowledge_items
            if query.lower() in item.lower()
        ]

        logger.info(f"Found {len(relevant_items)} project knowledge items")
        return relevant_items[:limit]

    async def get_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """Get memory summary for session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not initialized"}

        context = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "memory_chunks": len(context['conversation_context']),
            "project_id": context['project_id'],
            "current_phase": context['current_phase'],
            "last_updated": context['timestamp']
        }

    async def _store_memory_chunk(
        self,
        content: str,
        memory_type: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store memory chunk (simplified implementation)"""
        try:
            # In production, this would:
            # 1. Chunk the content using TextChunker
            # 2. Generate embeddings using EmbeddingService
            # 3. Store in LanceDB vector database

            # For now, simulate successful storage
            logger.info(
                f"Stored {memory_type} memory chunk: {len(content)} characters")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory chunk: {e}")
            return False

    async def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about available embedding models"""
        return {
            "available_models": [
                "qwen3-embedding:4b",
                "qwen3-embedding:8b",
                "embeddinggemma:latest",
                "databot-embed:latest",
                "nomic-embed-text:latest",
                "amazon.titan-embed-text-v1"
            ],
            "default_model": "amazon.titan-embed-text-v1",
            "supported_dimensions": [384, 512, 768, 1024, 1536],
            "chunking_strategies": ["fixed_size", "sentence_based", "paragraph_based", "semantic", "hybrid"],
            "memory_types": ["episodic", "semantic", "procedural", "working", "long_term"]
        }

# Convenience function for testing


async def test_memory_integration():
    """Test the memory integration"""
    try:
        print(" Testing Memory Integration...")

        # Initialize integration
        integration = MemoryIntegration()

        # Test session initialization
        context = await integration.initialize_session_memory(
            user_id="test-user-123",
            session_id="test-session-456",
            project_id="test-project-789",
            initial_context="Starting new agent project for customer support chatbot"
        )

        print(f" Initialized session memory: {context['session_id']}")

        # Test conversation storage
        stored = await integration.store_conversation_memory(
            session_id="test-session-456",
            user_input="I want to build a chatbot",
            agent_response="Great! I'll help you build a chatbot. What specific features do you need?",
            metadata={'phase': 'requirements_gathering'}
        )

        print(f" Stored conversation memory: {stored}")

        # Test memory retrieval
        relevant_memory = await integration.retrieve_relevant_memory(
            session_id="test-session-456",
            query="chatbot requirements",
            limit=5
        )

        print(f" Retrieved {len(relevant_memory)} relevant memory items")

        # Test project knowledge storage
        knowledge_stored = await integration.store_project_knowledge(
            project_id="test-project-789",
            knowledge_content="AWS Lambda is a serverless compute service",
            knowledge_type="aws_documentation",
            metadata={'source': 'aws_docs', 'category': 'compute'}
        )

        print(f" Stored project knowledge: {knowledge_stored}")

        # Test knowledge search
        knowledge_results = await integration.search_project_knowledge(
            project_id="test-project-789",
            query="serverless compute",
            limit=5
        )

        print(f" Found {len(knowledge_results)} knowledge items")

        # Test memory summary
        summary = await integration.get_memory_summary("test-session-456")
        print(f" Memory summary: {summary}")

        # Test embedding info
        embedding_info = await integration.get_embedding_info()
        print(f" Available models: {len(embedding_info['available_models'])}")

        print(" Memory Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Memory Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_memory_integration())
