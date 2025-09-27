"""Main coding assistant with Ollama integration and semantic memory management."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from strands import Agent

from database_manager import DatabaseManager
from enhanced_memory_manager import EnhancedMemoryManager
from ollama_model import OllamaModel, create_ollama_model
from coding_tools import (
    python_repl,
    file_read,
    file_write,
    file_append,
    list_files,
    shell_execute,
    code_analyze,
    code_format,
    code_test,
    git_status,
    search_code,
    editor,
    load_tool,
)

# Import existing assistants from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from embedding_assistant import EmbeddingAssistant, embed_document_hierarchy
from chunking_assistant import ChunkingAssistant, chunk_hierarchy_tool

logger = logging.getLogger(__name__)


class CodingAssistant:
    """Advanced coding assistant with hierarchical memory and semantic understanding."""

    def __init__(
        self,
        db_dir: Path = Path("./data"),
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid4())
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)

        # Initialize components
        self.db_manager = DatabaseManager(self.db_dir)
        self.ollama_model = create_ollama_model(model_name, ollama_host)

        # Initialize embedding and chunking assistants
        self.embedding_assistant = EmbeddingAssistant(
            provider="ollama",
            host=ollama_host,
            model_id="embeddinggemma"
        )
        self.chunking_assistant = ChunkingAssistant(
            chunk_size=900,
            chunk_overlap=180
        )

        # Initialize enhanced memory manager with existing assistants
        self.memory_manager = EnhancedMemoryManager(
            self.db_manager,
            self.ollama_model,
            self.embedding_assistant,
            self.chunking_assistant
        )

        # Initialize Agent with tools (including embedding and chunking tools)
        self.tools = [
            python_repl,
            file_read,
            file_write,
            file_append,
            list_files,
            shell_execute,
            code_analyze,
            code_format,
            code_test,
            git_status,
            search_code,
            editor,
            load_tool,
            embed_document_hierarchy,
            chunk_hierarchy_tool,
        ]

        self.agent = Agent(
            model=self.ollama_model,
            tools=self.tools,
            system_prompt=self._create_system_prompt(),
        )

        # Context classification categories
        self.context_categories = {
            "coding": ["code", "programming", "function", "class", "variable", "debug", "error"],
            "documentation": ["documentation", "docs", "readme", "comment", "explain"],
            "testing": ["test", "pytest", "unittest", "assert", "mock"],
            "debugging": ["debug", "error", "exception", "traceback", "fix", "bug"],
            "architecture": ["design", "pattern", "architecture", "structure", "organize"],
            "optimization": ["optimize", "performance", "speed", "memory", "efficient"],
            "learning": ["learn", "understand", "explain", "tutorial", "example"],
        }

        logger.info("CodingAssistant initialized with session: %s", self.session_id)

    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Main chat interface with semantic memory integration."""
        start_time = time.time()

        # Update conversation memory
        self.memory_manager.update_conversation(self.session_id, {
            "role": "user",
            "content": message,
            "timestamp": start_time,
        })

        # Classify the query and retrieve relevant context
        query_classification = self._classify_query(message)
        relevant_memories = self.memory_manager.retrieve_memory(
            message,
            self.session_id,
            memory_types=query_classification.get("categories", []),
            limit=5,
        )

        # Build enhanced context
        enhanced_context = self._build_context(message, relevant_memories, context)

        # Cache check
        cache_key = self._generate_cache_key(message, enhanced_context)
        cached_response = self.db_manager.cache_get(cache_key)

        if cached_response:
            logger.debug("Returning cached response for query")
            return cached_response

        # Generate response using agent
        try:
            response = self.agent.chat(enhanced_context)

            # Store response in memory and cache
            self._store_interaction(message, response, query_classification, enhanced_context)

            # Cache the response (TTL: 1 hour)
            self.db_manager.cache_set(cache_key, response, ttl_seconds=3600)

            return response

        except Exception as e:
            logger.error("Error generating response: %s", e)
            error_response = f"I encountered an error while processing your request: {e}"
            return error_response

    def analyze_code_context(self, file_path: str) -> Dict[str, Any]:
        """Analyze code file and store in semantic memory."""
        try:
            # Read and analyze the file
            file_content = self.agent.tool.file_read(file_path)
            if not file_content.get("success"):
                return file_content

            analysis = self.agent.tool.code_analyze(file_path)

            # Store in memory with high importance for code context
            self.memory_manager.add_memory(
                content=f"Code analysis for {file_path}:\n{json.dumps(analysis, indent=2)}",
                session_id=self.session_id,
                memory_type="code_analysis",
                importance=0.8,
                metadata={
                    "file_path": file_path,
                    "analysis_type": "code_structure",
                    "functions": analysis.get("analysis", {}).get("functions", []),
                    "classes": analysis.get("analysis", {}).get("classes", []),
                }
            )

            return analysis

        except Exception as e:
            logger.error("Error analyzing code context: %s", e)
            return {"success": False, "error": str(e)}

    def get_project_summary(self, project_path: str) -> Dict[str, Any]:
        """Generate and cache project summary with semantic understanding."""
        try:
            # List all Python files in project
            files_result = self.agent.tool.list_files(project_path, "*.py", recursive=True)
            if not files_result.get("success"):
                return files_result

            project_files = [f for f in files_result["files"] if f["is_file"]]

            # Analyze key files
            summaries = []
            for file_info in project_files[:10]:  # Limit to avoid overwhelming
                file_path = file_info["path"]
                analysis = self.analyze_code_context(file_path)
                if analysis.get("success"):
                    summaries.append({
                        "file": file_path,
                        "analysis": analysis.get("analysis", {}),
                    })

            # Generate project overview using LLM
            overview_prompt = f"""
            Analyze this Python project structure and provide a comprehensive summary:

            Project path: {project_path}
            Files analyzed: {len(summaries)}

            File analyses:
            {json.dumps(summaries, indent=2)}

            Please provide:
            1. Project purpose and main functionality
            2. Key components and their roles
            3. Architecture patterns used
            4. Potential improvements or issues
            5. Development recommendations
            """

            project_overview = self.ollama_model.generate(overview_prompt, max_tokens=1000)

            # Store project summary in memory
            self.memory_manager.add_memory(
                content=f"Project summary for {project_path}:\n{project_overview}",
                session_id=self.session_id,
                memory_type="project_analysis",
                importance=0.9,
                metadata={
                    "project_path": project_path,
                    "files_count": len(project_files),
                    "analyzed_files": len(summaries),
                }
            )

            return {
                "success": True,
                "project_path": project_path,
                "overview": project_overview,
                "files_analyzed": len(summaries),
                "total_files": len(project_files),
                "summaries": summaries,
            }

        except Exception as e:
            logger.error("Error generating project summary: %s", e)
            return {"success": False, "error": str(e)}

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query using semantic similarity to determine context categories."""
        query_lower = query.lower()

        # Get query embedding
        query_embedding = self.ollama_model.embed(query)

        # Calculate similarity to each category
        category_scores = {}
        for category, keywords in self.context_categories.items():
            # Simple keyword matching score
            keyword_score = sum(1 for keyword in keywords if keyword in query_lower) / len(keywords)

            # Semantic similarity score (simplified)
            category_text = " ".join(keywords)
            category_embedding = self.ollama_model.embed(category_text)
            semantic_score = self._cosine_similarity(query_embedding, category_embedding)

            # Combined score
            category_scores[category] = (keyword_score * 0.3) + (semantic_score * 0.7)

        # Get top categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, score in sorted_categories if score > 0.2][:3]

        return {
            "categories": top_categories,
            "scores": category_scores,
            "primary_category": sorted_categories[0][0] if sorted_categories else "general",
        }

    def _build_context(
        self,
        query: str,
        relevant_memories: List[Any],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build enhanced context for the agent."""
        context_parts = [f"User Query: {query}"]

        # Add conversation context
        conversation = self.memory_manager.get_conversation_context(self.session_id, limit=5)
        if conversation["recent_messages"]:
            context_parts.append("Recent Conversation:")
            for msg in conversation["recent_messages"][-3:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate long messages
                context_parts.append(f"{role.title()}: {content}")

        # Add conversation summary if available
        if conversation.get("summary"):
            context_parts.append(f"Conversation Summary: {conversation['summary']}")

        # Add relevant memories
        if relevant_memories:
            context_parts.append("Relevant Context from Memory:")
            for memory in relevant_memories[:3]:  # Limit to top 3
                content = memory.content[:300]  # Truncate long content
                context_parts.append(f"- [{memory.level}] {content}")

        # Add additional context
        if additional_context:
            context_parts.append("Additional Context:")
            for key, value in additional_context.items():
                context_parts.append(f"{key}: {value}")

        return "\n\n".join(context_parts)

    def _store_interaction(
        self,
        query: str,
        response: str,
        classification: Dict[str, Any],
        context: str,
    ):
        """Store the interaction in memory with appropriate classification."""
        # Store query
        self.memory_manager.add_memory(
            content=f"Query: {query}",
            session_id=self.session_id,
            memory_type="query",
            importance=0.6,
            metadata={
                "categories": classification.get("categories", []),
                "primary_category": classification.get("primary_category", "general"),
            }
        )

        # Store response
        self.memory_manager.add_memory(
            content=f"Response: {response}",
            session_id=self.session_id,
            memory_type="response",
            importance=0.7,
            metadata={
                "query_categories": classification.get("categories", []),
                "response_length": len(response),
            }
        )

        # Update conversation memory
        self.memory_manager.update_conversation(self.session_id, {
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "categories": classification.get("categories", []),
        })

    def _generate_cache_key(self, query: str, context: str) -> str:
        """Generate cache key for query and context."""
        import hashlib
        content = f"{query}:{context[:500]}"  # Limit context for cache key
        return hashlib.md5(content.encode()).hexdigest()

    def _create_system_prompt(self) -> str:
        """Create system prompt for the coding assistant."""
        return """
You are an advanced coding assistant with access to powerful tools and semantic memory.

Your capabilities include:
- Python code execution and testing
- File operations and project analysis
- Code formatting and debugging
- Git operations and project management
- Semantic memory with context awareness

Key principles:
1. Always use your tools to perform file operations, code analysis, and testing
2. Provide clear, actionable advice with code examples
3. Consider the conversation context and previous interactions
4. Break down complex problems into manageable steps
5. Suggest best practices and improvements
6. Ask clarifying questions when needed

Your memory system understands context categories like:
- Coding: Programming tasks, debugging, implementation
- Documentation: Explaining code, writing docs
- Testing: Writing and running tests
- Architecture: Design patterns, project structure
- Optimization: Performance improvements
- Learning: Educational content and explanations

Always be helpful, accurate, and provide working solutions.
"""

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
        """Get assistant statistics."""
        memory_stats = self.memory_manager.get_memory_stats(self.session_id)
        model_info = self.ollama_model.model_info

        return {
            "session_id": self.session_id,
            "memory_stats": memory_stats,
            "model_info": model_info,
            "tools_available": len(self.tools),
            "context_categories": list(self.context_categories.keys()),
        }

    def cleanup(self):
        """Cleanup resources and consolidate memories."""
        try:
            # Consolidate short-term memories to long-term
            self.memory_manager.consolidate_memories(self.session_id)

            # Clean up old cache entries
            self.db_manager.cache_cleanup_expired()

            logger.info("Cleanup completed for session: %s", self.session_id)

        except Exception as e:
            logger.error("Error during cleanup: %s", e)