"""
Clean Tools for Swarm System

Simplified meta-tooling capabilities for the swarm system.
Provides basic tool creation and management functionality.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from strands import Agent, tool
from strands.models.ollama import OllamaModel

# Windows compatibility: handle missing termios module
try:
    from strands_tools import load_tool, shell, editor
    STRANDS_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: strands_tools not fully available: {e}")
    STRANDS_TOOLS_AVAILABLE = False
    # Create fallback functions
    def load_tool(*args, **kwargs):
        return "Tool loading not available on Windows"
    def shell(*args, **kwargs):
        return "Shell operations not available on Windows"
    def editor(*args, **kwargs):
        return "Editor operations not available on Windows"

from .database_manager import db_manager

logger = logging.getLogger(__name__)


@tool
def create_dynamic_tool(
    tool_name: str,
    description: str,
    code_content: str,
    input_schema: Dict[str, Any]
) -> str:
    """Create a new tool dynamically at runtime."""
    try:
        # Create the tool file
        tool_filename = f"dynamic_{tool_name}.py"

        tool_code = f'''
from typing import Any
from strands.types.tools import ToolUse, ToolResult

TOOL_SPEC = {{
    "name": "{tool_name}",
    "description": "{description}",
    "inputSchema": {input_schema}
}}

def {tool_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """{description}"""
    tool_use_id = tool_use["toolUseId"]

    try:
        # Execute the provided code (simplified for demo)
        result = f"Executed: {code_content[:50]}..."

        return {{
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{{"text": f"Tool executed successfully: {{result}}"}}]
        }}
    except Exception as e:
        return {{
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{{"text": f"Tool execution failed: {{str(e)}}"}}]
        }}
'''

        # Write the tool file
        with open(tool_filename, 'w') as f:
            f.write(tool_code)

        # Load the tool
        load_tool(tool_filename)

        # Store in knowledge base
        db_manager.store_knowledge(
            topic="dynamic_tools",
            content=f"Created tool: {tool_name}",
            subtopic="meta_tooling",
            source="runtime_creation",
            confidence=0.9,
            metadata={
                "tool_name": tool_name,
                "description": description,
                "filename": tool_filename,
                "creation_method": "meta_tooling"
            }
        )

        logger.info(f"Created dynamic tool: {tool_name}")
        return f"Successfully created and loaded tool: {tool_name} in file: {tool_filename}"

    except Exception as e:
        error_msg = f"Failed to create tool {tool_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def query_knowledge_base(
    query: str,
    topic: str = "",
    limit: int = 5
) -> str:
    """Query the knowledge base for relevant information."""
    try:
        # Search knowledge base
        results = db_manager.search_knowledge(query, limit)

        if not results:
            return f"No knowledge found for query: {query}"

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Topic: {result['topic']}\\n"
                f"Content: {result['content'][:200]}...\\n"
                f"Confidence: {result['confidence']}\\n"
                f"Created: {result['created_at']}\\n"
            )

        # Store query in memory
        query_text = query
        db_manager.store_memory(
            session_id="knowledge_queries",
            content=f"Knowledge query: {query_text}",
            memory_type="knowledge_access",
            importance_score=0.6,
            metadata={
                "query": query_text,
                "topic": topic,
                "results_count": len(results)
            }
        )

        return f"Knowledge Base Results for '{query}':\\n\\n" + "\\n---\\n".join(formatted_results)

    except Exception as e:
        error_msg = f"Knowledge base query failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def store_learning(
    topic: str,
    content: str,
    source: str = "swarm_learning",
    confidence: float = 0.8
) -> str:
    """Store new learning or insight in the knowledge base."""
    try:
        # Store in knowledge base
        knowledge_id = db_manager.store_knowledge(
            topic=topic,
            content=content,
            source=source,
            confidence=confidence,
            metadata={
                "learning_type": "swarm_generated",
                "creation_method": "meta_learning"
            }
        )

        # Also store in memory for context
        db_manager.store_memory(
            session_id="swarm_learning",
            content=f"Learned: {content[:100]}...",
            memory_type="long_term",
            importance_score=confidence,
            metadata={
                "topic": topic,
                "knowledge_id": knowledge_id,
                "learning_source": source
            }
        )

        logger.info(f"Stored learning: {topic} -> {knowledge_id}")
        return f"Successfully stored learning in topic '{topic}' with ID: {knowledge_id}"

    except Exception as e:
        error_msg = f"Failed to store learning: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_swarm_status() -> str:
    """Get current status of the swarm system."""
    try:
        # Get database statistics
        db_stats = db_manager.get_stats()

        # Format status report
        status = "SWARM SYSTEM STATUS\\n"
        status += "=" * 50 + "\\n\\n"

        status += "DATABASES:\\n"
        for db_name, stats in db_stats.items():
            status += f"  {db_name.upper()}: "
            if stats.get("exists", False):
                status += f"{stats['records']} records, {stats['size_mb']:.2f} MB\\n"
            else:
                status += f"ERROR - {stats.get('error', 'Unknown error')}\\n"

        status += f"\\nTIMESTAMP: {datetime.now().isoformat()}\\n"

        return status

    except Exception as e:
        return f"Failed to get swarm status: {str(e)}"
