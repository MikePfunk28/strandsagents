"""
Tools for Swarm System

Meta-tooling capabilities and agent-as-tool functionality for the swarm system.
Provides dynamic tool creation, agent communication, and workflow management.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import load_tool, shell, editor

from .database_manager import db_manager
from .prompts import get_assistant_prompt, get_lightweight_prompt, get_orchestrator_prompt

logger = logging.getLogger(__name__)


# Meta-tooling: Create tools dynamically
@tool
def create_dynamic_tool(
    tool_name: str,
    description: str,
    code_content: str,
    input_schema: Dict[str, Any]
) -> str:
    """
    Create a new tool dynamically at runtime.

    Args:
        tool_name: Name of the tool to create
        description: Description of what the tool does
        code_content: Python code for the tool implementation
        input_schema: JSON schema for tool inputs

    Returns:
        Success message with tool creation details
    """
    try:
        # Create the tool file
        tool_filename = f"dynamic_{tool_name}.py"

        tool_code = f'''"""
Dynamic Tool: {tool_name}

Generated at runtime by meta-tooling system.
"""

from typing import Any
from strands.types.tools import ToolUse, ToolResult

TOOL_SPEC = {{
    "name": "{tool_name}",
    "description": "{description}",
    "inputSchema": {input_schema}
}}

def {tool_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """
    {description}

    Args:
        tool_use: Tool use parameters
        **kwargs: Additional arguments

    Returns:
        Tool result
    """
    tool_use_id = tool_use["toolUseId"]

    try:
        # Execute the provided code
        {code_content}

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
def create_assistant_as_tool(
    assistant_name: str,
    assistant_type: str,
    model_id: str = "llama3.2",
    custom_prompt: str = ""
) -> str:
    """
    Create an assistant as a tool that can be used by other agents.

    Args:
        assistant_name: Name for the assistant tool
        assistant_type: Type of assistant (research, creative, critical, etc.)
        model_id: Model to use for the assistant
        custom_prompt: Custom prompt for the assistant

    Returns:
        Success message with tool creation details
    """
    try:
        # Get the appropriate prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = get_assistant_prompt(assistant_type)

        # Create the assistant tool code
        tool_code = f'''"""
Assistant as Tool: {assistant_name}

Provides {assistant_type} capabilities as a reusable tool.
"""

from typing import Any
from strands.types.tools import ToolUse, ToolResult
from strands.models.ollama import OllamaModel
from strands import Agent

# Global assistant instance (created once, reused)
_assistant_instance = None

def get_assistant():
    global _assistant_instance
    if _assistant_instance is None:
        model = OllamaModel(host="http://localhost:11434", model_id="{model_id}")
        _assistant_instance = Agent(
            model=model,
            system_prompt="""{system_prompt}"""
        )
    return _assistant_instance

TOOL_SPEC = {{
    "name": "{assistant_name}",
    "description": "{assistant_type.title()} assistant capabilities as a tool",
    "inputSchema": {{
        "json": {{
            "type": "object",
            "properties": {{
                "query": {{
                    "type": "string",
                    "description": "The query or task for the assistant"
                }},
                "context": {{
                    "type": "string",
                    "description": "Additional context for the assistant"
                }}
            }},
            "required": ["query"]
        }}
    }}
}}

def {assistant_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Execute {assistant_type} assistant capabilities.

    Args:
        tool_use: Tool use parameters
        **kwargs: Additional arguments

    Returns:
        Assistant's response
    """
    tool_use_id = tool_use["toolUseId"]
    query = tool_use["input"]["query"]
    context = tool_use["input"].get("context", "")

    try:
        assistant = get_assistant()
        full_query = f"{{context}}\\n\\n{query}" if context else query
        response = assistant(full_query)

        return {{
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{{"text": str(response)}}]
        }}
    except Exception as e:
        return {{
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{{"text": f"Assistant execution failed: {{str(e)}}"}}]
        }}
'''

        # Write the assistant tool file
        tool_filename = f"assistant_{assistant_name}.py"
        with open(tool_filename, 'w') as f:
            f.write(tool_code)

        # Load the tool
        load_tool(tool_filename)

        # Store in knowledge base
        db_manager.store_knowledge(
            topic="assistant_tools",
            content=f"Created assistant tool: {assistant_name}",
            subtopic=assistant_type,
            source="meta_tooling",
            confidence=0.9,
            metadata={
                "assistant_name": assistant_name,
                "assistant_type": assistant_type,
                "model_id": model_id,
                "filename": tool_filename,
                "creation_method": "assistant_as_tool"
            }
        )

        logger.info(f"Created assistant as tool: {assistant_name}")
        return f"Successfully created assistant tool: {assistant_name} in file: {tool_filename}"

    except Exception as e:
        error_msg = f"Failed to create assistant tool {assistant_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def swarm_communicate(
    message: str,
    target_assistant: str,
    context: str = ""
) -> str:
    """
    Enable communication between swarm assistants.

    Args:
        message: Message to send to the target assistant
        target_assistant: Name of the target assistant
        context: Additional context for the communication

    Returns:
        Response from the target assistant
    """
    try:
        # Get the target assistant from registry
        target = global_registry.get_instance(target_assistant)

        # Prepare the communication
        full_message = f"Communication from swarm: {context}\\n\\n{message}" if context else message

        # Execute the target assistant
        response = target.execute(full_message)

        # Store communication in memory
        db_manager.store_memory(
            session_id="swarm_communication",
            content=f"Communication: {target_assistant} -> {message[:100]}...",
            memory_type="communication",
            importance_score=0.7,
            metadata={
                "source_assistant": "swarm_coordinator",
                "target_assistant": target_assistant,
                "message_length": len(message),
                "response_length": len(str(response))
            }
        )

        logger.info(f"Swarm communication: {target_assistant} -> {len(str(response))} chars")
        return str(response)

    except Exception as e:
        error_msg = f"Swarm communication failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def create_lightweight_agent(
    agent_name: str,
    role: str,
    model_id: str = "gemma3:27b"
) -> str:
    """
    Create a lightweight agent for the swarm using 270m models.

    Args:
        agent_name: Name for the agent
        role: Role type (fact_checker, idea_generator, etc.)
        model_id: Model to use (default: gemma3:27b)

    Returns:
        Success message with agent creation details
    """
    try:
        # Get the appropriate prompt for the role
        system_prompt = get_lightweight_prompt(role)

        # Create the lightweight agent code
        agent_code = f'''"""
Lightweight Agent: {agent_name}

Role: {role}
Generated for swarm operations.
"""

from typing import Any
from strands.types.tools import ToolUse, ToolResult
from strands.models.ollama import OllamaModel
from strands import Agent

# Global agent instance
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        model = OllamaModel(host="http://localhost:11434", model_id="{model_id}")
        _agent_instance = Agent(
            model=model,
            system_prompt="""{system_prompt}"""
        )
    return _agent_instance

TOOL_SPEC = {{
    "name": "{agent_name}",
    "description": "{role.replace('_', ' ').title()} agent for swarm operations",
    "inputSchema": {{
        "json": {{
            "type": "object",
            "properties": {{
                "task": {{
                    "type": "string",
                    "description": "The task for this agent"
                }},
                "context": {{
                    "type": "string",
                    "description": "Additional context"
                }}
            }},
            "required": ["task"]
        }}
    }}
}}

def {agent_name}(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Execute {role} agent task.

    Args:
        tool_use: Tool use parameters
        **kwargs: Additional arguments

    Returns:
        Agent's response
    """
    tool_use_id = tool_use["toolUseId"]
    task = tool_use["input"]["task"]
    context = tool_use["input"].get("context", "")

    try:
        agent = get_agent()
        full_task = f"{{context}}\\n\\n{task}" if context else task
        response = agent(full_task)

        return {{
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{{"text": str(response)}}]
        }}
    except Exception as e:
        return {{
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{{"text": f"Agent execution failed: {{str(e)}}"}}]
        }}
'''

        # Write the agent file
        agent_filename = f"lightweight_{agent_name}.py"
        with open(agent_filename, 'w') as f:
            f.write(agent_code)

        # Load as tool
        load_tool(agent_filename)

        # Store in knowledge base
        db_manager.store_knowledge(
            topic="lightweight_agents",
            content=f"Created lightweight agent: {agent_name}",
            subtopic=role,
            source="swarm_creation",
            confidence=0.9,
            metadata={
                "agent_name": agent_name,
                "role": role,
                "model_id": model_id,
                "filename": agent_filename,
                "agent_type": "lightweight_swarm"
            }
        )

        logger.info(f"Created lightweight agent: {agent_name}")
        return f"Successfully created lightweight agent: {agent_name} in file: {agent_filename}"

    except Exception as e:
        error_msg = f"Failed to create lightweight agent {agent_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def query_knowledge_base(
    query: str,
    topic: str = "",
    limit: int = 5
) -> str:
    """
    Query the knowledge base for relevant information.

    Args:
        query: Search query
        topic: Specific topic to search within
        limit: Maximum number of results to return

    Returns:
        Relevant knowledge base entries
    """
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
        db_manager.store_memory(
            session_id="knowledge_queries",
            content=f"Knowledge query: {query}",
            memory_type="knowledge_access",
            importance_score=0.6,
            metadata={
                "query": query,
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
    """
    Store new learning or insight in the knowledge base.

    Args:
        topic: Topic/category for the learning
        content: The learning content
        source: Source of the learning
        confidence: Confidence level in the information

    Returns:
        Success message
    """
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
    """
    Get current status of the swarm system.

    Returns:
        Status information about databases, assistants, and system health
    """
    try:
        # Get database statistics
        db_stats = db_manager.get_stats()

        # Get assistant registry info
        available_types = global_registry.list_available_types()
        instance_count = len(global_registry.list_instances())

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

        status += f"\\nASSISTANTS:\\n"
        status += f"  Available Types: {len(available_types)}\\n"
        status += f"  Active Instances: {instance_count}\\n"

        if available_types:
            status += "  Types: " + ", ".join(available_types) + "\\n"

        status += f"\\nTIMESTAMP: {datetime.now().isoformat()}\\n"

        return status

    except Exception as e:
        return f"Failed to get swarm status: {str(e)}"


@tool
def optimize_swarm_performance(
    operation: str,
    target: str = ""
) -> str:
    """
    Optimize swarm performance based on learning and analytics.

    Args:
        operation: Type of optimization (cleanup, analyze, optimize)
        target: Specific target for optimization

    Returns:
        Optimization results
    """
    try:
        if operation == "cleanup":
            # Clean up expired cache and vacuum databases
            expired_count = db_manager.cleanup_expired_cache()
            db_manager.vacuum_all()

            return f"Cleanup completed: {expired_count} expired cache entries removed, databases vacuumed"

        elif operation == "analyze":
            # Analyze system performance
            db_stats = db_manager.get_stats()
            analysis = "Performance Analysis:\\n"

            for db_name, stats in db_stats.items():
                if stats.get("exists", False):
                    analysis += f"{db_name}: {stats['records']} records, {stats['size_mb']:.2f} MB\\n"

            return analysis

        elif operation == "optimize":
            # Perform optimization tasks
            optimizations = []

            # Clean up expired data
            expired_count = db_manager.cleanup_expired_cache()
            if expired_count > 0:
                optimizations.append(f"Cleaned {expired_count} expired cache entries")

            # Vacuum databases
            db_manager.vacuum_all()
            optimizations.append("Vacuumed all databases")

            return "Optimizations completed:\\n" + "\\n".join(f"- {opt}" for opt in optimizations)

        else:
            return f"Unknown optimization operation: {operation}"

    except Exception as e:
        return f"Optimization failed: {str(e)}"


# Import the global registry (will be set up when assistants are loaded)
try:
    from ..assistants.registry import global_registry
except ImportError:
    # Fallback if not available
    global_registry = None
    logger.warning("Global registry not available - some tools may not function properly")
