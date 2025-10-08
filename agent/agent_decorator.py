# agent_decorator.py
"""
Enhanced @agent decorator for StrandsAgents with code execution capabilities.
Eliminates boilerplate model/agent creation while providing powerful features.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from strands.models.ollama import OllamaModel
from strands import Agent

# Global agent registry for runtime discovery
AGENT_REGISTRY = {}

# Setup logging
logger = logging.getLogger("agent_decorator")

# Import sandbox executor for code execution
try:
    from .sandbox_executor import SandboxExecutor
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    logger.warning("🔧 Sandbox executor not available - code execution disabled")

def agent(model_id: str = None, tools: List = None, system_prompt: str = "",
          enable_code_execution: bool = False, sandbox_timeout: int = 30):
    """
    Enhanced @agent decorator that eliminates boilerplate and adds powerful features.

    Args:
        model_id: Ollama model to use (qwen3:8b, qwen3:4b, llama3.2)
        tools: List of tools to provide to the agent
        system_prompt: System prompt for the agent
        enable_code_execution: Enable code execution capabilities
        sandbox_timeout: Timeout for code execution in seconds

    Returns:
        Decorated function that can be called directly as an agent
    """
    def decorator(func):
        # Auto-create model and agent (eliminates boilerplate!)
        model = OllamaModel(host="http://localhost:11434", model_id=model_id)
        agent_instance = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools or []
        )

        # Register globally for discovery and chaining
        AGENT_REGISTRY[func.__name__] = {
            'agent': agent_instance,
            'model_id': model_id,
            'tools': tools,
            'system_prompt': system_prompt,
            'enable_code_execution': enable_code_execution,
            'sandbox_timeout': sandbox_timeout,
            'function': None,  # Set after wrapper definition
            'original_function': func,
            'description': func.__doc__ or "No description available"
        }

        logger.info(f"🔧 Registered agent: {func.__name__} with model {model_id}")

        # Return wrapper function that handles both regular queries and code execution
        def wrapper(query: str) -> str:
            try:
                # Check if this is a code execution request
                if enable_code_execution and any(keyword in query.lower() for keyword in
                                                ['execute', 'run code', 'execute python', 'run python']):
                    return _handle_code_execution(query, agent_instance, sandbox_timeout)

                # Regular agent query
                response = str(agent_instance(query))

                # Check if agent wants to call other agents
                if "CALL_AGENT:" in response:
                    return _handle_agent_chaining(response, query)

                return response

            except Exception as e:
                error_msg = f"Error in agent {func.__name__}: {str(e)}"
                logger.error(f"🔧 Agent {func.__name__} error: {error_msg}")
                return error_msg

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        # Store wrapper in registry for direct invocation
        AGENT_REGISTRY[func.__name__]['function'] = wrapper

        return wrapper
    return decorator

def _handle_code_execution(query: str, agent: Agent, timeout: int) -> str:
    """Handle code execution requests within agent queries"""
    try:
        # Extract code from query (look for code blocks or python/bash keywords)
        code = _extract_code_from_query(query)

        if not code:
            return "No code found to execute. Please provide code in the query."

        # Use sandbox executor for safe code execution
        if not SANDBOX_AVAILABLE:
            return "Code execution not available - sandbox module not found"

        try:
            sandbox = SandboxExecutor(timeout=timeout)

            # Determine language
            language = "python"  # Default
            if "bash" in query.lower() or "shell" in query.lower():
                language = "bash"

            # Execute code
            result = sandbox.execute_code(code, language)

            if result.error:
                return f"Code execution failed: {result.error}"

            # Format result
            output = []
            if result.stdout:
                output.append(f"Output: {result.stdout}")
            if result.stderr:
                output.append(f"Errors: {result.stderr}")

            return "\\n".join(output) if output else "Code executed successfully (no output)"

        except Exception as e:
            return f"Code execution error: {str(e)}"

    except Exception as e:
        return f"Code execution failed: {str(e)}"

def _extract_code_from_query(query: str) -> Optional[str]:
    """Extract code from natural language query"""
    # Look for code blocks (```python ... ```)
    code_block_pattern = r'```(?:python|bash|shell)?\s*(.*?)\s*```'
    code_blocks = re.findall(code_block_pattern, query, re.DOTALL)

    if code_blocks:
        return code_blocks[0].strip()

    # Look for inline code after keywords
    keywords = ['execute:', 'run:', 'code:', 'python:']
    for keyword in keywords:
        if keyword in query.lower():
            code_part = query.lower().split(keyword)[1].strip()
            if code_part:
                return code_part

    return None

def _handle_agent_chaining(response: str, original_query: str) -> str:
    """Handle agent-to-agent chaining requests"""
    try:
        # Parse agent call from response
        call_pattern = r'CALL_AGENT:\s*(\w+)\s*(?:with)?\s*(.+)?'
        match = re.search(call_pattern, response, re.IGNORECASE)

        if match:
            target_agent = match.group(1)
            call_message = match.group(2) if match.group(2) else original_query

            if target_agent in AGENT_REGISTRY:
                target_agent_func = get_agent_function(target_agent)
                if target_agent_func:
                    logger.info(f"🔗 Agent chaining: calling {target_agent}")
                    return target_agent_func(call_message)

                return f"Agent {target_agent} found but not callable"
            else:
                available = ", ".join(AGENT_REGISTRY.keys())
                return f"Agent {target_agent} not found. Available agents: {available}"

        return response

    except Exception as e:
        logger.error(f"🔗 Agent chaining error: {str(e)}")
        return response

def get_agent_function(agent_name: str):
    """Get the wrapper function for an agent"""
    if agent_name in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_name].get('function')
    return None

def list_agents() -> List[str]:
    """List all registered agents"""
    return list(AGENT_REGISTRY.keys())

def get_agent_info(agent_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific agent"""
    return AGENT_REGISTRY.get(agent_name)

def call_agent(agent_name: str, query: str) -> str:
    """Call a specific agent by name"""
    if agent_name in AGENT_REGISTRY:
        agent_func = get_agent_function(agent_name)
        if agent_func:
            return agent_func(query)
        return f"Agent {agent_name} registered but not callable"
    else:
        available = ", ".join(AGENT_REGISTRY.keys())
        return f"Agent {agent_name} not found. Available: {available}"

# Convenience functions for agent management
def get_agents_with_code_execution() -> List[str]:
    """Get list of agents that have code execution enabled"""
    return [name for name, info in AGENT_REGISTRY.items()
            if info.get('enable_code_execution', False)]

def get_agents_by_model(model_id: str) -> List[str]:
    """Get list of agents using a specific model"""
    return [name for name, info in AGENT_REGISTRY.items()
            if info.get('model_id') == model_id]

def print_agent_summary():
    """Print a summary of all registered agents"""
    print("\\n🤖 Agent Registry Summary")
    print("=" * 50)

    for name, info in AGENT_REGISTRY.items():
        print(f"🔧 {name}:")
        print(f"   Model: {info['model_id']}")
        print(f"   Tools: {info['tools']}")
        print(f"   Code Execution: {info.get('enable_code_execution', False)}")
        print(f"   Description: {info['description'][:100]}...")
        print()

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the decorator
    @agent(
        model_id="qwen3:8b",
        tools=["http_request", "calculator"],
        system_prompt="You are a helpful assistant that can write and execute code.",
        enable_code_execution=True
    )
    def example_agent(query: str) -> str:
        """Example agent with code execution"""
        return f"Processed: {query}"

    # Test the agent
    result = example_agent("Hello, world!")
    print(f"Result: {result}")

    # Test code execution
    code_result = example_agent("Execute this Python code: print('Hello from code execution!')")
    print(f"Code execution result: {code_result}")

    # Show registry
    print_agent_summary()
