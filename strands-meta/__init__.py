"""
StrandsAgents @agent Package

A powerful decorator-based agent system with code execution capabilities.
Designed for contribution to the official StrandsAgents project.

This package provides:
- @agent decorator for simplified agent creation
- SandboxExecutor for safe code execution
- Agent registry for discovery and chaining
- Workflow template system
- Integration with StrandsAgents ecosystem

Example:
    from agent import agent, SandboxExecutor

    @agent(
        model_id="qwen3:8b",
        enable_code_execution=True,
        system_prompt="You are a helpful coding assistant"
    )
    def coding_agent(query: str) -> str:
        return f"Processing: {query}"

    # Use the agent
    result = coding_agent("Help me write a Python function")
"""

__version__ = "1.0.0"
__author__ = "StrandsAgents Community"

from .agent_decorator import (
    agent,
    list_agents,
    get_agent_info,
    call_agent,
    get_agents_with_code_execution,
    get_agents_by_model,
    print_agent_summary,
    AGENT_REGISTRY
)

from .sandbox_executor import SandboxExecutor, ExecutionResult
from .sandbox_tool import sandbox_test_code
from .scaffolder import (
    AgentConfig,
    ScaffoldingResult,
    QUESTIONNAIRE,
    materialise_agent_module,
    render_agent_module,
    validate_answers,
    generate_agent,
)

from .workflow_templates import (
    WorkflowTemplate,
    WorkflowStep,
    WorkflowTemplateManager,
    get_workflow_template,
    list_workflow_templates,
    execute_workflow,
    template_manager
)

from .model_selector import (
    ModelSelector,
    ModelInfo,
    get_best_model_for_task,
    list_available_models,
    get_model_recommendations,
    model_selector
)

from .meta_agent_demo import (
    meta_agent_builder,
    create_agent_from_description
)

__all__ = [
    'agent',
    'SandboxExecutor',
    'ExecutionResult',
    'sandbox_test_code',
    'AgentConfig',
    'ScaffoldingResult',
    'QUESTIONNAIRE',
    'materialise_agent_module',
    'render_agent_module',
    'validate_answers',
    'generate_agent',
    'list_agents',
    'get_agent_info',
    'call_agent',
    'get_agents_with_code_execution',
    'get_agents_by_model',
    'print_agent_summary',
    'AGENT_REGISTRY',
    'WorkflowTemplate',
    'WorkflowStep',
    'WorkflowTemplateManager',
    'get_workflow_template',
    'list_workflow_templates',
    'execute_workflow',
    'template_manager',
    'ModelSelector',
    'ModelInfo',
    'get_best_model_for_task',
    'list_available_models',
    'get_model_recommendations',
    'model_selector'
]
