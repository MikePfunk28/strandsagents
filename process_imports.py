#!/usr/bin/env python3
import re

def filter_imports(imports_file):
    """Filter out relative imports and local modules."""
    filtered_imports = []
    local_modules = {
        # Core project modules
        'strands', 'strands_tools', 'mem0', 'ollama',
        # Local assistant modules
        'memory_assistant', 'file_assistant', 'embedding_assistant',
        'chunking_assistant', 'thought_agent', 'meta_tool_assistant',
        'browser_assistant', 'research_assistant', 'financial_assistant',
        'reasoning_agent', 'memory_manager', 'coding_assistant',
        'language_engine', 'database_manager', 'ollama_model',
        'simple_coding_agent', 'coding_tools', 'enhanced_memory_manager',
        'adversarial_agents', 'adversarial_coding_system',
        'agent2agent_coordinator', 'computer_science_assistant',
        'math_assistant', 'mcp_integration', 'model_config',
        # Graph modules
        'graph', 'graph.advanced_analytics', 'graph.embedding_integration',
        'graph.enhanced_memory_graph', 'graph.feedback_workflow',
        'graph.graph_storage', 'graph.programming_graph', 'graph.workflow_engine',
        # Security modules
        'security', 'security.agent_authenticator', 'security.answer_validator',
        'security.message_verifier', 'security.secure_reporter', 'security.security_manager',
        # Swarm modules
        'swarm', 'swarm.agents.base_assistant', 'swarm.agents.creative_assistant.service',
        'swarm.agents.critical_assistant.service', 'swarm.agents.research_assistant.service',
        'swarm.agents.summarizer_assistant.service', 'swarm.communication.mcp_client',
        'swarm.communication.mcp_server', 'swarm.coordinator.orchestrator',
        'swarm.orchestration.feedback_graph', 'swarm.orchestrator',
        'swarm.tools.code_feedback_tool', 'swarm_system.assistants.base_assistant',
        'swarm_system.learning', 'swarm_system.learning.adaptive_benchmark',
        'swarm_system.learning.adaptive_challenge_manager', 'swarm_system.learning.agitator_assistant',
        'swarm_system.learning.benchmark', 'swarm_system.learning.code_feedback_loop',
        'swarm_system.learning.discriminator_assistant', 'swarm_system.learning.generator_assistant',
        'swarm_system.learning.schemas', 'swarm_system.utils.database_manager',
        # Web scraping modules
        'scraper.items', 'scraper.ollama_rerank',
        # AI agents modules
        'ai_agents', 'ai_agents.swarm.orchestrator', 'ai_agents.swarm.agents.base_assistant',
        'ai_agents.swarm.agents.creative_assistant.service', 'ai_agents.swarm.agents.creative_assistant.tools',
        'ai_agents.swarm.agents.creative_assistant', 'ai_agents.swarm.agents.critical_assistant.service',
        'ai_agents.swarm.agents.critical_assistant.tools', 'ai_agents.swarm.agents.critical_assistant',
        'ai_agents.swarm.agents.research_assistant.service', 'ai_agents.swarm.agents.research_assistant.tools',
        'ai_agents.swarm.agents.summarizer_assistant.service', 'ai_agents.swarm.agents.summarizer_assistant.tools',
        'ai_agents.swarm.agents.summarizer_assistant', 'ai_agents.swarm.communication.mcp_client',
        'ai_agents.swarm.communication.mcp_server', 'ai_agents.swarm.meta_tooling.tool_builder',
        # Example code assistant
        'Example_code_assistant', 'Example_code_assistant.code_assistant',
        'Example_code_assistant.main', 'Example_code_assistant.models',
        'Example_code_assistant.utils.tools', 'Example_code_assistant.utils.prompt',
        # Utility modules
        'utils.prompt', 'utils.tools', 'logging_config',
        # Communication modules
        'swarm.communication.feedback_channel', 'swarm.communication.feedback_service',
        'swarm.communication.inmemory_mcp',
        # Storage modules
        'swarm.storage.database_manager',
        # Meta tooling
        'swarm.meta_tooling.tool_builder',
        # Agents
        'swarm.agents.code_feedback.service', 'swarm.agents.code_feedback.tools',
        'swarm.agents.code_feedback',
        # Main modules
        'swarm.main', 'swarm.__init__',
        # Test modules (shouldn't be in requirements)
        'test', 'tests',
        # Websearch modules
        'websearch',
        # Coordination modules
        'coordination',
        # Modelfile
        'modelfile',
        # Memory store
        'memory', 'memory_store',
        # Knowledge context
        'knowledge_context',
        # Workflow runs
        'workflow_runs',
        # Documentation
        'docs',
        # Other local modules
        'strnads.models.ollama',  # typo in import
    }

    with open(imports_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip relative imports (starting with dots)
            if line.startswith('.'):
                continue

            # Skip local modules
            if line in local_modules:
                continue

            # Skip if it's a sub-module of a local module
            is_local = False
            for local_module in local_modules:
                if line.startswith(local_module + '.'):
                    is_local = True
                    break
            if is_local:
                continue

            # Skip standard library modules that don't need to be in requirements.txt
            stdlib_modules = {
                '__future__', 'abc', 'argparse', 'ast', 'asyncio', 'base64',
                'collections', 'contextvars', 'dataclasses', 'datetime', 'difflib',
                'enum', 'hashlib', 'hmac', 'importlib.util', 'inspect', 'json',
                'logging', 'logging.config', 'math', 'os', 'pathlib', 'random', 're',
                'secrets', 'shutil', 'signal', 'sqlite3', 'statistics', 'subprocess',
                'sys', 'tempfile', 'time', 'types', 'typing', 'unittest', 'unittest.mock',
                'urllib.parse', 'uuid'
            }

            # Extract the root module name (before any dots)
            root_module = line.split('.')[0]

            if root_module not in stdlib_modules:
                filtered_imports.append(line)

    return filtered_imports

def main():
    imports_file = 'all_imports.txt'
    filtered_imports = filter_imports(imports_file)

    print(f"Original imports: 161")
    print(f"Filtered imports: {len(filtered_imports)}")

    # Write filtered imports to file
    with open('filtered_imports.txt', 'w') as f:
        for imp in sorted(filtered_imports):
            f.write(f"{imp}\n")

    print("Filtered imports written to filtered_imports.txt")
    print("\nFiltered imports:")
    for imp in sorted(filtered_imports):
        print(f"  {imp}")

if __name__ == "__main__":
    main()
