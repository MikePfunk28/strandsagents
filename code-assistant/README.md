# Adversarial Coding System

A GAN-style coding assistant that uses multiple AI agents to iteratively improve code quality through adversarial feedback.

## 🎯 What We Built

**Your Vision Realized**: A "Claude Code + WARP" equivalent that goes beyond Python-only coding with:

- **Multi-language support**: Python, JavaScript, TypeScript, Rust, Go, Java, C++
- **GAN-like adversarial architecture**: Multiple specialized agents improve code iteratively
- **Model selection**: Choose from Llama 3.2, Gemma, and other local models
- **Agent2agent communication**: Parallel processing with StrandsAgents
- **MCP integration**: External system coordination
- **Zero cost**: Completely local with Ollama

## 🏗️ Architecture

### Core Agents
- **Generator**: Creates initial code solutions
- **Discriminator**: Finds flaws and suggests improvements (GAN-style)
- **Optimizer**: Performance and efficiency improvements
- **Security**: Vulnerability analysis and safety checks
- **Tester**: Comprehensive test case generation
- **Reviewer**: Overall code quality assessment

### Model Configurations
- **Speed**: Fast models for quick iteration (270M discriminator)
- **Balanced**: Mix of quality and speed (default)
- **Quality**: Larger models for best results
- **Custom**: Select individual models for each agent

## 🚀 Quick Setup (Windows PowerShell)

### 1. Prerequisites
```powershell
# Install Ollama first
winget install Ollama.Ollama
# OR download from https://ollama.ai

# Start Ollama service
ollama serve
```

### 2. Install Models
```powershell
# Required models (pick based on your strategy)
ollama pull llama3.2:3b   # Main generator
ollama pull llama3.2:1b   # Fast processing
ollama pull gemma2:2b     # Balanced performance
ollama pull gemma:270m    # Ultra-fast discriminator

# Optional quality models
ollama pull qwen2.5:4b    # High quality generation
```

### 3. Setup Environment
```powershell
# Navigate to code-assistant directory
cd M:\strandsagents\code-assistant

# Run the Windows setup script
.\setup_windows.ps1

# OR manual setup:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage
```python
import asyncio
from adversarial_agents import AdversarialCodingCoordinator, CodeGenerationRequest, LanguageType

async def generate_code():
    # Initialize coordinator
    coordinator = AdversarialCodingCoordinator()

    # Setup agents with your preferred strategy
    await coordinator.initialize_agents(strategy="balanced")

    # Create request
    request = CodeGenerationRequest(
        requirements="Create a secure user authentication system",
        language=LanguageType.PYTHON,
        constraints=["Include error handling", "Add comprehensive tests"]
    )

    # Generate code through adversarial process
    result = await coordinator.generate_code_adversarially(request)

    print(f"Generated code (Score: {result['final_score']:.1f}/10):")
    print(result['final_code'])

# Run
asyncio.run(generate_code())
```

### Interactive Mode
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run interactive interface
python main.py
```

### Model Selection
```python
from adversarial_agents import ModelConfiguration

# See available models
config = ModelConfiguration()
print(config.available_models)

# Use custom models
coordinator = AdversarialCodingCoordinator(
    model_name="llama3.2:3b",  # Your choice
    host="localhost:11434"     # Ollama host
)
```

## Architecture

### Core Components

1. **CodingAgent**: Main orchestrator with workflow management
2. **CodingAssistant**: Core assistant with memory and tools
3. **DatabaseManager**: SQLite database management (cache.db, knowledge.db, memory.db)
4. **EnhancedMemoryManager**: Hierarchical memory with semantic search
5. **OllamaModel**: Local AI model interface

### Memory System

The memory system uses three levels:

- **Short-term Memory**: Recent context in RAM cache
- **Long-term Memory**: Important information in knowledge database
- **Conversation Memory**: Session-specific conversation history with summaries

### Semantic Classification

Content is automatically classified into categories:
- **coding**: Programming tasks, debugging, implementation
- **documentation**: Explanations, guides, documentation
- **testing**: Test writing and execution
- **architecture**: Design patterns, project structure
- **optimization**: Performance improvements
- **learning**: Educational content and explanations

## Tools Available

### Code Tools
- `python_repl`: Execute Python code
- `code_analyze`: Analyze code structure and complexity
- `code_format`: Format code using Black
- `code_test`: Run tests with pytest/unittest

### File Tools
- `file_read`, `file_write`, `file_append`: File operations
- `list_files`: Directory listing with patterns
- `search_code`: Search for patterns in code files

### Development Tools
- `git_status`: Git repository status
- `shell_execute`: Execute shell commands
- `editor`: Advanced code editing operations

### Embedding & Chunking Tools
- `embed_document_hierarchy`: Hierarchical document embeddings
- `chunk_hierarchy_tool`: Intelligent text chunking

## Configuration

### Environment Variables

```bash
# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_EMBED_MODEL="embeddinggemma"
export OLLAMA_EMBED_TIMEOUT="60"

# Assistant configuration
export ASSISTANT_DB_DIR="./assistant_data"
export ASSISTANT_SESSION_ID="my-session"
```

### Database Files

The assistant creates three SQLite databases:

- **cache.db**: Temporary cache for query results
- **knowledge.db**: Long-term knowledge storage with embeddings
- **memory.db**: Session memory and conversation history

## Workflow Types

### Analysis Workflow
```python
result = await agent.execute_task(
    "Analyze the code structure of this project",
    task_type="analysis",
    project_path="/path/to/project"
)
```

### Feature Implementation
```python
result = await agent.execute_task(
    "Implement a user authentication system",
    task_type="feature",
    file_path="auth.py"
)
```

### Debugging
```python
result = await agent.execute_task(
    "Debug the error in line 42 of main.py",
    task_type="debug",
    file_path="main.py"
)
```

### Testing
```python
result = await agent.execute_task(
    "Write comprehensive tests for the calculator module",
    task_type="testing",
    file_path="calculator.py"
)
```

## Advanced Usage

### Custom Memory Management

```python
from coding_assistant import CodingAssistant

assistant = CodingAssistant()

# Add important information to memory
memory_id = assistant.memory_manager.add_memory(
    content="This project uses FastAPI for the web framework",
    session_id=assistant.session_id,
    memory_type="project_info",
    importance=0.9
)

# Retrieve relevant memories
memories = assistant.memory_manager.retrieve_memory(
    query="What web framework does this project use?",
    session_id=assistant.session_id,
    limit=5
)
```

### Code Context Analysis

```python
# Analyze a specific file
analysis = assistant.analyze_code_context("./src/main.py")

# Get project overview
summary = assistant.get_project_summary("./src")
```

### Streaming Task Execution

```python
async for event in agent.stream_task(
    "Create a REST API with authentication",
    task_type="feature"
):
    print(f"Event: {event['type']}")
    if event['type'] == 'complete':
        print(f"Result: {event['result']}")
```

## Memory and Performance

### Memory Consolidation

The assistant automatically consolidates important short-term memories into long-term storage based on:
- Importance scores
- Access frequency
- Semantic relevance

### Cache Management

- Query results are cached for 1 hour by default
- Expired cache entries are automatically cleaned up
- Cache keys are generated based on query and context

### Database Optimization

- Embeddings use cosine similarity for fast search
- Indexes on frequently queried fields
- Configurable cleanup for old entries

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check host configuration
   - Verify models are installed

2. **Memory Issues**
   - Large embeddings consume memory
   - Configure shorter conversation limits
   - Enable periodic cleanup

3. **Performance Issues**
   - Use smaller chunk sizes for faster processing
   - Limit search results
   - Enable caching

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = create_coding_agent()
```

## Integration with Existing Assistants

This coding assistant integrates with your existing:
- `embedding_assistant.py`: Uses existing Ollama embedding functionality
- `chunking_assistant.py`: Uses existing hierarchical chunking
- `meta_tool_assistant.py`: Compatible with meta-tooling workflows

The integration preserves all existing functionality while adding coding-specific capabilities.

## Contributing

The coding assistant is designed to be extensible:
- Add new tools by creating functions with the `@tool` decorator
- Extend workflow types in the orchestrator
- Add new memory classification categories
- Implement custom callback handlers

## License

This project follows the same license as the StrandsAgents framework.