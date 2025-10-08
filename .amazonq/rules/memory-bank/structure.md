# Project Structure & Architecture

## Root Directory Organization

### Core System Components
- **`graph/`** - Graph-based memory and analytics system (4-phase implementation)
- **`swarm/`** - Multi-agent coordination and orchestration
- **`agent/`** - Agent building, scaffolding, and workflow templates
- **`assistants/`** - Specialized AI assistants for various domains
- **`code-assistant/`** - Programming and development-focused agents

### Framework Integration
- **`strands-meta/`** - Meta-orchestration and configuration management
- **`ai_agents/`** - Alternative agent implementation with swarm coordination
- **`swarm_system/`** - Lightweight swarm system with learning capabilities

### Specialized Modules
- **`websearch/`** - Web scraping and search capabilities using Scrapy
- **`security/`** - Authentication, validation, and secure communication
- **`context7/`** - Context lookup and search integration
- **`coordination/`** - Task coordination and orchestration utilities

### Data & Memory
- **`memory/`** - Session and agent memory storage
- **`memory_store/`** - Persistent memory with thought agent integration
- **`knowledge_context/`** - Knowledge summaries and context management
- **`workflow_runs/`** - Workflow execution logs and results

### Configuration & Rules
- **`.amazonq/rules/`** - Amazon Q IDE integration rules and memory bank
- **`.claude/`** - Claude AI integration with agents, commands, and helpers
- **`.hive-mind/`** - Hive mind configuration and session management
- **`.clinerules/`** - Cline AI integration rules and context

## Core Architecture Patterns

### Graph System Architecture
```
Storage Layer (Parquet/JSON) → Embedding Integration → Enhanced Memory Graph → Advanced Analytics
```

### Swarm Coordination Flow
```
Agent Registration → Task Assignment → Memory Storage → Performance Analytics → Optimization
```

### Multi-Agent Communication
```
MCP Protocol → Agent-to-Agent Messaging → Feedback Channels → Coordination
```

## Key Architectural Components

### Graph System (graph/)
- **`graph_storage.py`** - Multiple storage backends with vector search
- **`embedding_integration.py`** - Auto-embedding generation and management
- **`enhanced_memory_graph.py`** - Swarm integration with MCP communication
- **`advanced_analytics.py`** - Real-time monitoring and pattern detection
- **`improved_cleanup_assistant.py`** - Safe file management with risk assessment

### Swarm System (swarm/)
- **`main.py`** - Main swarm orchestration entry point
- **`agents/base_assistant.py`** - Base class for all swarm agents
- **`communication/mcp_client.py`** - MCP client for agent messaging
- **`communication/mcp_server.py`** - MCP server for coordination
- **`orchestration/feedback_graph.py`** - Feedback loop management

### Agent Framework (agent/)
- **`agent_builder.py`** - Dynamic agent creation and configuration
- **`scaffolder.py`** - Agent scaffolding and template generation
- **`workflow_templates.py`** - Predefined workflow patterns
- **`model_selector.py`** - Intelligent model selection for tasks

### Specialized Assistants (assistants/)
- **`embedding_assistant.py`** - Vector embedding and similarity search
- **`research_assistant.py`** - Comprehensive research and analysis
- **`memory_assistant.py`** - Memory management and context retrieval
- **`file_assistant.py`** - File processing and management
- **`garbage_cleanup.py`** - Intelligent cleanup and maintenance

## Data Flow Architecture

### Memory & Context Flow
1. **Input Processing** - Tasks and data ingestion
2. **Embedding Generation** - Vector representation creation
3. **Graph Storage** - Relationship mapping and storage
4. **Context Retrieval** - Semantic similarity matching
5. **Agent Assignment** - Capability-based task routing

### Swarm Coordination Flow
1. **Agent Registration** - Capability and model registration
2. **Task Distribution** - Intelligent task-agent matching
3. **Execution Monitoring** - Real-time performance tracking
4. **Result Aggregation** - Output collection and synthesis
5. **Learning & Optimization** - Pattern detection and improvement

### Communication Architecture
- **MCP Protocol** - Standardized agent-to-agent messaging
- **Feedback Channels** - Bidirectional communication for coordination
- **Event System** - Asynchronous event handling and notifications
- **Memory Sharing** - Shared context and knowledge base access

## Storage & Database Architecture

### Graph Storage
- **Parquet Backend** - Vector-optimized storage with compression
- **JSON Backend** - Human-readable storage for debugging
- **Vector Similarity** - Cosine similarity for content matching
- **File History** - Complete audit trail for all operations

### Database Systems
- **`memory.db`** - SQLite database for agent memory
- **`knowledge.db`** - Knowledge base with semantic search
- **`cache.db`** - Performance optimization caching
- **`coderl.db`** - Code-related learning and patterns

### Memory Management
- **Short-term Memory** - Active context and working memory
- **Long-term Memory** - Persistent knowledge and experiences
- **Shared Memory** - Cross-agent knowledge sharing
- **Context Windows** - Efficient context management for large datasets

## Integration Points

### External Services
- **AWS Services** - Bedrock, S3, EC2 integration via boto3
- **Azure AI** - Azure AI services and cognitive capabilities
- **OpenAI/Anthropic** - LLM providers for agent intelligence
- **Ollama** - Local model deployment and management

### Development Tools
- **IDE Integration** - Amazon Q and Claude AI IDE plugins
- **Testing Framework** - Pytest with async support
- **Monitoring** - OpenTelemetry, LangSmith, Phoenix tracing
- **Documentation** - Comprehensive inline and external docs