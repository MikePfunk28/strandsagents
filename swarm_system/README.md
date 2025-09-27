# 🧠 Swarm System - Hierarchical Assistant → Agent → Swarm Architecture

A sophisticated multi-layered AI system that implements the vision of simple, focused assistants composing into complex agents, orchestrated as intelligent swarms.

## 🎯 **Vision**

This system implements a hierarchical architecture where:
- **Assistants** are simple, focused building blocks (prompt + tools + model)
- **Agents** are compositions of multiple assistants with additional logic
- **Swarms** are collections of lightweight agents coordinated by orchestrators

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                        SWARM LAYER                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                ORCHESTRATOR                          │    │
│  │              (llama3.2 model)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             LIGHTWEIGHT AGENTS                      │    │
│  │           (gemma3:27b models)                       │    │
│  │  • Fact Checker    • Idea Generator                │    │
│  │  • Risk Analyzer   • Solution Optimizer            │    │
│  │  • Pattern Recog.  • Communicator                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                         AGENT LAYER                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 COMPLEX AGENTS                       │    │
│  │  • Research Agent  • Creative Agent                 │    │
│  │  • Analysis Agent  • Execution Agent                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                       ASSISTANT LAYER                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 CORE ASSISTANTS                     │    │
│  │  • Text Processor  • Calculator                     │    │
│  │  • Data Analyzer   • File Manager                   │    │
│  │  • Web Scraper     • Code Executor                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Project Structure**

```
swarm_system/
├── assistants/                    # Assistant foundation layer
│   ├── __init__.py               # Assistant package exports
│   ├── registry.py               # Assistant registration & management
│   ├── base_assistant.py         # Base assistant class
│   └── core/                     # Core assistant implementations
│       ├── text_processor.py     # Text processing assistant
│       └── calculator_assistant.py # Mathematical operations assistant
├── agents/                       # Agent composition layer
│   └── composition/              # Complex agent compositions
├── swarm/                        # Swarm intelligence layer
│   └── lightweight/              # Lightweight 270m model agents
├── meta_tooling/                 # Dynamic creation capabilities
├── communication/                # Inter-component communication
├── workflows/                    # Async workflows & callbacks
├── utils/                        # Utility functions
│   ├── database_manager.py       # Multi-database management
│   ├── prompts.py                # Centralized prompt management
│   └── tools.py                  # Meta-tooling capabilities
├── swarm_demo.py                 # System demonstration
└── README.md                     # This file
```

## 🗄️ **Database Architecture**

The system uses **4 specialized SQLite databases** as requested:

### **cache.db** - Caching Layer
- Temporary data storage with TTL
- Frequently accessed computation results
- Session-based caching with automatic cleanup

### **memory.db** - Context & Memory
- Short-term and long-term memory storage
- Session-based memory management
- Importance scoring and memory relationships

### **knowledge.db** - External Knowledge
- Learned information from external sources
- Research findings and discoveries
- Topic-based organization with confidence scoring

### **coderl.db** - Code Explanations
- Line-by-line code explanations with embeddings
- File scope analysis and understanding
- Similarity search for code patterns
- Human-in-the-loop feedback integration

## 🚀 **Key Features Implemented**

### ✅ **Assistant Foundation Layer**
- **BaseAssistant** abstract class with standardized interface
- **AssistantRegistry** for dynamic registration and management
- **AssistantConfig** for flexible configuration
- Core assistants: TextProcessor, Calculator

### ✅ **Database Layer**
- **DatabaseManager** class handling all 4 databases
- Automatic database initialization and schema management
- CRUD operations for all data types
- Performance optimization with vacuuming and cleanup

### ✅ **Meta-Tooling Capabilities**
- **Dynamic tool creation** at runtime
- **Assistant-as-tool** functionality
- **Lightweight agent creation** for swarm operations
- **Knowledge base integration** with learning storage

### ✅ **Prompt Management**
- **Centralized prompts** for all assistant types
- **Lightweight agent prompts** optimized for 270m models
- **Orchestrator prompts** for complex reasoning with llama3.2
- **Dynamic prompt customization** and context injection

### ✅ **Communication Infrastructure**
- **Swarm communication** protocols
- **Knowledge base querying** and retrieval
- **Learning storage** and retrieval
- **System status monitoring**

## 💡 **Usage Examples**

### **Basic Assistant Usage**
```python
from swarm_system.assistants.registry import global_registry
from swarm_system.assistants.base_assistant import AssistantConfig
from swarm_system.assistants.core.text_processor import TextProcessorAssistant

# Register assistant type
global_registry.register("text_processor", TextProcessorAssistant)

# Create assistant instance
config = AssistantConfig(
    name="my_text_processor",
    description="Text processing assistant",
    model_id="llama3.2"
)
assistant = global_registry.create_instance("text_processor", "my_instance", config=config)

# Use assistant
result = await assistant.execute_async("Analyze this text: Hello world!")
```

### **Database Operations**
```python
from swarm_system.utils.database_manager import db_manager

# Store knowledge
db_manager.store_knowledge(
    topic="ai_systems",
    content="Swarm systems use multiple AI agents working together",
    confidence=0.9
)

# Query knowledge
results = db_manager.search_knowledge("swarm", limit=5)

# Cache data
db_manager.set_cache("my_key", {"data": "value"}, ttl_seconds=3600)
cached = db_manager.get_cache("my_key")
```

### **Meta-Tooling**
```python
from swarm_system.utils.tools import create_dynamic_tool, create_assistant_as_tool

# Create dynamic tool
create_dynamic_tool(
    tool_name="my_tool",
    description="My custom tool",
    code_content="result = 'Hello from dynamic tool!'",
    input_schema={"type": "object", "properties": {}}
)

# Create assistant as tool
create_assistant_as_tool(
    assistant_name="research_helper",
    assistant_type="research",
    model_id="llama3.2"
)
```

## 🔧 **Model Strategy**

### **Lightweight Agents (270m models)**
- **gemma3:27b** - Fast, efficient for simple tasks
- Optimized prompts for minimal resource usage
- Specialized roles: fact_checker, idea_generator, etc.

### **Orchestrator/Executor (llama3.2)**
- **llama3.2** - More capable for complex reasoning
- Advanced prompts for coordination and planning
- Meta-learning and performance optimization

## 🔄 **Workflow Integration**

The system integrates with **Strands workflows** through:
- **Async iterators** for streaming responses
- **Callback handlers** for real-time updates
- **Event-driven architecture** for component communication
- **State management** across workflow executions

## 📊 **System Capabilities**

### **Current Implementation Status**
- ✅ **Assistant Foundation**: Base classes and core assistants
- ✅ **Database Layer**: All 4 databases with full CRUD operations
- ✅ **Meta-Tooling**: Dynamic creation of tools and assistants
- ✅ **Prompt Management**: Comprehensive prompt library
- ✅ **Communication**: Inter-component communication protocols
- 🔄 **Agent Composition**: Framework ready for implementation
- 🔄 **Swarm Coordination**: Architecture designed, implementation pending
- 🔄 **Advanced Features**: Learning loops and optimization (designed)

### **Performance Characteristics**
- **Local Only**: Uses Ollama models, no cloud dependencies
- **Scalable Architecture**: From single assistant to full swarm
- **Memory Efficient**: Database optimization and cleanup
- **Extensible Design**: Easy to add new assistant types

## 🚀 **Getting Started**

1. **Install Dependencies**
```bash
pip install strands strands-tools
# Ensure Ollama is running with required models
ollama pull llama3.2
ollama pull gemma3:27b
```

2. **Run Demonstration**
```python
cd swarm_system
python swarm_demo.py
```

3. **Explore Components**
```python
# Examine database status
from swarm_system.utils.database_manager import db_manager
print(db_manager.get_stats())

# List available assistants
from swarm_system.assistants.registry import global_registry
print(global_registry.list_available_types())
```

## 🎯 **Next Development Phases**

### **Phase 1: Agent Composition** ✅ *Completed*
- Assistant foundation layer
- Database infrastructure
- Meta-tooling capabilities

### **Phase 2: Complex Agents** 🔄 *In Progress*
- Multi-assistant agent compositions
- Advanced workflow orchestration
- Agent-to-agent communication

### **Phase 3: Swarm Intelligence** 📋 *Planned*
- Lightweight agent swarm (20+ agents)
- Advanced orchestration with llama3.2
- Meta-learning and optimization

### **Phase 4: Advanced Features** 📋 *Planned*
- Human-in-the-loop feedback integration
- Advanced embeddings and similarity search
- Performance analytics and optimization
- Production deployment patterns

## 🔍 **Key Innovations**

1. **Hierarchical Architecture**: Clean separation of concerns across layers
2. **Meta-Tooling**: Runtime creation and modification of components
3. **Multi-Database Design**: Specialized storage for different data types
4. **Local-First Approach**: No cloud dependencies, full local operation
5. **Extensible Framework**: Easy to add new assistant types and capabilities

## 📈 **Benefits**

- **Modularity**: Mix and match components as needed
- **Scalability**: From simple to complex use cases
- **Maintainability**: Clean, organized codebase
- **Performance**: Optimized for local execution
- **Flexibility**: Adapt to various problem domains
- **Learning**: Self-improving through meta-learning

This swarm system represents a sophisticated approach to AI agent orchestration, providing both the simplicity of focused assistants and the power of coordinated swarm intelligence.
