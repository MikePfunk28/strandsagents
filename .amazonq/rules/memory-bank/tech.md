# Technology Stack & Dependencies

## Programming Languages
- **Python 3.8+** - Primary development language
- **JavaScript** - Configuration and helper scripts
- **PowerShell** - Windows automation and setup scripts
- **Shell/Bash** - Unix/Linux automation scripts

## Core Framework
- **Strands Agents SDK** - Primary multi-agent framework
  - `strands-agents>=1.0.0` - Base framework
  - `strands-agents-builder` - Agent building tools
  - `strands-agents-tools[mem0_memory,local_chromium_browser,a2a_client,diagram,rss,use_computer]` - Tool ecosystem

## AI & Machine Learning
### Model Providers
- **Anthropic** - Claude models for advanced reasoning
- **OpenAI** - GPT models for general intelligence
- **Ollama** - Local model deployment and management
- **Azure AI Services** - Enterprise AI capabilities
  - `azure-ai-agents` - Azure agent framework
  - `azure-ai-inference` - Model inference services
  - `azure-ai-projects` - Project management

### Multi-Agent Frameworks
- **AutoGen** - Microsoft's multi-agent framework
  - `autogen-agentchat` - Agent communication
  - `autogen-core` - Core functionality
  - `autogen-ext` - Extensions
  - `autogenstudio` - Visual agent design

## Data Processing & Storage
### Vector & Graph Storage
- **LanceDB** - Vector database for embeddings
- **Qdrant** - Vector similarity search engine
- **Mem0** - Agent memory built on FAISS (local/free)
- **PyArrow** - Columnar data processing (Parquet support)

### Traditional Databases
- **SQLAlchemy** - SQL toolkit and ORM
- **SQLModel** - Modern SQL database integration
- **PostgreSQL** - Production database via `psycopg`
- **SQLite** - Local development databases
- **Alembic** - Database migration management

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **PyArrow** - Columnar data processing

## Web & Network
### Web Frameworks
- **Flask** - Lightweight web framework
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Starlette** - Async web toolkit

### Web Scraping & Automation
- **Scrapy** - Web scraping framework
- **Playwright** - Browser automation
- **BeautifulSoup4** - HTML parsing
- **HTTPX** - Async HTTP client
- **lxml** - XML/HTML processing
- **html2text** - HTML to text conversion

## Communication & Protocols
- **MCP (Model Context Protocol)** - Agent-to-agent communication
- **Slack SDK** - Slack integration
  - `slack-bolt` - Slack app framework
  - `slack-sdk` - Slack API client

## File Processing
### Document Processing
- **PyPDF** - PDF processing
- **python-pptx** - PowerPoint processing
- **openpyxl** - Excel file processing
- **xlrd/xlsxwriter** - Excel read/write
- **mammoth** - Word document processing
- **markitdown** - Markdown conversion

### Media Processing
- **Pillow** - Image processing
- **OpenCV** - Computer vision
- **PyTesseract** - OCR capabilities
- **PyDub** - Audio processing
- **SpeechRecognition** - Speech-to-text

## Development Tools
### Testing
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support

### Configuration & Environment
- **python-dotenv** - Environment variable management
- **Pydantic** - Data validation and settings
- **pydantic-settings** - Settings management

### Utilities
- **Rich** - Terminal formatting and progress bars
- **Typer** - CLI application framework
- **Loguru** - Advanced logging
- **tqdm** - Progress bars
- **tenacity** - Retry mechanisms
- **tiktoken** - Token counting for LLMs

## Monitoring & Observability
### Tracing & Analytics
- **OpenTelemetry** - Distributed tracing
  - `opentelemetry-api` - Core API
  - `opentelemetry-sdk` - SDK implementation
  - `opentelemetry-instrumentation` - Auto-instrumentation
  - `opentelemetry-exporter-otlp` - OTLP exporter

### AI Observability
- **LangSmith** - LLM tracing and monitoring
- **Weights & Biases** - Experiment tracking
- **Phoenix (Arize)** - AI observability platform

## Cloud Services
### AWS Integration
- **boto3** - AWS SDK for Python
- **nest-asyncio** - Async AWS operations

### Azure Integration
- **azure-identity** - Azure authentication
- **azure-search-documents** - Azure Cognitive Search
- **azure-storage-blob** - Azure Blob Storage

## Automation & System Integration
- **PyAutoGUI** - Desktop automation
- **youtube-transcript-api** - YouTube transcript extraction

## Build & Deployment
### Package Management
- **pip** - Python package installer
- **uv** - Fast Python package manager (used in examples)

### Configuration Files
- **requirements.txt** - Python dependencies
- **pyproject.toml** - Modern Python project configuration
- **pytest.ini** - Testing configuration
- **.env** - Environment variables

## Development Commands
### Setup & Installation
```bash
# Virtual environment setup
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Browser automation setup
playwright install chromium
```

### Testing
```bash
# Run all tests
pytest

# Run specific test files
pytest test/test_graph_system.py
pytest test/test_swarm_simple.py
```

### Development Scripts
- **`fix-venv.ps1`** - Virtual environment repair
- **`quick-fix-venv.ps1`** - Quick environment fixes
- **`claude-flow.ps1`** - Claude AI workflow automation
- **`start.ps1`** - Web scraper startup (Windows)
- **`start.sh`** - Web scraper startup (Unix/Linux)

## Environment Variables
### Required Configuration
- `ANTHROPIC_API_KEY` - Claude model access
- `OPENAI_API_KEY` - OpenAI model access
- `TAVILY_API_KEY` - Web search capabilities
- `AWS_REGION` - AWS services region
- `STABILITY_API_KEY` - Image generation

### Optional Configuration
- `BYPASS_TOOL_CONSENT=true` - Skip confirmation prompts
- `MEM0_API_KEY` - Mem0 platform access
- `LANGSMITH_API_KEY` - LangSmith tracing
- `WANDB_API_KEY` - Weights & Biases tracking
- `PHOENIX_COLLECTOR_ENDPOINT` - Phoenix observability

## Platform Support
- **Windows** - Full support with PowerShell scripts
- **Linux/Mac** - Full support with shell scripts
- **Docker** - Containerized deployment support

## Performance Characteristics
- **Node Creation**: ~50ms per node (including embeddings)
- **Similarity Search**: ~100ms for 10,000 nodes
- **Graph Traversal**: ~200ms for depth-3 traversal
- **Memory Usage**: ~100MB for 10,000 nodes with embeddings
- **Storage Efficiency**: Parquet compression reduces size by ~70%