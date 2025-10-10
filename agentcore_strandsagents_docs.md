# StrandsAgents & AgentCore Documentation Summary

## 🎯 STRANDS AGENTS FRAMEWORK

**Model-driven SDK for building AI agents**

### Key Features:
- Lightweight, production-ready framework
- Supports multiple models and providers (Anthropic, OpenAI, Bedrock, etc.)
- Built-in tools: calculator, http_request, shell, current_time
- MCP server integration
- Streaming and async support
- Comprehensive observability

### Installation:
```bash
pip install strands-agents
pip install strands-agents-tools
```

### Basic Usage:
```python
from strands import Agent, tool
from strands_tools import calculator, current_time

agent = Agent(tools=[calculator, current_time])
response = agent("What is 25 * 48 and what time is it?")
```

### Custom Tools:
```python
@tool
def letter_counter(text: str, letter: str) -> int:
    """Count occurrences of a letter in text"""
    return text.lower().count(letter.lower())

agent = Agent(tools=[letter_counter])
```

##  AMAZON BEDROCK AGENTCORE

**Enterprise-grade agent deployment platform**

### Key Features:
- Zero infrastructure management
- Built-in memory, security, observability
- Browser automation and code interpreter
- Gateway for MCP protocol integration
- Identity and authentication
- Auto-scaling and monitoring

### Installation:
```bash
pip install bedrock-agentcore-starter-toolkit
pip install bedrock-agentcore
```

### Deployment:
```bash
# Configure agent
agentcore configure -e my_agent.py

# Deploy to AWS
agentcore launch

# Test deployment
agentcore invoke '{"prompt": "Hello!"}'
```

### Runtime Integration:
```python
from strands import Agent
from bedrock_agentcore import BedrockAgentCoreApp

agent = Agent()
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    response = agent(payload.get('prompt', 'Hello'))
    return str(response)

if __name__ == '__main__':
    app.run()
```

## 🔗 INTEGRATION PATTERNS

### 1. Strands Agent → AgentCore Runtime
Deploy local Strands agents to enterprise AgentCore platform:

```python
# Local development with Strands
from strands import Agent
from strands_tools import http_request

agent = Agent(tools=[http_request])

# Deploy to AgentCore
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def my_agent(payload):
    return agent(payload.get('prompt'))
```

### 2. MCP Server Integration
Connect external tools via Model Context Protocol:

```python
# MCP configuration
{
  "mcpServers": {
    "strands-agents": {
      "command": "uvx",
      "args": ["strands-agents-mcp-server"]
    }
  }
}
```

### 3. Gateway for External APIs
Route agent requests through AgentCore Gateway:

```python
# Gateway configuration
gateway_config = {
    "name": "MyGateway",
    "protocolType": "MCP",
    "authorizerType": "CUSTOM_JWT"
}
```

## 🛠️ KEY TOOLS & CAPABILITIES

### Built-in Tools (Strands):
- `calculator` - Mathematical operations
- `http_request` - API calls
- `shell` - System commands
- `current_time` - Date/time functions
- `use_computer` - Desktop automation

### AgentCore Services:
- **Browser Tool** - Web automation with Playwright
- **Code Interpreter** - Python code execution
- **Memory** - Conversation persistence
- **Gateway** - MCP protocol routing
- **Identity** - Authentication & authorization

### Browser Automation Example:
```python
from bedrock_agentcore.tools.browser_client import browser_session

with browser_session("us-west-2") as client:
    ws_url, headers = client.generate_ws_headers()
    # Use with Playwright or Nova Act
```

### Code Interpreter Example:
```python
from bedrock_agentcore.tools.code_interpreter_client import code_session

with code_session("us-west-2") as client:
    result = client.invoke("print('Hello from AgentCore!')")
```

##  OBSERVABILITY & MONITORING

### Strands Telemetry:
```python
from strands.telemetry import StrandsTelemetry

telemetry = StrandsTelemetry()
telemetry.setup_otlp_exporter()
telemetry.setup_console_exporter()

agent = Agent(model="claude-3-sonnet")
response = agent("Analyze this data")  # Automatically traced
```

### AgentCore Monitoring:
- Built-in CloudWatch integration
- OpenTelemetry support
- Performance metrics
- Error tracking

##  DEPLOYMENT OPTIONS

### Local Development:
```bash
# Strands only
python my_agent.py

# AgentCore local testing
agentcore launch --local
```

### Production Deployment:
```bash
# Deploy to AgentCore Runtime
agentcore configure -e my_agent.py
agentcore launch

# Import existing agents
agentcore import-agent --agent-id ABCD1234
```

## 💡 BEST PRACTICES

1. **Start with Strands** for rapid prototyping
2. **Use AgentCore** for production deployment
3. **Leverage MCP** for tool integration
4. **Enable observability** from day one
5. **Test locally** before deploying
6. **Use typed strategies** for memory configuration
7. **Implement proper error handling**

## 🔧 TROUBLESHOOTING

### Common Issues:
- **Model access**: Ensure proper AWS credentials
- **Tool registration**: Check tool imports and decorators
- **Memory limits**: Configure appropriate instance sizes
- **Network access**: Verify VPC and security group settings

### Debug Mode:
```python
import logging
logging.getLogger("strands").setLevel(logging.DEBUG)
```

This comprehensive framework combination provides everything needed to build, deploy, and scale production AI agents from prototype to enterprise.