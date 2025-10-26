# Agent Builder Architecture Diagrams - Summary

## Generated Diagrams

### 1. **agent_builder_architecture_enhanced.png** ✨ **RECOMMENDED**
- **Size:** 747 KB
- **Location:** `m:\strandsagents\diagram\agent_builder_architecture_enhanced.png`
- **Features:**
  - ✅ **BOLD, larger text** throughout for better readability
  - ✅ **8 clearly defined environments** with color coding
  - ✅ **Thicker connection lines** (penwidth 2-4) for better visibility
  - ✅ **Environment-specific background colors**
  - ✅ **Clear visual hierarchy** with font sizes 12-24pt
  - ✅ **Enhanced spacing** for better layout

### 2. agent_builder_architecture.png (Original)
- **Size:** Standard
- **Location:** `m:\strandsagents\diagram\agent_builder_architecture.png`
- **Features:** Original architecture diagram with standard formatting

---

## Architecture Overview - 8 Environments

### 🌐 **Environment 1: User & Frontend Layer** (Blue - #E3F2FD)
**Components:**
- End Users
- React + Vite Frontend (TypeScript)
- Authentication Providers:
  - GitHub OAuth 2.0
  - Google OAuth 2.0
  - AWS Cognito Identity

**Purpose:** User interaction and authentication

---

### ⚡ **Environment 2: Convex Backend Layer** (Orange - #FFF3E0)
**Components:**
- Convex API Gateway
- Core Microservices:
  - **Agents Service** - CRUD operations for AI agents
  - **Code Generator** - Generates Python agent code with decorators
  - **Deployment Router** - Routes deployments based on user tier
  - **MCP Client** - Tool integration via Model Context Protocol
- **Convex Database** - Stores users, agents, deployments, tests

**Purpose:** Serverless backend with real-time capabilities

---

### 🆓 **Environment 3: Tier 1 - Freemium Deployment** (Green - #E8F5E9)
**Components:**
- AWS Bedrock AgentCore Serverless Sandbox
- AgentCore Execution Logs

**Features:**
- ✅ 10 tests/month limit
- ✅ Zero AWS setup required
- ✅ Instant deployment
- ✅ Perfect for testing and prototyping

**Target Users:** Free tier users

---

### 👤 **Environment 4: Tier 2 - Personal Deployment** (Purple - #F3E5F5)
**Components:**
- Cross-Account Security:
  - STS AssumeRole with External ID
- User's AWS Account:
  - User VPC (Isolated Network)
  - ECS Fargate (Production-ready agent containers)
  - ECR Repository (Docker images)
  - S3 Storage (Agent data & logs)
  - CloudWatch Logs (Full monitoring)

**Features:**
- ✅ Full control over AWS resources
- ✅ No usage limits
- ✅ Production-ready deployment
- ✅ Secure cross-account access

**Target Users:** Personal tier users with their own AWS accounts

---

### 🏢 **Environment 5: Tier 3 - Enterprise** (Yellow - #FFF9C4) ⏳ **COMING SOON**
**Components:**
- AWS SSO (Centralized authentication)
- Enterprise Deployment (Multi-tenant)

**Planned Features:**
- ⏳ Centralized management
- ⏳ Compliance controls
- ⏳ Cost allocation
- ⏳ Large-scale enterprise deployments

**Target Users:** Enterprise customers

---

### 🔌 **Environment 6: MCP Integration Layer** (Teal - #E0F2F1)
**Components:**
- Bedrock MCP Server
- Custom MCP Servers (User configured)

**Purpose:**
- ✅ Tool integration via Model Context Protocol
- ✅ Extends agent capabilities
- ✅ Custom tool development
- ✅ Agent-to-agent communication

---

### 🧪 **Environment 7: Testing Infrastructure** (Pink - #FCE4EC)
**Components:**
- Test Queue (Async processing via SQS)
- Docker Test Environment (Isolated)
- Test Execution Logs

**Purpose:**
- ✅ Automated agent testing
- ✅ Isolated test execution
- ✅ Async test processing
- ✅ Comprehensive test logging

---

### 📊 **Environment 8: Observability & Monitoring** (Light Blue - #E1F5FE)
**Components:**
- CloudWatch Metrics & Alarms
- Error Logs (Centralized)
- Audit Logs (Compliance)

**Purpose:**
- ✅ Performance monitoring
- ✅ Error tracking
- ✅ Security auditing
- ✅ Compliance reporting

---

## Data Flow Legend

### Primary Flows (Bold, Thick Lines)
- **Blue (#1976D2):** User → Frontend (HTTPS traffic)
- **Purple (#7B1FA2):** Authentication flows (OAuth)
- **Green (#388E3C):** Frontend → Backend API calls
- **Orange (#F57C00):** Backend service routing

### Deployment Flows (Extra Thick Lines)
- **Green (#43A047):** Freemium deployments to AgentCore
- **Purple (#5E35B1):** Personal deployments to User's AWS
- **Yellow (#F57F17):** Enterprise deployments (Future, dashed)

### Integration Flows
- **Teal (#00897B):** MCP protocol communications
- **Pink (#C2185B):** Testing infrastructure flows
- **Blue (#0288D1):** Database read/write operations

### Monitoring Flows (Dotted Lines)
- **Gray (#546E7A):** CloudWatch metrics
- **Red (#D32F2F):** Error logs
- **Blue (#1565C0):** Audit logs

---

## Key Architectural Decisions

### 1. Multi-Tier Deployment Strategy
- **Tier 1 (Freemium):** Serverless AgentCore for quick start
- **Tier 2 (Personal):** User's AWS account for production
- **Tier 3 (Enterprise):** Centralized enterprise deployment (future)

### 2. Security Model
- Cross-account access via STS AssumeRole with External ID
- OAuth 2.0 for authentication (GitHub, Google, AWS Cognito)
- Least-privilege IAM policies
- Isolated VPC environments

### 3. Technology Stack
- **Frontend:** React + Vite (TypeScript)
- **Backend:** Convex (Serverless, real-time)
- **Compute:** AWS ECS Fargate, Bedrock AgentCore
- **Database:** Convex Database, DynamoDB
- **Monitoring:** CloudWatch
- **Container Registry:** ECR

### 4. Integration Strategy
- Model Context Protocol (MCP) for tool integration
- EventBridge for event-driven architecture
- SQS for async processing
- Docker for isolated testing

---

## File Locations

### Diagrams
```
m:\strandsagents\diagram\
  ├── agent_builder_architecture_enhanced.png  ← ENHANCED VERSION (RECOMMENDED)
  ├── agent_builder_architecture.png           ← Original version
  ├── generate_enhanced_architecture.py        ← Generator script
  ├── architecture_diagram.py                  ← Original generator
  └── ARCHITECTURE_DIAGRAM.md                  ← Original documentation
```

### Other Generated Diagrams
```
m:\strandsagents\
  ├── aws_production_infrastructure.png        ← AWS Infrastructure
  └── full_stack_architecture.png              ← Full stack with AgentCore
```

---

## Usage

### Viewing the Diagram
Open `agent_builder_architecture_enhanced.png` with any image viewer to see:
- Clear visual separation of 8 environments
- Bold, readable text
- Color-coded components
- Comprehensive data flow visualization

### Regenerating the Diagram
```powershell
cd m:\strandsagents\diagram
$env:Path += ";C:\Program Files\Graphviz\bin"
python generate_enhanced_architecture.py
```

---

## Next Steps

1. ✅ Review the enhanced architecture diagram
2. ✅ Validate environment separation
3. ✅ Confirm data flow paths
4. ⏳ Implement Tier 3 Enterprise features
5. ⏳ Expand MCP integration capabilities
6. ⏳ Enhance testing infrastructure

---

**Generated:** October 18, 2025
**Tool:** AWS Diagrams (Python diagrams library)
**Format:** PNG (High Resolution)
