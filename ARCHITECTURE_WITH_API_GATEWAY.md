# Agent Builder - Full Architecture with API Gateway

**Generated**: 2025-10-19
**Diagram File**: `agent_builder_full_architecture_with_api_gateway.png` (943 KB)

---

## Architecture Overview

This diagram shows the **complete Agent Builder application architecture** with the **Convex API backend integrated behind AWS API Gateway**, enabling secure, scalable deployment routing across three tiers.

---

## Key Architecture Decision: Convex Behind API Gateway

### Why This Design?

**Convex provides its own serverless API endpoint**, but we're placing it **behind AWS API Gateway + Lambda** for these critical reasons:

1. ✅ **Deployment Routing**: Lambda intelligently routes deployments to the correct tier (Freemium/Personal/Enterprise)
2. ✅ **AWS Integration**: Seamless bridge between Convex (data layer) and AWS services (deployment layer)
3. ✅ **Security**: API Gateway authorizer validates JWT tokens from Convex Auth
4. ✅ **Flexibility**: Easy to add WAF, rate limiting, custom domains, caching
5. ✅ **Separation of Concerns**: Convex handles data, API Gateway handles AWS operations

### What Gets Routed Through API Gateway?

**Deployment requests ONLY:**
```
Convex → API Gateway → Lambda → AgentCore/ECS/SSO
```

**All other operations go direct to Convex:**
```
Frontend → Convex (WebSocket/HTTPS)
  - Real-time queries
  - Mutations
  - Authentication
  - Code generation
```

---

## Architecture Layers

### 1. Frontend Layer (Cloudflare Pages)
- **Technology**: React + Vite + TypeScript
- **Production URL**: https://ai-forge.mikepfunk.com
- **CDN**: Cloudflare global edge network
- **Features**:
  - Static asset serving via CDN
  - Real-time WebSocket connection to Convex
  - Progressive Web App (PWA) capable

### 2. Backend Layer (Convex Serverless)
- **Deployment URL**: https://resolute-kudu-325.convex.site
- **Type**: Serverless real-time database + functions
- **Core Services**:

  #### Convex Auth (Session Management)
  - Multi-provider authentication
  - JWT token generation
  - Session persistence

  #### Code Generator (Agent Templates)
  - Takes user input from UI
  - Applies agent templates
  - Validates configurations

  #### Package Generator (4-File Bundle)
  Generates deployment artifacts:
  ```
  1. agent.py        - Agent code with @agent decorator
  2. mcp.json        - MCP server configuration
  3. Dockerfile      - Container build instructions
  4. cloudformation.yaml - AWS infrastructure template
  ```

  #### Deployment Router (Tier Selection)
  - Determines user's tier (Freemium/Personal/Enterprise)
  - Routes to API Gateway with tier information
  - Tracks deployment status

### 3. API Gateway Layer (AWS Integration Bridge)

#### API Gateway (REST API)
- **Endpoint**: `/deploy` (POST)
- **Purpose**: Receive deployment requests from Convex
- **Features**:
  - JWT validation via authorizer Lambda
  - Request/response logging
  - Optional: WAF, rate limiting, custom domain

#### Routing Lambda (Tier Selection Logic)
```python
def handler(event):
    user_id = validate_jwt(event['headers']['Authorization'])
    user_tier = get_user_tier_from_convex(user_id)

    if user_tier == 'freemium':
        return deploy_to_agentcore(agent_data)
    elif user_tier == 'personal':
        return deploy_to_user_aws(agent_data, user_credentials)
    elif user_tier == 'enterprise':
        return deploy_via_sso(agent_data)
```

#### Authorizer Lambda (JWT Validation)
- Validates JWT tokens issued by Convex Auth
- Fetches user information from Convex
- Returns authorization decision to API Gateway

---

## Authentication Providers

### 1. GitHub OAuth
- **Flow**: Frontend → Convex Auth → GitHub → Convex
- **Token**: GitHub access token stored in Convex
- **Scope**: `user:email`

### 2. Google OAuth
- **Flow**: Frontend → Convex Auth → Google → Convex
- **Token**: Google ID token stored in Convex
- **Scope**: `email`, `profile`

### 3. AWS Cognito (Federated Identity)
- **User Pool**: `us-east-1_hMFTc7CNL`
- **Client ID**: `fk09hmkpbk7sral3cj9ofh5vc`
- **Purpose**: Provides temporary AWS credentials for Personal tier
- **Flow**: Convex Auth → Cognito → STS AssumeRole

### 4. Anonymous (Guest Access)
- **Limitations**: View-only mode
- **Features**: Can browse templates, view demos
- **No deployments**: Cannot deploy agents

---

## Deployment Tiers

### Tier 1: Freemium (Bedrock AgentCore)

**Target Users**: Free tier users
**Limit**: 10 tests/month
**Infrastructure**: Platform-managed

#### AWS Bedrock AgentCore
- **Type**: Serverless agent runtime
- **No infrastructure management**: Fully managed by AWS
- **Features**:
  - Serverless execution
  - Auto-scaling
  - Pay-per-invocation
  - Built-in observability

#### AgentCore Services

##### AgentCore Memory
- **Event Memory**: Stores conversation history
- **Semantic Memory**: Vector database for knowledge
- **Persistence**: Across sessions

##### Code Interpreter (Isolated Sandbox)
- **Purpose**: Execute Python code safely
- **Isolation**: Sandboxed environment
- **Timeout**: 30 seconds
- **Filesystem**: Ephemeral

##### AgentCore Browser
- **Purpose**: Web interaction for agents
- **Features**: Headless Chrome, screenshot capture
- **Security**: Isolated browsing sessions

#### Platform Resources

##### Platform ECR (Shared Images)
- **Purpose**: Store agent container images
- **Access**: Read-only for freemium users
- **Retention**: 30 days

##### CloudWatch Logs
- **Retention**: 7 days for freemium
- **Access**: View-only via UI

##### DynamoDB (Agent State)
- **Purpose**: Store agent execution state
- **TTL**: 7 days for freemium

---

### Tier 2: Personal (User AWS Account)

**Target Users**: Personal tier subscribers
**Infrastructure**: User-owned AWS account
**Deployment**: Cross-account via STS AssumeRole

#### Cross-Account Deployment Flow

```
1. User provides AWS credentials in UI
2. Convex stores encrypted credentials
3. Deployment request → API Gateway → Lambda
4. Lambda assumes role in user's AWS account
5. Lambda deploys to user's ECS cluster
```

#### STS AssumeRole (Cross-Account Access)
- **Role ARN**: Provided by user
- **External ID**: Generated by platform (security)
- **Temporary Credentials**: 1-hour duration
- **Permissions**: Deploy to ECS, write to S3, CloudWatch

#### User VPC (10.0.0.0/16)
- **Subnets**: 3 public, 3 private (multi-AZ)
- **NAT Gateway**: For private subnet internet access
- **Internet Gateway**: For public subnet access

#### Application Load Balancer (HTTPS)
- **Purpose**: Route traffic to Fargate tasks
- **Certificate**: ACM certificate (auto-provisioned)
- **Health Checks**: HTTP /health endpoint

#### ECS Cluster (User-Managed)
- **Capacity Provider**: Fargate
- **Auto-scaling**: Based on CPU/memory
- **Service Discovery**: AWS Cloud Map

#### Fargate Tasks (512 CPU / 1024 MB)
- **vCPU**: 0.5 vCPU (512 units)
- **Memory**: 1 GB
- **Network**: awsvpc mode (dedicated ENI)
- **Storage**: 20 GB ephemeral

#### User Storage & Logs

##### User ECR (Private Registry)
- **Purpose**: Store user's agent images
- **Scan on Push**: Enabled
- **Retention**: User-controlled

##### CloudWatch Logs (30-day retention)
- **Log Groups**: Per agent
- **Retention**: 30 days (configurable)
- **Export**: To S3 for analysis

##### S3 Storage (Encrypted at Rest)
- **Purpose**: Agent artifacts, logs, data
- **Encryption**: AES-256 (SSE-S3)
- **Versioning**: Enabled
- **Lifecycle**: User-defined

##### DynamoDB (Agent State & Metrics)
- **Purpose**: Store agent state, metrics, history
- **Read/Write Capacity**: On-demand
- **Backup**: Point-in-time recovery enabled

---

### Tier 3: Enterprise (Coming Soon)

**Target Users**: Enterprise customers
**Infrastructure**: AWS Organizations + SSO
**Status**: Planned for Q1 2026

#### AWS SSO / Identity Center
- **Purpose**: Centralized authentication
- **SAML**: Integration with corporate IdP
- **MFA**: Required for all users

#### AWS Organizations (Multi-Account)
- **Structure**: Organization → OUs → Accounts
- **SCPs**: Service control policies for governance
- **Consolidated Billing**: Across all accounts

#### Secrets Manager (API Keys & Config)
- **Purpose**: Store sensitive configuration
- **Rotation**: Automatic rotation enabled
- **Access**: IAM role-based

#### KMS Encryption (Data Keys)
- **Purpose**: Encrypt data at rest
- **Key Rotation**: Annual automatic rotation
- **Audit**: CloudTrail logging

---

## MCP Integration Layer

### AWS Diagram MCP (Architecture Visualization)
- **Package**: `awslabs.aws-diagram-mcp-server`
- **Purpose**: Generate architecture diagrams
- **Features**:
  - List available icons
  - Generate diagrams from code
  - Export to PNG/SVG

### Bedrock AgentCore MCP (Deployment & Testing)
- **Purpose**: Deploy and test agents on AgentCore
- **Control Plane API**: Manage agent lifecycle
- **Features**:
  - Deploy agents
  - Test invocations
  - Monitor logs

### Terraform MCP (Infrastructure as Code)
- **Purpose**: Generate Terraform templates
- **Features**:
  - Search providers/modules
  - Get latest versions
  - Generate IaC code

---

## External Services

### GitHub Repository (Agent Code Storage)
- **Purpose**: Version control for agent code
- **4-File Bundle**: Committed to user's repo
- **CI/CD**: GitHub Actions integration

### Cloudflare CDN (Global Edge Network)
- **Purpose**: Serve static assets globally
- **Edge Locations**: 300+ cities worldwide
- **Features**: DDoS protection, auto-minification

---

## Data Flow Examples

### Example 1: User Creates Agent (Freemium)

```
1. User designs agent in UI (Cloudflare Pages)
2. Frontend → Convex: Save agent config (WebSocket)
3. Convex Code Generator: Generate agent.py
4. Convex Package Generator: Create 4-file bundle
5. User clicks "Deploy" in UI
6. Convex Deployment Router → API Gateway (/deploy)
7. API Gateway Authorizer: Validate JWT
8. Routing Lambda: Determine tier = freemium
9. Routing Lambda → Bedrock AgentCore API: Deploy
10. AgentCore: Pull image from Platform ECR
11. AgentCore: Execute agent
12. Logs → CloudWatch Logs
13. State → DynamoDB
14. Frontend receives deployment status (via Convex)
```

### Example 2: User Deploys to Personal AWS Account

```
1. User provides AWS credentials in settings (UI)
2. Convex: Encrypt and store credentials
3. User clicks "Deploy to My AWS"
4. Convex Deployment Router → API Gateway (/deploy)
5. API Gateway Authorizer: Validate JWT
6. Routing Lambda: Determine tier = personal
7. Routing Lambda: Fetch encrypted AWS credentials from Convex
8. Routing Lambda → STS: AssumeRole in user's account
9. Lambda (with temp credentials) → User's ECS: Deploy Fargate task
10. User's ECR: Pull agent image
11. Fargate Task: Start agent
12. Logs → User's CloudWatch Logs
13. Artifacts → User's S3
14. State → User's DynamoDB
15. Frontend receives deployment status (via Convex)
```

### Example 3: Real-time UI Update (No API Gateway)

```
1. Agent execution status changes (running → completed)
2. AgentCore/Fargate → CloudWatch Logs
3. CloudWatch Logs Subscription → Lambda
4. Lambda → Convex Mutation: Update agent status
5. Convex → Frontend: Real-time WebSocket update
6. UI automatically shows "Completed" status
```

---

## Security Architecture

### 1. Authentication Layer
- **Convex Auth**: Session management, token issuance
- **API Gateway Authorizer**: JWT validation
- **Multi-provider**: GitHub, Google, Cognito, Anonymous

### 2. Authorization Layer
- **Tier-based**: Freemium vs Personal vs Enterprise
- **Resource-based**: Users can only access their own agents
- **IAM Roles**: Least privilege for AWS services

### 3. Network Security
- **HTTPS Everywhere**: All traffic encrypted in transit
- **VPC Isolation**: Personal tier agents in isolated VPC
- **Security Groups**: Restrict traffic to necessary ports
- **NACLs**: Network-level access control

### 4. Data Security
- **Encryption at Rest**: S3, DynamoDB, Secrets Manager
- **Encryption in Transit**: TLS 1.3
- **Credential Storage**: Encrypted in Convex, decrypted only in Lambda
- **External ID**: Prevents confused deputy problem

### 5. Audit & Compliance
- **CloudTrail**: All API calls logged
- **CloudWatch Logs**: Application logs retained
- **Convex Logs**: User actions logged
- **Access Logs**: API Gateway access logs

---

## Cost Optimization

### Freemium Tier (Bedrock AgentCore)
- **AgentCore**: $0.00075 per second of execution
- **10 tests/month** @ 30 seconds each = **$0.23/month**
- **Platform cost**: Absorbed by us

### Personal Tier (User AWS Account)
User pays directly to AWS:
- **Fargate**: ~$0.04/hour → **$30/month** (24/7)
- **S3**: ~$0.50/month (50 GB)
- **CloudWatch Logs**: ~$1/month (10 GB ingested)
- **DynamoDB**: ~$2/month (on-demand)
- **Total**: ~**$35/month** (user pays AWS directly)

### Enterprise Tier (Coming Soon)
- Custom pricing
- Volume discounts
- Reserved capacity options

---

## Scalability

### Convex (Backend)
- **Auto-scaling**: Managed by Convex
- **Limits**: 1000 concurrent connections per deployment
- **Upgrade**: Can scale to enterprise plan

### API Gateway
- **Throttling**: 10,000 requests/second (default)
- **Burst**: 5,000 requests
- **Scalability**: Regional auto-scaling

### Lambda (Routing)
- **Concurrency**: 1,000 concurrent executions (default)
- **Scaling**: Automatic
- **Reserved**: Can provision for guaranteed capacity

### Bedrock AgentCore (Freemium)
- **Auto-scaling**: Managed by AWS
- **Limits**: Per-account quotas
- **Cold Start**: ~2-5 seconds

### ECS Fargate (Personal)
- **Auto-scaling**: Based on CPU/memory metrics
- **Limits**: 1,000 tasks per region (default)
- **Scaling**: Can request quota increase

---

## Monitoring & Observability

### Frontend (Cloudflare)
- **Analytics**: Cloudflare Web Analytics
- **Performance**: Lighthouse CI
- **Errors**: Sentry integration

### Backend (Convex)
- **Dashboard**: Convex built-in dashboard
- **Logs**: Function execution logs
- **Metrics**: Query performance, mutation latency

### API Gateway
- **CloudWatch Metrics**: Request count, latency, errors
- **Access Logs**: Detailed request/response logs
- **X-Ray**: Distributed tracing

### Lambda
- **CloudWatch Metrics**: Invocations, duration, errors
- **CloudWatch Logs**: Function execution logs
- **X-Ray**: Trace Lambda → AWS service calls

### AgentCore (Freemium)
- **CloudWatch Logs**: Agent execution logs
- **CloudWatch Metrics**: Invocation count, duration
- **Bedrock Dashboard**: Agent performance metrics

### ECS Fargate (Personal)
- **CloudWatch Container Insights**: CPU, memory, network
- **CloudWatch Logs**: Container stdout/stderr
- **Service-level Metrics**: Task count, CPU/memory utilization

---

## Disaster Recovery

### Backups
- **Convex**: Automatic daily backups (retained 30 days)
- **S3**: Versioning enabled, lifecycle policies
- **DynamoDB**: Point-in-time recovery (35 days)
- **ECR**: Image replication to secondary region

### High Availability
- **Convex**: Multi-region replication
- **Cloudflare**: Global CDN with automatic failover
- **API Gateway**: Regional service with multiple AZs
- **Lambda**: Runs in multiple AZs automatically
- **Fargate**: Multi-AZ deployment via ECS service

### Recovery Time Objective (RTO)
- **Convex**: < 5 minutes (automatic)
- **API Gateway + Lambda**: < 10 minutes
- **Fargate**: < 15 minutes (redeploy tasks)

### Recovery Point Objective (RPO)
- **Convex**: < 1 minute (real-time replication)
- **DynamoDB**: < 5 minutes (point-in-time recovery)
- **S3**: 0 (versioning)

---

## Diagram Legend

### Color Coding

| Color | Environment | Purpose |
|-------|-------------|---------|
| **Light Blue** (#E3F2FD) | Frontend Layer | User-facing React app |
| **Light Yellow** (#FFF9C4) | Backend Layer | Convex serverless backend |
| **Light Green** (#C8E6C9) | API Gateway Layer | AWS integration bridge |
| **Misty Rose** (#FFEBEE) | Authentication | Multi-provider auth |
| **Light Green** (#E8F5E9) | Tier 1 Freemium | Bedrock AgentCore |
| **Lavender** (#E1BEE7) | Tier 2 Personal | User AWS account |
| **Light Coral** (#FFCCBC) | Tier 3 Enterprise | Coming soon |
| **Light Cyan** (#B2EBF2) | MCP Integration | Model Context Protocol |
| **Light Gray** (#F5F5F5) | External Services | GitHub, Cloudflare |

### Edge Styles

| Style | Color | Purpose |
|-------|-------|---------|
| **Solid Bold** | Blue | User traffic (HTTPS/WebSocket) |
| **Solid Bold** | Purple | Deployment requests |
| **Solid Bold** | Green/Blue/Red | Tier routing |
| **Solid** | Various | Service-to-service communication |
| **Dashed** | Various | Optional/future connections |

### Line Thickness

| Thickness | Usage |
|-----------|-------|
| **4pt** | Critical paths (user → backend, deployment routing) |
| **3pt** | Important connections (tier selection) |
| **2pt** | Standard connections (service-to-service) |

---

## File Locations

- **Diagram**: `m:\strandsagents\agent_builder_full_architecture_with_api_gateway.png` (943 KB)
- **Generator**: `m:\strandsagents\generate_full_architecture_with_api_gateway.py`
- **Architecture Spec**: `m:\strandsagents\GENERATED_ARCHITECTURE.md`
- **Integration Guide**: `m:\strandsagents\diagram\CONVEX_API_GATEWAY_INTEGRATION.md`

---

## Next Steps

### For Developers

1. **Review** the architecture diagram
2. **Understand** the API Gateway integration flow
3. **Implement** the Routing Lambda function
4. **Test** deployment routing for each tier
5. **Deploy** to staging environment

### For Infrastructure

1. **Provision** API Gateway REST API
2. **Create** Authorizer Lambda
3. **Create** Routing Lambda
4. **Configure** IAM roles for cross-account access
5. **Set up** CloudWatch dashboards

### For Security

1. **Review** JWT validation logic
2. **Audit** STS AssumeRole permissions
3. **Configure** WAF rules for API Gateway
4. **Enable** CloudTrail logging
5. **Set up** security alerts

---

**Generated**: 2025-10-19
**Version**: 1.0
**Last Updated**: 2025-10-19 18:20 PST
