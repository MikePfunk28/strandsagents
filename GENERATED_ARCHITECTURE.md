# Agent Builder Application - Generated Architecture Diagram

**Generated**: 2025-10-19T21:50:26.050Z
**Resources**: 20

## System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        CloudflarePages["Cloudflare Pages<br/>React + Vite<br/>ai-forge.mikepfunk.com"]
    end

    subgraph "Backend Layer - Convex"
        ConvexBackend["Convex Backend<br/>resolute-kudu-325.convex.site"]
        ConvexAuth["Convex Auth"]
        CodeGen["Code Generator"]
        PackageGen["Package Generator<br/>4-file bundle"]
        DeployRouter["Deployment Router"]
    end

    subgraph "Authentication Providers"
        GitHub["GitHub OAuth"]
        Google["Google OAuth"]
        Cognito["AWS Cognito<br/>Federated Identity<br/>us-east-1_hMFTc7CNL"]
    end

    subgraph "AWS Tier 1: Freemium"
        Tier1Cluster["Platform ECS Cluster"]
        Tier1Fargate["Fargate Tasks<br/>256 CPU / 512 MB"]
        Tier1ECR["Platform ECR"]
        AgentCore["Bedrock AgentCore<br/>Serverless Runtime"]
    end

    subgraph "AWS Tier 2: Personal"
        STSRole["STS AssumeRole<br/>Cross-Account"]
        UserVPC["User VPC<br/>10.0.0.0/16"]
        UserCluster["User ECS Cluster"]
        UserFargate["Fargate Tasks<br/>512 CPU / 1024 MB"]
        UserLogs["CloudWatch Logs<br/>30-day retention"]
        UserS3["S3 Storage<br/>Encrypted"]
    end

    subgraph "AWS Tier 3: Enterprise"
        SSO["AWS SSO /<br/>Identity Center"]
        Orgs["AWS Organizations"]
        Secrets["Secrets Manager"]
    end

    subgraph "MCP Integration"
        DiagramMCP["AWS Diagram MCP<br/>Architecture Visualization"]
        AgentCoreMCP["Bedrock AgentCore MCP<br/>Agent Deployment"]
    end

    %% Frontend connections
    CloudflarePages --> ConvexBackend

    %% Backend internal connections
    ConvexBackend --> ConvexAuth
    ConvexBackend --> CodeGen
    ConvexBackend --> PackageGen
    ConvexBackend --> DeployRouter

    %% Authentication
    ConvexAuth --> GitHub
    ConvexAuth --> Google
    ConvexAuth --> Cognito

    %% Deployment routing
    DeployRouter --> |Tier 1<br/>Freemium| Tier1Cluster
    DeployRouter --> |Tier 2<br/>Personal| STSRole
    DeployRouter --> |Tier 3<br/>Enterprise| SSO

    %% Tier 1 flow
    Tier1Cluster --> Tier1Fargate
    Tier1ECR --> Tier1Fargate
    Tier1Fargate --> AgentCore

    %% Tier 2 flow
    STSRole --> UserVPC
    UserVPC --> UserCluster
    UserCluster --> UserFargate
    UserFargate --> UserLogs
    UserFargate --> UserS3

    %% Tier 3 flow
    SSO --> Orgs
    Orgs --> Secrets

    %% MCP connections
    ConvexBackend --> DiagramMCP
    ConvexBackend --> AgentCoreMCP

    %% Styling
    style CloudflarePages fill:#f96
    style ConvexBackend fill:#4a9eff
    style ConvexAuth fill:#4a9eff
    style Tier1Fargate fill:#ff9900
    style UserFargate fill:#ff9900
    style AgentCore fill:#00d4aa
    style STSRole fill:#ff6b6b
    style Cognito fill:#ff6b6b
```

## Architecture Overview

### Frontend
- **Platform**: Cloudflare Pages
- **Framework**: React + Vite + TypeScript
- **Production URL**: https://ai-forge.mikepfunk.com
- **Deployment URL**: https://633051e6.agent-builder-application.pages.dev

### Backend
- **Platform**: Convex (Serverless)
- **URL**: https://resolute-kudu-325.convex.site
- **Features**:
  - Real-time data synchronization
  - Serverless functions
  - Built-in authentication
  - Code generation pipeline
  - Deployment routing

### Authentication
1. **GitHub OAuth** - GitHub authentication
2. **Google OAuth** - Google authentication
3. **AWS Cognito** - Federated identity with STS AssumeRole
   - User Pool: us-east-1_hMFTc7CNL
   - Client ID: fk09hmkpbk7sral3cj9ofh5vc

### Deployment Tiers

#### Tier 1: Freemium (Platform Fargate)
- **Target**: Free tier users
- **Limit**: 10 tests/month
- **Infrastructure**:
  - Platform-managed ECS cluster
  - Fargate tasks (256 CPU, 512 MB memory)
  - Shared ECR repository
  - AWS Bedrock AgentCore runtime

#### Tier 2: Personal (User AWS Account)
- **Target**: Personal tier users
- **Features**:
  - Cross-account deployment via STS AssumeRole
  - User-owned VPC (10.0.0.0/16)
  - User ECS cluster
  - Fargate tasks (512 CPU, 1024 MB memory)
  - CloudWatch Logs (30-day retention)
  - S3 storage (encrypted)

#### Tier 3: Enterprise (SSO)
- **Target**: Enterprise customers
- **Features**:
  - AWS SSO / Identity Center integration
  - AWS Organizations management
  - Secrets Manager for sensitive data
  - Enhanced security and compliance

### MCP Integration
- **AWS Diagram MCP**: Architecture visualization tool
- **Bedrock AgentCore MCP**: Agent deployment and testing

## Resource List

1. **cloudflare-pages**: Agent Builder Frontend
2. **convex-backend**: Convex Backend
3. **auth-provider**: GitHub OAuth
4. **auth-provider**: Google OAuth
5. **cognito-user-pool**: AWS Cognito Auth (us-east-1_hMFTc7CNL)
6. **ecs-cluster**: Platform ECS Cluster
7. **ecs-fargate**: Agent Runtime (Tier 1)
8. **ecr**: Platform ECR Repository
9. **bedrock-agentcore**: AWS Bedrock AgentCore
10. **iam-role**: Cross-Account Role
11. **vpc**: User VPC (Tier 2)
12. **ecs-cluster**: User ECS Cluster
13. **ecs-fargate**: Agent Runtime (Tier 2)
14. **cloudwatch-logs**: Agent Logs
15. **s3**: Deployment Artifacts
16. **sso**: AWS SSO
17. **organizations**: AWS Organizations
18. **secrets-manager**: Enterprise Secrets
19. **mcp-server**: AWS Diagram MCP
20. **mcp-server**: Bedrock AgentCore MCP

## Key Features

### 4-File Deployment Bundle
Every agent deployment generates:
1. **agent.py** - Agent code with @agent decorator
2. **mcp.json** - MCP server configuration
3. **Dockerfile** - Container configuration
4. **cloudformation.yaml** - AWS infrastructure template

### Security
- ✅ Proper access control using Convex user document IDs
- ✅ STS temporary credentials for cross-account deployment
- ✅ External ID validation for AssumeRole
- ✅ Encrypted S3 storage
- ✅ Secrets Manager for sensitive data

### Agent Capabilities
- Preprocessing and postprocessing hooks
- MCP tool integration
- Meta-tooling for dynamic tool creation
- Memory and interleaved reasoning
- AWS Bedrock AgentCore runtime

---

**Note**: This diagram was generated using a simplified Mermaid generator.
For deployment-specific diagrams, use the built-in `awsDiagramGenerator` module.
