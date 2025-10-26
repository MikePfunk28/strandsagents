# Convex API Backend Integration with AWS API Gateway

## Architecture Overview

```
┌─────────────┐
│   Frontend  │
│ (React/Web) │
└──────┬──────┘
       │
       │ HTTPS
       ▼
┌─────────────────────┐
│  AWS API Gateway    │
│  - Custom Domain    │
│  - Rate Limiting    │
│  - WAF Protection   │
│  - Caching          │
└──────┬──────────────┘
       │
       │ HTTP/HTTPS Proxy
       ▼
┌─────────────────────┐
│   Convex Backend    │
│  - Real-time DB     │
│  - Functions        │
│  - Actions          │
│  - Auth             │
└─────────────────────┘
```

---

## Option 1: Direct Convex (Recommended - No API Gateway)

### Why Direct?
- ✅ **Simpler**: No additional AWS configuration
- ✅ **Lower Latency**: One less hop
- ✅ **Built-in Features**: Convex has rate limiting, auth, real-time subscriptions
- ✅ **WebSocket Support**: Real-time updates work seamlessly
- ✅ **Lower Cost**: No API Gateway charges

### Implementation

#### 1. Frontend Configuration
```typescript
// convex.config.ts
import { defineConfig } from "convex/server";

export default defineConfig({
  projectSlug: "your-project-slug",
});
```

#### 2. Environment Variables
```bash
# .env.local
NEXT_PUBLIC_CONVEX_URL=https://your-project.convex.cloud
CONVEX_DEPLOY_KEY=your_deploy_key
```

#### 3. Direct API Calls
```typescript
// React Component
import { useQuery, useMutation } from "convex/react";
import { api } from "../convex/_generated/api";

function AgentsList() {
  const agents = useQuery(api.agents.list);
  const createAgent = useMutation(api.agents.create);

  // Direct Convex queries - no API Gateway needed
  return (
    <div>
      {agents?.map(agent => (
        <div key={agent._id}>{agent.name}</div>
      ))}
    </div>
  );
}
```

---

## Option 2: API Gateway as Proxy (Advanced Use Cases)

### When to Use API Gateway?
- 🔒 Need AWS WAF for additional security
- 🌐 Custom domain with AWS Certificate Manager
- 🚦 Fine-grained rate limiting beyond Convex
- 📊 Centralized logging with CloudWatch
- 🔀 Combining Convex with other AWS services in one API
- 💰 Need AWS cost allocation tags

### Implementation Steps

#### Step 1: Set Up API Gateway HTTP API

```yaml
# serverless.yml or SAM template
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ConvexProxyApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: prod
      CorsConfiguration:
        AllowOrigins:
          - "https://yourdomain.com"
        AllowMethods:
          - GET
          - POST
          - OPTIONS
        AllowHeaders:
          - Content-Type
          - Authorization
      DefinitionBody:
        openapi: 3.0.1
        info:
          title: Convex Proxy API
          version: 1.0.0
        paths:
          /{proxy+}:
            x-amazon-apigateway-any-method:
              x-amazon-apigateway-integration:
                type: http_proxy
                httpMethod: ANY
                uri: https://your-project.convex.cloud/{proxy}
                requestParameters:
                  integration.request.path.proxy: method.request.path.proxy
                payloadFormatVersion: "1.0"
              parameters:
                - name: proxy
                  in: path
                  required: true
                  schema:
                    type: string

  # Optional: Custom Domain
  CustomDomain:
    Type: AWS::ApiGatewayV2::DomainName
    Properties:
      DomainName: api.yourdomain.com
      DomainNameConfigurations:
        - EndpointType: REGIONAL
          CertificateArn: !Ref Certificate

  ApiMapping:
    Type: AWS::ApiGatewayV2::ApiMapping
    Properties:
      DomainName: !Ref CustomDomain
      ApiId: !Ref ConvexProxyApi
      Stage: prod
```

#### Step 2: Configure Frontend to Use API Gateway

```typescript
// convex.config.ts
import { defineConfig } from "convex/server";

export default defineConfig({
  // Point to API Gateway instead of Convex directly
  address: process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
           "https://your-api-id.execute-api.us-east-1.amazonaws.com"
});
```

```bash
# .env.local
NEXT_PUBLIC_API_GATEWAY_URL=https://api.yourdomain.com
NEXT_PUBLIC_CONVEX_URL=https://your-project.convex.cloud
```

#### Step 3: Add Rate Limiting (Optional)

```yaml
# Add to API Gateway template
Resources:
  ConvexProxyApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      # ... previous config
      ThrottleSettings:
        RateLimit: 1000  # requests per second
        BurstLimit: 2000

  # Or use Usage Plan for more control
  ApiUsagePlan:
    Type: AWS::ApiGateway::UsagePlan
    Properties:
      UsagePlanName: ConvexProxyPlan
      Throttle:
        RateLimit: 100
        BurstLimit: 200
      Quota:
        Limit: 10000
        Period: DAY
```

#### Step 4: Add WAF Protection (Optional)

```yaml
Resources:
  WebACL:
    Type: AWS::WAFv2::WebACL
    Properties:
      Name: ConvexProxyWAF
      Scope: REGIONAL
      DefaultAction:
        Allow: {}
      Rules:
        - Name: RateLimitRule
          Priority: 1
          Statement:
            RateBasedStatement:
              Limit: 2000
              AggregateKeyType: IP
          Action:
            Block: {}
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: RateLimitRule

  WebACLAssociation:
    Type: AWS::WAFv2::WebACLAssociation
    Properties:
      ResourceArn: !GetAtt ConvexProxyApi.Arn
      WebACLArn: !GetAtt WebACL.Arn
```

---

## Option 3: Hybrid Approach (Best of Both Worlds)

Use Convex directly for real-time features, API Gateway for specific endpoints:

```typescript
// convex-client.ts
import { ConvexReactClient } from "convex/react";

// Direct Convex for real-time subscriptions
export const convexClient = new ConvexReactClient(
  process.env.NEXT_PUBLIC_CONVEX_URL!
);

// API Gateway for specific actions that need AWS features
export const apiGatewayClient = {
  async deployAgent(agentId: string) {
    // Use API Gateway for deployment (might trigger Lambda, ECS, etc.)
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_API_GATEWAY_URL}/deploy`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agentId })
      }
    );
    return response.json();
  }
};
```

```tsx
// App.tsx
import { ConvexProvider } from "convex/react";
import { convexClient, apiGatewayClient } from "./convex-client";

function App() {
  return (
    <ConvexProvider client={convexClient}>
      {/* Real-time features use Convex directly */}
      <AgentsList />

      {/* AWS-specific features use API Gateway */}
      <button onClick={() => apiGatewayClient.deployAgent(id)}>
        Deploy to AWS
      </button>
    </ConvexProvider>
  );
}
```

---

## Comparison Matrix

| Feature | Direct Convex | API Gateway Proxy | Hybrid |
|---------|--------------|-------------------|--------|
| **Setup Complexity** | ⭐ Simple | ⭐⭐⭐ Complex | ⭐⭐ Moderate |
| **Latency** | ⭐⭐⭐ Lowest | ⭐⭐ Higher | ⭐⭐⭐ Low |
| **Real-time Support** | ✅ Native | ⚠️ Limited | ✅ Native |
| **AWS WAF** | ❌ No | ✅ Yes | ✅ Yes (partial) |
| **Custom Domain** | ⚠️ Convex only | ✅ Yes | ✅ Yes (partial) |
| **Rate Limiting** | ✅ Built-in | ✅ Fine-grained | ✅ Both |
| **Cost** | 💰 Low | 💰💰 Higher | 💰 Moderate |
| **AWS Integration** | ⚠️ Limited | ✅ Full | ✅ Full |

---

## Recommended Architecture for Your Use Case

Based on your Agent Builder application with 3-tier deployment:

```typescript
// Recommended: Hybrid Approach

// 1. Convex for data operations (direct)
const agents = useQuery(api.agents.list);
const updateAgent = useMutation(api.agents.update);

// 2. API Gateway + Lambda for AWS deployments
async function deployToAWS(agentId: string, tier: string) {
  // API Gateway routes to appropriate deployment target
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_API_GATEWAY_URL}/deploy`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${await getAuthToken()}`
      },
      body: JSON.stringify({
        agentId,
        tier, // 'freemium', 'personal', 'enterprise'
      })
    }
  );
  return response.json();
}

// API Gateway Lambda routes to correct tier
export async function handler(event: APIGatewayEvent) {
  const { agentId, tier } = JSON.parse(event.body);

  switch (tier) {
    case 'freemium':
      return deployToAgentCore(agentId);
    case 'personal':
      return deployToUserAWS(agentId);
    case 'enterprise':
      return deployToEnterprise(agentId);
  }
}
```

---

## Implementation Guide

### Quick Start: Direct Convex (Recommended)

```bash
# 1. Install Convex
npm install convex

# 2. Initialize
npx convex dev

# 3. Deploy
npx convex deploy
```

### Advanced: Add API Gateway Proxy

```bash
# 1. Create SAM template
cat > template.yaml << 'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Parameters:
  ConvexUrl:
    Type: String
    Description: Your Convex deployment URL

Resources:
  ConvexProxyApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: prod
      DefinitionBody:
        openapi: 3.0.1
        paths:
          /{proxy+}:
            x-amazon-apigateway-any-method:
              x-amazon-apigateway-integration:
                type: http_proxy
                httpMethod: ANY
                uri: !Sub '${ConvexUrl}/{proxy}'

Outputs:
  ApiUrl:
    Value: !Sub 'https://${ConvexProxyApi}.execute-api.${AWS::Region}.amazonaws.com'
EOF

# 2. Deploy
sam build
sam deploy --guided --parameter-overrides ConvexUrl=https://your-project.convex.cloud
```

---

## Best Practices

### 1. Use Direct Convex for Most Cases
```typescript
// ✅ Good: Direct Convex
const agents = useQuery(api.agents.list);
```

### 2. Use API Gateway Only When Needed
```typescript
// ✅ Good: API Gateway for AWS-specific operations
const deployment = await apiGateway.deployToECS(agentId);
```

### 3. Implement Proper Error Handling
```typescript
async function deployAgent(agentId: string) {
  try {
    const response = await fetch('/deploy', {
      method: 'POST',
      body: JSON.stringify({ agentId })
    });

    if (!response.ok) {
      throw new Error(`Deployment failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Deployment error:', error);
    throw error;
  }
}
```

### 4. Monitor Both Endpoints
```typescript
// CloudWatch metrics for API Gateway
// Convex dashboard for Convex metrics
// Combined view in your monitoring dashboard
```

---

## Security Considerations

1. **Authentication**: Use Convex Auth + API Gateway authorizers
2. **CORS**: Configure properly on both Convex and API Gateway
3. **Rate Limiting**: Implement on API Gateway and/or Convex
4. **Secrets**: Store in AWS Secrets Manager, access from Lambda
5. **Network**: Use VPC endpoints if needed for private connectivity

---

## Cost Optimization

- **Direct Convex**: ~$25-50/month for moderate usage
- **+ API Gateway**: Additional $3.50/million requests
- **+ WAF**: $5/month + $1/million requests
- **Recommendation**: Start with direct Convex, add API Gateway only if needed

---

## Summary

**For your Agent Builder application, I recommend:**

1. ✅ **Use Convex directly** for all CRUD operations (agents, users, deployments)
2. ✅ **Use API Gateway + Lambda** for AWS deployment routing (tier selection)
3. ✅ **Keep real-time features** on direct Convex (subscriptions, live updates)
4. ✅ **Route AWS operations** through API Gateway (ECS, AgentCore, cross-account)

This hybrid approach gives you:
- Fast, real-time UI updates via Convex
- Secure, flexible AWS deployments via API Gateway
- Best of both worlds! 🚀
