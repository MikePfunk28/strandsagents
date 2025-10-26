"""
Agent Builder Application - AWS Architecture Diagram
Generates a comprehensive visualization of the 3-tier deployment architecture
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS, Fargate, Lambda
from diagrams.aws.database import Dynamodb
from diagrams.aws.network import APIGateway, ELB, VPC
from diagrams.aws.security import IAM, SecretsManager, Cognito
from diagrams.aws.storage import S3
from diagrams.aws.integration import SQS
from diagrams.aws.management import Cloudwatch, CloudwatchLogs
from diagrams.aws.ml import Bedrock
from diagrams.aws.devtools import Codebuild
from diagrams.onprem.client import Users
from diagrams.onprem.vcs import Github
from diagrams.saas.identity import Auth0
from diagrams.programming.framework import React

graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "pad": "0.5",
}

with Diagram("Agent Builder Application - 3-Tier Architecture", 
             show=False, 
             direction="TB",
             graph_attr=graph_attr,
             filename="agent_builder_architecture"):
    
    # Users and Frontend
    users = Users("Users")
    
    with Cluster("Frontend (Vite + React)"):
        frontend = React("React App\n(Vite)")
    
    # Authentication Layer
    with Cluster("Authentication (Convex Auth)"):
        github_oauth = Github("GitHub OAuth")
        google_oauth = Auth0("Google OAuth")
        cognito_oauth = Cognito("AWS Cognito")
    
    # Convex Backend
    with Cluster("Backend (Convex)"):
        convex_api = APIGateway("Convex API")
        
        with Cluster("Core Services"):
            agents_service = Lambda("Agents Service")
            code_gen = Lambda("Code Generator")
            deployment_router = Lambda("Deployment Router")
            mcp_client = Lambda("MCP Client")
        
        with Cluster("Data Layer"):
            convex_db = Dynamodb("Convex Database\n(Users, Agents,\nDeployments)")
    
    # Tier 1: Freemium (AgentCore)
    with Cluster("Tier 1: Freemium Deployment"):
        with Cluster("AWS Bedrock AgentCore"):
            agentcore = Bedrock("AgentCore\nSandbox")
            agentcore_logs = CloudwatchLogs("AgentCore Logs")
    
    # Tier 2: Personal (User's AWS Account)
    with Cluster("Tier 2: Personal Deployment\n(User's AWS Account)"):
        with Cluster("Cross-Account Access"):
            sts_role = IAM("STS AssumeRole\n(External ID)")
        
        with Cluster("User's VPC"):
            user_vpc = VPC("User VPC")
            
            with Cluster("ECS Fargate"):
                user_fargate = Fargate("Agent Container")
                user_ecr = Codebuild("ECR Repository")
            
            user_s3 = S3("Agent Storage")
            user_logs = CloudwatchLogs("CloudWatch Logs")
    
    # Tier 3: Enterprise (Future)
    with Cluster("Tier 3: Enterprise\n(Coming Soon)"):
        enterprise_sso = IAM("AWS SSO")
        enterprise_fargate = Fargate("Enterprise\nDeployment")
    
    # MCP Integration
    with Cluster("MCP Servers"):
        mcp_bedrock = Bedrock("Bedrock MCP")
        mcp_custom = Lambda("Custom MCP\nServers")
    
    # Monitoring & Logging
    with Cluster("Observability"):
        cloudwatch = Cloudwatch("CloudWatch\nMetrics")
        error_logs = CloudwatchLogs("Error Logs")
        audit_logs = CloudwatchLogs("Audit Logs")
    
    # Testing Infrastructure
    with Cluster("Agent Testing"):
        test_queue = SQS("Test Queue")
        docker_test = ECS("Docker Test\nEnvironment")
        test_logs = CloudwatchLogs("Test Logs")
    
    # User Flow
    users >> Edge(label="HTTPS") >> frontend
    
    # Authentication Flow
    frontend >> Edge(label="OAuth") >> [github_oauth, google_oauth, cognito_oauth]
    [github_oauth, google_oauth, cognito_oauth] >> convex_api
    
    # Frontend to Backend
    frontend >> Edge(label="API Calls") >> convex_api
    
    # Backend Services
    convex_api >> agents_service
    convex_api >> code_gen
    convex_api >> deployment_router
    convex_api >> mcp_client
    
    # Data Access
    [agents_service, code_gen, deployment_router] >> convex_db
    
    # Tier 1 Deployment Flow
    deployment_router >> Edge(label="Freemium\nDeploy", color="green") >> agentcore
    agentcore >> agentcore_logs
    
    # Tier 2 Deployment Flow
    deployment_router >> Edge(label="Personal\nDeploy", color="blue") >> sts_role
    sts_role >> Edge(label="Assume Role") >> user_vpc
    user_vpc >> user_fargate
    user_fargate >> user_ecr
    user_fargate >> user_s3
    user_fargate >> user_logs
    
    # Tier 3 Deployment Flow (Future)
    deployment_router >> Edge(label="Enterprise\n(Future)", color="orange", style="dashed") >> enterprise_sso
    enterprise_sso >> enterprise_fargate
    
    # MCP Integration
    mcp_client >> [mcp_bedrock, mcp_custom]
    mcp_bedrock >> agentcore
    
    # Testing Flow
    agents_service >> Edge(label="Submit Test") >> test_queue
    test_queue >> docker_test
    docker_test >> test_logs
    
    # Monitoring
    [agentcore, user_fargate, docker_test] >> cloudwatch
    [agents_service, deployment_router] >> error_logs
    [convex_api, deployment_router] >> audit_logs

print("Architecture diagram generated: agent_builder_architecture.png")
