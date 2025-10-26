"""
Agent Builder Application - Enhanced AWS Architecture Diagram
With bold text, larger fonts, and clearly defined environments
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS, Fargate, Lambda
from diagrams.aws.database import Dynamodb
from diagrams.aws.network import APIGateway, VPC
from diagrams.aws.security import IAM, Cognito
from diagrams.aws.storage import S3
from diagrams.aws.integration import SQS
from diagrams.aws.management import Cloudwatch, CloudwatchLogs
from diagrams.aws.ml import Bedrock
from diagrams.aws.devtools import Codebuild
from diagrams.onprem.client import Users
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import React
from diagrams.programming.language import Python

print("Generating Enhanced Agent Builder Architecture Diagram...")
print("=" * 80)

# Enhanced graph attributes for bold, larger text
graph_attr = {
    "fontsize": "24",
    "fontname": "Arial Bold",
    "bgcolor": "white",
    "pad": "1.0",
    "nodesep": "1.2",
    "ranksep": "2.0",
    "splines": "spline"
}

cluster_attr = {
    "fontsize": "20",
    "fontname": "Arial Bold",
    "labeljust": "l",
    "penwidth": "3.0"
}

node_attr = {
    "fontsize": "14",
    "fontname": "Arial Bold"
}

edge_attr = {
    "fontsize": "12",
    "fontname": "Arial Bold"
}

with Diagram(
    "Agent Builder Application - 3-Tier Architecture",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    filename="agent_builder_architecture_enhanced",
    outformat="png"
):

    # ========================================================================
    # ENVIRONMENT 1: USER & FRONTEND LAYER
    # ========================================================================
    with Cluster("🌐 ENVIRONMENT 1: USER & FRONTEND LAYER", graph_attr={"bgcolor": "#E3F2FD", "penwidth": "4"}):
        users = Users("End Users")

        with Cluster("Frontend Application", graph_attr={"bgcolor": "#BBDEFB"}):
            frontend = React("React + Vite\nTypeScript")

        with Cluster("Authentication Providers", graph_attr={"bgcolor": "#90CAF9"}):
            github_oauth = Github("GitHub\nOAuth 2.0")
            google_oauth = Python("Google\nOAuth 2.0")
            cognito_oauth = Cognito("AWS Cognito\nIdentity")

    # ========================================================================
    # ENVIRONMENT 2: CONVEX BACKEND LAYER
    # ========================================================================
    with Cluster("⚡ ENVIRONMENT 2: CONVEX BACKEND LAYER", graph_attr={"bgcolor": "#FFF3E0", "penwidth": "4"}):
        convex_api = APIGateway("Convex API\nGateway")

        with Cluster("Core Microservices", graph_attr={"bgcolor": "#FFE0B2"}):
            agents_service = Lambda("Agents Service\nCRUD Operations")
            code_gen = Lambda("Code Generator\nPython Agent Code")
            deployment_router = Lambda("Deployment Router\nTier Selection")
            mcp_client = Lambda("MCP Client\nTool Integration")

        with Cluster("Data Persistence", graph_attr={"bgcolor": "#FFCC80"}):
            convex_db = Dynamodb(
                "Convex Database\nUsers | Agents\nDeployments | Tests")

    # ========================================================================
    # ENVIRONMENT 3: TIER 1 - FREEMIUM (AWS AGENTCORE)
    # ========================================================================
    with Cluster("🆓 ENVIRONMENT 3: TIER 1 - FREEMIUM DEPLOYMENT", graph_attr={"bgcolor": "#E8F5E9", "penwidth": "4"}):
        with Cluster("AWS Bedrock AgentCore", graph_attr={"bgcolor": "#C8E6C9"}):
            agentcore = Bedrock(
                "AgentCore\nServerless Sandbox\n10 Tests/Month")
            agentcore_logs = CloudwatchLogs("AgentCore\nExecution Logs")

    # ========================================================================
    # ENVIRONMENT 4: TIER 2 - PERSONAL (USER'S AWS ACCOUNT)
    # ========================================================================
    with Cluster("👤 ENVIRONMENT 4: TIER 2 - PERSONAL DEPLOYMENT", graph_attr={"bgcolor": "#F3E5F5", "penwidth": "4"}):
        with Cluster("Cross-Account Security", graph_attr={"bgcolor": "#E1BEE7"}):
            sts_role = IAM("STS AssumeRole\nExternal ID\nCross-Account Access")

        with Cluster("USER'S AWS ACCOUNT", graph_attr={"bgcolor": "#CE93D8", "penwidth": "3"}):
            with Cluster("User VPC Network", graph_attr={"bgcolor": "#BA68C8"}):
                user_vpc = VPC("User VPC\nIsolated Network")

                with Cluster("ECS Fargate Compute", graph_attr={"bgcolor": "#AB47BC"}):
                    user_fargate = Fargate("Agent Container\nProduction Ready")
                    user_ecr = Codebuild("ECR Repository\nDocker Images")

                user_s3 = S3("S3 Storage\nAgent Data & Logs")
                user_logs = CloudwatchLogs("CloudWatch Logs\nFull Monitoring")

    # ========================================================================
    # ENVIRONMENT 5: TIER 3 - ENTERPRISE (FUTURE)
    # ========================================================================
    with Cluster("🏢 ENVIRONMENT 5: TIER 3 - ENTERPRISE (COMING SOON)", graph_attr={"bgcolor": "#FFF9C4", "penwidth": "4", "style": "dashed"}):
        with Cluster("Enterprise Security", graph_attr={"bgcolor": "#FFF59D", "style": "dashed"}):
            enterprise_sso = IAM("AWS SSO\nCentralized Auth")
            enterprise_fargate = Fargate(
                "Enterprise\nDeployment\nMulti-Tenant")

    # ========================================================================
    # ENVIRONMENT 6: MCP INTEGRATION LAYER
    # ========================================================================
    with Cluster("🔌 ENVIRONMENT 6: MCP INTEGRATION LAYER", graph_attr={"bgcolor": "#E0F2F1", "penwidth": "4"}):
        mcp_bedrock = Bedrock("Bedrock MCP\nServer")
        mcp_custom = Lambda("Custom MCP\nServers\nUser Configured")

    # ========================================================================
    # ENVIRONMENT 7: TESTING INFRASTRUCTURE
    # ========================================================================
    with Cluster("🧪 ENVIRONMENT 7: TESTING INFRASTRUCTURE", graph_attr={"bgcolor": "#FCE4EC", "penwidth": "4"}):
        test_queue = SQS("Test Queue\nAsync Processing")
        docker_test = ECS("Docker Test\nEnvironment\nIsolated")
        test_logs = CloudwatchLogs("Test Execution\nLogs")

    # ========================================================================
    # ENVIRONMENT 8: OBSERVABILITY & MONITORING
    # ========================================================================
    with Cluster("📊 ENVIRONMENT 8: OBSERVABILITY & MONITORING", graph_attr={"bgcolor": "#E1F5FE", "penwidth": "4"}):
        cloudwatch = Cloudwatch("CloudWatch\nMetrics & Alarms")
        error_logs = CloudwatchLogs("Error Logs\nCentralized")
        audit_logs = CloudwatchLogs("Audit Logs\nCompliance")

    # ========================================================================
    # DATA FLOWS
    # ========================================================================

    # User Flow
    users >> Edge(label="HTTPS", color="#1976D2",
                  style="bold", penwidth="3") >> frontend

    # Authentication Flow
    frontend >> Edge(label="OAuth Login", color="#7B1FA2",
                     style="bold", penwidth="2") >> github_oauth
    frontend >> Edge(label="OAuth Login", color="#7B1FA2",
                     style="bold", penwidth="2") >> google_oauth
    frontend >> Edge(label="OAuth Login", color="#7B1FA2",
                     style="bold", penwidth="2") >> cognito_oauth

    github_oauth >> Edge(label="Auth Token", color="#7B1FA2",
                         penwidth="2") >> convex_api
    google_oauth >> Edge(label="Auth Token", color="#7B1FA2",
                         penwidth="2") >> convex_api
    cognito_oauth >> Edge(label="Auth Token",
                          color="#7B1FA2", penwidth="2") >> convex_api

    # Frontend to Backend
    frontend >> Edge(label="API Calls", color="#388E3C",
                     style="bold", penwidth="3") >> convex_api

    # Backend Services
    convex_api >> Edge(label="Route", color="#F57C00",
                       penwidth="2") >> agents_service
    convex_api >> Edge(label="Route", color="#F57C00",
                       penwidth="2") >> code_gen
    convex_api >> Edge(label="Route", color="#F57C00",
                       penwidth="2") >> deployment_router
    convex_api >> Edge(label="Route", color="#F57C00",
                       penwidth="2") >> mcp_client

    # Data Access
    agents_service >> Edge(
        label="Read/Write", color="#0288D1", penwidth="2") >> convex_db
    code_gen >> Edge(label="Read/Write", color="#0288D1",
                     penwidth="2") >> convex_db
    deployment_router >> Edge(
        label="Read/Write", color="#0288D1", penwidth="2") >> convex_db

    # Tier 1 Deployment Flow
    deployment_router >> Edge(
        label="FREEMIUM\nDEPLOY", color="#43A047", style="bold", penwidth="4") >> agentcore
    agentcore >> Edge(label="Logs", color="#757575",
                      penwidth="2") >> agentcore_logs

    # Tier 2 Deployment Flow
    deployment_router >> Edge(
        label="PERSONAL\nDEPLOY", color="#5E35B1", style="bold", penwidth="4") >> sts_role
    sts_role >> Edge(label="Assume Role\nExternal ID",
                     color="#5E35B1", style="bold", penwidth="3") >> user_vpc
    user_vpc >> Edge(label="Deploy", color="#5E35B1",
                     penwidth="3") >> user_fargate
    user_fargate >> Edge(label="Pull Image",
                         color="#5E35B1", penwidth="2") >> user_ecr
    user_fargate >> Edge(label="Store Data",
                         color="#5E35B1", penwidth="2") >> user_s3
    user_fargate >> Edge(label="Send Logs", color="#5E35B1",
                         penwidth="2") >> user_logs

    # Tier 3 Deployment Flow (Future)
    deployment_router >> Edge(label="ENTERPRISE\n(FUTURE)", color="#F57F17",
                              style="dashed", penwidth="3") >> enterprise_sso
    enterprise_sso >> Edge(label="SSO Auth", color="#F57F17",
                           style="dashed", penwidth="2") >> enterprise_fargate

    # MCP Integration
    mcp_client >> Edge(label="MCP Protocol", color="#00897B",
                       style="bold", penwidth="3") >> mcp_bedrock
    mcp_client >> Edge(label="MCP Protocol", color="#00897B",
                       style="bold", penwidth="3") >> mcp_custom
    mcp_bedrock >> Edge(label="Execute", color="#00897B",
                        penwidth="2") >> agentcore

    # Testing Flow
    agents_service >> Edge(label="Submit Test", color="#C2185B",
                           style="bold", penwidth="3") >> test_queue
    test_queue >> Edge(label="Process", color="#C2185B",
                       penwidth="2") >> docker_test
    docker_test >> Edge(label="Results", color="#C2185B",
                        penwidth="2") >> test_logs

    # Monitoring Flows
    agentcore >> Edge(label="Metrics", color="#546E7A",
                      style="dotted", penwidth="2") >> cloudwatch
    user_fargate >> Edge(label="Metrics", color="#546E7A",
                         style="dotted", penwidth="2") >> cloudwatch
    docker_test >> Edge(label="Metrics", color="#546E7A",
                        style="dotted", penwidth="2") >> cloudwatch

    agents_service >> Edge(label="Errors", color="#D32F2F",
                           style="dotted", penwidth="2") >> error_logs
    deployment_router >> Edge(
        label="Errors", color="#D32F2F", style="dotted", penwidth="2") >> error_logs

    convex_api >> Edge(label="Audit", color="#1565C0",
                       style="dotted", penwidth="2") >> audit_logs
    deployment_router >> Edge(
        label="Audit", color="#1565C0", style="dotted", penwidth="2") >> audit_logs

print("\n" + "=" * 80)
print("✅ Enhanced Architecture Diagram Generated Successfully!")
print("=" * 80)
print("\n📁 Location: agent_builder_architecture_enhanced.png")
print("\n✨ Enhanced Features:")
print("  • BOLD, larger text throughout")
print("  • 8 clearly defined environments with color coding")
print("  • Thicker connection lines for better visibility")
print("  • Environment-specific background colors")
print("  • Clear visual hierarchy")
print("\n🏗️ Architecture Environments:")
print("  1️⃣  User & Frontend Layer (Blue)")
print("  2️⃣  Convex Backend Layer (Orange)")
print("  3️⃣  Tier 1 - Freemium Deployment (Green)")
print("  4️⃣  Tier 2 - Personal Deployment (Purple)")
print("  5️⃣  Tier 3 - Enterprise (Yellow - Coming Soon)")
print("  6️⃣  MCP Integration Layer (Teal)")
print("  7️⃣  Testing Infrastructure (Pink)")
print("  8️⃣  Observability & Monitoring (Light Blue)")
print("\n" + "=" * 80)
