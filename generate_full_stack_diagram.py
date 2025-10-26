"""
Full Stack Architecture Diagram Generator
Creates diagram for AgentCore + Strands Agents + Convex + Cloudflare + AWS setup
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import Lambda, Fargate, ECS
from diagrams.aws.database import Dynamodb
from diagrams.aws.security import IAM, SecretsManager, Cognito
from diagrams.aws.integration import Eventbridge
from diagrams.aws.ml import Bedrock
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.client import User, Users
from diagrams.onprem.vcs import Github
from diagrams.programming.framework import React
from diagrams.programming.language import Python, Javascript

print("Generating Full Stack Architecture Diagram...")
print("=" * 80)

with Diagram(
    "AgentCore + Strands Agents Full Stack Architecture",
    show=False,
    direction="TB",
    filename="full_stack_architecture",
    outformat="png",
    graph_attr={
        "splines": "spline",
        "nodesep": "1.0",
        "ranksep": "1.5"
    }
):

    # ============================================================================
    # USER LAYER - Different types of users and authentication
    # ============================================================================
    with Cluster("Users & Authentication"):
        users = Users("End Users")

        with Cluster("Auth Providers"):
            aws_auth = Cognito("AWS Cognito\nLogin")
            github_auth = Github("GitHub\nOAuth")
            google_auth = User("Google\nOAuth")
            anon_auth = User("Anonymous\nAccess")

    # ============================================================================
    # FRONTEND LAYER - Cloudflare hosted
    # ============================================================================
    with Cluster("Frontend Layer - Cloudflare"):
        cloudflare = React("Cloudflare\nCDN & Edge")
        frontend_app = React("Web Application\nUI/UX")

    # ============================================================================
    # BACKEND LAYER - Convex
    # ============================================================================
    with Cluster("Backend Layer - Convex"):
        convex_backend = Javascript("Convex Backend\nRealtime Database")
        convex_functions = Javascript("Convex Functions\nAPI Endpoints")
        convex_actions = Javascript("Convex Actions\nServer-Side Logic")

    # ============================================================================
    # AI AGENT LAYER - Strands Agents + AgentCore
    # ============================================================================
    with Cluster("AI Agent Layer"):
        with Cluster("Strands Agents Framework"):
            strands_orchestrator = Python(
                "Strands Orchestrator\nAgent Coordination")
            strands_agents = [
                Python("Research Agent"),
                Python("Code Agent"),
                Python("Analysis Agent")
            ]

        with Cluster("AWS Bedrock AgentCore"):
            agentcore_runtime = Bedrock(
                "AgentCore Runtime\nServerless Execution")
            agentcore_memory = Bedrock("AgentCore Memory\nContext & Knowledge")
            agentcore_sandbox = Lambda("AgentCore Sandbox\nCode Interpreter")
            agentcore_browser = Lambda("AgentCore Browser\nWeb Interaction")

    # ============================================================================
    # AWS INFRASTRUCTURE LAYER
    # ============================================================================
    with Cluster("AWS Infrastructure"):
        with Cluster("Compute"):
            ecs_cluster = ECS("ECS Cluster")
            fargate_tasks = [
                Fargate("Fargate Task 1"),
                Fargate("Fargate Task 2")
            ]

        with Cluster("Database & Storage"):
            dynamodb = Dynamodb("DynamoDB\nAgent State & Data")
            dynamodb_sessions = Dynamodb("DynamoDB\nUser Sessions")

        with Cluster("Integration & Events"):
            eventbridge = Eventbridge("EventBridge\nEvent Routing")

        with Cluster("Security & Secrets"):
            iam_policies = IAM("IAM Policies\nLeast Privilege")
            secrets = SecretsManager("Secrets Manager\nAPI Keys & Tokens")

        with Cluster("Monitoring"):
            cloudwatch_logs = Cloudwatch("CloudWatch Logs\nApplication Logs")
            cloudwatch_metrics = Cloudwatch(
                "CloudWatch Metrics\nPerformance & Alarms")

    # ============================================================================
    # DATA FLOWS - User Authentication
    # ============================================================================
    users >> Edge(label="access", color="blue") >> cloudflare

    # Auth flows
    users >> Edge(label="login", color="purple", style="dashed") >> aws_auth
    users >> Edge(label="login", color="purple", style="dashed") >> github_auth
    users >> Edge(label="login", color="purple", style="dashed") >> google_auth
    users >> Edge(label="access", color="purple", style="dashed") >> anon_auth

    # Auth to Cognito
    aws_auth >> Edge(label="verify", color="purple") >> frontend_app
    github_auth >> Edge(label="verify", color="purple") >> frontend_app
    google_auth >> Edge(label="verify", color="purple") >> frontend_app
    anon_auth >> Edge(label="access", color="purple") >> frontend_app

    # ============================================================================
    # DATA FLOWS - Frontend to Backend
    # ============================================================================
    cloudflare >> Edge(label="serves", color="blue") >> frontend_app
    frontend_app >> Edge(label="API calls", color="green") >> convex_backend

    # Convex internal flows
    convex_backend >> Edge(label="executes", color="green") >> convex_functions
    convex_functions >> Edge(label="triggers", color="green") >> convex_actions

    # ============================================================================
    # DATA FLOWS - Backend to AI Agents
    # ============================================================================
    convex_actions >> Edge(label="invoke agents",
                           color="orange") >> strands_orchestrator

    # Strands Orchestrator to individual agents
    strands_orchestrator >> Edge(
        label="coordinates", color="orange") >> strands_agents[0]
    strands_orchestrator >> Edge(
        label="coordinates", color="orange") >> strands_agents[1]
    strands_orchestrator >> Edge(
        label="coordinates", color="orange") >> strands_agents[2]

    # Strands to AgentCore
    strands_orchestrator >> Edge(
        label="uses runtime", color="red") >> agentcore_runtime
    strands_agents[0] >> Edge(
        label="retrieves context", color="red") >> agentcore_memory
    strands_agents[1] >> Edge(label="executes code",
                              color="red") >> agentcore_sandbox
    strands_agents[2] >> Edge(label="web scraping",
                              color="red") >> agentcore_browser

    # ============================================================================
    # DATA FLOWS - AgentCore to AWS Infrastructure
    # ============================================================================
    agentcore_runtime >> Edge(label="runs on", color="brown") >> ecs_cluster
    ecs_cluster >> Edge(label="manages", color="brown") >> fargate_tasks[0]
    ecs_cluster >> Edge(label="manages", color="brown") >> fargate_tasks[1]

    # AgentCore to DynamoDB
    agentcore_memory >> Edge(label="stores/retrieves",
                             color="darkgreen") >> dynamodb
    agentcore_runtime >> Edge(
        label="agent state", color="darkgreen") >> dynamodb
    convex_backend >> Edge(label="session data",
                           color="darkgreen") >> dynamodb_sessions

    # AgentCore to EventBridge
    agentcore_runtime >> Edge(label="publishes events",
                              color="orange") >> eventbridge
    eventbridge >> Edge(label="triggers", color="orange") >> convex_actions

    # ============================================================================
    # DATA FLOWS - Security & Monitoring
    # ============================================================================
    # IAM governs everything
    iam_policies >> Edge(label="governs", color="darkred",
                         style="dashed") >> ecs_cluster
    iam_policies >> Edge(label="governs", color="darkred",
                         style="dashed") >> dynamodb
    iam_policies >> Edge(label="governs", color="darkred",
                         style="dashed") >> agentcore_runtime
    iam_policies >> Edge(label="governs", color="darkred",
                         style="dashed") >> convex_backend

    # Secrets management
    fargate_tasks[0] >> Edge(label="retrieves secrets",
                             color="purple") >> secrets
    fargate_tasks[1] >> Edge(label="retrieves secrets",
                             color="purple") >> secrets
    convex_backend >> Edge(label="API keys", color="purple") >> secrets

    # Monitoring - All services to CloudWatch
    frontend_app >> Edge(label="logs", color="gray",
                         style="dotted") >> cloudwatch_logs
    convex_backend >> Edge(label="logs", color="gray",
                           style="dotted") >> cloudwatch_logs
    strands_orchestrator >> Edge(
        label="logs", color="gray", style="dotted") >> cloudwatch_logs
    agentcore_runtime >> Edge(
        label="logs", color="gray", style="dotted") >> cloudwatch_logs
    fargate_tasks[0] >> Edge(label="logs", color="gray",
                             style="dotted") >> cloudwatch_logs

    # Metrics
    convex_backend >> Edge(label="metrics", color="gray",
                           style="dotted") >> cloudwatch_metrics
    agentcore_runtime >> Edge(
        label="metrics", color="gray", style="dotted") >> cloudwatch_metrics
    dynamodb >> Edge(label="metrics", color="gray",
                     style="dotted") >> cloudwatch_metrics

print("\n" + "=" * 80)
print("✅ Full Stack Architecture Diagram Generated Successfully!")
print("=" * 80)
print("\n📁 Location: full_stack_architecture.png")
print("\n🏗️ Architecture Components:")
print("  ┌─ Frontend Layer:")
print("  │  ├─ Cloudflare CDN & Edge")
print("  │  └─ React Web Application")
print("  │")
print("  ┌─ Authentication:")
print("  │  ├─ AWS Cognito")
print("  │  ├─ GitHub OAuth")
print("  │  ├─ Google OAuth")
print("  │  └─ Anonymous Access")
print("  │")
print("  ┌─ Backend Layer:")
print("  │  ├─ Convex Backend (Realtime Database)")
print("  │  ├─ Convex Functions (API)")
print("  │  └─ Convex Actions (Server Logic)")
print("  │")
print("  ┌─ AI Agent Layer:")
print("  │  ├─ Strands Agents Framework")
print("  │  │  ├─ Orchestrator")
print("  │  │  ├─ Research Agent")
print("  │  │  ├─ Code Agent")
print("  │  │  └─ Analysis Agent")
print("  │  └─ AWS Bedrock AgentCore")
print("  │     ├─ Runtime (Serverless)")
print("  │     ├─ Memory (Context)")
print("  │     ├─ Sandbox (Code Interpreter)")
print("  │     └─ Browser (Web Interaction)")
print("  │")
print("  └─ AWS Infrastructure:")
print("     ├─ ECS Fargate (Compute)")
print("     ├─ DynamoDB (Agent State & Sessions)")
print("     ├─ EventBridge (Event Routing)")
print("     ├─ IAM (Security)")
print("     ├─ Secrets Manager")
print("     └─ CloudWatch (Monitoring)")
print("\n" + "=" * 80)
