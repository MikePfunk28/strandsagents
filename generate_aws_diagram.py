"""
Direct AWS Infrastructure Diagram Generator
Creates the AWS production infrastructure diagram.
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS, Fargate
from diagrams.aws.network import ALB, InternetGateway, VPC
from diagrams.aws.storage import S3
from diagrams.aws.security import SecretsManager, IAM, KMS
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.client import User

print("Generating AWS Production Infrastructure Diagram...")

with Diagram("AWS Production Infrastructure", show=False, direction="LR", filename="aws_production_infrastructure", outformat="png"):
    # Users/Clients
    users = User("End Users")

    # Network Layer
    igw = InternetGateway("Internet Gateway")

    # VPC and Networking
    with Cluster("VPC"):
        with Cluster("Public Subnet"):
            alb = ALB("Application\nLoad Balancer")

            # ECS/Fargate Cluster
            with Cluster("ECS Fargate Cluster"):
                ecs = ECS("ECS Service")
                fargate1 = Fargate("Fargate Task 1")
                fargate2 = Fargate("Fargate Task 2")

    # Storage and Secrets Layer
    with Cluster("Storage & Secrets"):
        s3 = S3("S3 Buckets\n(Encrypted at Rest)")
        kms = KMS("KMS\nEncryption Keys")
        secrets = SecretsManager("Secrets Manager\n(Credentials)")

    # Security and Monitoring Layer
    with Cluster("Security & Monitoring"):
        iam = IAM("IAM\nLeast-Privilege\nPolicies")
        cloudwatch = Cloudwatch("CloudWatch\nMonitoring & Alarms")

    # Main Data Flow: Users -> Internet -> VPC -> Load Balancer -> ECS
    users >> Edge(label="HTTPS", color="blue") >> igw
    igw >> Edge(label="traffic", color="blue") >> alb
    alb >> Edge(label="routes", color="blue") >> ecs

    # ECS to Fargate Tasks
    ecs >> Edge(label="manages", color="green") >> fargate1
    ecs >> Edge(label="manages", color="green") >> fargate2

    # Fargate Tasks to S3 Storage
    fargate1 >> Edge(label="read/write", color="orange") >> s3
    fargate2 >> Edge(label="read/write", color="orange") >> s3

    # Fargate Tasks to Secrets Manager
    fargate1 >> Edge(label="retrieve secrets", color="purple") >> secrets
    fargate2 >> Edge(label="retrieve secrets", color="purple") >> secrets

    # Encryption: S3 and Secrets encrypted by KMS
    s3 >> Edge(label="encrypted by", color="red", style="dashed") >> kms
    secrets >> Edge(label="encrypted by", color="red", style="dashed") >> kms

    # Monitoring: All services send metrics to CloudWatch
    fargate1 >> Edge(label="metrics & logs", color="gray",
                     style="dotted") >> cloudwatch
    fargate2 >> Edge(label="metrics & logs", color="gray",
                     style="dotted") >> cloudwatch
    alb >> Edge(label="metrics", color="gray", style="dotted") >> cloudwatch
    s3 >> Edge(label="metrics", color="gray", style="dotted") >> cloudwatch

    # IAM Policies control access to all services
    iam >> Edge(label="governs", color="darkred", style="dashed") >> ecs
    iam >> Edge(label="governs", color="darkred", style="dashed") >> s3
    iam >> Edge(label="governs", color="darkred", style="dashed") >> secrets
    iam >> Edge(label="governs", color="darkred", style="dashed") >> cloudwatch

print("\n✅ Diagram generated successfully!")
print("📁 Location: aws_production_infrastructure.png")
print("\nThis diagram shows:")
print("  • S3 encryption at rest (KMS)")
print("  • Least-privilege IAM policies")
print("  • Cost monitoring alarms (CloudWatch)")
print("  • Secrets management")
print("  • VPC with public subnet for Fargate")
print("  • ECS Fargate cluster ready for containers")
