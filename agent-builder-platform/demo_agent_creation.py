#!/usr/bin/env python3
"""
Demo Agent Creation - Demonstrates @agent decorator system
Shows how to create, build, and deploy agents using the deterministic approach
"""

from strandsagents.agent_decorator import (
    agent, system_prompt, tool, function,
    mcp, aws_service, environment_variable, deployment_config
)
import asyncio
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the @agent decorator system

print(" Agent Builder Platform - @agent Decorator System Demo")
print("=" * 70)

# Example 1: Simple Chatbot Agent
print("\n📦 Creating Customer Support Chatbot...")


@agent(
    name="customer_support_bot",
    type="chatbot",
    description="AI-powered customer support agent with AWS Bedrock integration"
)
@system_prompt("""
You are a helpful and professional customer support agent.
Always be polite, accurate, and helpful in your responses.
If you don't know something, admit it and offer to escalate to a human.
Keep responses clear and concise.
Ask clarifying questions when needed.
""")
@tool('bedrock', 'dynamodb')
@aws_service('bedrock', 'dynamodb', 'cloudwatch')
@environment_variable(
    LOG_LEVEL='INFO',
    AWS_REGION='us-east-1',
    SUPPORT_EMAIL='support@company.com'
)
class CustomerSupportBot:
    """Example customer support chatbot"""

    @function("check_order_status")
    async def check_order_status(self, order_id: str) -> str:
        """Check the status of a customer order"""
        # This would integrate with your order management system
        return f"Order {order_id} is currently being processed and will be shipped within 2-3 business days."

    @function("escalate_to_human")
    async def escalate_to_human(self, reason: str) -> str:
        """Escalate complex issues to human support"""
        return f"I've escalated this issue to our human support team. Reason: {reason}. You should hear from them within 24 hours."

    @function("get_faq_answer")
    async def get_faq_answer(self, question: str) -> str:
        """Get answer from FAQ database"""
        faq_responses = {
            "hours": "Our customer support hours are Monday-Friday 9AM-6PM EST.",
            "returns": "You can return items within 30 days with original packaging.",
            "shipping": "Standard shipping takes 3-5 business days.",
            "payment": "We accept all major credit cards and PayPal."
        }

        for key, response in faq_responses.items():
            if key in question.lower():
                return response

        return "I don't have that specific information in our FAQ. Let me escalate this to a human agent."


# Example 2: API Agent
print("\n📦 Creating Product API Agent...")


@agent(
    name="product_api_agent",
    type="api",
    description="REST API agent for product management"
)
@system_prompt("""
You are an API agent that processes product-related requests.
Validate all inputs and provide structured responses.
Handle errors gracefully and provide helpful error messages.
Log all requests for monitoring and debugging.
""")
@tool('dynamodb', 's3')
@aws_service('dynamodb', 's3', 'api_gateway', 'lambda')
class ProductAPIAgent:
    """Example API agent for product management"""

    @function("get_product")
    async def get_product(self, product_id: str) -> dict:
        """Get product information"""
        return {
            "product_id": product_id,
            "name": "Sample Product",
            "price": 29.99,
            "available": True,
            "description": "High-quality product with excellent features"
        }

    @function("update_inventory")
    async def update_inventory(self, product_id: str, quantity: int) -> bool:
        """Update product inventory"""
        return True

    @function("search_products")
    async def search_products(self, query: str) -> list:
        """Search for products"""
        return [
            {"id": "1", "name": "Product 1", "price": 19.99},
            {"id": "2", "name": "Product 2", "price": 29.99}
        ]


# Example 3: Data Processing Agent
print("\n📦 Creating Data Processing Agent...")


@agent(
    name="data_processor_agent",
    type="data_processing",
    description="Agent for processing and analyzing data"
)
@system_prompt("""
You are a data processing agent specialized in data analysis and transformation.
Process data efficiently and provide insights.
Handle large datasets and complex transformations.
Validate data quality and provide cleaning recommendations.
""")
@tool('s3', 'dynamodb', 'lambda')
@aws_service('s3', 'dynamodb', 'lambda', 'glue', 'athena')
class DataProcessorAgent:
    """Example data processing agent"""

    @function("process_csv_data")
    async def process_csv_data(self, s3_bucket: str, file_key: str) -> dict:
        """Process CSV data from S3"""
        return {
            "status": "processed",
            "rows_processed": 1000,
            "columns_analyzed": 15,
            "data_quality_score": 0.92,
            "insights": [
                "Data quality is good with 92% completeness",
                "No significant outliers detected",
                "Data follows expected patterns"
            ]
        }

    @function("generate_report")
    async def generate_report(self, data_summary: dict) -> str:
        """Generate analysis report"""
        return f"Data analysis complete. Processed {data_summary.get('rows_processed', 0)} rows with {data_summary.get('data_quality_score', 0):.0%} quality score."


# Example 4: Custom Agent with All Features
print("\n📦 Creating Enterprise Assistant...")


@agent(
    name="enterprise_assistant",
    type="custom",
    description="Enterprise-grade AI assistant with full feature set"
)
@system_prompt("""
You are an enterprise AI assistant with advanced capabilities.
You have access to multiple tools and can handle complex tasks.
Always maintain security and compliance standards.
Provide detailed, accurate responses with proper error handling.
""")
@tool('bedrock', 'dynamodb', 's3', 'lambda', 'cloudwatch')
@aws_service('bedrock', 'dynamodb', 's3', 'lambda', 'cloudwatch', 'iam', 'kms')
@mcp('aws_documentation', 'aws_pricing', 'github_analysis', 'strands_patterns')
@environment_variable(
    LOG_LEVEL='DEBUG',
    AWS_REGION='us-east-1',
    ENTERPRISE_MODE='true',
    COMPLIANCE_FRAMEWORK='SOC2'
)
@deployment_config(
    memory_size=1024,
    timeout=600,
    environment='prod',
    auto_scaling=True,
    monitoring_enabled=True
)
class EnterpriseAssistant:
    """Example enterprise agent with all features"""

    @function("analyze_requirements")
    async def analyze_requirements(self, requirements: str) -> dict:
        """Analyze and validate requirements"""
        return {
            "analysis": "Requirements analyzed successfully",
            "confidence": 0.95,
            "recommendations": [
                "Use serverless architecture for cost efficiency",
                "Implement comprehensive monitoring",
                "Follow security best practices"
            ]
        }

    @function("generate_architecture")
    async def generate_architecture(self, use_case: str) -> dict:
        """Generate architecture recommendations"""
        return {
            "architecture": "Serverless microservices",
            "services": ["Lambda", "API Gateway", "DynamoDB", "S3"],
            "estimated_cost": "$50-100/month",
            "scalability": "Auto-scaling based on demand"
        }

    @function("deploy_solution")
    async def deploy_solution(self, architecture: dict) -> str:
        """Deploy the solution to AWS"""
        return "Solution deployed successfully with monitoring and alerting configured"


async def main():
    """Demonstrate the @agent decorator system"""
    print("\n🎯 Demo: Creating and Building Agents")
    print("-" * 50)

    try:
        # Import the agent builder
        from build_agent import AgentBuilder

        builder = AgentBuilder()

        # Build the customer support bot
        print("\n Building Customer Support Bot...")
        output_dir = await builder.build_agent(
            agent_type="chatbot",
            name="customer_support_bot",
            description="AI-powered customer support agent",
            system_prompt="You are a helpful customer support agent. Always be polite and professional.",
            tools=["bedrock", "dynamodb"],
            aws_services=["bedrock", "dynamodb", "cloudwatch"]
        )

        print(f" Customer Support Bot built in: {output_dir}")

        # Build the product API agent
        print("\n Building Product API Agent...")
        output_dir2 = await builder.build_agent(
            agent_type="api",
            name="product_api_agent",
            description="REST API agent for product management",
            tools=["dynamodb", "s3"],
            aws_services=["dynamodb", "s3", "api_gateway"]
        )

        print(f" Product API Agent built in: {output_dir2}")

        # Build the data processor agent
        print("\n Building Data Processor Agent...")
        output_dir3 = await builder.build_agent(
            agent_type="data_processing",
            name="data_processor_agent",
            description="Agent for processing and analyzing data",
            tools=["s3", "dynamodb", "lambda"],
            aws_services=["s3", "dynamodb", "lambda", "glue", "athena"]
        )

        print(f" Data Processor Agent built in: {output_dir3}")

        print("\n🎉 All agents built successfully!")

        # Show what was created
        print("\n📁 Generated Files Structure:")
        for i, output_dir in enumerate([output_dir, output_dir2, output_dir3], 1):
            print(f"\nAgent {i} ({output_dir.split('/')[-1]}):")
            try:
                files = os.listdir(output_dir)
                for file in files:
                    print(f"   📄 {file}")
            except:
                print("   📄 Generated agent files")

        print("\n Next Steps:")
        print("   1. Review the generated agent files")
        print("   2. Customize the agent code if needed")
        print("   3. Deploy to AWS using: ./scripts/deploy.sh")
        print("   4. Test your agents with the API endpoints")
        print("   5. Monitor costs and performance in AWS Console")

        print("\n💡 Example Usage:")
        print("   # Test the chatbot")
        print("   python -c \"from generated_agents/customer_support_bot/agent import CustomerSupportBotAgent; import asyncio; asyncio.run(CustomerSupportBotAgent().process_message('Hello'))\"")

        print("\n   # Deploy to AWS")
        print("   cd generated_agents/customer_support_bot")
        print("   ./scripts/deploy.sh prod us-east-1")

        print("\n🎯 Key Benefits of @agent Decorator System:")
        print("    Deterministic: Same inputs = same outputs")
        print("    Scriptable: Can be automated and batched")
        print("    Customizable: Easy to modify and extend")
        print("    Fast: No consultation delays")
        print("    Powerful: Clean API with decorators")
        print("    Production-Ready: Built for AWS deployment")
        print("    Cost-Effective: Optimized for hackathon budgets")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if dependencies aren't installed yet.")

        print("\n📝 To run the demo manually:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run: python demo_agent_creation.py")
        print("   3. Build agents: python build_agent.py --type chatbot --name my_agent")
        print("   4. Deploy: cd generated_agents/my_agent && ./scripts/deploy.sh")

if __name__ == "__main__":
    asyncio.run(main())
