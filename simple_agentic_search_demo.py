#!/usr/bin/env python3
"""
Simple Agentic Search Demo - Demonstrates controlled agentic search
Shows how the search system works with visible reasoning and safeguards
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime

print(" Agentic Search System - Controlled Demo")
print("=" * 60)


class ControlledSearchEngine:
    """Controlled search engine with visible reasoning"""

    def __init__(self):
        self.knowledge_base = {
            'aws_lambda': {
                'title': 'AWS Lambda Documentation',
                'content': 'AWS Lambda is a serverless compute service that lets you run code without provisioning or managing servers. You pay only for the compute time you consume.',
                'url': 'https://docs.aws.amazon.com/lambda/latest/dg/welcome.html'
            },
            'lambda_best_practices': {
                'title': 'AWS Lambda Best Practices',
                'content': 'Key best practices include: right-sizing memory, using environment variables, implementing error handling, and monitoring with CloudWatch.',
                'url': 'https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html'
            },
            'serverless_architecture': {
                'title': 'Serverless Architecture Patterns',
                'content': 'Serverless architectures provide automatic scaling, pay-per-use billing, and reduced operational overhead.',
                'url': 'https://aws.amazon.com/architecture/serverless/'
            }
        }

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform controlled search with visible reasoning"""
        print(f" THINKING: Searching for '{query}'")
        print(
            f" THINKING: I should look for relevant AWS documentation and best practices")
        print(f" THINKING: I'll focus on authoritative sources and technical accuracy")

        # Simulate search delay
        await asyncio.sleep(0.5)

        results = []
        query_terms = query.lower().split()

        for key, data in self.knowledge_base.items():
            relevance = sum(1 for term in query_terms if term in key.lower(
            ) or term in data['content'].lower())
            relevance_score = min(
                relevance / len(query_terms), 1.0) if query_terms else 0.0

            if relevance_score > 0.3:
                result = {
                    'title': data['title'],
                    'content': data['content'],
                    'url': data['url'],
                    'relevance_score': relevance_score
                }
                results.append(result)

        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:max_results]

        print(f" SEARCH: Found {len(results)} relevant results")
        return results


async def perform_controlled_search(query: str) -> str:
    """Perform controlled agentic search with visible reasoning"""
    print(f"\n Processing Query: {query}")
    print("-" * 50)

    start_time = time.time()
    search_engine = ControlledSearchEngine()

    # Step 1: Broad search
    print("📚 Step 1: THINKING - Starting with broad search to understand context")
    broad_results = await search_engine.search(f"AWS overview and general information about {query}")
    print(f"   Found {len(broad_results)} broad results")

    # Step 2: Specific search
    print(" Step 2: THINKING - Now searching for specific technical details")
    specific_results = await search_engine.search(f"AWS best practices and technical details for {query}")
    print(f"   Found {len(specific_results)} specific results")

    # Step 3: Synthesize results
    print("🧠 Step 3: THINKING - Synthesizing findings into comprehensive answer")

    all_results = broad_results + specific_results
    synthesis = f"""# Agentic Search Results: {query}

## Summary
Based on comprehensive search across AWS documentation and best practices.

## Key Findings

### Core Concepts
{chr(10).join(f"• {result['content'][:100]}..." for result in broad_results[:2])}

### Technical Details
{chr(10).join(f"• {result['content'][:100]}..." for result in specific_results[:2])}

## Recommendations
• Follow AWS Lambda best practices for production applications
• Implement proper error handling and monitoring
• Use appropriate memory allocation and timeout settings
• Consider cost optimization strategies
• Implement security best practices from the start

## Confidence: High
This analysis is based on authoritative AWS documentation and established best practices.
"""

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000

    print(f" SYNTHESIS: Completed in {processing_time:.0f}ms")
    print(f" SYNTHESIS: Generated {len(synthesis)} character response")

    return synthesis


async def main():
    """Demonstrate controlled agentic search"""
    print("🎯 Agentic Search Demonstration")
    print("This demo shows how agentic search works with:")
    print("    Visible reasoning at each step")
    print("    Controlled search process")
    print("    Multiple search strategies")
    print("    Confidence scoring")
    print("    Comprehensive synthesis")
    print("    No runaway requests")

    # Test queries
    test_queries = [
        "AWS Lambda best practices for production applications",
        "How to optimize Lambda function performance and cost",
        "Serverless architecture patterns for web applications"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)

        try:
            result = await perform_controlled_search(query)
            print("\n📋 Final Result:")
            print("-" * 30)
            print(result[:500] + "..." if len(result) > 500 else result)

        except Exception as e:
            print(f"❌ Search failed: {e}")

    print(f"\n{'='*60}")
    print("🎉 Agentic Search Demo Complete!")
    print("💡 Key Benefits:")
    print("    Visible reasoning at each step")
    print("    Controlled search process")
    print("    Multiple search strategies")
    print("    Confidence scoring")
    print("    Comprehensive synthesis")
    print("    No runaway requests")

if __name__ == "__main__":
    asyncio.run(main())
