#!/usr/bin/env python3
"""
Demonstration script for the agent.agentic_search module.

Runs several sample queries through the AgenticSearchEngine and prints the
step-by-step reasoning along with the final answer. The demo intentionally
keeps the knowledge base local so it can be executed without network access.
"""

from __future__ import annotations

import asyncio
from textwrap import indent

from agent.agentic_search import AgenticSearchEngine, SearchOutcome


KNOWLEDGE_BASE = [
    {
        "title": "AWS Lambda Overview",
        "content": "AWS Lambda provides serverless compute that integrates well "
        "with Bedrock agents, allowing secure invocation of backend logic.",
        "relevance": 0.92,
        "url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
        "metadata": {"tags": ["lambda", "aws", "serverless"], "source": "AWS Documentation"},
    },
    {
        "title": "AWS Bedrock Agents Guide",
        "content": "Bedrock Agents can orchestrate multi-step workflows and call "
        "Lambda functions for custom actions such as data access or analytics.",
        "relevance": 0.89,
        "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
        "metadata": {"tags": ["bedrock", "agent"], "source": "AWS Documentation"},
    },
    {
        "title": "Lambda Cost Optimisation",
        "content": "Right-size memory, use provisioned concurrency only when required, "
        "and monitor usage with CloudWatch to keep AWS Lambda costs predictable.",
        "relevance": 0.84,
        "metadata": {"tags": ["cost", "lambda"], "source": "AWS Best Practices"},
    },
]


async def run_demo() -> None:
    engine = AgenticSearchEngine(knowledge_base=KNOWLEDGE_BASE)

    queries = [
        "How do I connect an AWS Bedrock agent to a Lambda function?",
        "What are the cost considerations for running Lambda with agents?",
        "Outline the architecture for an AWS serverless chatbot using Bedrock.",
    ]

    for index, query in enumerate(queries, start=1):
        print("\n" + "=" * 80)
        print(f"Query {index}: {query}")
        print("=" * 80)

        outcome: SearchOutcome = await engine.search(query)

        # Print reasoning steps
        for step in outcome.steps:
            header = f"[{step.status.upper()}] {step.description}"
            print(header)
            for evidence in step.findings:
                bullet = f"- {evidence.title}: {evidence.content}"
                print(indent(bullet, "    "))

        # Print final answer
        print("\nFinal Answer")
        print("-" * 12)
        print(outcome.answer)
        print(f"\nConfidence: {outcome.confidence:.1%}")


if __name__ == "__main__":
    asyncio.run(run_demo())
