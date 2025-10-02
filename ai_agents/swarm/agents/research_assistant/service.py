"""Research Assistant microservice implementation."""

import asyncio
import logging
from typing import List, Any

from swarm.agents.base_assistant import BaseAssistant
from .prompts import RESEARCH_ASSISTANT_PROMPT, DOCUMENT_ANALYSIS_PROMPT, FACT_CHECKING_PROMPT
from .tools import get_research_tools
from strands.tools.executors import SequentialToolExecutor

logger = logging.getLogger(__name__)

class ResearchAssistant(BaseAssistant):
    """Research Assistant specializing in information gathering and analysis.

    Uses 270M model for fast research tasks:
    - Document search and analysis
    - Fact verification
    - Information synthesis
    - Data extraction
    """

    def __init__(self, assistant_id: str, model_name: str = "gemma:270m",
                 host: str = "localhost:11434"):
        super().__init__(
            assistant_id=assistant_id,
            assistant_type="research",
            capabilities=[
                "document_search",
                "fact_extraction",
                "data_verification",
                "information_synthesis",
                "source_evaluation"
            ],
            model_name=model_name,
            host=host
        )

        # Research-specific state
        self.research_context = {}
        self.fact_database = {}
        self.search_history = []

    def get_system_prompt(self) -> str:
        """Get research assistant system prompt."""
        return RESEARCH_ASSISTANT_PROMPT

    def get_tools(self) -> List[Any]:
        """Get research-specific tools."""
        return get_research_tools()

    async def process_research_task(self, query: str, context: dict = None) -> dict:
        """Process a research task with specialized handling."""
        try:
            # Track search history
            self.search_history.append({
                "query": query,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context
            })

            # Limit history size
            if len(self.search_history) > 50:
                self.search_history = self.search_history[-50:]

            # Build enhanced prompt for research
            if context:
                enhanced_prompt = f"""Research Context: {context}

Research Query: {query}

Focus on:
- Factual accuracy
- Source reliability
- Key insights and patterns
- Data verification
- Knowledge gaps identification

Provide structured findings with confidence levels."""
            else:
                enhanced_prompt = query

            # Process with specialized research prompt
            result = await self.agent.invoke_async(enhanced_prompt)

            return {
                "query": query,
                "findings": result,
                "assistant_type": "research",
                "confidence": self._assess_confidence(result),
                "sources_checked": self._count_sources(result),
                "research_depth": "fast_270m"
            }

        except Exception as e:
            logger.error(f"Research task failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "assistant_type": "research"
            }

    def _assess_confidence(self, result: str) -> float:
        """Simple confidence assessment based on result content."""
        confidence_indicators = [
            "verified", "confirmed", "according to", "data shows",
            "study indicates", "research suggests", "evidence"
        ]

        uncertainty_indicators = [
            "might", "possibly", "unclear", "uncertain", "needs verification",
            "insufficient data", "conflicting", "unknown"
        ]

        result_lower = result.lower()
        confidence_score = 0.5  # baseline

        for indicator in confidence_indicators:
            if indicator in result_lower:
                confidence_score += 0.1

        for indicator in uncertainty_indicators:
            if indicator in result_lower:
                confidence_score -= 0.1

        return max(0.0, min(1.0, confidence_score))

    def _count_sources(self, result: str) -> int:
        """Count potential sources mentioned in result."""
        source_indicators = [
            "source:", "according to", "study", "research", "report",
            "paper", "article", "document", "file:", "reference"
        ]

        result_lower = result.lower()
        source_count = 0

        for indicator in source_indicators:
            source_count += result_lower.count(indicator)

        return source_count

    async def collaborate_research(self, other_agents: List[str],
                                 research_topic: str) -> dict:
        """Collaborate with other agents on research topic."""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        collaboration_id = await self.mcp_client.request_collaboration(
            other_agents,
            f"Research collaboration on: {research_topic}"
        )

        # Perform initial research
        initial_findings = await self.process_research_task(
            research_topic,
            {"collaboration": True, "agents": other_agents}
        )

        return {
            "collaboration_id": collaboration_id,
            "initial_findings": initial_findings,
            "research_topic": research_topic,
            "collaborating_agents": other_agents
        }

    def get_research_stats(self) -> dict:
        """Get research performance statistics."""
        return {
            "searches_performed": len(self.search_history),
            "cached_results": len(self.results_cache),
            "fact_database_size": len(self.fact_database),
            "recent_queries": [
                entry["query"] for entry in self.search_history[-5:]
            ],
            "capabilities": self.capabilities,
            "model": self.model_name
        }

# Factory function for creating research assistant
def create_research_assistant(assistant_id: str = None) -> ResearchAssistant:
    """Create a research assistant instance."""
    if assistant_id is None:
        import uuid
        assistant_id = f"research_{str(uuid.uuid4())[:8]}"

    return ResearchAssistant(assistant_id)

# Example usage and testing
async def demo_research_assistant():
    """Demonstrate research assistant functionality."""
    print("Research Assistant Demo")
    print("=" * 30)

    # Create research assistant
    assistant = create_research_assistant("research_demo_001")

    try:
        # Start the assistant service
        await assistant.start_service()

        # Test research tasks
        test_queries = [
            "What are the latest trends in renewable energy efficiency?",
            "Compare solar panel efficiency across different technologies",
            "Find information about wind turbine capacity factors"
        ]

        for query in test_queries:
            print(f"\nResearching: {query}")
            result = await assistant.process_research_task(query)
            print(f"Findings: {result.get('findings', 'No findings')[:200]}...")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Sources: {result.get('sources_checked', 0)}")

        # Show research stats
        stats = assistant.get_research_stats()
        print(f"\nResearch Stats: {stats}")

        # Test collaboration request
        collaboration_result = await assistant.collaborate_research(
            ["creative_001", "critical_001"],
            "Sustainable energy solutions for urban environments"
        )
        print(f"Collaboration: {collaboration_result}")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await assistant.stop_service()

if __name__ == "__main__":
    asyncio.run(demo_research_assistant())