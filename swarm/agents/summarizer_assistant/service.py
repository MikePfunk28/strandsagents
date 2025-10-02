"""Summarizer Assistant microservice implementation."""

import asyncio
import logging
from typing import List, Any, Dict

from ...base_assistant import BaseAssistant
from .prompts import SUMMARIZER_ASSISTANT_PROMPT, MEETING_SUMMARY_PROMPT, PROGRESS_TRACKING_PROMPT
from .tools import get_summarizer_tools

logger = logging.getLogger(__name__)

class SummarizerAssistant(BaseAssistant):
    """Summarizer Assistant specializing in content synthesis and distillation.

    Uses 270M model for fast summarization tasks:
    - Text summarization and synthesis
    - Key point extraction
    - Meeting notes creation
    - Progress tracking
    """

    def __init__(self, assistant_id: str, model_name: str = "gemma:270m",
                 host: str = "localhost:11434"):
        super().__init__(
            assistant_id=assistant_id,
            assistant_type="summarizer",
            capabilities=[
                "text_summarization",
                "key_extraction",
                "multi_source_synthesis",
                "meeting_notes",
                "progress_tracking",
                "status_reporting"
            ],
            model_name=model_name,
            host=host
        )

        # Summarization-specific state
        self.summary_history = []
        self.meeting_notes = []
        self.progress_reports = []
        self.synthesis_templates = ["executive", "technical", "status", "decision"]

    def get_system_prompt(self) -> str:
        """Get summarizer assistant system prompt."""
        return SUMMARIZER_ASSISTANT_PROMPT

    def get_tools(self) -> List[Any]:
        """Get summarization tools."""
        return get_summarizer_tools()

    async def create_summary(self, content: str, context: dict = None,
                           summary_type: str = "comprehensive") -> dict:
        """Create structured summary of content."""
        try:
            # Select prompt based on summary type
            if summary_type == "meeting":
                system_prompt = MEETING_SUMMARY_PROMPT
                summary_prompt = f"Meeting Summary: {content}"
            elif summary_type == "progress":
                system_prompt = PROGRESS_TRACKING_PROMPT
                summary_prompt = f"Progress Tracking: {content}"
            else:
                system_prompt = self.get_system_prompt()
                summary_prompt = self._build_summary_prompt(content, context, summary_type)

            # Temporarily switch system prompt for focused summarization
            original_prompt = self.agent.system_prompt
            self.agent.system_prompt = system_prompt

            result = await self.agent.run_async(summary_prompt)

            # Restore original system prompt
            self.agent.system_prompt = original_prompt

            # Generate summary metrics
            summary_metrics = self._generate_summary_metrics(result, content)

            # Store summary in history
            summary_entry = {
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "summary_type": summary_type,
                "summary": result,
                "metrics": summary_metrics,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context
            }
            self.summary_history.append(summary_entry)

            # Limit history size
            if len(self.summary_history) > 100:
                self.summary_history = self.summary_history[-100:]

            return {
                "content_length": len(content),
                "summary_type": summary_type,
                "summary": result,
                "compression_ratio": summary_metrics["compression_ratio"],
                "key_points_count": summary_metrics["key_points"],
                "clarity_score": summary_metrics["clarity_score"],
                "assistant_type": "summarizer"
            }

        except Exception as e:
            logger.error(f"Summary creation failed: {e}")
            return {
                "content_length": len(content),
                "summary_type": summary_type,
                "error": str(e),
                "assistant_type": "summarizer"
            }

    def _build_summary_prompt(self, content: str, context: dict, summary_type: str) -> str:
        """Build enhanced summary prompt with context and type."""
        base_prompt = f"Content to Summarize: {content}\n\n"

        if context:
            base_prompt += f"Context: {context}\n\n"

        type_instructions = {
            "comprehensive": "Create comprehensive summary with key points, insights, and action items",
            "executive": "Create executive summary focusing on high-level insights and decisions",
            "technical": "Create technical summary focusing on implementation details and specifications",
            "brief": "Create brief summary highlighting only the most essential points",
            "structured": "Create structured summary with clear sections and hierarchy"
        }

        instruction = type_instructions.get(summary_type, type_instructions["comprehensive"])
        base_prompt += f"Summary Type: {instruction}\n\n"

        # Add summarization guidelines
        base_prompt += """Summarization Guidelines:
1. Preserve essential information
2. Use clear, structured format
3. Include key metrics and data
4. Highlight action items
5. Maintain logical flow
6. Use headings and bullet points
7. Ensure accuracy and completeness

Create well-structured, actionable summary."""

        return base_prompt

    def _generate_summary_metrics(self, summary: str, original_content: str) -> dict:
        """Generate metrics for summary quality."""
        original_words = len(original_content.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0

        # Count structural elements
        lines = summary.split('\n')
        bullet_points = len([line for line in lines if line.strip().startswith('•') or line.strip().startswith('-')])
        headings = len([line for line in lines if line.strip().isupper() and len(line.strip()) > 2])

        # Clarity indicators
        clarity_indicators = [
            "key", "important", "main", "primary", "essential",
            "summary", "overview", "conclusion", "result"
        ]
        summary_lower = summary.lower()
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in summary_lower)

        # Calculate clarity score
        clarity_score = min(1.0, (clarity_count + bullet_points + headings) / 20)

        return {
            "compression_ratio": compression_ratio,
            "key_points": bullet_points,
            "headings": headings,
            "clarity_score": clarity_score,
            "structure_elements": bullet_points + headings
        }

    async def synthesize_agent_outputs(self, agent_outputs: List[Dict[str, Any]],
                                     synthesis_focus: str = "comprehensive") -> dict:
        """Synthesize outputs from multiple agents into coherent summary."""
        try:
            if not agent_outputs:
                return {"error": "No agent outputs to synthesize"}

            # Prepare sources for synthesis
            sources = []
            for i, output in enumerate(agent_outputs):
                agent_type = output.get("assistant_type", f"agent_{i}")
                content = ""

                # Extract content from different output formats
                if "result" in output:
                    content = str(output["result"])
                elif "ideas" in output:
                    content = str(output["ideas"])
                elif "evaluation" in output:
                    content = str(output["evaluation"])
                elif "findings" in output:
                    content = str(output["findings"])
                else:
                    content = str(output)

                sources.append({
                    "name": f"{agent_type.title()} Output",
                    "content": content
                })

            synthesis_prompt = f"""Multi-Agent Output Synthesis:

Sources: {len(sources)} agent outputs
Focus: {synthesis_focus}

Agent Outputs:
"""

            for source in sources:
                synthesis_prompt += f"\n{source['name']}:\n{source['content'][:500]}...\n"

            synthesis_prompt += f"""

Create comprehensive synthesis that:
1. Identifies common themes and patterns
2. Highlights complementary insights
3. Resolves any conflicts or contradictions
4. Extracts actionable recommendations
5. Provides unified conclusion

Focus on {synthesis_focus} perspective."""

            result = await self.agent.run_async(synthesis_prompt)

            # Store synthesis result
            synthesis_entry = {
                "agent_count": len(agent_outputs),
                "synthesis_focus": synthesis_focus,
                "synthesis": result,
                "timestamp": asyncio.get_event_loop().time()
            }

            return {
                "agent_count": len(agent_outputs),
                "synthesis_focus": synthesis_focus,
                "synthesis": result,
                "unified_insights": self._extract_unified_insights(result),
                "assistant_type": "summarizer"
            }

        except Exception as e:
            logger.error(f"Agent output synthesis failed: {e}")
            return {
                "agent_count": len(agent_outputs),
                "synthesis_focus": synthesis_focus,
                "error": str(e),
                "assistant_type": "summarizer"
            }

    def _extract_unified_insights(self, synthesis: str) -> List[str]:
        """Extract unified insights from synthesis."""
        lines = synthesis.split('\n')
        insights = []

        for line in lines:
            line = line.strip()
            # Look for insight indicators
            if any(indicator in line.lower() for indicator in
                   ['insight:', 'conclusion:', 'key finding:', 'important:', 'recommendation:']):
                insights.append(line)
            elif line.startswith('•') and len(line) > 20:
                insights.append(line)

        return insights[:10]  # Limit to 10 insights

    async def create_progress_report(self, progress_data: Dict[str, Any],
                                   timeframe: str = "current") -> dict:
        """Create structured progress report."""
        try:
            progress_prompt = f"""Progress Report Generation:

Timeframe: {timeframe}
Progress Data: {progress_data}

Create structured progress report with:
1. Executive Summary
2. Key Accomplishments
3. Current Status
4. Metrics and KPIs
5. Challenges and Blockers
6. Next Steps
7. Timeline Updates

Format as professional progress report."""

            result = await self.agent.run_async(progress_prompt)

            # Store progress report
            report_entry = {
                "timeframe": timeframe,
                "report": result,
                "data_points": len(progress_data) if isinstance(progress_data, dict) else 0,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.progress_reports.append(report_entry)

            # Limit reports size
            if len(self.progress_reports) > 50:
                self.progress_reports = self.progress_reports[-50:]

            return {
                "timeframe": timeframe,
                "progress_report": result,
                "data_points_analyzed": report_entry["data_points"],
                "report_sections": self._count_report_sections(result),
                "assistant_type": "summarizer"
            }

        except Exception as e:
            logger.error(f"Progress report creation failed: {e}")
            return {
                "timeframe": timeframe,
                "error": str(e),
                "assistant_type": "summarizer"
            }

    def _count_report_sections(self, report: str) -> int:
        """Count the number of sections in a report."""
        section_indicators = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "SUMMARY", "STATUS", "NEXT"]
        return sum(1 for indicator in section_indicators if indicator in report.upper())

    async def collaborate_on_summary(self, other_agents: List[str],
                                   content_to_summarize: str) -> dict:
        """Collaborate with other agents on summarization tasks."""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        collaboration_id = await self.mcp_client.request_collaboration(
            other_agents,
            f"Collaborative summarization: {content_to_summarize[:100]}..."
        )

        # Create initial summary
        initial_summary = await self.create_summary(
            content_to_summarize,
            {"collaboration": True, "agents": other_agents},
            "comprehensive"
        )

        return {
            "collaboration_id": collaboration_id,
            "initial_summary": initial_summary,
            "content_length": len(content_to_summarize),
            "collaborating_agents": other_agents
        }

    def get_summarizer_stats(self) -> dict:
        """Get summarization performance statistics."""
        total_summaries = len(self.summary_history)
        avg_compression = sum(
            summary["metrics"]["compression_ratio"]
            for summary in self.summary_history
        ) / max(1, total_summaries)

        avg_clarity = sum(
            summary["metrics"]["clarity_score"]
            for summary in self.summary_history
        ) / max(1, total_summaries)

        return {
            "summaries_created": total_summaries,
            "progress_reports": len(self.progress_reports),
            "meeting_notes": len(self.meeting_notes),
            "average_compression_ratio": avg_compression,
            "average_clarity_score": avg_clarity,
            "summary_types_used": list(set(summary["summary_type"] for summary in self.summary_history)),
            "capabilities": self.capabilities,
            "model": self.model_name
        }

# Factory function for creating summarizer assistant
def create_summarizer_assistant(assistant_id: str = None) -> SummarizerAssistant:
    """Create a summarizer assistant instance."""
    if assistant_id is None:
        import uuid
        assistant_id = f"summarizer_{str(uuid.uuid4())[:8]}"

    return SummarizerAssistant(assistant_id)

# Example usage and testing
async def demo_summarizer_assistant():
    """Demonstrate summarizer assistant functionality."""
    print("Summarizer Assistant Demo")
    print("=" * 30)

    assistant = create_summarizer_assistant("summarizer_demo_001")

    try:
        await assistant.start_service()

        # Test content summarization
        test_content = """
        The quarterly meeting discussed several important topics including budget allocations,
        project timelines, and resource planning. Key decisions were made regarding the
        development roadmap for Q4. The team agreed to prioritize feature development
        over bug fixes, with a target of 85% test coverage. Budget approval was given
        for additional developer resources. Next steps include finalizing technical
        specifications and beginning implementation planning.
        """

        print("\n1. Content Summarization:")
        result = await assistant.create_summary(
            test_content,
            {"meeting_type": "quarterly", "department": "engineering"},
            "structured"
        )
        print(f"Compression Ratio: {result.get('compression_ratio', 0):.2f}")
        print(f"Clarity Score: {result.get('clarity_score', 0):.2f}")

        # Test multi-agent synthesis
        print("\n2. Multi-Agent Synthesis:")
        mock_outputs = [
            {"assistant_type": "research", "findings": "Market analysis shows 25% growth potential"},
            {"assistant_type": "creative", "ideas": "Three innovative approaches for user engagement"},
            {"assistant_type": "critical", "evaluation": "Risk assessment identifies implementation challenges"}
        ]

        synthesis_result = await assistant.synthesize_agent_outputs(
            mock_outputs,
            "strategic_planning"
        )
        print(f"Synthesized {synthesis_result.get('agent_count', 0)} agent outputs")

        # Test progress report
        print("\n3. Progress Report:")
        progress_data = {
            "completed_tasks": 8,
            "pending_tasks": 3,
            "blockers": 1,
            "milestone_progress": "75%"
        }

        progress_result = await assistant.create_progress_report(
            progress_data,
            "weekly"
        )
        print(f"Report Sections: {progress_result.get('report_sections', 0)}")

        # Show summarizer stats
        stats = assistant.get_summarizer_stats()
        print(f"\nSummarizer Stats: {stats}")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await assistant.stop_service()

if __name__ == "__main__":
    asyncio.run(demo_summarizer_assistant())