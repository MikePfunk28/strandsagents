"""Creative Assistant microservice implementation."""

import asyncio
import logging
import random
from typing import List, Any, Dict

from swarm.agents.base_assistant import BaseAssistant
from .prompts import CREATIVE_ASSISTANT_PROMPT, BRAINSTORMING_PROMPT, INNOVATION_PROMPT
from .tools import get_creative_tools

logger = logging.getLogger(__name__)

class CreativeAssistant(BaseAssistant):
    """Creative Assistant specializing in ideation and innovation.

    Uses 270M model for fast creative tasks:
    - Idea generation and brainstorming
    - Creative problem solving
    - Innovation suggestions
    - Concept combinations
    """

    def __init__(self, assistant_id: str, model_name: str = "gemma:270m",
                 host: str = "localhost:11434"):
        super().__init__(
            assistant_id=assistant_id,
            assistant_type="creative",
            capabilities=[
                "idea_generation",
                "brainstorming",
                "creative_combination",
                "problem_reframing",
                "innovation_thinking",
                "concept_development"
            ],
            model_name=model_name,
            host=host
        )

        # Creative-specific state
        self.idea_bank = []
        self.inspiration_sources = []
        self.brainstorming_sessions = []
        self.creative_modes = ["divergent", "convergent", "lateral", "associative"]

    def get_system_prompt(self) -> str:
        """Get creative assistant system prompt."""
        return CREATIVE_ASSISTANT_PROMPT

    def get_tools(self) -> List[Any]:
        """Get creative-specific tools."""
        return get_creative_tools()

    async def generate_ideas(self, prompt: str, context: dict = None,
                           creative_mode: str = "divergent") -> dict:
        """Generate creative ideas with specified mode."""
        try:
            # Select prompt based on creative mode
            if creative_mode == "brainstorming":
                system_prompt = BRAINSTORMING_PROMPT
            elif creative_mode == "innovation":
                system_prompt = INNOVATION_PROMPT
            else:
                system_prompt = self.get_system_prompt()

            # Build enhanced creative prompt
            enhanced_prompt = self._build_creative_prompt(prompt, context, creative_mode)

            # Temporarily switch system prompt for this task
            original_prompt = self.agent.system_prompt
            self.agent.system_prompt = system_prompt

            result = await self.agent.invoke_async(enhanced_prompt)

            # Restore original system prompt
            self.agent.system_prompt = original_prompt

            # Store ideas in idea bank
            idea_entry = {
                "prompt": prompt,
                "mode": creative_mode,
                "ideas": result,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context
            }
            self.idea_bank.append(idea_entry)

            # Limit idea bank size
            if len(self.idea_bank) > 100:
                self.idea_bank = self.idea_bank[-100:]

            return {
                "prompt": prompt,
                "creative_mode": creative_mode,
                "ideas": result,
                "idea_count": self._count_ideas(result),
                "novelty_score": self._assess_novelty(result),
                "assistant_type": "creative"
            }

        except Exception as e:
            logger.error(f"Idea generation failed: {e}")
            return {
                "prompt": prompt,
                "error": str(e),
                "assistant_type": "creative"
            }

    def _build_creative_prompt(self, prompt: str, context: dict, mode: str) -> str:
        """Build enhanced creative prompt with context and mode."""
        base_prompt = f"Creative Challenge: {prompt}\n\n"

        if context:
            base_prompt += f"Context: {context}\n\n"

        mode_instructions = {
            "divergent": "Generate as many diverse ideas as possible. Think wild and unconventional!",
            "convergent": "Focus on refining and developing the most promising ideas.",
            "lateral": "Use lateral thinking - make unexpected connections and associations.",
            "associative": "Build ideas by associating with different domains and concepts.",
            "brainstorming": "Rapid-fire idea generation - quantity over quality initially.",
            "innovation": "Focus on disruptive, paradigm-shifting innovations."
        }

        instruction = mode_instructions.get(mode, mode_instructions["divergent"])
        base_prompt += f"Creative Mode: {instruction}\n\n"

        # Add inspiration triggers
        if len(self.inspiration_sources) > 0:
            recent_sources = self.inspiration_sources[-3:]
            base_prompt += f"Draw inspiration from: {', '.join(recent_sources)}\n\n"

        base_prompt += "Generate multiple creative options with brief explanations."

        return base_prompt

    def _count_ideas(self, result: str) -> int:
        """Count the number of ideas in the result."""
        # Look for numbered lists, bullet points, or line breaks
        idea_patterns = [r'^\d+\.', r'^•', r'^-', r'^\*']
        lines = result.split('\n')
        idea_count = 0

        for line in lines:
            line = line.strip()
            if line:
                for pattern in idea_patterns:
                    import re
                    if re.match(pattern, line):
                        idea_count += 1
                        break

        # If no structured list found, estimate based on content
        if idea_count == 0:
            sentences = result.split('.')
            idea_count = max(1, len([s for s in sentences if len(s.strip()) > 20]) // 2)

        return idea_count

    def _assess_novelty(self, result: str) -> float:
        """Assess the novelty of generated ideas."""
        novelty_indicators = [
            "unprecedented", "revolutionary", "breakthrough", "innovative",
            "disruptive", "unconventional", "unique", "original", "creative",
            "new approach", "never before", "reimagine", "transform"
        ]

        common_indicators = [
            "typical", "standard", "conventional", "usual", "traditional",
            "common", "basic", "simple", "obvious", "expected"
        ]

        result_lower = result.lower()
        novelty_score = 0.5  # baseline

        for indicator in novelty_indicators:
            if indicator in result_lower:
                novelty_score += 0.05

        for indicator in common_indicators:
            if indicator in result_lower:
                novelty_score -= 0.05

        # Assess idea diversity (different topics/approaches)
        unique_words = set(result_lower.split())
        if len(unique_words) > 50:  # Rich vocabulary suggests diverse thinking
            novelty_score += 0.1

        return max(0.0, min(1.0, novelty_score))

    async def brainstorm_session(self, topic: str, participants: List[str] = None,
                               duration_minutes: int = 10) -> dict:
        """Conduct a structured brainstorming session."""
        session_id = f"brainstorm_{len(self.brainstorming_sessions) + 1}"

        session = {
            "session_id": session_id,
            "topic": topic,
            "participants": participants or [self.assistant_id],
            "start_time": asyncio.get_event_loop().time(),
            "duration_minutes": duration_minutes,
            "phases": []
        }

        try:
            # Phase 1: Divergent thinking
            divergent_result = await self.generate_ideas(
                f"Brainstorm ideas for: {topic}",
                {"session": session_id, "phase": "divergent"},
                "divergent"
            )
            session["phases"].append({"phase": "divergent", "result": divergent_result})

            # Phase 2: Build on ideas
            build_prompt = f"Build on and expand these initial ideas about {topic}: {divergent_result.get('ideas', '')[:500]}"
            build_result = await self.generate_ideas(
                build_prompt,
                {"session": session_id, "phase": "building"},
                "associative"
            )
            session["phases"].append({"phase": "building", "result": build_result})

            # Phase 3: Wild ideas
            wild_prompt = f"Generate wild, unconventional ideas for {topic} - no limits!"
            wild_result = await self.generate_ideas(
                wild_prompt,
                {"session": session_id, "phase": "wild"},
                "lateral"
            )
            session["phases"].append({"phase": "wild", "result": wild_result})

            session["end_time"] = asyncio.get_event_loop().time()
            session["total_ideas"] = sum(phase["result"].get("idea_count", 0) for phase in session["phases"])

            self.brainstorming_sessions.append(session)

            return {
                "session_id": session_id,
                "topic": topic,
                "total_ideas": session["total_ideas"],
                "phases_completed": len(session["phases"]),
                "session_summary": f"Generated {session['total_ideas']} ideas across {len(session['phases'])} creative phases",
                "assistant_type": "creative"
            }

        except Exception as e:
            logger.error(f"Brainstorming session failed: {e}")
            return {
                "session_id": session_id,
                "topic": topic,
                "error": str(e),
                "assistant_type": "creative"
            }

    async def collaborate_creatively(self, other_agents: List[str],
                                   creative_challenge: str) -> dict:
        """Collaborate with other agents on creative challenges."""
        if not self.mcp_client:
            return {"error": "MCP client not initialized"}

        collaboration_id = await self.mcp_client.request_collaboration(
            other_agents,
            f"Creative collaboration: {creative_challenge}"
        )

        # Generate initial creative concepts
        initial_ideas = await self.generate_ideas(
            creative_challenge,
            {"collaboration": True, "agents": other_agents},
            "innovation"
        )

        return {
            "collaboration_id": collaboration_id,
            "initial_ideas": initial_ideas,
            "creative_challenge": creative_challenge,
            "collaborating_agents": other_agents
        }

    def add_inspiration_source(self, source: str):
        """Add a new inspiration source."""
        self.inspiration_sources.append(source)
        if len(self.inspiration_sources) > 20:
            self.inspiration_sources = self.inspiration_sources[-20:]

    def get_creative_stats(self) -> dict:
        """Get creative performance statistics."""
        return {
            "ideas_generated": len(self.idea_bank),
            "brainstorming_sessions": len(self.brainstorming_sessions),
            "inspiration_sources": len(self.inspiration_sources),
            "creative_modes_used": list(set(idea["mode"] for idea in self.idea_bank)),
            "average_novelty": sum(idea.get("novelty_score", 0.5) for idea in self.idea_bank) / max(1, len(self.idea_bank)),
            "capabilities": self.capabilities,
            "model": self.model_name
        }

# Factory function for creating creative assistant
def create_creative_assistant(assistant_id: str = None) -> CreativeAssistant:
    """Create a creative assistant instance."""
    if assistant_id is None:
        import uuid
        assistant_id = f"creative_{str(uuid.uuid4())[:8]}"

    return CreativeAssistant(assistant_id)

# Example usage and testing
async def demo_creative_assistant():
    """Demonstrate creative assistant functionality."""
    print("Creative Assistant Demo")
    print("=" * 30)

    assistant = create_creative_assistant("creative_demo_001")

    try:
        await assistant.start_service()

        # Test idea generation
        print("\n1. Generating Ideas:")
        result = await assistant.generate_ideas(
            "How to make remote work more engaging and collaborative",
            {"domain": "workplace", "constraints": "budget-friendly"},
            "divergent"
        )
        print(f"Generated {result.get('idea_count', 0)} ideas")
        print(f"Novelty score: {result.get('novelty_score', 0):.2f}")

        # Test brainstorming session
        print("\n2. Brainstorming Session:")
        session_result = await assistant.brainstorm_session(
            "Sustainable urban transportation solutions",
            ["creative_001", "research_001"],
            5
        )
        print(f"Session: {session_result.get('session_summary', 'No summary')}")

        # Add inspiration and test
        assistant.add_inspiration_source("biomimicry")
        assistant.add_inspiration_source("gaming mechanics")

        print("\n3. Inspired Ideas:")
        inspired_result = await assistant.generate_ideas(
            "Improve online learning experiences",
            {"inspiration": True},
            "lateral"
        )
        print(f"Ideas with inspiration: {inspired_result.get('idea_count', 0)}")

        # Show creative stats
        stats = assistant.get_creative_stats()
        print(f"\nCreative Stats: {stats}")

    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await assistant.stop_service()

if __name__ == "__main__":
    asyncio.run(demo_creative_assistant())