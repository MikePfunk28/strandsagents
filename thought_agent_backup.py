"""
Thought Agent - Advanced Reasoning with Thinking-First Architecture

This agent implements a comprehensive thinking-first approach where structured
thinking pipelines analyze tasks and intelligently orchestrate agent workflows.
"""

from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands_tools import http_request, handoff_to_user, retrieve, think, use_llm
from strands.models.ollama import OllamaModel
from strands import Agent, tool
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from dotenv import load_dotenv
import os
from mem0 import MemoryClient
from mem0 import Memory
from file_assistant import FileAssistant
from memory_assistant import MemoryAssistant
from meta_tool_assistant import MetaToolAssistant
# Load environment variables at startup
load_dotenv()

# Verify MEM_API_KEY is loaded
MEM_API_KEY = os.getenv('MEM_API_KEY')
if not MEM_API_KEY:
    raise ValueError("MEM_API_KEY environment variable is required")
else:
    client = MemoryClient(api_key=MEM_API_KEY)

WORKFLOW_RUN_OUTPUT_DIR = "workflow_runs"
KNOWLEDGE_CONTEXT_OUTPUT_DIR = "knowledge_context"
KNOWLEDGE_LOG_FILE = "knowledge_base.txt"
MEM0_ENABLED = False
LAST_WORKFLOW_ARTIFACTS: Dict[str, str] = {}

# Core strands imports

# Browser tool setup with proper error handling
try:
    from strands_tools.browser import LocalChromiumBrowser
    import nest_asyncio
    nest_asyncio.apply()
    browser_instance = LocalChromiumBrowser()
    browser_tool = browser_instance.browser
    print("✅ Browser tool initialized successfully")
except (Exception) as e:
    browser_tool = None
    print(
        f"⚠️ Browser tool not available ({e}) - using http_request and retrieve only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create model (keeping user's existing configuration)
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2"
)

# Create a base agent for thinking operations
thinking_agent = Agent(
    model=ollama_model,
    tools=[think],
    conversation_manager=SlidingWindowConversationManager(window_size=30)
)


@tool
def structured_thinking_pipeline(goal: str, context: str = "") -> Dict[str, Any]:
    """
    Execute a comprehensive structured thinking pipeline with multiple reasoning cycles.

    Args:
        goal: The user's goal or query
        context: Additional context information

    Returns:
        Dictionary containing structured thinking results
    """
    logger.info("Starting structured thinking pipeline for goal: %s", goal)

    thinking_results = {}

    # Step 1: Goal Analysis (3 cycles for depth)
    logger.info("Step 1: Goal Analysis")
    thinking_results['goal_analysis'] = thinking_agent.tool.think(
        thought=f"Analyze the goal: {goal}. Context: {context}. What are the core objectives, constraints, and success criteria? What are the key components and requirements?",
        system_prompt="You are an expert analyst. Provide detailed, structured analysis.",
        cycle_count=3
    )

    # Step 2: Task Decomposition (4 cycles for thoroughness)
    logger.info("Step 2: Task Decomposition")
    thinking_results['task_breakdown'] = thinking_agent.tool.think(
        thought=f"Break down the goal into actionable tasks. Consider dependencies, resources needed, potential challenges, and alternative approaches. Goal: {goal}",
        system_prompt="You are a strategic planner. Break down complex goals into manageable tasks.",
        cycle_count=4
    )

    # Step 3: Risk Assessment (3 cycles for comprehensive analysis)
    logger.info("Step 3: Risk Assessment")
    thinking_results['risk_assessment'] = thinking_agent.tool.think(
        thought=f"Identify potential risks, uncertainties, failure modes, and edge cases for: {goal}. What could go wrong? What assumptions am I making?",
        system_prompt="You are a risk assessment expert. Identify all potential issues and uncertainties.",
        cycle_count=3
    )

    # Step 4: Resource Planning (2 cycles for efficiency)
    logger.info("Step 4: Resource Planning")
    thinking_results['resource_planning'] = thinking_agent.tool.think(
        thought=f"What tools, information, capabilities, and agents are needed to accomplish: {goal}? Consider data sources, validation methods, and quality checks.",
        system_prompt="You are a resource planning specialist. Identify required tools and capabilities.",
        cycle_count=2
    )

    # Step 5: Meta-Thinking (2 cycles for self-improvement)
    logger.info("Step 5: Meta-Thinking")
    thinking_results['meta_thinking'] = thinking_agent.tool.think(
        thought=f"Evaluate my thinking quality so far. How thorough was the analysis? What could be improved? Are there any blind spots or biases?",
        system_prompt="You are a meta-cognition expert. Evaluate thinking quality and identify improvements.",
        cycle_count=2
    )

    logger.info("Structured thinking pipeline completed")
    return thinking_results


@tool
def similarity_ranker(content: str, query: str) -> str:
    """Semantic similarity ranking and content filtering."""
    logger.info("Executing similarity ranker for query: %s", query)

    ranker = Agent(
        model=ollama_model,
        system_prompt="""You are a semantic similarity expert:
        1. Rank information by relevance to original query (1-10 scale)
        2. Group similar concepts and identify patterns
        3. Filter low-relevance information (threshold: 6+)
        4. Create attention-weighted summary (focus on high-relevance)
        5. Identify key insights and supporting evidence

        Use sliding window attention for long content."""
    )

    result = str(ranker(f"Rank by similarity to '{query}':\n\n{content}"))
    logger.info("Similarity ranker completed")
    return result



def synthesize_workflow_results(workflow_state: Dict[str, Any]) -> str:
    """Synthesize all workflow results into final output."""
    logger.info("Synthesizing workflow results")

    # Extract key information from workflow state
    original_query = workflow_state.get("original_query", "Unknown query")
    completed_steps = workflow_state.get("completed_steps", [])
    thinking_results = workflow_state.get("thinking_results", {})
    total_steps = workflow_state.get("total_steps", 0)
    errors = workflow_state.get("errors", [])

    # Build comprehensive summary
    summary_parts = []

    # Add original query
    summary_parts.append(f"## Original Query\n{original_query}\n")

    # Add thinking results summary if available
    if thinking_results:
        summary_parts.append("## Thinking Analysis\n")
        for key, value in thinking_results.items():
            if isinstance(value, str) and len(value) > 100:
                summary_parts.append(f"**{key}:** {value[:100]}...\n")
            else:
                summary_parts.append(f"**{key}:** {value}\n")
        summary_parts.append("\n")

    # Add completed steps summary
    if completed_steps:
        summary_parts.append(f"## Workflow Steps ({total_steps} total)\n")
        for step in completed_steps:
            agent = step.get("agent", "Unknown")
            confidence = step.get("confidence", "Unknown")
            reasoning = step.get("reasoning", "No reasoning provided")
            summary_parts.append(f"- **{agent}** (confidence: {confidence})\n")
            summary_parts.append(f"  - Reasoning: {reasoning}\n")
        summary_parts.append("\n")

    # Add errors if any occurred
    if errors:
        summary_parts.append("## Issues Encountered\n")
        for error in errors:
            summary_parts.append(f"- {error}\n")
        summary_parts.append("\n")

    # Add final status
    summary_parts.append("## Final Status\n")
    summary_parts.append("✅ Workflow completed successfully\n")

    final_result = "".join(summary_parts)

    logger.info("Synthesizing final results - %d characters", len(final_result))
    if MEM0_ENABLED:
        try:
            mem = Memory()
            mem.add(
                final_result,
                user_id="demo_user",
                metadata={"source": "workflow_summary"}
            )
        except Exception as mem_error:
            logger.warning("mem0 write skipped: %s", mem_error)

    return final_result




def save_workflow_state(workflow_state: Dict[str, Any]):
    """Save workflow state for analysis and learning."""
    try:
        timestamp = workflow_state.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_state_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(workflow_state, f, indent=2)

        logger.info("Workflow state saved to %s", filename)
    except Exception as e:
        logger.error("Failed to save workflow state: %s", str(e))

    return workflow_state, filename


@tool
def intelligent_workflow_orchestrator(user_query: str, context: str = "", history: List = None) -> Dict[str, Any]:
    """
    Main orchestrator that thinks first, then decides which agent to call next.

    Args:
        user_query: The user's original query
        context: Additional context
        history: Previous interaction history

    Returns:
        Dictionary containing orchestration decision and thinking results
    """
    logger.info(
        "Starting intelligent workflow orchestration for: %s", user_query)

# Step 1: Execute structured thinking pipeline
thinking_results = structured_thinking_pipeline(user_query, context)

logger.info("Step 2: Decision Analysis via MetaToolAssistant")
history_list = history if isinstance(history, list) else []
meta_decision = meta_tool_assistant.select_next_action(
    user_query=user_query,
    context=context,
    history=history_list,
    thinking_results=thinking_results,
    current_results={},
)

next_agent_name = meta_decision.get("next_agent", "planner_agent")
parameters = _build_agent_parameters(
    next_agent_name,
    user_query,
    context,
    thinking_results,
    {},
)

next_action = {
    "next_agent": next_agent_name,
    "parameters": parameters,
    "reasoning": meta_decision.get("reasoning", "Decision based on meta-tool analysis"),
    "confidence": meta_decision.get("confidence", "medium"),
}
if meta_decision.get("follow_up"):
    next_action["follow_up"] = meta_decision["follow_up"]

orchestration_result = {
    "thinking_results": thinking_results,
    "meta_decision": meta_decision,
    "decision": next_action,
    "user_query": user_query,
    "timestamp": datetime.now().isoformat(),
}

logger.info(
    "Orchestration decision: %s with confidence: %s",
    next_action.get("next_agent", "unknown"),
    next_action.get("confidence", "unknown"),
)
return orchestration_result


def parse_decision_text(decision_text: str) -> Dict[str, Any]:
    """
    Parse the thinking decision into actionable format.
    Enhanced parsing logic for better decision extraction.
    """
    # Look for JSON-like structure in the text
    import re

    # Try to extract JSON from the text
    json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback parsing based on keywords
    decision_lower = decision_text.lower()

    if "planner" in decision_lower or "plan" in decision_lower:
        agent = "planner_agent"
    elif "research" in decision_lower or "search" in decision_lower:
        agent = "researcher_agent"
    elif "analyst" in decision_lower or "analyze" in decision_lower:
        agent = "analyst_agent"
    elif "writer" in decision_lower or "write" in decision_lower:
        agent = "writer_agent"
    elif "similarity" in decision_lower or "rank" in decision_lower:
        agent = "similarity_ranker"
    else:
        agent = "planner_agent"  # Default fallback

    return {
        "next_agent": agent,
        "parameters": {"goal": "parsed_goal"},
        "reasoning": decision_text[:200] + "..." if len(decision_text) > 200 else decision_text,
        "confidence": "medium"
    }


@tool
def planner_agent(goal: str, context: str = "") -> str:
    """Creates research brief and step-by-step outline with task board."""
    logger.info("Executing planner agent for goal: %s", goal)

    planner = Agent(
        model=ollama_model,
        tools=[http_request, retrieve],
        system_prompt="""You are the Planner Agent. Create a research brief:
        1. BRIEF: Restate user's goal, key constraints, success criteria
        2. OUTLINE: Step-by-step breakdown (decompose tasks, note data/tool needs, flag unknowns)
        3. TASK BOARD: Goals, subgoals, status tracking
        4. ASSUMPTIONS: Track what's assumed vs verified

        Keep outline visible and update as you learn. Format structured for other agents."""
    )

    result = str(
        planner(f"Create research plan for: {goal}\nContext: {context}"))
    logger.info("Planner agent completed")
    return result


@tool
def researcher_agent(task: str, constraints: str = "") -> str:
    """Executes multi-strategy searches with provenance tracking."""
    logger.info("Executing researcher agent for task: %s", task)

    # Build tools list based on availability
    tools = [http_request, retrieve]
    if browser_tool:
        tools.append(browser_tool)
    logger.info("Researcher tools: %s", [t.__name__ for t in tools])
    researcher = Agent(
        model=ollama_model,
        tools=tools,
        system_prompt="""You are the Researcher Agent. Execute comprehensive research:
        1. Query decomposition (core concepts, specific questions, search strategies)
        2. Multi-strategy search (http_request for APIs, retrieve for web scraping, browser for interactive sites)
        3. Evidence extraction with provenance (cite source for every assertion)
        4. Gap identification and uncertainty flagging
        5. Source credibility assessment
        6. Cross-reference multiple sources for validation

        Tools available:
        - http_request: For API calls and simple web requests
        - retrieve: For web scraping and content extraction
        - browser: For interactive websites requiring JavaScript


        Return: {evidence: [], sources: [], gaps: [], confidence: score, provenance: []}"""
    )

    result = str(researcher(f"Research: {task}\nConstraints: {constraints}"))
    logger.info("Researcher agent completed %s", task)
    return result


@tool
def analyst_agent(findings: str, goal: str, assumptions: str = "") -> str:
    """Critical analysis with logic checking and risk assessment."""
    logger.info("Executing analyst agent for goal: %s", goal)

    analyst = Agent(
        model=ollama_model,
        system_prompt="""You are the Analyst/Critic Agent. Perform critical analysis:
        1. Logic validation (check reasoning chains, spot fallacies)
        2. Evidence quality assessment (source reliability, bias detection)
        3. Gap analysis (missing cases, alternative explanations)
        4. Risk logging (uncertainties, potential errors)
        5. Assumption verification (separate facts from assumptions)

        Ask: What could be wrong? What's unverified? What contradicts? What's missing?"""
    )

    result = str(analyst(
        f"Analyze for goal: {goal}\nFindings: {findings}\nAssumptions: {assumptions}"))
    logger.info("Analyst agent completed %s", goal)
    logger.info("Analyst result length: %d characters", len(result))
    logger.info("Findings %s", findings[:200] +
                "..." if len(findings) > 200 else findings)
    logger.info("Assumptions %s",
                assumptions[:200] + "..." if len(assumptions) > 200 else assumptions)
    return result


@tool
def writer_agent(vetted_reasoning: str, goal: str, sources: str = "") -> str:
    """Synthesizes final report with confidence levels and citations."""
    logger.info("Executing writer agent for goal: %s", goal)
    logger.info("Sources %s", sources[:200] +
                "..." if len(sources) > 200 else sources)

    writer = Agent(
        model=ollama_model,
        system_prompt="""You are the Writer/Synthesizer Agent:
        1. Direct answer to original question (clear, structured)
        2. Supporting evidence with full citations
        3. Confidence levels for key claims (High/Medium/Low)
        4. Limitations and uncertainties acknowledgment
        5. Open issues and areas needing verification

        Format: Executive summary + detailed analysis + sources + confidence assessment"""
    )

    result = str(writer(
        f"Synthesize report for: {goal}\nVetted reasoning: {vetted_reasoning}\nSources: {sources}"))
    logger.info("Writer agent completed reasoning %s", vetted_reasoning)
    return result


file_assistant = FileAssistant(
    run_dir=WORKFLOW_RUN_OUTPUT_DIR,
    context_dir=KNOWLEDGE_CONTEXT_OUTPUT_DIR,
    knowledge_log=KNOWLEDGE_LOG_FILE,
)

memory_assistant = MemoryAssistant(user_id="thought_agent")

meta_tool_assistant = MetaToolAssistant(
    model=ollama_model,
    available_tools={name: meta["description"] for name, meta in AGENT_REGISTRY.items()},
)

knowledge_extraction_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You extract concise, durable memories from context. Output 0-5 bullet facts.
Each fact must be standalone, <=20 words, and omit speculation.""",
)

knowledge_summary_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You maintain a rolling summary of the conversation context.
Keep it under 120 words, preserve key facts, and avoid duplication.""",
)


AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    "planner_agent": {
        "description": "Creates structured plans, decomposes the goal, and identifies next steps.",
    },
    "researcher_agent": {
        "description": "Gathers external information using http_request, retrieve, and browser tools.",
    },
    "analyst_agent": {
        "description": "Critiques findings, surfaces risks, and checks logic gaps.",
    },
    "writer_agent": {
        "description": "Synthesizes final reports with evidence, confidence, and open issues.",
    },
    "similarity_ranker": {
        "description": "Ranks content by semantic similarity to the query for relevance filtering.",
    },
}


def _summarize_thinking_results(thinking_results: Dict[str, Any], max_length: int = 400) -> str:
    if not thinking_results:
        return ""
    parts: List[str] = []
    for key, value in thinking_results.items():
        snippet = str(value)
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        parts.append(f"{key}: {snippet}")
    summary = "\n".join(parts)
    if len(summary) > max_length:
        summary = summary[: max_length - 3] + "..."
    return summary


def _build_agent_parameters(
    agent_name: str,
    user_query: str,
    context: str,
    thinking_results: Dict[str, Any],
    current_results: Dict[str, Any],
) -> Dict[str, Any]:
    thinking_summary = _summarize_thinking_results(thinking_results)

    if agent_name == "planner_agent":
        return {
            "goal": user_query,
            "context": context or thinking_summary,
        }

    if agent_name == "researcher_agent":
        constraints = context.strip()
        if thinking_summary:
            supplemental = f"Planning insights:\n{thinking_summary}"
            constraints = f"{constraints}\n\n{supplemental}" if constraints else supplemental
        return {
            "task": user_query,
            "constraints": constraints or "None",
        }

    if agent_name == "analyst_agent":
        findings = (
            current_results.get("researcher_agent")
            or current_results.get("planner_agent")
            or thinking_summary
            or user_query
        )
        assumptions = context or thinking_summary or "None"
        return {
            "findings": findings,
            "goal": user_query,
            "assumptions": assumptions,
        }

    if agent_name == "writer_agent":
        vetted_reasoning = (
            current_results.get("analyst_agent")
            or thinking_summary
            or user_query
        )
        sources = current_results.get("researcher_agent", "")
        return {
            "vetted_reasoning": vetted_reasoning,
            "goal": user_query,
            "sources": sources,
        }

    if agent_name == "similarity_ranker":
        content = current_results.get("researcher_agent") or thinking_summary or user_query
        return {
            "content": content,
            "query": user_query,
        }

    return {
        "goal": user_query,
        "context": context or thinking_summary,
    }


file_assistant = FileAssistant(
    run_dir=WORKFLOW_RUN_OUTPUT_DIR,
    context_dir=KNOWLEDGE_CONTEXT_OUTPUT_DIR,
    knowledge_log=KNOWLEDGE_LOG_FILE,
)

memory_assistant = MemoryAssistant(user_id="thought_agent")

meta_tool_assistant = MetaToolAssistant(
    model=ollama_model,
    available_tools={name: meta["description"] for name, meta in AGENT_REGISTRY.items()},
)


knowledge_extraction_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You extract concise, durable memories from context. Output 0-5 bullet facts.
Each fact must be standalone, <=20 words, and omit speculation.""",
)

knowledge_summary_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You maintain a rolling summary of the conversation context.
Keep it under 120 words, preserve key facts, and avoid duplication.""",
)



AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    "planner_agent": {
        "description": "Creates structured plans, decomposes the goal, and identifies next steps.",
    },
    "researcher_agent": {
        "description": "Gathers external information using http_request, retrieve, and browser tools.",
    },
    "analyst_agent": {
        "description": "Critiques findings, surfaces risks, and checks logic gaps.",
    },
    "writer_agent": {
        "description": "Synthesizes final reports with evidence, confidence, and open issues.",
    },
    "similarity_ranker": {
        "description": "Ranks content by semantic similarity to the query for relevance filtering.",
    },
}


def _summarize_thinking_results(thinking_results: Dict[str, Any], max_length: int = 400) -> str:
    if not thinking_results:
        return ""
    parts: List[str] = []
    for key, value in thinking_results.items():
        snippet = str(value)
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        parts.append(f"{key}: {snippet}")
    summary = "\n".join(parts)
    if len(summary) > max_length:
        summary = summary[: max_length - 3] + "..."
    return summary


def _build_agent_parameters(
    agent_name: str,
    user_query: str,
    context: str,
    thinking_results: Dict[str, Any],
    current_results: Dict[str, Any],
) -> Dict[str, Any]:
    thinking_summary = _summarize_thinking_results(thinking_results)

    if agent_name == "planner_agent":
        return {
            "goal": user_query,
            "context": context or thinking_summary,
        }

    if agent_name == "researcher_agent":
        constraints = context.strip()
        if thinking_summary:
            supplemental = f"Planning insights:\n{thinking_summary}"
            constraints = f"{constraints}\n\n{supplemental}" if constraints else supplemental
        return {
            "task": user_query,
            "constraints": constraints or "None",
        }

    if agent_name == "analyst_agent":
        findings = (
            current_results.get("researcher_agent")
            or current_results.get("planner_agent")
            or thinking_summary
            or user_query
        )
        assumptions = context or thinking_summary or "None"
        return {
            "findings": findings,
            "goal": user_query,
            "assumptions": assumptions,
        }

    if agent_name == "writer_agent":
        vetted_reasoning = (
            current_results.get("analyst_agent")
            or thinking_summary
            or user_query
        )
        sources = current_results.get("researcher_agent", "")
        return {
            "vetted_reasoning": vetted_reasoning,
            "goal": user_query,
            "sources": sources,
        }

    if agent_name == "similarity_ranker":
        content = current_results.get("researcher_agent") or thinking_summary or user_query
        return {
            "content": content,
            "query": user_query,
        }

    return {
        "goal": user_query,
        "context": context or thinking_summary,
    }


file_assistant = FileAssistant(
    run_dir=WORKFLOW_RUN_OUTPUT_DIR,
    context_dir=KNOWLEDGE_CONTEXT_OUTPUT_DIR,
    knowledge_log=KNOWLEDGE_LOG_FILE,
)

memory_assistant = MemoryAssistant(user_id="thought_agent")

meta_tool_assistant = MetaToolAssistant(
    model=ollama_model,
    available_tools={name: meta["description"] for name, meta in AGENT_REGISTRY.items()},
)

knowledge_extraction_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You extract concise, durable memories from context. Output 0-5 bullet facts.
Each fact must be standalone, <=20 words, and omit speculation.""",
)

knowledge_summary_agent = Agent(
    model=ollama_model,
    tools=[use_llm],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    system_prompt="""You maintain a rolling summary of the conversation context.
Keep it under 120 words, preserve key facts, and avoid duplication.""",
)



AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    "planner_agent": {
        "description": "Creates structured plans, decomposes the goal, and identifies next steps.",
    },
    "researcher_agent": {
        "description": "Gathers external information using http_request, retrieve, and browser tools.",
    },
    "analyst_agent": {
        "description": "Critiques findings, surfaces risks, and checks logic gaps.",
    },
    "writer_agent": {
        "description": "Synthesizes final reports with evidence, confidence, and open issues.",
    },
    "similarity_ranker": {
        "description": "Ranks content by semantic similarity to the query for relevance filtering.",
    },
}


def _summarize_thinking_results(thinking_results: Dict[str, Any], max_length: int = 400) -> str:
    if not thinking_results:
        return ""
    parts: List[str] = []
    for key, value in thinking_results.items():
        snippet = str(value)
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        parts.append(f"{key}: {snippet}")
    summary = "
".join(parts)
    if len(summary) > max_length:
        summary = summary[: max_length - 3] + "..."
    return summary


def _build_agent_parameters(
    agent_name: str,
    user_query: str,
    context: str,
    thinking_results: Dict[str, Any],
    current_results: Dict[str, Any],
) -> Dict[str, Any]:
    thinking_summary = _summarize_thinking_results(thinking_results)

    if agent_name == "planner_agent":
        return {
            "goal": user_query,
            "context": context or thinking_summary,
        }

    if agent_name == "researcher_agent":
        constraints = context.strip()
        if thinking_summary:
            supplemental = f"Planning insights:\n{thinking_summary}"
            constraints = f"{constraints}

{sup supplemental}" if constraints else supplemental
        return {
            "task": user_query,
            "constraints": constraints or "None",
        }

    if agent_name == "analyst_agent":
        findings = current_results.get("researcher_agent") or current_results.get("planner_agent") or thinking_summary or user_query
        assumptions = context or thinking_summary or "None"
        return {
            "findings": findings,
            "goal": user_query,
            "assumptions": assumptions,
        }

    if agent_name == "writer_agent":
        vetted_reasoning = current_results.get("analyst_agent") or thinking_summary or user_query
        sources = current_results.get("researcher_agent", "")
        return {
            "vetted_reasoning": vetted_reasoning,
            "goal": user_query,
            "sources": sources,
        }

    if agent_name == "similarity_ranker":
        content = current_results.get("researcher_agent") or thinking_summary or user_query
        return {
            "content": content,
            "query": user_query,
        }

    return {
        "goal": user_query,
        "context": context or thinking_summary,
    }



def _build_context_sources(
    workflow_state: Dict[str, Any],
    final_result: str,
    rolling_summary: str,
    recent_limit: int = 3,
) -> Dict[str, str]:
    completed = workflow_state.get("completed_steps", []) or []
    recent_steps = completed[-recent_limit:]
    recent_chunks: List[str] = []
    for step in recent_steps:
        agent_name = step.get("agent", "Unknown agent")
        result_payload = step.get("result")
        if not result_payload:
            continue
        recent_chunks.append(f"{agent_name}: {result_payload}")
    return {
        "latest_exchange": final_result.strip(),
        "rolling_summary": (rolling_summary or "").strip(),
        "recent_messages": "\n".join(recent_chunks).strip(),
    }


def _parse_memory_bullets(raw_output: str) -> List[str]:
    memories: List[str] = []
    bullet_prefix_chars = "-*0123456789.) " + chr(8226)
    for line in raw_output.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in {"none", "no facts", "n/a"}:
            return []
        while cleaned and cleaned[0] in bullet_prefix_chars:
            cleaned = cleaned[1:].lstrip()
        cleaned = cleaned.strip()
        if cleaned:
            memories.append(cleaned)
    return memories


def _extract_candidate_memories(context_sources: Dict[str, str]) -> List[str]:
    prompt = """You are a knowledge extraction analyst.
Use the context sources to propose at most five durable memory facts.
Return each fact on its own line prefixed with "- ". If nothing is worth storing, respond with "NONE"."""
    prompt += "\n\nContext Sources:\n"
    prompt += f"1. Latest exchange:\n{context_sources.get('latest_exchange') or 'None'}\n\n"
    prompt += f"2. Rolling summary:\n{context_sources.get('rolling_summary') or 'None'}\n\n"
    prompt += f"3. Recent messages:\n{context_sources.get('recent_messages') or 'None'}"
    try:
        raw = knowledge_extraction_agent.tool.use_llm(prompt=prompt)
    except Exception as exc:
        logger.error("Knowledge extraction failed: %s", exc)
        return []
    memories = _parse_memory_bullets(str(raw))
    return memories[:5]


def _refresh_rolling_summary(existing_summary: str, candidate_memories: List[str]) -> str:
    existing_summary = (existing_summary or "").strip()
    if not candidate_memories:
        return existing_summary
    bullet_block = "\n".join(f"- {fact}" for fact in candidate_memories)
    prompt = """Update the rolling summary with the new facts.
Keep it under 120 words and avoid repeating information."""
    prompt += "\n\nCurrent summary:\n" + (existing_summary or "None")
    prompt += "\n\nNew facts:\n" + bullet_block
    try:
        updated = knowledge_summary_agent.tool.use_llm(prompt=prompt)
    except Exception as exc:
        logger.warning("Rolling summary update failed: %s", exc)
        return existing_summary
    return str(updated).strip()


def _run_knowledge_pipeline(workflow_state: Dict[str, Any], final_result: str) -> Dict[str, Optional[str]]:
    rolling_summary = file_assistant.load_rolling_summary()
    context_sources = _build_context_sources(workflow_state, final_result, rolling_summary)
    candidate_memories = _extract_candidate_memories(context_sources)
    updated_summary = _refresh_rolling_summary(rolling_summary, candidate_memories)

    workflow_state["candidate_memories"] = candidate_memories
    if updated_summary:
        workflow_state["rolling_summary"] = updated_summary

    summary_changed = updated_summary.strip() != (rolling_summary or "").strip()
    snapshot_summary = updated_summary or rolling_summary or ""

    artifacts = file_assistant.persist_artifacts(
        workflow_state=workflow_state,
        final_result=final_result,
        candidate_memories=candidate_memories,
        rolling_summary=snapshot_summary.strip() or None,
        store_summary=summary_changed and bool(snapshot_summary.strip()),
    )

    logger.info("Knowledge pipeline stored %d candidate memories", len(candidate_memories))
    return artifacts

def execute_thinking_driven_workflow(user_query: str, context: str = "", history: List = None) -> str:
    """Execute workflow where thinking determines which agents to call and in what order."""
    logger.info("Starting thinking-driven workflow for: %s", user_query)

    run_started_at = datetime.now()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")

    # Track workflow state
    workflow_state = {
        "original_query": user_query,
        "completed_steps": [],
        "current_results": {},
        "next_actions": [],
        "start_time": run_started_at.isoformat(),
        "run_id": run_id
    }

    # Initial thinking and planning
    orchestration_result = intelligent_workflow_orchestrator(
        user_query, context, history)
    workflow_state["thinking_results"] = orchestration_result["thinking_results"]

    # Execute first agent based on thinking decision
    current_decision = orchestration_result["decision"]
    max_steps = 5  # Prevent infinite loops
    step_count = 0

    while current_decision and step_count < max_steps:
        step_count += 1
        logger.info("Executing step %d: %s", step_count,
                    current_decision['next_agent'])

        try:
            # Execute the chosen agent
            agent_name = current_decision["next_agent"]
            parameters = current_decision["parameters"]

            if agent_name not in AGENT_REGISTRY:
                logger.warning("Unknown agent '%s' selected; defaulting to planner_agent", agent_name)
                agent_name = "planner_agent"
                parameters = _build_agent_parameters(
                    agent_name,
                    user_query,
                    context,
                    workflow_state.get("thinking_results", {}),
                    workflow_state.get("current_results", {}),
                )
                current_decision["next_agent"] = agent_name
                current_decision["parameters"] = parameters

            # Call the appropriate agent
            if agent_name == "planner_agent":
                result = planner_agent(**parameters)
            elif agent_name == "researcher_agent":
                result = researcher_agent(**parameters)
            elif agent_name == "analyst_agent":
                result = analyst_agent(**parameters)
            elif agent_name == "similarity_ranker":
                result = "Similarity ranker not implemented"
            elif agent_name == "writer_agent":
                result = writer_agent(**parameters)
            else:
                result = f"Unknown agent: {agent_name}"

            result_text = str(result)

            # Store results
            workflow_state["completed_steps"].append({
                "step": step_count,
                "agent": agent_name,
                "result": result_text,
                "reasoning": current_decision["reasoning"],
                "confidence": current_decision.get("confidence", "unknown"),
                "timestamp": datetime.now().isoformat()
            })

            workflow_state["current_results"][agent_name] = result_text
            workflow_state["last_decision"] = current_decision

            # For now, just break after first step to avoid complexity
            break

        except Exception as e:
            logger.error("Error in workflow step %d: %s", step_count, str(e))
            workflow_state["errors"] = workflow_state.get(
                "errors", []) + [str(e)]
            break

    workflow_state["total_steps"] = step_count
    workflow_state["end_time"] = datetime.now().isoformat()

    final_result = synthesize_workflow_results(workflow_state)
    workflow_state["final_result"] = final_result

    workflow_state, workflow_state_path = save_workflow_state(workflow_state)
    artifact_paths = _run_knowledge_pipeline(workflow_state, final_result)
    artifact_paths["workflow_state"] = workflow_state_path
    workflow_state["artifact_paths"] = artifact_paths

    logger.info("Workflow artifacts saved: %s", artifact_paths)
    logger.info("Thinking-driven workflow completed in %d steps", step_count)

    global LAST_WORKFLOW_ARTIFACTS
    LAST_WORKFLOW_ARTIFACTS = artifact_paths

    return final_result


if __name__ == "__main__":
    print("\n🧠 Advanced Thinking-First Reasoning Agent")
    print("=" * 60)
    print("Multi-agent reasoning workflow with:")
    print("• Structured thinking pipeline")
    print("• Intelligent workflow orchestration")
    print("• Dynamic agent selection")
    print("• Comprehensive logging and analysis")
    print("• Meta-thinking capabilities")
    print("=" * 60)

    # Store results for comparison
    research_history = []

    while True:
        try:
            user_input = input(
                "\nWhat would you like me to research? (type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            # Check if this is a repeat query
            is_repeat = any(entry['query'].lower() == user_input.lower()
                            for entry in research_history)

            print(f"\n🔍 Starting thinking-first workflow for: '{user_input}'")
            if is_repeat:
                print("🔄 Repeat query detected - will compare with previous results")
            print(
                "📋 Workflow: Think → Plan → Research → Analyze → Rank → Synthesize → Report")

            # Execute thinking-driven workflow
            result = execute_thinking_driven_workflow(user_input)

            print("\n" + "="*70)
            print("🧠 ADVANCED THINKING-FIRST REASONING REPORT")
            print("="*70)
            print(result)
            print("\n" + "="*70)

            # Store result for comparison
            research_entry = {
                'query': user_input,
                'result': str(result),
                'timestamp': datetime.now().isoformat(),
                'result_length': len(str(result)),
                'artifacts': dict(LAST_WORKFLOW_ARTIFACTS)
            }
            research_history.append(research_entry)

            artifacts = research_entry['artifacts']
            if artifacts:
                full_run_path = artifacts.get("full_run")
                context_path = artifacts.get("context")
                knowledge_log_path = artifacts.get("knowledge_log")
                workflow_state_path = artifacts.get("workflow_state")

                if full_run_path:
                    print(f"Saved full workflow transcript to: {full_run_path}")
                if context_path:
                    print(f"Saved distilled knowledge context to: {context_path}")
                if workflow_state_path:
                    print(f"Workflow state JSON: {workflow_state_path}")
                if knowledge_log_path:
                    print(f"Knowledge log updated: {knowledge_log_path}")

            # Compare with previous results if repeat
            if is_repeat:
                previous_results = [entry for entry in research_history[:-1]
                                    if entry['query'].lower() == user_input.lower()]
                if previous_results:
                    print("\n🔄 COMPARISON WITH PREVIOUS RESULT(S):")
                    print(
                        f"Current result length: {len(str(result))} characters")
                    for prev in previous_results:
                        print(
                            f"Previous result length: {prev['result_length']} characters")
                    print(
                        "Note: Results may vary due to different thinking paths and timing.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error("Error in main loop: %s", str(e))
            print(f"\nError: {str(e)}")
            print("Please try a different question.")
