from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import diagram, think, http_request, handoff_to_user, retrieve, file_read, file_write, editor
from strands.agent.conversation_manager import SlidingWindowConversationManager
import logging
from dotenv import load_dotenv
import os
# from mem0 import MemoryClient
from datetime import datetime
import json

# Load environment variables at startup
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Try to import browser tool safely
try:
    from strands_tools.browser import LocalChromiumBrowser
    import nest_asyncio
    nest_asyncio.apply()
    browser_instance = LocalChromiumBrowser()
    browser_tool = browser_instance.browser  # Use the .browser attribute
except (ImportError, AttributeError) as e:
    browser_tool = None
    print(
        f"Browser tool not available ({e}) - using http_request and retrieve only")


# Create model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:4b"
)

# Creating Orchestrator Agent
# Define the orchestrator system prompt with clear tool selection guidance
MAIN_SYSTEM_PROMPT = """
You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_assistant tool
- For product recommendations and shopping advice → Use the product_recommendation_assistant tool
- For travel planning and itineraries → Use the trip_planning_assistant tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.
"""


@tool
def planner_agent(goal: str, context: str = "") -> str:
    """Creates research brief and step-by-step outline with task board."""
    logger.info(
        "Planner Agent invoked with goal %s:  and context length: %d", goal, len(context))
    planner = Agent(
        model=ollama_model,
        tools=[think],
        conversation_manager=SlidingWindowConversationManager(window_size=25),
        system_prompt="""You are the Planner Agent. Create a research brief:
        1. BRIEF: Restate user's goal, key constraints, success criteria
        2. OUTLINE: Step-by-step breakdown (decompose tasks, note data/tool needs, flag unknowns)
        3. TASK BOARD: Goals, subgoals, status tracking
        4. ASSUMPTIONS: Track what's assumed vs verified

        Keep outline visible and update as you learn. Format structured for other agents."""
    )
    return str(planner(f"Create research plan for: {goal}\nContext: {context}"))


@tool
def researcher_agent(task: str, constraints: str = "") -> str:
    """Executes multi-strategy searches with provenance tracking."""
    logger.info(
        f"Researcher Agent invoked with task %s:  and constraints length: {len(constraints)}", task)
    # Build tools list based on availability
    tools = [http_request, retrieve]
    if browser_tool:
        tools.append(browser_tool)

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
    return str(researcher(f"Research: {task}\nConstraints: {constraints}"))


@tool
def analyst_agent(findings: str, goal: str, assumptions: str = "") -> str:
    """Critical analysis with logic checking and risk assessment."""
    logger.info(
        f"Analyst Agent invoked with goal %s:  and findings length: {len(findings)}", goal)
    analyst = Agent(
        model=ollama_model,
        conversation_manager=SlidingWindowConversationManager(window_size=20),
        system_prompt="""You are the Analyst/Critic Agent. Perform critical analysis:
        1. Logic validation (check reasoning chains, spot fallacies)
        2. Evidence quality assessment (source reliability, bias detection)
        3. Gap analysis (missing cases, alternative explanations)
        4. Risk logging (uncertainties, potential errors)
        5. Assumption verification (separate facts from assumptions)

        Ask: What could be wrong? What's unverified? What contradicts? What's missing?"""
    )
    return str(analyst(f"Analyze for goal: {goal}\nFindings: {findings}\nAssumptions: {assumptions}"))


@tool
def similarity_ranker(content: str, query: str) -> str:
    """Semantic similarity ranking and content filtering."""
    logger.info(
        f"Similarity Ranker invoked with query %s:  and content length: {len(content)}", query)
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
    return str(ranker(f"Rank by similarity to '{query}':\n\n{content}"))


@tool
def writer_agent(vetted_reasoning: str, goal: str, sources: str = "") -> str:
    """Synthesizes final report with confidence levels and citations."""
    logger.info(
        f"Writer Agent invoked with goal %s:  and vetted_reasoning length: {len(vetted_reasoning)}", goal)
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
    return str(writer(f"Synthesize report for: {goal}\nVetted reasoning: {vetted_reasoning}\nSources: {sources}"))


@tool
def diagram_agent(description: str) -> str:
    """
    Create a diagram based on the provided description.

    :param description: The description of the diagram to create.
    :return: The generated diagram as a string (e.g., URL or file path).
    """
    # create a diagram from the results of the research
    # use the diagram tool to do so, using the research file
    # research_output_{timestamp}.txt
    diagram_filename = f"research_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger.info("Creating diagram for description: %s", description)
    diagram_builder = Agent(
        model=ollama_model,
        tools=[diagram],
        conversation_manager=SlidingWindowConversationManager(window_size=10),
        system_prompt="""You are the Diagram Creation Agent. Generate diagrams based on Research findings:
        1. Interpret the description to identify key components and relationships.
        2. Use the diagram tool to create a visual representation.
        3. Ensure clarity and accuracy in the diagram.
        4. Provide a brief explanation of the diagram created.

        Format: Diagram + Explanation"""
    )
    return str(diagram_builder(f"Create diagram for: {description}, {diagram_filename}"))


def similarity_rank(content: str, query: str) -> str:
    """
    Semantic similarity ranking and content filtering.

    :param content: The content to be ranked and filtered.
    :param query: The query used for ranking.
    :return: The ranked and filtered content.

    """
    logger.info(
        f"Similarity Ranker invoked with query %s:  and content length: {len(content)}", query)
    # define code for a similarity ranker
    similarity_search = Agent(
        model=ollama_model,
        system_prompt="""You are a semantic similarity expert:
        1. Rank information by relevance to original query (1-10 scale)
        2. Group similar concepts and identify patterns
        3. Filter low-relevance information (threshold: 6+)
        4. Create attention-weighted summary (focus on high-relevance)
        5. Identify key insights and supporting evidence

        Use sliding window attention for long content."""
    )
    return similarity_search(content=content, query=query)


def reasoning_result(user_query: str, context: str) -> str:
    """
    Execute the comprehensive reasoning research workflow.

    :param user_query: The user's research query.
    :return: The final synthesized report.
    """
    context = knowledge_base_memory() + "\n\n" + context
    # Master reasoning agent with advanced workflow
    orchestrator = Agent(
        model=ollama_model,

        tools=[planner_agent, researcher_agent, analyst_agent,
               similarity_ranker, writer_agent, diagram_agent, handoff_to_user],
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        system_prompt="""You are the Master Reasoning Agent. You MUST execute ALL 6 steps of the workflow systematically:

        MANDATORY WORKFLOW - EXECUTE ALL STEPS:
        1. PLAN: Call planner_agent(goal=user_query, context="") - CREATE research brief
        2. RESEARCH: Call researcher_agent(task=plan_output, constraints="") - GATHER evidence
        3. ANALYZE: Call analyst_agent(findings=research_output, goal=user_query, assumptions="") - CRITIQUE findings
        4. RANK: Call similarity_ranker(content=analysis_output, query=user_query) - FILTER content
        5. SYNTHESIZE: Call writer_agent(vetted_reasoning=ranked_output, goal=user_query, sources="") - FINAL report

        DO NOT STOP until all 6 tools have been called. Show your thinking between each step.

        PARAMETER PASSING:
        - Always pass the original user query as 'goal' parameter
        - Pass previous agent outputs as content parameters
        - Include context and constraints where relevant

        SHARED MEMORY & TRACKING:
        - Maintain task board (goals, subgoals, status)
        - Track assumptions vs facts with sources
        - Keep reasoning transcript for auditability
        - Update outline as you learn

        QUALITY CONTROL:
        - Force citation of sources for all assertions
        - Separate assumptions from verified facts
        - Generate checklists before finalizing
        - Break loops when analyst confirms acceptance criteria met
        - Use sliding window attention for long content

        ADVANCED FEATURES:
        - Query decomposition and multi-strategy search
        - Semantic similarity ranking and filtering
        - Logic validation and bias detection
        - Confidence scoring and uncertainty tracking
        - Provenance tracking for all evidence

        You must complete the entire workflow. Think step-by-step and use each tool in sequence.""",
    )
    return orchestrator(f"Execute comprehensive reasoning research workflow for: {user_query}")

# Knowledge Base Memory- Need to make sure to only get what
# is relevant to the current query


def knowledge_base_memory():
    """
    Load and return the knowledge base memory.
    Creates knowledgebase.txt if it doesn't exist.

    :return: The knowledge base memory content.
    """
    base_memory = "knowledgebase.txt"

    # Check if file exists first
    if not os.path.exists(base_memory):
        logger.info("Knowledge base file not found, creating new one")
        try:
            # Create the file with initial content
            with open(base_memory, "w", encoding="utf-8") as f:
                initial_content = "Knowledge Base Initialized.\n"
                f.write(initial_content)
            logger.info("Created knowledgebase.txt with %d characters",
                        (len(initial_content),))
            return initial_content
        except Exception as e:
            logger.error("❌ Failed to create knowledgebase.txt: %s", e)
            return "Error: Could not create knowledge base file."

    # File exists, try to read it
    try:
        with open(base_memory, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(
                " Loaded knowledge base with %d characters", len(content))
            return content
    except Exception as e:
        logger.error("❌ Failed to read knowledgebase.txt: %s", e)
        return "Error: Could not read knowledge base file."


def append_to_knowledge_base(content: str, category: str = "general"):
    """
    Append important information to the knowledge base with categorization.

    :param content: The content to store
    :param category: The category/topic of the information
    :return: Success status
    """
    try:
        base_memory = "knowledgebase.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the entry with metadata
        entry = f"\n[{timestamp}] [{category.upper()}]\n{content}\n{'='*50}\n"

        with open(base_memory, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.info(
            "Added %d characters to knowledge base in category '%s'", len(content), category)
        return True
    except Exception as e:
        logger.error("Failed to append to knowledge base: %s", e)
        return False


def retrieve_memory(query: str, context: str) -> str:
    """
    Retrieve relevant information from the knowledge base.

    :param query: The query to search for in the knowledge base.
    :return: The relevant information from the knowledge base.
    """
    logger.info(
        "Retrieving information from knowledge base for query: %s", query)
    # Retrieve relevant information from the knowledge base
    # This is a placeholder and should be replaced with actual knowledge base retrieval
    knowledge_base = knowledge_base_memory()
    return knowledge_base


def save_research_output(query: str, result):
    """
    Save research results to a timestamped file for comparison.

    :param query: The research query
    :param result: The research result (can be AgentResult or string)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{timestamp}.txt"

        # Convert result to string if it's an AgentResult object
        if hasattr(result, '__str__'):
            result_str = str(result)
        else:
            result_str = str(result)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Result Length: {len(result_str)} characters\n")
            f.write("="*70 + "\n\n")
            f.write(result_str)

        logger.info("Saved research output to %s", filename)
        return filename
    except Exception as e:
        logger.error("Failed to save research output: %s", e)
        return None


if __name__ == "__main__":
    print("\n Advanced Reasoning Research Agent")
    print("Multi-agent reasoning workflow with query decomposition, similarity ranking, and provenance tracking.")

    # Store results for comparison
    research_history = []
    reasoning_agent = Agent(
        model=ollama_model,
        tools=[planner_agent, researcher_agent, analyst_agent,
               similarity_ranker, writer_agent, diagram_agent, handoff_to_user],
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        system_prompt="""You are the Master Reasoning Agent. You MUST execute ALL 6 steps of the workflow systematically:"""
    )
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

            print(
                f"\n Starting advanced reasoning workflow for: '{user_input}'")
            if is_repeat:
                print("🔄 Repeat query detected - will compare with previous results")
            print(" Workflow: Plan → Research → Analyze → Rank → Synthesize → Report")

            result = reasoning_agent(
                f"Execute comprehensive reasoning research workflow for: {user_input}")

            print("\n" + "="*70)
            print(" ADVANCED REASONING RESEARCH REPORT")
            print("="*70)
            print(result)
            print("\n" + "="*70)

            # Store research result in knowledge base
            append_to_knowledge_base(
                f"Research Query: {user_input}", "research")
            append_to_knowledge_base(f"Research Result: {result}", "findings")

            # Save research output to file for comparison
            output_file = save_research_output(user_input, result)
            if output_file:
                print(f"📄 Research output saved to: {output_file}")

            # Store result for comparison
            research_entry = {
                'query': user_input,
                'result': str(result),
                'timestamp': len(research_history) + 1
            }
            research_history.append(research_entry)

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
                            f"Previous result #{prev['timestamp']} length: {len(prev['result'])} characters")
                    print(
                        "Note: Results may vary due to different web sources and timing.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try a different question.")
