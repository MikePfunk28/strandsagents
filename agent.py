from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import http_request, handoff_to_user, retrieve
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Try to import browser tool safely
try:
    from strands_tools.browser import LocalChromiumBrowser
    import nest_asyncio
    nest_asyncio.apply()
    browser_instance = LocalChromiumBrowser()
    browser_tool = browser_instance.browser  # Use the .browser attribute
except (ImportError, AttributeError) as e:
    browser_tool = None
    print(f"Browser tool not available ({e}) - using http_request and retrieve only")

# Create model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2"
)

@tool
def planner_agent(goal: str, context: str = "") -> str:
    """Creates research brief and step-by-step outline with task board."""
    planner = Agent(
        model=ollama_model,
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

# Master reasoning agent with advanced workflow
reasoning_agent = Agent(
    model=ollama_model,
    tools=[planner_agent, researcher_agent, analyst_agent, similarity_ranker, writer_agent, handoff_to_user],
    conversation_manager=SlidingWindowConversationManager(window_size=40),
    system_prompt="""You are the Master Reasoning Agent. You MUST execute ALL 5 steps of the workflow systematically:

    MANDATORY WORKFLOW - EXECUTE ALL STEPS:
    1. PLAN: Call planner_agent(goal=user_query, context="") - CREATE research brief
    2. RESEARCH: Call researcher_agent(task=plan_output, constraints="") - GATHER evidence
    3. ANALYZE: Call analyst_agent(findings=research_output, goal=user_query, assumptions="") - CRITIQUE findings
    4. RANK: Call similarity_ranker(content=analysis_output, query=user_query) - FILTER content
    5. SYNTHESIZE: Call writer_agent(vetted_reasoning=ranked_output, goal=user_query, sources="") - FINAL report

    DO NOT STOP until all 5 tools have been called. Show your thinking between each step.

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

    You must complete the entire workflow. Think step-by-step and use each tool in sequence."""
)

if __name__ == "__main__":
    print("\n🧠 Advanced Reasoning Research Agent")
    print("Multi-agent reasoning workflow with query decomposition, similarity ranking, and provenance tracking.")

    # Store results for comparison
    research_history = []

    while True:
        try:
            user_input = input("\nWhat would you like me to research? (type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            # Check if this is a repeat query
            is_repeat = any(entry['query'].lower() == user_input.lower() for entry in research_history)

            print(f"\n🔍 Starting advanced reasoning workflow for: '{user_input}'")
            if is_repeat:
                print("🔄 Repeat query detected - will compare with previous results")
            print("📋 Workflow: Plan → Research → Analyze → Rank → Synthesize → Report")

            result = reasoning_agent(f"Execute comprehensive reasoning research workflow for: {user_input}")

            print("\n" + "="*70)
            print("🧠 ADVANCED REASONING RESEARCH REPORT")
            print("="*70)
            print(result)
            print("\n" + "="*70)

            # Store result for comparison
            research_entry = {
                'query': user_input,
                'result': str(result),
                'timestamp': len(research_history) + 1
            }
            research_history.append(research_entry)

            # Compare with previous results if repeat
            if is_repeat:
                previous_results = [entry for entry in research_history[:-1] if entry['query'].lower() == user_input.lower()]
                if previous_results:
                    print("\n🔄 COMPARISON WITH PREVIOUS RESULT(S):")
                    print(f"Current result length: {len(str(result))} characters")
                    for prev in previous_results:
                        print(f"Previous result #{prev['timestamp']} length: {len(prev['result'])} characters")
                    print("Note: Results may vary due to different web sources and timing.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try a different question.")