from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import http_request
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Create model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2"
)

@tool
def planner_agent(goal: str, context: str = "") -> str:
    """Orchestrates research workflow and maintains task outline."""
    planner = Agent(
        model=ollama_model,
        conversation_manager=SlidingWindowConversationManager(window_size=25),
        system_prompt="""You are the Planner Agent. Your role:
        1. Restate user's goal, constraints, and success criteria
        2. Create step-by-step outline with data/tool needs
        3. Track assumptions vs facts with sources
        4. Assign work to other agents
        5. Decide when research is complete
        
        Always maintain a visible task board and update it as you learn."""
    )
    return str(planner(f"Plan research for: {goal}\nContext: {context}"))

@tool
def researcher_agent(task: str, constraints: str = "") -> str:
    """Executes searches and returns structured findings with provenance."""
    researcher = Agent(
        model=ollama_model,
        tools=[http_request],
        system_prompt="""You are the Researcher Agent. Your role:
        1. Execute searches, read files, query APIs
        2. Return structured findings with clear provenance
        3. Cite sources for every assertion
        4. Flag gaps and uncertainties
        5. Provide evidence quality assessment
        
        Format: {evidence: [], sources: [], gaps: [], confidence: score}"""
    )
    return str(researcher(f"Research task: {task}\nConstraints: {constraints}"))

@tool
def analyst_agent(findings: str, goal: str) -> str:
    """Inspects answers, checks logic, spots missing cases."""
    analyst = Agent(
        model=ollama_model,
        system_prompt="""You are the Analyst/Critic Agent. Your role:
        1. Inspect partial answers for logic flaws
        2. Check evidence quality and source reliability
        3. Spot missing cases or alternative explanations
        4. Maintain risk log of uncertainties
        5. Validate against success criteria
        
        Ask: What could be wrong? What's unverified? What's missing?"""
    )
    return str(analyst(f"Analyze findings for goal: {goal}\nFindings: {findings}"))

@tool
def writer_agent(vetted_reasoning: str, goal: str) -> str:
    """Synthesizes final response with sources and open issues."""
    writer = Agent(
        model=ollama_model,
        system_prompt="""You are the Writer/Synthesizer Agent. Your role:
        1. Turn vetted reasoning into final response
        2. Reference all sources clearly
        3. Note open issues and limitations
        4. Use clear, structured formatting
        5. Include confidence levels for claims
        
        Create comprehensive, well-sourced final report."""
    )
    return str(writer(f"Synthesize final report for: {goal}\nVetted reasoning: {vetted_reasoning}"))

# Main orchestrating agent
def create_reasoning_agent():
    """Creates the main reasoning agent that orchestrates the workflow."""
    return Agent(
        model=ollama_model,
        tools=[planner_agent, researcher_agent, analyst_agent, writer_agent],
        conversation_manager=SlidingWindowConversationManager(window_size=30),
        system_prompt="""You are the Master Reasoning Agent. You orchestrate a multi-agent research workflow:

        WORKFLOW:
        1. Use planner_agent to create research plan and task outline
        2. Use researcher_agent to gather evidence (may loop multiple times)
        3. Use analyst_agent to critique and validate findings
        4. Use writer_agent to synthesize final report
        5. Conduct mini-retros: what changed, what's pending, what to verify next

        SHARED MEMORY:
        - Maintain task board with goals, subgoals, status
        - Track assumptions vs facts with sources
        - Keep reasoning transcript for auditability
        
        QUALITY CONTROL:
        - Force citation of sources for all assertions
        - Generate checklists/tests before finalizing
        - Break loops when analyst confirms acceptance criteria met
        
        Execute this workflow systematically for any research request."""
    )

if __name__ == "__main__":
    print("\n🧠 Reasoning Research Agent")
    print("Multi-agent reasoning workflow with planner, researcher, analyst, and writer roles.")
    
    # Create the main reasoning agent
    reasoning_agent = create_reasoning_agent()
    
    while True:
        try:
            user_input = input("\nWhat would you like me to research? (type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break
            
            # Let the reasoning agent orchestrate the entire workflow
            result = reasoning_agent(f"Execute comprehensive research workflow for: {user_input}")
            
            print("\n" + "="*60)
            print("🧠 REASONING RESEARCH REPORT")
            print("="*60)
            print(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try a different question.")