from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import http_request
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Create model with sliding window
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2"
)

@tool
def query_decomposer(query: str) -> str:
    """Break down complex queries into specific searchable components."""
    decomposer = Agent(
        model=ollama_model,
        system_prompt="""You are a query decomposition expert. Break down the user's question into:
        1. Core concepts (3-5 key terms)
        2. Specific questions to answer
        3. Search strategies (broad vs specific)
        4. Information gaps to fill
        Format as structured output."""
    )
    return str(decomposer(f"Decompose this query: {query}"))

@tool
def multi_search(search_terms: str) -> str:
    """Execute multiple search strategies with different approaches."""
    searcher = Agent(
        model=ollama_model,
        tools=[http_request],
        system_prompt="""You are an advanced web researcher. For each search:
        1. Use broad searches for context
        2. Use specific searches for details
        3. Cross-reference multiple sources
        4. Focus on authoritative domains
        5. Extract key facts and quotes with sources"""
    )
    return str(searcher(f"Execute comprehensive search for: {search_terms}"))

@tool
def content_analyzer(content: str, query: str) -> str:
    """Analyze content relevance and extract key information."""
    analyzer = Agent(
        model=ollama_model,
        conversation_manager=SlidingWindowConversationManager(window_size=15),
        system_prompt="""You are a content analysis expert. For the given content:
        1. Score relevance to the original query (1-10)
        2. Extract key facts and evidence
        3. Identify contradictions or gaps
        4. Rank information by importance
        5. Note source credibility indicators"""
    )
    return str(analyzer(f"Analyze this content for query '{query}':\n\n{content}"))

@tool
def similarity_ranker(findings: str, query: str) -> str:
    """Rank and filter findings by semantic similarity to query."""
    ranker = Agent(
        model=ollama_model,
        system_prompt="""You are a semantic similarity expert. For the findings:
        1. Rank each piece of information by relevance to the query
        2. Group similar concepts together
        3. Identify the most important insights
        4. Filter out low-relevance information
        5. Create a priority-ordered summary"""
    )
    return str(ranker(f"Rank findings by similarity to '{query}':\n\n{findings}"))

def run_research_workflow(user_input):
    """Advanced research workflow with query decomposition and similarity analysis."""
    
    print(f"\n🔍 Processing: '{user_input}'")
    
    # Step 1: Query Decomposition
    print("\n📋 Step 1: Breaking down query...")
    decomposition = query_decomposer(user_input)
    print("✓ Query decomposed")
    
    # Step 2: Multi-Strategy Search
    print("\n🌐 Step 2: Executing multi-strategy search...")
    search_results = multi_search(decomposition)
    print("✓ Search complete")
    
    # Step 3: Content Analysis
    print("\n📊 Step 3: Analyzing content relevance...")
    analysis = content_analyzer(search_results, user_input)
    print("✓ Content analyzed")
    
    # Step 4: Similarity Ranking
    print("\n🎯 Step 4: Ranking by similarity...")
    ranked_findings = similarity_ranker(analysis, user_input)
    print("✓ Findings ranked")
    
    # Step 5: Final Synthesis
    print("\n📝 Step 5: Synthesizing final report...")
    synthesizer = Agent(
        model=ollama_model,
        conversation_manager=SlidingWindowConversationManager(window_size=20),
        system_prompt="""You are an expert research synthesizer. Create a comprehensive report that:
        1. Directly answers the original question
        2. Provides supporting evidence with sources
        3. Acknowledges limitations or uncertainties
        4. Uses clear, structured formatting
        5. Includes confidence levels for key claims"""
    )
    
    final_report = synthesizer(
        f"Create a comprehensive research report for: '{user_input}'\n\nBased on these ranked findings:\n\n{ranked_findings}"
    )
    
    print("✓ Report synthesized")
    return final_report

if __name__ == "__main__":
    print("\n🔬 Advanced Research Assistant")
    print("Multi-agent system with query decomposition, similarity ranking, and sliding window analysis.")
    print("\nType 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break
            
            final_report = run_research_workflow(user_input)
            print("\n" + "="*60)
            print("📋 RESEARCH REPORT")
            print("="*60)
            print(final_report)
        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try a different request.")