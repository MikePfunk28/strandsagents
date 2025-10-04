import logging
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
import uuid

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools import http_request
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Create model with sliding window
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:1.7b"
)

# Setup comprehensive logging
def setup_logging():
    """Configure comprehensive logging for the research system."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("advanced_research")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler for detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_file = log_dir / f"research_detailed_{timestamp}.log"
    file_handler = logging.FileHandler(detailed_log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler for user feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logger
logger = setup_logging()

# Log system startup
logger.info("🚀 Advanced Research System Starting Up")
logger.info("📊 Model: qwen3:1.7b")
logger.info(f"🌐 Host: {ollama_model.host}")
logger.info("=" * 80)

@tool
def query_decomposer(query: str) -> str:
    """Break down complex queries into specific searchable components."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"🔍 [Tool #{session_id}] QUERY_DECOMPOSER - Starting query decomposition")
    logger.info(f"🔍 [Tool #{session_id}] Input query: {query}")
    logger.info(f"🔍 [Tool #{session_id}] Query length: {len(query)} characters")

    start_time = time.time()
    try:
        logger.debug(f"🔍 [Tool #{session_id}] Creating decomposer agent...")

        decomposer = Agent(
            model=ollama_model,
            system_prompt="""You are a query decomposition expert. Break down the user's question into:
            1. Core concepts (3-5 key terms)
            2. Specific questions to answer
            3. Search strategies (broad vs specific)
            4. Information gaps to fill
            Format as structured output."""
        )

        logger.debug(f"🔍 [Tool #{session_id}] Agent created, processing query...")
        prompt = f"Decompose this query: {query}"
        logger.debug(f"🔍 [Tool #{session_id}] Full prompt: {prompt}")

        result = str(decomposer(prompt))

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"🔍 [Tool #{session_id}] QUERY_DECOMPOSER - Completed successfully")
        logger.info(f"🔍 [Tool #{session_id}] Processing time: {duration:.2f}")
        logger.info(f"🔍 [Tool #{session_id}] Output length: {len(result)} characters")
        logger.debug(f"🔍 [Tool #{session_id}] Decomposition result: {result[:500]}{'...' if len(result) > 500 else ''}")

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error(f"🔍 [Tool #{session_id}] QUERY_DECOMPOSER - Failed after {duration:.2f}")
        logger.error(f"🔍 [Tool #{session_id}] Error type: {type(e).__name__}")
        logger.error(f"🔍 [Tool #{session_id}] Error message: {str(e)}")
        logger.debug(f"🔍 [Tool #{session_id}] Full traceback: {traceback.format_exc()}")

        raise

@tool
def multi_search(search_terms: str) -> str:
    """Execute multiple search strategies with different approaches."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"🌐 [Tool #{session_id}] MULTI_SEARCH - Starting multi-strategy search")
    logger.info(f"🌐 [Tool #{session_id}] Search terms length: {len(search_terms)} characters")
    logger.debug(f"🌐 [Tool #{session_id}] Search terms: {search_terms[:200]}{'...' if len(search_terms) > 200 else ''}")

    start_time = time.time()
    try:
        logger.debug(f"🌐 [Tool #{session_id}] Creating searcher agent with http_request tool...")

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

        logger.debug(f"🌐 [Tool #{session_id}] Agent created, executing search...")
        prompt = f"Execute comprehensive search for: {search_terms}"
        logger.debug(f"🌐 [Tool #{session_id}] Full prompt: {prompt}")

        result = str(searcher(prompt))

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"🌐 [Tool #{session_id}] MULTI_SEARCH - Completed successfully")
        logger.info(f"🌐 [Tool #{session_id}] Processing time: {duration:.2f}s")
        logger.info(f"🌐 [Tool #{session_id}] Results length: {len(result)} characters")
        logger.debug(f"🌐 [Tool #{session_id}] Search results preview: {result[:500]}{'...' if len(result) > 500 else ''}")

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error(f"🌐 [Tool #{session_id}] MULTI_SEARCH - Failed after {duration:.2f}s")
        logger.error(f"🌐 [Tool #{session_id}] Error type: {type(e).__name__}")
        logger.error(f"🌐 [Tool #{session_id}] Error message: {str(e)}")
        logger.debug(f"🌐 [Tool #{session_id}] Full traceback: {traceback.format_exc()}")

        raise

@tool
def content_analyzer(content: str, query: str) -> str:
    """Analyze content relevance and extract key information."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"📊 [Tool #{session_id}] CONTENT_ANALYZER - Starting content analysis")
    logger.info(f"📊 [Tool #{session_id}] Content length: {len(content)} characters")
    logger.info(f"📊 [Tool #{session_id}] Query length: {len(query)} characters")
    logger.debug(f"📊 [Tool #{session_id}] Query: {query}")
    logger.debug(f"📊 [Tool #{session_id}] Content preview: {content[:300]}{'...' if len(content) > 300 else ''}")

    start_time = time.time()
    try:
        logger.debug(f"📊 [Tool #{session_id}] Creating analyzer agent with sliding window...")

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

        logger.debug(f"📊 [Tool #{session_id}] Agent created, analyzing content...")
        prompt = f"Analyze this content for query '{query}':\n\n{content}"
        logger.debug(f"📊 [Tool #{session_id}] Full prompt length: {len(prompt)} characters")

        result = str(analyzer(prompt))

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"📊 [Tool #{session_id}] CONTENT_ANALYZER - Completed successfully")
        logger.info(f"📊 [Tool #{session_id}] Processing time: {duration:.2f}s")
        logger.info(f"📊 [Tool #{session_id}] Analysis length: {len(result)} characters")
        logger.debug(f"📊 [Tool #{session_id}] Analysis preview: {result[:500]}{'...' if len(result) > 500 else ''}")

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error(f"📊 [Tool #{session_id}] CONTENT_ANALYZER - Failed after {duration:.2f}s")
        logger.error(f"📊 [Tool #{session_id}] Error type: {type(e).__name__}")
        logger.error(f"📊 [Tool #{session_id}] Error message: {str(e)}")
        logger.debug(f"📊 [Tool #{session_id}] Full traceback: {traceback.format_exc()}")

        raise

@tool
def similarity_ranker(findings: str, query: str) -> str:
    """Rank and filter findings by semantic similarity to query."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"🎯 [Tool #{session_id}] SIMILARITY_RANKER - Starting similarity ranking")
    logger.info(f"🎯 [Tool #{session_id}] Findings length: {len(findings)} characters")
    logger.info(f"🎯 [Tool #{session_id}] Query length: {len(query)} characters")
    logger.debug(f"🎯 [Tool #{session_id}] Query: {query}")
    logger.debug(f"🎯 [Tool #{session_id}] Findings preview: {findings[:300]}{'...' if len(findings) > 300 else ''}")

    start_time = time.time()
    try:
        logger.debug(f"🎯 [Tool #{session_id}] Creating ranker agent...")

        ranker = Agent(
            model=ollama_model,
            system_prompt="""You are a semantic similarity expert. For the findings:
            1. Rank each piece of information by relevance to the query
            2. Group similar concepts together
            3. Identify the most important insights
            4. Filter out low-relevance information
            5. Create a priority-ordered summary"""
        )

        logger.debug(f"🎯 [Tool #{session_id}] Agent created, ranking findings...")
        prompt = f"Rank findings by similarity to '{query}':\n\n{findings}"
        logger.debug(f"🎯 [Tool #{session_id}] Full prompt length: {len(prompt)} characters")

        result = str(ranker(prompt))

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"🎯 [Tool #{session_id}] SIMILARITY_RANKER - Completed successfully")
        logger.info(f"🎯 [Tool #{session_id}] Processing time: {duration:.2f}s")
        logger.info(f"🎯 [Tool #{session_id}] Ranked results length: {len(result)} characters")
        logger.debug(f"🎯 [Tool #{session_id}] Ranked results preview: {result[:500]}{'...' if len(result) > 500 else ''}")

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        logger.error(f"🎯 [Tool #{session_id}] SIMILARITY_RANKER - Failed after {duration:.2f}s")
        logger.error(f"🎯 [Tool #{session_id}] Error type: {type(e).__name__}")
        logger.error(f"🎯 [Tool #{session_id}] Error message: {str(e)}")
        logger.debug(f"🎯 [Tool #{session_id}] Full traceback: {traceback.format_exc()}")

        raise

def run_research_workflow(user_input):
    """Advanced research workflow with query decomposition and similarity analysis."""
    workflow_id = str(uuid.uuid4())[:8]
    total_start_time = time.time()

    logger.info(f"🚀 [Workflow #{workflow_id}] RESEARCH_WORKFLOW - Starting research workflow")
    logger.info(f"🚀 [Workflow #{workflow_id}] User input: {user_input}")
    logger.info(f"🚀 [Workflow #{workflow_id}] Input length: {len(user_input)} characters")

    try:
        # Step 1: Query Decomposition
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 1 - Query Decomposition starting...")
        step1_start = time.time()

        decomposition = query_decomposer(user_input)

        step1_end = time.time()
        step1_duration = step1_end - step1_start
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 1 - Query Decomposition completed in {step1_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 1 - Decomposition length: {len(decomposition)} characters")

        # Step 2: Multi-Strategy Search
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 2 - Multi-Strategy Search starting...")
        step2_start = time.time()

        search_results = multi_search(decomposition)

        step2_end = time.time()
        step2_duration = step2_end - step2_start
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 2 - Multi-Strategy Search completed in {step2_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 2 - Search results length: {len(search_results)} characters")

        # Step 3: Content Analysis
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 3 - Content Analysis starting...")
        step3_start = time.time()

        analysis = content_analyzer(search_results, user_input)

        step3_end = time.time()
        step3_duration = step3_end - step3_start
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 3 - Content Analysis completed in {step3_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 3 - Analysis length: {len(analysis)} characters")

        # Step 4: Similarity Ranking
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 4 - Similarity Ranking starting...")
        step4_start = time.time()

        ranked_findings = similarity_ranker(analysis, user_input)

        step4_end = time.time()
        step4_duration = step4_end - step4_start
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 4 - Similarity Ranking completed in {step4_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 4 - Ranked findings length: {len(ranked_findings)} characters")

        # Step 5: Final Synthesis
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 5 - Final Synthesis starting...")
        step5_start = time.time()

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

        logger.debug(f"🚀 [Workflow #{workflow_id}] STEP 5 - Synthesizer agent created, generating final report...")
        synthesis_prompt = f"Create a comprehensive research report for: '{user_input}'\n\nBased on these ranked findings:\n\n{ranked_findings}"
        logger.debug(f"🚀 [Workflow #{workflow_id}] STEP 5 - Synthesis prompt length: {len(synthesis_prompt)} characters")

        final_report = synthesizer(synthesis_prompt)

        step5_end = time.time()
        step5_duration = step5_end - step5_start

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 5 - Final Synthesis completed in {step5_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}] STEP 5 - Final report length: {len(final_report)} characters")
        logger.info(f"🚀 [Workflow #{workflow_id}] RESEARCH_WORKFLOW - Completed successfully in {total_duration:.2f}s")

        # Log step-by-step timing summary
        logger.info(f"🚀 [Workflow #{workflow_id}] TIMING SUMMARY:")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Step 1 (Decomposition): {step1_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Step 2 (Search): {step2_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Step 3 (Analysis): {step3_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Step 4 (Ranking): {step4_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Step 5 (Synthesis): {step5_duration:.2f}s")
        logger.info(f"🚀 [Workflow #{workflow_id}]   Total: {total_duration:.2f}s")

        return final_report

    except Exception as e:
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        logger.error(f"🚀 [Workflow #{workflow_id}] RESEARCH_WORKFLOW - Failed after {total_duration:.2f}s")
        logger.error(f"🚀 [Workflow #{workflow_id}] Error type: {type(e).__name__}")
        logger.error(f"🚀 [Workflow #{workflow_id}] Error message: {str(e)}")
        logger.debug(f"🚀 [Workflow #{workflow_id}] Full traceback: {traceback.format_exc()}")

        raise

if __name__ == "__main__":
    logger.info("🚀 MAIN - Advanced Research Assistant starting...")
    print("\n🔬 Advanced Research Assistant")
    print("Multi-agent system with query decomposition, similarity ranking, and sliding window analysis.")
    print("\nType 'exit' to quit.")

    session_count = 0

    try:
        while True:
            try:
                user_input = input("\n> ")
                if user_input.lower() == "exit":
                    logger.info("🚀 MAIN - User requested exit")
                    print("\nGoodbye!")
                    break

                session_count += 1
                logger.info(f"🚀 MAIN - Starting research session #{session_count}")
                logger.info(f"🚀 MAIN - User input: {user_input}")

                final_report = run_research_workflow(user_input)

                logger.info(f"🚀 MAIN - Research session #{session_count} completed successfully")
                logger.info(f"🚀 MAIN - Final report length: {len(final_report)} characters")

                print("\n" + "="*60)
                print("📋 RESEARCH REPORT")
                print("="*60)
                print(final_report)

            except KeyboardInterrupt:
                logger.info("🚀 MAIN - Keyboard interrupt received")
                print("\n\nExecution interrupted. Exiting...")
                break
            except Exception as e:
                session_count += 1
                logger.error(f"🚀 MAIN - Error in session #{session_count}: {str(e)}")
                logger.error(f"🚀 MAIN - Error type: {type(e).__name__}")
                logger.debug(f"🚀 MAIN - Full traceback: {traceback.format_exc()}")

                print(f"\nError: {str(e)}")
                print("Please try a different request.")

    except Exception as e:
        logger.critical("🚀 MAIN - Critical error in main loop")
        logger.critical(f"🚀 MAIN - Error type: {type(e).__name__}")
        logger.critical(f"🚀 MAIN - Error message: {str(e)}")
        logger.critical(f"🚀 MAIN - Full traceback: {traceback.format_exc()}")

        print(f"\nCritical error: {str(e)}")
        print("Shutting down...")

    finally:
        logger.info("🚀 MAIN - Advanced Research Assistant shutting down...")
        logger.info(f"🚀 MAIN - Total sessions processed: {session_count}")
        logger.info("👋 MAIN - Goodbye!")
