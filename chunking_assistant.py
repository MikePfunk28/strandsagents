import asyncio
import re
from strands import Agent, tool
from langchain.schema import Document
from strands_tools import agent_graph, retrieve
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


@tool
def check_chunks_relevance(results: str, question: str):
    """
    Evaluates the relevance of retrieved chunks to the user question using RAGAs.

    Args:
        results (str): Retrieval output as a string with 'Score:' and 'Content:' patterns.
        question (str): Original user question.

    Returns:
        dict: A binary score ('yes' or 'no') and the numeric relevance score, or an error message.
    """
    try:
        if not results or not isinstance(results, str):
            raise ValueError(
                "Invalid input: 'results' must be a non-empty string.")
        if not question or not isinstance(question, str):
            raise ValueError(
                "Invalid input: 'question' must be a non-empty string.")

        # Extract content chunks using regex
        pattern = r"Score:.*?\nContent:\s*(.*?)(?=Score:|\Z)"
        docs = [chunk.strip()
                for chunk in re.findall(pattern, results, re.DOTALL)]

        if not docs:
            raise ValueError("No valid content chunks found in 'results'.")

        # Prepare evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            response="placeholder-response",  # required dummy response
            retrieved_contexts=docs
        )

        # Evaluate using context precision metric
        scorer = LLMContextPrecisionWithoutReference(llm=llm_for_evaluation)
        score = asyncio.run(scorer.single_turn_ascore(sample))

        print("------------------------")
        print("Context evaluation")
        print("------------------------")
        print(f"chunk_relevance_score: {score}")

        return {
            "chunk_relevance_score": "yes" if score > 0.5 else "no",
            "chunk_relevance_value": score
        }

    except Exception as e:
        return {
            "error": str(e),
            "chunk_relevance_score": "unknown",
            "chunk_relevance_value": None
        }
