from strands import Agent
import json
from strands.models.ollama import OllamaModel

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini-reasoning:latest"
)
# Create the agent to evaluate
agent = Agent(model=ollama_model)

# Create an evaluator agent with a stronger model
evaluator = Agent(
    model=ollama_model,
    system_prompt="""
    You are an expert AI evaluator. Your job is to assess the quality of AI responses based on:
    1. Accuracy - factual correctness of the response
    2. Relevance - how well the response addresses the query
    3. Completeness - whether all aspects of the query are addressed
    4. Tool usage - appropriate use of available tools

    Score each criterion from 1-5, where 1 is poor and 5 is excellent.
    Provide an overall score and brief explanation for your assessment.
    """
)

# Load test cases
with open("test_cases.json", "a", encoding="utf-8") as f:
    test_cases = json.load(f)

# Run evaluations
evaluation_results = []
for case in test_cases:
    # Get agent response
    agent_response = agent(case["query"])

    # Create evaluation prompt
    eval_prompt = f"""
    Query: {case['query']}

    Response to evaluate:
    {agent_response}

    Expected response (if available):
    {case.get('expected', 'Not provided')}

    Please evaluate the response based on accuracy, relevance, completeness, and tool usage.
    """

    # Get evaluation
    evaluation = evaluator(eval_prompt)

    # Store results
    evaluation_results.append({
        "test_id": case.get("id", ""),
        "query": case["query"],
        "agent_response": str(agent_response),
        "evaluation": evaluation.message['content']
    })

# Save evaluation results
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)
