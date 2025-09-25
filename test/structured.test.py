from strands import Agent
import json
import pandas as pd

# Load test cases from JSON file
with open("test_cases.json", "r") as f:
    test_cases = json.load(f)

# Create agent
agent = Agent(model="us.anthropic.claude-sonnet-4-20250514-v1:0")

# Run tests and collect results
results = []
for case in test_cases:
    query = case["query"]
    expected = case.get("expected")

    # Execute the agent query
    response = agent(query)

    # Store results for analysis
    results.append({
        "test_id": case.get("id", ""),
        "query": query,
        "expected": expected,
        "actual": str(response),
        "timestamp": pd.Timestamp.now()
    })

# Export results for review
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
# Example output:
# |test_id    |query                         |expected                       |actual                          |timestamp                 |
# |-----------|------------------------------|-------------------------------|--------------------------------|--------------------------|
# |knowledge-1|What is the capital of France?|The capital of France is Paris.|The capital of France is Paris. |2025-05-13 18:37:22.673230|
#
