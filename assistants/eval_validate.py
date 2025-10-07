from strands import Agent
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os


class AgentEvaluator:
    def __init__(self, test_cases_path, output_dir="evaluation_results"):
        """Initialize evaluator with test cases"""
        with open(test_cases_path, "r") as f:
            self.test_cases = json.load(f)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_agent(self, agent, agent_name):
        """Run evaluation on an agent"""
        results = []
        start_time = datetime.datetime.now()

        print(f"Starting evaluation of {agent_name} at {start_time}")

        for case in self.test_cases:
            case_start = datetime.datetime.now()
            response = agent(case["query"])
            case_duration = (datetime.datetime.now() -
                             case_start).total_seconds()

            results.append({
                "test_id": case.get("id", ""),
                "category": case.get("category", ""),
                "query": case["query"],
                "expected": case.get("expected", ""),
                "actual": str(response),
                "response_time": case_duration
            })

        total_duration = (datetime.datetime.now() - start_time).total_seconds()

        # Save raw results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.output_dir, f"{agent_name}_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation completed in {total_duration:.2f} seconds")
        print(f"Results saved to {results_path}")

        return results

    def analyze_results(self, results, agent_name):
        """Generate analysis of evaluation results"""
        df = pd.DataFrame(results)

        # Calculate metrics
        metrics = {
            "total_tests": len(results),
            "avg_response_time": df["response_time"].mean(),
            "max_response_time": df["response_time"].max(),
            "categories": df["category"].value_counts().to_dict()
        }

        # Generate charts
        plt.figure(figsize=(10, 6))
        df.groupby("category")["response_time"].mean().plot(kind="bar")
        plt.title(f"Average Response Time by Category - {agent_name}")
        plt.ylabel("Seconds")
        plt.tight_layout()

        chart_path = os.path.join(
            self.output_dir, f"{agent_name}_response_times.png")
        plt.savefig(chart_path)

        return metrics


# Usage example
if __name__ == "__main__":
    # Create agents with different configurations
    agent1 = Agent(
        model="ollama_model",
        system_prompt="You are a helpful assistant."
    )

    agent2 = Agent(
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
        system_prompt="You are a helpful assistant."
    )

    # Create evaluator
    evaluator = AgentEvaluator("test_cases.json")

    # Evaluate agents
    results1 = evaluator.evaluate_agent(agent1, "claude-sonnet")
    metrics1 = evaluator.analyze_results(results1, "claude-sonnet")

    results2 = evaluator.evaluate_agent(agent2, "claude-haiku")
    metrics2 = evaluator.analyze_results(results2, "claude-haiku")

    # Compare results
    print("\nPerformance Comparison:")
    print(f"Sonnet avg response time: {metrics1['avg_response_time']:.2f}s")
    print(f"Haiku avg response time: {metrics2['avg_response_time']:.2f}s")
