# Claude 4 Orchestrator with Interleaved Thinking using Strands

from strands import Agent, tool
from strands.models import OllamaModel
from strands.tools import think

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="gemma3:latest"
)

agent = Agent(
    model=ollama_model,
    tools=[think],
)
class StrandsInterleavedWorkflowOrchestrator:
    def __init__(self):
        # Define the orchestrator system prompt for intelligent workflow coordination
        self.system_prompt = """You are an intelligent workflow orchestrator with access to specialist agents.

        Your role is to intelligently coordinate a workflow using these specialist agents:
        - researcher: Gathers factual information on any topic
        - data_analyst: Analyzes data and extracts insights
        - fact_checker: Verifies accuracy of information
        - report_writer: Creates polished final reports

        """

    def run_workflow(self, task: str, enable_interleaved_thinking: bool = True) -> str:
        """Execute an intelligent workflow for the given task.

        Args:
            task: The task to complete
            enable_interleaved_thinking: Whether to enable interleaved thinking (default: True)

        The orchestrator will:
        1. Understand the task requirements
        2. Think about the best approach
        3. Coordinate specialist agents
        4. Reflect on results between steps
        5. Produce a comprehensive output
        """
        thinking_mode = "WITH interleaved thinking" if enable_interleaved_thinking else "WITHOUT interleaved thinking"
        print(f"\nStarting intelligent workflow {thinking_mode} for: {task}")
        print("=" * 70)

        # Configure Claude 4 with or without interleaved thinking via Bedrock
        if enable_interleaved_thinking:
            claude4_model = ollama_model,
            max_tokens=4096,
            temperature=1,  # Required to be 1 when thinking is enabled
            additional_request_fields={
                # Enable interleaved thinking beta feature
                "interleaved-thinking-2025-05-14": [],
                # Configure reasoning parameters
                "reasoning_config": {
                    "type": "enabled",  # Turn on thinking
                    "budget_tokens": 3000  # Thinking token budget
                }
            }

        else:
            claude4_model = BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                max_tokens=4096,
                temperature=1
            )

        # Create the orchestrator agent with Claude 4 and specialist tools
        orchestrator = Agent(
            model=claude4_model,
            system_prompt=self.system_prompt,
            tools=[researcher, data_analyst, fact_checker, report_writer]
        )

        prompt = f"""Complete this task using intelligent workflow coordination: {task}

        Instructions:
        1. Think carefully about what information you need to accomplish this task
        2. Use the specialist agents strategically - each has unique strengths
        3. After each tool use, reflect on the results and adapt your approach
        4. Coordinate multiple agents as needed for comprehensive results
        5. Ensure accuracy by fact-checking when appropriate
        6. Provide a comprehensive final response that addresses all aspects

        Remember: Your thinking between tool calls helps you make better decisions.
        Use it to plan, evaluate results, and adjust your strategy.
        """

        try:
            result = orchestrator(prompt)
            return str(result)
        except Exception as e:
            return f"Workflow failed: {e}"
