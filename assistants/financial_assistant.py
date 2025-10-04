"""
# Task management example
tasks = {
    "data_extraction": {
        "description": "Extract key financial data from the quarterly report",
        "status": "pending",
        "agent": financial_agent,
        "dependencies": []
    },
    "trend_analysis": {
        "description": "Analyze trends in the extracted data",
        "status": "pending",
        "agent": analyst_agent,
        "dependencies": ["data_extraction"]
    }
}

def get_ready_tasks(tasks, completed_tasks):
    ready_tasks = []
    for task_id, task in tasks.items():
        if task["status"] == "pending":
            deps = task.get("dependencies", [])
            if all(dep in completed_tasks for dep in deps):
                ready_tasks.append(task_id)
    return ready_tasks
"""


from strands import Agent
from strands_tools import workflow

# Create an agent with workflow capability
agent = Agent(tools=[workflow])

# Create a multi-agent workflow
agent.tool.workflow(
    action="create",
    workflow_id="data_analysis",
    tasks=[
        {
            "task_id": "data_extraction",
            "description": "Extract key financial data from the quarterly report",
            "system_prompt": "You extract and structure financial data from reports.",
            "priority": 5
        },
        {
            "task_id": "trend_analysis",
            "description": "Analyze trends in the data compared to previous quarters",
            "dependencies": ["data_extraction"],
            "system_prompt": "You identify trends in financial time series.",
            "priority": 3
        },
        {
            "task_id": "report_generation",
            "description": "Generate a comprehensive analysis report",
            "dependencies": ["trend_analysis"],
            "system_prompt": "You create clear financial analysis reports.",
            "priority": 2
        }
    ]
)

# Execute workflow (parallel processing where possible)
agent.tool.workflow(action="start", workflow_id="data_analysis")

# Check results
status = agent.tool.workflow(action="status", workflow_id="data_analysis")


def build_task_context(task_id, tasks, results):
    """Build context from dependent tasks"""
    context = []
    for dep_id in tasks[task_id].get("dependencies", []):
        if dep_id in results:
            context.append(f"Results from {dep_id}: {results[dep_id]}")

    prompt = tasks[task_id]["description"]
    if context:
        prompt = "Previous task results:\n" + \
            "\n\n".join(context) + "\n\nTask:\n" + prompt

    return prompt
