from strands import Agent, tools
from strands_tools import workflow, think, SlidingWindowConversationManager
from strnads.models.ollama import OllamaModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Create an agent with workflow capability
agent = Agent(tools=[workflow])
ollama_model = OllamaModel(host="localhost:11434", model="llama3.2")
think_agent = Agent(
    model=ollama_model,
    tools=[think],
    conversation_manager=SlidingWindowConversationManager(window_size=30))

# Create a multi-agent workflow
agent.tool.workflow(
    action="create",
    workflow_id="GarbagaeCleanup",
    tasks=[
        {
            "task_id": "file_indexing",
            "description": "Index all the files in the project directory, and create a csv.",
            "system_prompt": "You index files, create a timestamped CSV summary and file list.",
            "priority": 5

        },
        {
            "task_id": "analyze_files",
            "description": "Analyze all the files and their contents, and create a summary report.",
            "dependencies": ["file_indexing"],
            "system_prompt": "You analyze file contents and create a summary report.",
            "priority": 4
        },
        {
            "task_id": "file_traces",
            "description": "Trace files, dependencies and references, and create a graph database.",
            "dependencies": ["analyze_files"],
            "system_prompt": "You create a graph database of file dependencies and references.",
            "priority": 3
        },
        {
            "task_id": "file_usage",
            "description": "If the file is active, where its being used, if its needed, or not.",
            "dependencies": ["analyze_files"],
            "system_prompt": "You check file activity, unused files, add to the list and report.",
            "priority": 2
        },
        {
            "task_id": "garbage_cleanup",
            "description": "Check file usage, identify and remove unused files and dependencies.",
            "dependencies": ["file_usage"],
            "system_prompt": "You identify and remove unused files and dependencies. Gen a report",
            "priority": 1
        }
    ]
)

agent.tool.think(
    action="set_agent",
    agent=think_agent,
    workflow_id="AreWeClean",
    description="Check the files, and the directory to see if we are clean.",
    system_prompt="You check the files and the directory to see if we are clean."
)

# Execute workflow (parallel processing where possible)
agent.tool.workflow(action="start", workflow_id="data_analysis")

# Check results
status = agent.tool.workflow(action="status", workflow_id="data_analysis")
