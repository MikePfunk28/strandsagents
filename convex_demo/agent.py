from strandsagents import agent, Agent
from agentcore.memory import MemoryManager
from agentcore.interpreter import CodeInterpreter
import requests
from bs4 import BeautifulSoup
import httpx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sqlite3


class WebSearchTool:
    def __init__(self):
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 5):
        """Search the web for information"""
        # Implementation for web search
        pass

# API Client tool configuration

# Memory Manager tool configuration

# Data Analysis tool configuration


class FileOperationsTool:
    def __init__(self, base_path: str = "./"):
        self.base_path = base_path
    
    def read_file(self, filepath: str):
        """Read contents of a file"""
        with open(os.path.join(self.base_path, filepath), 'r') as f:
            return f.read()
    
    def write_file(self, filepath: str, content: str):
        """Write content to a file"""
        with open(os.path.join(self.base_path, filepath), 'w') as f:
            f.write(content)


class DatabaseTool:
    def __init__(self, db_path: str = "agent.db"):
        self.db_path = db_path
    
    def query(self, sql: str):
        """Execute SQL query"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result

# Code Interpreter tool configuration


@agent(
    model="claude-3-haiku-20240307",
    system_prompt="""you are a research assistant that goes to search for credible sources about a topic.  You think about what you need before, and then you think about if that covers it or if you need more information, then you get it or aggregate and present it.""",
    tools=["Web Search", "API Client", "Memory Manager", "Data Analysis", "File Operations", "Database Access", "Code Interpreter"],
    memory=True,
    code_interpreter=True,
    reasoning="interleaved"
)
class researcheragentAgent(Agent):
    def __init__(self):
        super().__init__()
        self.memory = MemoryManager()
        self.interpreter = CodeInterpreter()
        self.web search = WebSearchTool()
        self.api client = GenericTool()
        self.memory manager = GenericTool()
        self.data analysis = GenericTool()
        self.file operations = FileOperationsTool()
        self.database access = DatabaseTool()
        self.code interpreter = GenericTool()
    
    async def process_message(self, message: str, context: dict = None):
        """Process incoming message with interleaved reasoning"""
        # Store message in memory
        self.memory.store_message(message, context)
        
        # Use Claude Sonnet 4.5 with interleaved reasoning
        response = await self.generate_response(
            message=message,
            context=context,
            reasoning_mode="interleaved"
        )
        
        return response
    
    async def execute_tool(self, tool_name: str, **kwargs):
        """Execute a tool by name"""
        tool = getattr(self, tool_name.lower(), None)
        if tool and hasattr(tool, kwargs.get('method', 'execute')):
            return await getattr(tool, kwargs.get('method', 'execute'))(**kwargs)
        raise ValueError(f"Tool {tool_name} not found or method not available")


# Docker Deployment Configuration
if __name__ == "__main__":
    import asyncio
    from agentcore.hosting import DockerHost
    
    async def main():
        # Create agent instance
        agent = Agent()
        
        # Deploy to Docker container
        host = DockerHost()
        await host.deploy_agent(agent)
    
    asyncio.run(main())