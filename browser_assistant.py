from strands import Agent
from strands_tools.browser import LocalChromiumBrowser

# Create browser tool
browser = LocalChromiumBrowser()
agent = Agent(tools=[browser.browser])

# Simple navigation
result = agent.tool.browser({
    "action": {
        "type": "navigate",
        "url": "https://google.com"
    }
})

# Initialize a session first
result = agent.tool.browser({
    "action": {
        "type": "initSession",
        "session_name": "main-session",
        "description": "Web automation session"
    }
})
