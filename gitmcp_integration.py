"""
GitMCP Integration for StrandsAgents
Demonstrates how to use MCP (Model Context Protocol) for Git operations
"""

import asyncio
import json
from typing import Dict, List, Optional

class GitMCPClient:
    """Simple MCP client for Git operations"""
    
    def __init__(self, server_path: str = "git"):
        self.server_path = server_path
        self.connected = False
    
    async def connect(self):
        """Connect to GitMCP server"""
        # In a real implementation, this would establish MCP connection
        self.connected = True
        print("Connected to GitMCP server")
    
    async def list_tools(self) -> List[Dict]:
        """List available Git tools"""
        if not self.connected:
            await self.connect()
        
        # Mock tools that GitMCP typically provides
        return [
            {"name": "git_status", "description": "Get repository status"},
            {"name": "git_log", "description": "Get commit history"},
            {"name": "git_diff", "description": "Show changes"},
            {"name": "git_branch", "description": "List/create branches"},
            {"name": "git_commit", "description": "Create commits"},
            {"name": "git_push", "description": "Push changes"},
            {"name": "git_pull", "description": "Pull changes"}
        ]
    
    async def git_status(self, repo_path: str = ".") -> Dict:
        """Get Git status"""
        # Mock implementation - in real use, this would call MCP
        return {
            "branch": "main",
            "modified": ["context7_info.py"],
            "untracked": ["gitmcp_integration.py"],
            "staged": [],
            "clean": False
        }
    
    async def git_log(self, repo_path: str = ".", limit: int = 10) -> List[Dict]:
        """Get commit history"""
        # Mock implementation
        return [
            {
                "hash": "abc123",
                "message": "Fix GitHub API requests",
                "author": "developer",
                "date": "2024-01-15T10:30:00Z"
            }
        ]
    
    async def git_diff(self, repo_path: str = ".", file_path: Optional[str] = None) -> str:
        """Show Git diff"""
        # Mock implementation
        return """
diff --git a/context7_info.py b/context7_info.py
index 1234567..abcdefg 100644
--- a/context7_info.py
+++ b/context7_info.py
@@ -1,10 +1,15 @@
 import requests
 import json
 import base64
+
+def get_repo_info(repo_url):
+    \"\"\"Get repository information with error handling\"\"\"
"""

class StrandsGitIntegration:
    """Integration between StrandsAgents and GitMCP"""
    
    def __init__(self):
        self.git_client = GitMCPClient()
        self.project_path = "m:/strandsagents"
    
    async def initialize(self):
        """Initialize Git integration"""
        await self.git_client.connect()
        tools = await self.git_client.list_tools()
        print(f"Available Git tools: {[t['name'] for t in tools]}")
    
    async def get_project_status(self) -> Dict:
        """Get comprehensive project status"""
        status = await self.git_client.git_status(self.project_path)
        log = await self.git_client.git_log(self.project_path, limit=5)
        
        return {
            "repository_status": status,
            "recent_commits": log,
            "needs_commit": len(status["modified"]) > 0 or len(status["untracked"]) > 0
        }
    
    async def analyze_changes(self) -> Dict:
        """Analyze current changes for agent context"""
        diff = await self.git_client.git_diff(self.project_path)
        status = await self.git_client.git_status(self.project_path)
        
        return {
            "changed_files": status["modified"] + status["untracked"],
            "diff_summary": diff[:500] + "..." if len(diff) > 500 else diff,
            "change_type": self._classify_changes(status)
        }
    
    def _classify_changes(self, status: Dict) -> str:
        """Classify the type of changes"""
        modified = status["modified"]
        untracked = status["untracked"]
        
        if any("test" in f for f in modified + untracked):
            return "testing"
        elif any(f.endswith(".py") for f in modified + untracked):
            return "code_changes"
        elif any(f.endswith(".md") for f in modified + untracked):
            return "documentation"
        else:
            return "general"

async def demo_gitmcp_integration():
    """Demonstrate GitMCP integration"""
    print("=== StrandsAgents GitMCP Integration Demo ===\n")
    
    # Initialize integration
    git_integration = StrandsGitIntegration()
    await git_integration.initialize()
    
    # Get project status
    print("\n1. Project Status:")
    status = await git_integration.get_project_status()
    print(json.dumps(status, indent=2))
    
    # Analyze changes
    print("\n2. Change Analysis:")
    changes = await git_integration.analyze_changes()
    print(json.dumps(changes, indent=2))
    
    # Show how this integrates with your graph system
    print("\n3. Graph System Integration:")
    print("- Changes can be stored as nodes in your graph system")
    print("- Commit history becomes part of project memory")
    print("- Agent decisions can be tracked with Git context")

if __name__ == "__main__":
    asyncio.run(demo_gitmcp_integration())