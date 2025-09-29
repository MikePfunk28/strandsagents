"""Example usage of the GRAPH system for programming analysis.

This demonstrates how to use the deterministic graph workflows
for code analysis, data structure mapping, and debugging.
"""

import asyncio
import logging
from graph import ProgrammingGraph, create_code_analysis_graph, create_data_structure_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def example_code_analysis():
    """Example: Analyze a Python file using graph workflow."""
    print("📊 Starting code analysis graph example...")

    # Create code analysis graph
    graph = create_code_analysis_graph("llama3.2:3b")

    try:
        # Analyze the swarm system main file
        result = await graph.analyze_code("swarm/main.py", {
            "focus": "architecture_analysis",
            "include_patterns": True
        })

        print(f"✅ Analysis completed for: {result['file_path']}")
        print(f"📋 Graph type: {result['graph_type']}")
        print(f"🔍 Result summary: {str(result['result'])[:200]}...")

        # Check graph status
        status = graph.get_graph_status()
        print(f"📊 Graph status: {status}")

    except Exception as e:
        print(f"❌ Error in code analysis: {e}")

async def example_data_structure_analysis():
    """Example: Analyze data structures in code."""
    print("\n🏗️  Starting data structure analysis...")

    # Sample code to analyze
    sample_code = """
class DatabaseManager:
    def __init__(self):
        self.cache = {}
        self.connections = []
        self.metadata = {
            'version': '1.0',
            'tables': ['users', 'sessions']
        }

    def process_data(self, items: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        result = {'processed': [], 'errors': []}
        for item in items:
            if self.validate(item):
                result['processed'].append(self.transform(item))
            else:
                result['errors'].append(item)
        return result
"""

    graph = create_data_structure_graph("llama3.2:3b")

    try:
        result = await graph.analyze_data_structures(sample_code, {
            "focus": "data_flow_mapping",
            "include_optimizations": True
        })

        print(f"✅ Data structure analysis completed")
        print(f"📋 Graph type: {result['graph_type']}")
        print(f"🔍 Analysis: {str(result['result'])[:200]}...")

    except Exception as e:
        print(f"❌ Error in data structure analysis: {e}")

async def example_debugging_workflow():
    """Example: Debug an issue using graph workflow."""
    print("\n🐛 Starting debugging workflow...")

    # Create debugging graph
    graph = ProgrammingGraph("debugging", "llama3.2:3b")

    error_info = """
AttributeError: 'NoneType' object has no attribute 'connect'
  File "swarm/main.py", line 74, in initialize
    await self.mcp_client.connect()
"""

    code_context = """
def initialize(self):
    self.mcp_client = SwarmMCPClient(...)
    if some_condition:
        self.mcp_client = None  # This could cause the issue
    await self.mcp_client.connect()  # Error happens here
"""

    try:
        result = await graph.debug_issue(error_info, code_context, {
            "severity": "high",
            "component": "swarm_initialization"
        })

        print(f"✅ Debugging analysis completed")
        print(f"🔍 Debug result: {str(result['result'])[:200]}...")

    except Exception as e:
        print(f"❌ Error in debugging: {e}")

async def compare_swarm_vs_graph():
    """Compare when to use SWARM vs GRAPH patterns."""
    print("\n⚖️  SWARM vs GRAPH Pattern Comparison:")

    print("🐝 SWARM (swarm/main.py):")
    print("   - Emergent collaboration between agents")
    print("   - Best for: Research, creative tasks, open-ended problems")
    print("   - Agents communicate freely and adapt")

    print("\n📊 GRAPH (graph/programming_graph.py):")
    print("   - Deterministic, structured workflows")
    print("   - Best for: Code analysis, data flows, debugging")
    print("   - Clear dependencies and execution order")

    print("\n🎯 Use Cases:")
    print("   SWARM: 'Research AI trends and write a report'")
    print("   GRAPH: 'Analyze this codebase and find optimization opportunities'")

async def main():
    """Run all graph examples."""
    print("🌟 Programming Graph System Examples")
    print("=" * 50)

    # Run code analysis example
    await example_code_analysis()

    # Run data structure analysis
    await example_data_structure_analysis()

    # Run debugging workflow
    await example_debugging_workflow()

    # Show comparison
    await compare_swarm_vs_graph()

    print("\n✨ All graph examples completed!")

if __name__ == "__main__":
    asyncio.run(main())