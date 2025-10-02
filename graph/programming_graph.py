"""Programming analysis graph using Strands GRAPH pattern.

This module implements deterministic graph workflows for:
- Code analysis and structure understanding
- Data structure flow visualization
- Programming workflow coordination
- Debugging and optimization flows
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from strands import Agent
from strands.multiagent import GraphBuilder

logger = logging.getLogger(__name__)

class ProgrammingGraph:
    """Graph-based system for programming analysis and workflows."""

    def __init__(self, graph_type: str = "code_analysis", model_name: str = "llama3.2:3b"):
        self.graph_type = graph_type
        self.model_name = model_name
        self.graph = None
        self.agents: Dict[str, Agent] = {}

    def _create_code_analysis_agents(self) -> Dict[str, Agent]:
        """Create agents for code analysis workflow."""
        agents = {
            "parser": Agent(
                name="code_parser",
                system_prompt="""You are a code parsing specialist. Analyze code structure, identify:
                - Functions, classes, and modules
                - Import dependencies
                - Key data structures
                - Entry points and main flows
                Return structured analysis in clear format."""
            ),
            "analyzer": Agent(
                name="structure_analyzer",
                system_prompt="""You are a code structure analyzer. Given parsed code information:
                - Identify design patterns
                - Analyze complexity and dependencies
                - Find potential issues or optimizations
                - Map data flow between components
                Provide detailed architectural insights."""
            ),
            "documenter": Agent(
                name="code_documenter",
                system_prompt="""You are a code documentation specialist. Create clear documentation:
                - Generate function/class descriptions
                - Create architecture diagrams in text
                - Explain data flow and relationships
                - Suggest improvements
                Produce comprehensive, readable documentation."""
            )
        }
        return agents

    def _create_data_structure_agents(self) -> Dict[str, Agent]:
        """Create agents for data structure analysis."""
        agents = {
            "identifier": Agent(
                name="structure_identifier",
                system_prompt="""You are a data structure identification specialist. Analyze code to:
                - Identify all data structures (lists, dicts, classes, etc.)
                - Determine data types and relationships
                - Map data transformations
                - Find data access patterns
                Return structured data structure inventory."""
            ),
            "flow_mapper": Agent(
                name="data_flow_mapper",
                system_prompt="""You are a data flow mapping specialist. Create flow diagrams showing:
                - How data moves between functions/classes
                - Transformation points and operations
                - Input/output relationships
                - Dependencies and cycles
                Generate clear data flow visualizations."""
            ),
            "optimizer": Agent(
                name="structure_optimizer",
                system_prompt="""You are a data structure optimization specialist. Analyze and suggest:
                - More efficient data structures
                - Performance improvements
                - Memory optimization opportunities
                - Algorithmic improvements
                Provide actionable optimization recommendations."""
            )
        }
        return agents

    def _create_debugging_agents(self) -> Dict[str, Agent]:
        """Create agents for debugging workflow."""
        agents = {
            "error_analyzer": Agent(
                name="error_analyzer",
                system_prompt="""You are an error analysis specialist. Given error information:
                - Identify root causes
                - Trace error propagation
                - Find related code sections
                - Suggest debugging approaches
                Provide systematic error analysis."""
            ),
            "fix_generator": Agent(
                name="fix_generator",
                system_prompt="""You are a fix generation specialist. Based on error analysis:
                - Generate specific fix suggestions
                - Provide code examples
                - Explain implementation steps
                - Consider side effects
                Create actionable fix recommendations."""
            ),
            "validator": Agent(
                name="fix_validator",
                system_prompt="""You are a fix validation specialist. Review proposed fixes:
                - Verify fix correctness
                - Check for new issues
                - Validate against requirements
                - Suggest improvements
                Ensure fixes are robust and complete."""
            )
        }
        return agents

    async def build_graph(self, graph_type: Optional[str] = None):
        """Build the graph based on type."""
        if graph_type:
            self.graph_type = graph_type

        builder = GraphBuilder()

        if self.graph_type == "code_analysis":
            self.agents = self._create_code_analysis_agents()

            # Add nodes
            builder.add_node(self.agents["parser"], "parse")
            builder.add_node(self.agents["analyzer"], "analyze")
            builder.add_node(self.agents["documenter"], "document")

            # Define flow: parse -> analyze -> document
            builder.add_edge("parse", "analyze")
            builder.add_edge("analyze", "document")
            builder.set_entry_point("parse")

        elif self.graph_type == "data_structures":
            self.agents = self._create_data_structure_agents()

            builder.add_node(self.agents["identifier"], "identify")
            builder.add_node(self.agents["flow_mapper"], "map_flow")
            builder.add_node(self.agents["optimizer"], "optimize")

            # Parallel flow: identify -> (map_flow + optimize)
            builder.add_edge("identify", "map_flow")
            builder.add_edge("identify", "optimize")
            builder.set_entry_point("identify")

        elif self.graph_type == "debugging":
            self.agents = self._create_debugging_agents()

            builder.add_node(self.agents["error_analyzer"], "analyze_error")
            builder.add_node(self.agents["fix_generator"], "generate_fix")
            builder.add_node(self.agents["validator"], "validate_fix")

            # Sequential debugging flow
            builder.add_edge("analyze_error", "generate_fix")
            builder.add_edge("generate_fix", "validate_fix")
            builder.set_entry_point("analyze_error")

        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        # Set execution limits
        builder.set_max_node_executions(10)
        builder.set_execution_timeout(300)  # 5 minutes

        self.graph = builder.build()
        logger.info(f"Built {self.graph_type} graph with {len(self.agents)} agents")

    async def analyze_code(self, code_path: Union[str, Path], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze code using the graph workflow."""
        if not self.graph:
            await self.build_graph("code_analysis")

        code_path = Path(code_path)
        if not code_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")

        # Read code content
        with open(code_path, 'r', encoding='utf-8') as f:
            code_content = f.read()

        # Prepare input
        analysis_input = f"""
        File: {code_path}
        Content:
        {code_content}

        Context: {context or {}}

        Please analyze this code thoroughly.
        """

        logger.info(f"Starting code analysis for {code_path}")
        result = await self.graph.invoke_async(analysis_input)

        return {
            "file_path": str(code_path),
            "graph_type": self.graph_type,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def analyze_data_structures(self, code_content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze data structures using the graph workflow."""
        if not self.graph or self.graph_type != "data_structures":
            await self.build_graph("data_structures")

        analysis_input = f"""
        Code Content:
        {code_content}

        Context: {context or {}}

        Please analyze all data structures and their relationships.
        """

        logger.info("Starting data structure analysis")
        result = await self.graph.invoke_async(analysis_input)

        return {
            "graph_type": self.graph_type,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def debug_issue(self, error_info: str, code_context: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Debug an issue using the debugging workflow."""
        if not self.graph or self.graph_type != "debugging":
            await self.build_graph("debugging")

        debug_input = f"""
        Error Information:
        {error_info}

        Code Context:
        {code_context}

        Additional Context: {context or {}}

        Please analyze this error and provide fixes.
        """

        logger.info("Starting debugging workflow")
        result = await self.graph.invoke_async(debug_input)

        return {
            "graph_type": self.graph_type,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }

    def get_graph_status(self) -> Dict[str, Any]:
        """Get current graph status."""
        return {
            "graph_type": self.graph_type,
            "model_name": self.model_name,
            "agents_count": len(self.agents),
            "agents": list(self.agents.keys()),
            "graph_built": self.graph is not None
        }

# Factory functions for different graph types

def create_code_analysis_graph(model_name: str = "llama3.2:3b") -> ProgrammingGraph:
    """Create a graph optimized for code analysis."""
    return ProgrammingGraph("code_analysis", model_name)

def create_data_structure_graph(model_name: str = "llama3.2:3b") -> ProgrammingGraph:
    """Create a graph optimized for data structure analysis."""
    return ProgrammingGraph("data_structures", model_name)

def create_debugging_graph(model_name: str = "llama3.2:3b") -> ProgrammingGraph:
    """Create a graph optimized for debugging workflows."""
    return ProgrammingGraph("debugging", model_name)

# Utility functions

async def quick_code_analysis(file_path: Union[str, Path], model_name: str = "llama3.2:3b") -> Dict[str, Any]:
    """Quick code analysis using default graph."""
    graph = create_code_analysis_graph(model_name)
    return await graph.analyze_code(file_path)

async def quick_data_structure_analysis(code_content: str, model_name: str = "llama3.2:3b") -> Dict[str, Any]:
    """Quick data structure analysis using default graph."""
    graph = create_data_structure_graph(model_name)
    return await graph.analyze_data_structures(code_content)