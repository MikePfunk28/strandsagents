"""Graph-based programming analysis system using Strands GRAPH pattern.

This package provides deterministic graph workflows for code analysis,
data structure visualization, and programming flow coordination.

Example:
    from graph import ProgrammingGraph, create_code_analysis_graph

    graph = create_code_analysis_graph()
    result = await graph.analyze_code("path/to/code.py")
"""

from .programming_graph import ProgrammingGraph, create_code_analysis_graph, create_data_structure_graph

__version__ = "1.0.0"
__all__ = ["ProgrammingGraph", "create_code_analysis_graph", "create_data_structure_graph"]