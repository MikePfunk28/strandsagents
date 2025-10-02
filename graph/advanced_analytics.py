"""Advanced graph analytics and algorithms for the swarm system.

This module provides sophisticated graph algorithms and real-time analytics
for performance insights, pattern detection, and intelligent decision making.
"""

import asyncio
import json
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
import math

import numpy as np
from graph.graph_storage import GraphNode, GraphEdge, GraphStorageManager, create_graph_storage

logger = logging.getLogger(__name__)

@dataclass
class GraphMetrics:
    """Comprehensive graph analytics metrics."""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    average_degree: float
    clustering_coefficient: float
    density: float
    connected_components: int
    average_path_length: float
    modularity: float
    timestamp: datetime

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for swarm agents."""
    agent_id: str
    tasks_completed: int
    average_completion_time: float
    success_rate: float
    capability_scores: Dict[str, float]
    collaboration_score: float
    learning_rate: float
    last_active: datetime

@dataclass
class TaskPattern:
    """Identified patterns in task execution."""
    pattern_id: str
    task_type: str
    common_sequence: List[str]
    frequency: int
    success_rate: float
    required_capabilities: List[str]
    optimal_agents: List[str]

class GraphAnalyticsEngine:
    """Advanced analytics engine for graph-based insights."""

    def __init__(self, storage_manager: GraphStorageManager):
        self.storage = storage_manager
        self.metrics_cache: Optional[GraphMetrics] = None
        self.last_metrics_update: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=5)

    def calculate_graph_metrics(self) -> GraphMetrics:
        """Calculate comprehensive graph metrics."""
        nodes_data = getattr(self.storage.storage, 'nodes_data', [])
        edges_data = getattr(self.storage.storage, 'edges_data', [])

        if not nodes_data:
            return GraphMetrics(
                total_nodes=0, total_edges=0, node_types={}, edge_types={},
                average_degree=0, clustering_coefficient=0, density=0,
                connected_components=0, average_path_length=0, modularity=0,
                timestamp=datetime.now()
            )

        # Basic counts
        total_nodes = len(nodes_data)
        total_edges = len(edges_data)

        # Node and edge type distributions
        node_types = Counter(node.get('node_type', 'unknown') for node in nodes_data)
        edge_types = Counter(edge.get('edge_type', 'unknown') for edge in edges_data)

        # Degree analysis
        degrees = self._calculate_degrees(edges_data)
        average_degree = sum(degrees.values()) / total_nodes if total_nodes > 0 else 0

        # Clustering coefficient
        clustering_coefficient = self._calculate_clustering_coefficient(edges_data)

        # Density
        max_edges = total_nodes * (total_nodes - 1) / 2
        density = total_edges / max_edges if max_edges > 0 else 0

        # Connected components
        connected_components = self._count_connected_components(edges_data, total_nodes)

        # Average path length (simplified)
        average_path_length = self._estimate_average_path_length(total_nodes, total_edges)

        # Modularity (simplified community detection)
        modularity = self._calculate_modularity(edges_data, node_types)

        return GraphMetrics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            node_types=dict(node_types),
            edge_types=dict(edge_types),
            average_degree=average_degree,
            clustering_coefficient=clustering_coefficient,
            density=density,
            connected_components=connected_components,
            average_path_length=average_path_length,
            modularity=modularity,
            timestamp=datetime.now()
        )

    def _calculate_degrees(self, edges_data: List[Dict]) -> Dict[str, int]:
        """Calculate node degrees."""
        degrees = defaultdict(int)

        for edge in edges_data:
            degrees[edge['source_id']] += 1
            degrees[edge['target_id']] += 1

        return dict(degrees)

    def _calculate_clustering_coefficient(self, edges_data: List[Dict]) -> float:
        """Calculate global clustering coefficient."""
        if not edges_data:
            return 0.0

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in edges_data:
            adjacency[edge['source_id']].add(edge['target_id'])
            adjacency[edge['target_id']].add(edge['source_id'])

        # Calculate local clustering coefficients
        clustering_coeffs = []
        for node, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in adjacency[neighbor_list[i]]:
                        triangles += 1

            # Calculate clustering coefficient
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            if possible_triangles > 0:
                clustering_coeffs.append(triangles / possible_triangles)

        return sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0.0

    def _count_connected_components(self, edges_data: List[Dict], total_nodes: int) -> int:
        """Count connected components using DFS."""
        if not edges_data:
            return total_nodes

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in edges_data:
            adjacency[edge['source_id']].add(edge['target_id'])
            adjacency[edge['target_id']].add(edge['source_id'])

        # Find all nodes
        all_nodes = set()
        for edge in edges_data:
            all_nodes.add(edge['source_id'])
            all_nodes.add(edge['target_id'])

        # DFS to count components
        visited = set()
        components = 0

        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

        for node in all_nodes:
            if node not in visited:
                dfs(node)
                components += 1

        return components

    def _estimate_average_path_length(self, nodes: int, edges: int) -> float:
        """Estimate average path length using network theory approximations."""
        if nodes <= 1:
            return 0.0

        # Simplified estimation based on random graph theory
        if edges == 0:
            return float('inf')

        # For sparse graphs, use approximation
        if edges < nodes * math.log(nodes):
            return math.log(nodes) / math.log(edges / nodes) if edges > nodes else 1.0

        # For dense graphs
        return math.log(nodes) / math.log(1 + edges / (nodes * (nodes - 1) / 2))

    def _calculate_modularity(self, edges_data: List[Dict], node_types: Counter) -> float:
        """Calculate graph modularity based on node types."""
        if not edges_data or not node_types:
            return 0.0

        # Group nodes by type
        type_groups = defaultdict(set)
        for node_data in getattr(self.storage.storage, 'nodes_data', []):
            node_type = node_data.get('node_type', 'unknown')
            type_groups[node_type].add(node_data['node_id'])

        # Calculate modularity
        total_edges = len(edges_data)
        if total_edges == 0:
            return 0.0

        modularity = 0.0
        total_degree = sum(self._calculate_degrees(edges_data).values())

        for group_nodes in type_groups.values():
            if not group_nodes:
                continue

            # Count internal edges
            internal_edges = 0
            group_set = set(group_nodes)

            for edge in edges_data:
                if edge['source_id'] in group_set and edge['target_id'] in group_set:
                    internal_edges += 1

            # Calculate expected edges
            group_degree = sum(
                1 for edge in edges_data
                if edge['source_id'] in group_set or edge['target_id'] in group_set
            )

            if total_degree > 0:
                expected_edges = (group_degree ** 2) / (2 * total_edges)
                if expected_edges > 0:
                    modularity += (internal_edges - expected_edges) / total_edges

        return modularity

    def get_agent_performance_metrics(self) -> List[AgentPerformanceMetrics]:
        """Calculate performance metrics for all agents."""
        agent_nodes = []
        for node_data in getattr(self.storage.storage, 'nodes_data', []):
            if node_data.get('node_type') == 'agent':
                agent_nodes.append(node_data)

        metrics = []
        for agent_data in agent_nodes:
            agent_id = agent_data.get('node_id')
            metadata = json.loads(agent_data.get('metadata', '{}'))

            # Get agent-related edges
            agent_edges = self.storage.get_edges(agent_id)

            # Calculate metrics
            tasks_completed = len([e for e in agent_edges if e.edge_type == 'task_completed'])
            success_rate = self._calculate_agent_success_rate(agent_edges)
            capability_scores = self._calculate_capability_scores(agent_id, agent_edges)
            collaboration_score = self._calculate_collaboration_score(agent_id, agent_edges)
            learning_rate = self._calculate_learning_rate(agent_id, agent_edges)

            metrics.append(AgentPerformanceMetrics(
                agent_id=agent_id,
                tasks_completed=tasks_completed,
                average_completion_time=self._calculate_average_completion_time(agent_edges),
                success_rate=success_rate,
                capability_scores=capability_scores,
                collaboration_score=collaboration_score,
                learning_rate=learning_rate,
                last_active=datetime.fromisoformat(agent_data.get('updated_at', datetime.now().isoformat()))
            ))

        return metrics

    def _calculate_agent_success_rate(self, edges: List[GraphEdge]) -> float:
        """Calculate agent's task success rate."""
        completed_tasks = len([e for e in edges if e.edge_type == 'task_completed'])
        failed_tasks = len([e for e in edges if e.edge_type == 'task_failed'])

        total_tasks = completed_tasks + failed_tasks
        return completed_tasks / total_tasks if total_tasks > 0 else 0.0

    def _calculate_capability_scores(self, agent_id: str, edges: List[GraphEdge]) -> Dict[str, float]:
        """Calculate scores for each agent capability."""
        # This would analyze which capabilities lead to successful outcomes
        # Simplified implementation
        return {"general": 0.8}  # Placeholder

    def _calculate_collaboration_score(self, agent_id: str, edges: List[GraphEdge]) -> float:
        """Calculate how well the agent collaborates with others."""
        collaboration_edges = [e for e in edges if e.edge_type == 'collaborated_with']
        return len(collaboration_edges) / max(len(edges), 1) * 1.0

    def _calculate_learning_rate(self, agent_id: str, edges: List[GraphEdge]) -> float:
        """Calculate agent's learning rate over time."""
        # Analyze improvement in performance over time
        # Simplified implementation
        return 0.1  # Placeholder

    def _calculate_average_completion_time(self, edges: List[GraphEdge]) -> float:
        """Calculate average task completion time."""
        # This would analyze timing data from edges
        # Simplified implementation
        return 300.0  # 5 minutes placeholder

    def detect_task_patterns(self) -> List[TaskPattern]:
        """Detect common patterns in task execution."""
        patterns = []

        # Analyze task sequences and outcomes
        task_edges = [e for e in getattr(self.storage.storage, 'edges_data', [])
                     if e.get('edge_type', '').startswith('task_')]

        # Group by task type
        task_types = defaultdict(list)
        for edge in task_edges:
            task_type = edge.get('edge_type', 'unknown')
            task_types[task_type].append(edge)

        for task_type, edges in task_types.items():
            if len(edges) >= 3:  # Need minimum samples
                pattern = self._analyze_task_pattern(task_type, edges)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _analyze_task_pattern(self, task_type: str, edges: List[Dict]) -> Optional[TaskPattern]:
        """Analyze pattern for a specific task type."""
        # Analyze common sequences and outcomes
        # Simplified implementation
        return TaskPattern(
            pattern_id=f"pattern_{task_type}",
            task_type=task_type,
            common_sequence=["start", "process", "complete"],
            frequency=len(edges),
            success_rate=0.85,
            required_capabilities=["general"],
            optimal_agents=["agent_001"]
        )

    def get_real_time_insights(self) -> Dict[str, Any]:
        """Get real-time insights about the graph and swarm performance."""
        current_metrics = self.calculate_graph_metrics()

        # Cache metrics
        self.metrics_cache = current_metrics
        self.last_metrics_update = datetime.now()

        # Get agent performance
        agent_metrics = self.get_agent_performance_metrics()

        # Detect patterns
        task_patterns = self.detect_task_patterns()

        # Generate insights
        insights = {
            "graph_health": self._assess_graph_health(current_metrics),
            "agent_performance": self._summarize_agent_performance(agent_metrics),
            "task_patterns": [p.pattern_id for p in task_patterns],
            "recommendations": self._generate_recommendations(current_metrics, agent_metrics),
            "alerts": self._generate_alerts(current_metrics, agent_metrics)
        }

        return insights

    def _assess_graph_health(self, metrics: GraphMetrics) -> Dict[str, Any]:
        """Assess overall graph health."""
        health_score = 0.0
        issues = []

        # Check connectivity
        if metrics.connected_components > 1:
            health_score -= 0.3
            issues.append("Graph is disconnected")

        # Check density
        if metrics.density < 0.01:
            health_score -= 0.1
            issues.append("Graph is very sparse")

        # Check clustering
        if metrics.clustering_coefficient < 0.1:
            health_score -= 0.1
            issues.append("Low clustering coefficient")

        health_score = max(0.0, min(1.0, health_score + 0.8))  # Normalize

        return {
            "score": health_score,
            "status": "healthy" if health_score > 0.7 else "warning" if health_score > 0.4 else "critical",
            "issues": issues
        }

    def _summarize_agent_performance(self, agent_metrics: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Summarize agent performance metrics."""
        if not agent_metrics:
            return {"summary": "No agents found"}

        avg_success_rate = sum(a.success_rate for a in agent_metrics) / len(agent_metrics)
        total_tasks = sum(a.tasks_completed for a in agent_metrics)
        avg_collaboration = sum(a.collaboration_score for a in agent_metrics) / len(agent_metrics)

        return {
            "total_agents": len(agent_metrics),
            "average_success_rate": avg_success_rate,
            "total_tasks_completed": total_tasks,
            "average_collaboration_score": avg_collaboration,
            "top_performers": sorted(agent_metrics, key=lambda x: x.success_rate, reverse=True)[:3]
        }

    def _generate_recommendations(self, graph_metrics: GraphMetrics,
                                agent_metrics: List[AgentPerformanceMetrics]) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []

        # Graph-based recommendations
        if graph_metrics.density < 0.05:
            recommendations.append("Consider increasing agent interactions to improve knowledge sharing")

        if graph_metrics.connected_components > 1:
            recommendations.append("Graph has disconnected components - consider improving agent coordination")

        # Agent-based recommendations
        low_performers = [a for a in agent_metrics if a.success_rate < 0.5]
        if low_performers:
            recommendations.append(f"Consider retraining {len(low_performers)} low-performing agents")

        return recommendations

    def _generate_alerts(self, graph_metrics: GraphMetrics,
                        agent_metrics: List[AgentPerformanceMetrics]) -> List[str]:
        """Generate alerts for critical issues."""
        alerts = []

        # Critical graph issues
        if graph_metrics.connected_components > 3:
            alerts.append("CRITICAL: Graph is highly fragmented")

        # Critical agent issues
        failing_agents = [a for a in agent_metrics if a.success_rate < 0.2]
        if failing_agents:
            alerts.append(f"CRITICAL: {len(failing_agents)} agents have very low success rates")

        return alerts

    def get_metrics(self, force_refresh: bool = False) -> GraphMetrics:
        """Get cached or fresh graph metrics."""
        if (not force_refresh and self.metrics_cache and
            self.last_metrics_update and
            datetime.now() - self.last_metrics_update < self.cache_duration):
            return self.metrics_cache

        return self.calculate_graph_metrics()

    def export_analytics_report(self, output_path: str) -> str:
        """Export comprehensive analytics report."""
        try:
            metrics = self.get_metrics(force_refresh=True)
            agent_metrics = self.get_agent_performance_metrics()
            task_patterns = self.detect_task_patterns()
            insights = self.get_real_time_insights()

            report = {
                "timestamp": datetime.now().isoformat(),
                "graph_metrics": {
                    "total_nodes": metrics.total_nodes,
                    "total_edges": metrics.total_edges,
                    "node_types": metrics.node_types,
                    "edge_types": metrics.edge_types,
                    "average_degree": metrics.average_degree,
                    "clustering_coefficient": metrics.clustering_coefficient,
                    "density": metrics.density,
                    "connected_components": metrics.connected_components,
                    "average_path_length": metrics.average_path_length,
                    "modularity": metrics.modularity
                },
                "agent_performance": [
                    {
                        "agent_id": a.agent_id,
                        "tasks_completed": a.tasks_completed,
                        "success_rate": a.success_rate,
                        "collaboration_score": a.collaboration_score,
                        "learning_rate": a.learning_rate
                    }
                    for a in agent_metrics
                ],
                "task_patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "task_type": p.task_type,
                        "frequency": p.frequency,
                        "success_rate": p.success_rate
                    }
                    for p in task_patterns
                ],
                "insights": insights
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Analytics report exported to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export analytics report: {e}")
            raise

# Factory function
def create_analytics_engine(storage_manager: GraphStorageManager) -> GraphAnalyticsEngine:
    """Create a graph analytics engine."""
    return GraphAnalyticsEngine(storage_manager)

# Example usage
async def demo_analytics():
    """Demonstrate advanced graph analytics."""
    print("Graph Analytics Demo")
    print("=" * 50)

    # Create sample storage with data
    storage = create_graph_storage("json", "graph/analytics_demo")

    # Add sample nodes and edges
    nodes = [
        GraphNode("agent_1", "agent", "Research Agent", metadata={"capabilities": ["research"]}),
        GraphNode("agent_2", "agent", "Creative Agent", metadata={"capabilities": ["creative"]}),
        GraphNode("task_1", "task", "Research Task"),
        GraphNode("task_2", "task", "Creative Task"),
    ]

    edges = [
        GraphEdge("edge_1", "task_1", "agent_1", "assigned_to"),
        GraphEdge("edge_2", "task_2", "agent_2", "assigned_to"),
        GraphEdge("edge_3", "agent_1", "agent_2", "collaborates_with"),
    ]

    for node in nodes:
        storage.add_node(node)

    for edge in edges:
        storage.add_edge(edge)

    # Create analytics engine
    analytics = create_analytics_engine(storage)

    # Calculate metrics
    metrics = analytics.calculate_graph_metrics()
    print(f"Graph Metrics: {metrics.total_nodes} nodes, {metrics.total_edges} edges")
    print(f"Average degree: {metrics.average_degree:.2f}")
    print(f"Clustering coefficient: {metrics.clustering_coefficient:.3f}")

    # Get agent performance
    agent_metrics = analytics.get_agent_performance_metrics()
    print(f"Agent metrics: {len(agent_metrics)} agents analyzed")

    # Detect patterns
    patterns = analytics.detect_task_patterns()
    print(f"Task patterns detected: {len(patterns)}")

    # Get real-time insights
    insights = analytics.get_real_time_insights()
    print(f"Graph health score: {insights['graph_health']['score']:.2f}")

    # Export report
    report_path = "graph/analytics_report.json"
    analytics.export_analytics_report(report_path)
    print(f"Analytics report exported to {report_path}")

    print("Analytics demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_analytics())
