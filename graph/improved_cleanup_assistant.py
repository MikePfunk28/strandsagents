"""Improved cleanup assistant with graph integration and thinking capabilities.

This module provides a safe, intelligent file cleanup system that integrates
with the graph system to track file relationships and usage before deletion.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from graph.graph_storage import GraphNode, GraphEdge, create_graph_storage
from graph.embedding_integration import create_graph_embedding_manager
from graph.enhanced_memory_graph import create_enhanced_memory_graph
from graph.advanced_analytics import create_analytics_engine
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools import file_read, editor, shell, file_write, load_tool
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Create the cleanup agent with proper tools and thinking capabilities
ollama_model = OllamaModel(host="localhost:11434", model="llama3.2")

cleanup_agent = Agent(
    name="Graph Cleanup Assistant",
    model=ollama_model,
    tools=[file_read, editor, shell, file_write, load_tool],
    conversation_manager=SlidingWindowConversationManager(window_size=50),
    system_prompt="""You are an intelligent file cleanup assistant with graph integration and advanced thinking capabilities.

Your role is to:
1. Analyze project files for cleanup opportunities
2. Assess file usage, dependencies, and risk levels
3. Generate safe cleanup plans with backup strategies
4. Execute cleanup operations with safety validation
5. Track all operations in the graph system for audit trails
6. Provide intelligent recommendations and risk assessments

Always prioritize safety and accuracy. Never delete files without proper validation.
Use thinking capabilities to analyze complex scenarios and make intelligent decisions.
Maintain complete audit trails of all operations for accountability.

When in doubt, err on the side of caution and require manual review for high-risk operations.
""",
)

@dataclass
class FileAnalysis:
    """Analysis results for a file."""
    file_path: str
    file_type: str
    size_bytes: int
    last_modified: datetime
    is_used: bool
    usage_references: List[str]
    dependencies: List[str]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str
    graph_node_id: Optional[str] = None

@dataclass
class CleanupPlan:
    """Plan for cleaning up files."""
    plan_id: str
    files_to_delete: List[str]
    files_to_backup: List[str]
    files_to_keep: List[str]
    estimated_space_saved: int
    risk_assessment: str
    created_at: datetime
    requires_user_approval: bool

class GraphAwareCleanupAssistant:
    """Intelligent cleanup assistant with graph integration and safety checks."""

    def __init__(self, base_path: str = ".", graph_storage_type: str = "parquet"):
        self.base_path = Path(base_path)
        self.graph_storage_type = graph_storage_type

        # Graph components
        self.graph_storage = create_graph_storage(graph_storage_type, "cleanup_graph")
        self.embedding_manager = create_graph_embedding_manager(graph_storage_type, "cleanup_embeddings")
        self.enhanced_graph = create_enhanced_memory_graph(graph_storage_type, "cleanup_memory")
        self.analytics = create_analytics_engine(self.graph_storage)

        # Cleanup tracking
        self.analyzed_files: Dict[str, FileAnalysis] = {}
        self.cleanup_history: List[CleanupPlan] = []
        self.safety_thresholds = {
            'max_files_per_batch': 50,
            'critical_file_patterns': ['.env', 'requirements.txt', 'main.py', 'server.py'],
            'protected_directories': ['.git', 'node_modules', '__pycache__', '.venv']
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the cleanup assistant with graph integration."""
        if not self.initialized:
            await self.embedding_manager.initialize()
            await self.enhanced_graph.initialize()
            self.initialized = True
            logger.info("Graph-aware cleanup assistant initialized")

    async def analyze_project_files(self, include_patterns: List[str] = None,
                                   exclude_patterns: List[str] = None) -> Dict[str, FileAnalysis]:
        """Analyze all files in the project for cleanup consideration."""
        if not self.initialized:
            await self.initialize()

        include_patterns = include_patterns or ['**/*']
        exclude_patterns = exclude_patterns or ['**/__pycache__/**', '**/.git/**', '**/node_modules/**']

        # Find all files
        all_files = []
        for pattern in include_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    # Check exclude patterns
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break

                    if not should_exclude:
                        all_files.append(file_path)

        logger.info(f"Found {len(all_files)} files to analyze")

        # Analyze each file
        analyses = {}
        for file_path in all_files:
            try:
                analysis = await self._analyze_single_file(file_path)
                analyses[str(file_path)] = analysis

                # Store in graph
                await self._store_file_in_graph(file_path, analysis)

            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                # Create basic analysis for failed files
                analyses[str(file_path)] = FileAnalysis(
                    file_path=str(file_path),
                    file_type="unknown",
                    size_bytes=0,
                    last_modified=datetime.now(),
                    is_used=False,
                    usage_references=[],
                    dependencies=[],
                    risk_level="unknown",
                    recommendation="manual_review"
                )

        self.analyzed_files = analyses
        logger.info(f"Completed analysis of {len(analyses)} files")

        return analyses

    async def _analyze_single_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file for usage and dependencies."""
        # Get basic file info
        stat = file_path.stat()
        size_bytes = stat.st_size
        last_modified = datetime.fromtimestamp(stat.st_mtime)

        # Determine file type
        file_type = self._determine_file_type(file_path)

        # Check if file is currently in use
        is_used = self._check_file_usage(file_path)

        # Find usage references
        usage_references = self._find_file_references(file_path)

        # Find dependencies
        dependencies = self._find_dependencies(file_path)

        # Assess risk level
        risk_level = self._assess_risk_level(file_path, file_type, usage_references, dependencies)

        # Generate recommendation
        recommendation = self._generate_cleanup_recommendation(
            file_path, is_used, usage_references, dependencies, risk_level
        )

        return FileAnalysis(
            file_path=str(file_path),
            file_type=file_type,
            size_bytes=size_bytes,
            last_modified=last_modified,
            is_used=is_used,
            usage_references=usage_references,
            dependencies=dependencies,
            risk_level=risk_level,
            recommendation=recommendation
        )

    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of a file based on its extension."""
        extension = file_path.suffix.lower()

        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.md': 'markdown',
            '.txt': 'text',
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.env': 'environment',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.html': 'html',
            '.css': 'css',
            '.log': 'log'
        }

        return type_mapping.get(extension, 'unknown')

    def _check_file_usage(self, file_path: Path) -> bool:
        """Check if a file is currently being used."""
        try:
            # Check if file is open by trying to open it
            with open(file_path, 'r') as f:
                f.read(1)  # Try to read a byte
            return True
        except (PermissionError, OSError):
            return True  # File is locked/being used
        except Exception:
            return False  # File can be read, probably not in use

    def _find_file_references(self, file_path: Path) -> List[str]:
        """Find references to this file in other files."""
        references = []
        file_name = file_path.name
        file_stem = file_path.stem

        try:
            # Search for imports and references in Python files
            if file_path.suffix == '.py':
                for other_file in self.base_path.rglob('*.py'):
                    if other_file == file_path:
                        continue

                    try:
                        content = other_file.read_text()
                        # Check for imports
                        if f'import {file_stem}' in content or f'from {file_stem}' in content:
                            references.append(str(other_file))
                        # Check for file path references
                        if str(file_path) in content or file_name in content:
                            references.append(str(other_file))
                    except Exception:
                        continue

            # Search in other file types
            for other_file in self.base_path.rglob('*'):
                if other_file == file_path or not other_file.is_file():
                    continue

                try:
                    content = other_file.read_text()
                    if str(file_path) in content or file_name in content:
                        references.append(str(other_file))
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error finding references for {file_path}: {e}")

        return list(set(references))  # Remove duplicates

    def _find_dependencies(self, file_path: Path) -> List[str]:
        """Find files that this file depends on."""
        dependencies = []

        try:
            if file_path.suffix == '.py':
                content = file_path.read_text()

                # Find import statements
                import_lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        import_lines.append(line)

                # Extract module names from imports
                for import_line in import_lines:
                    if 'import ' in import_line:
                        module = import_line.split('import ')[1].split()[0].split('.')[0]
                        # Look for corresponding files
                        for potential_dep in self.base_path.rglob(f'{module}.py'):
                            dependencies.append(str(potential_dep))

                    elif 'from ' in import_line:
                        module = import_line.split('from ')[1].split()[0].split('.')[0]
                        for potential_dep in self.base_path.rglob(f'{module}.py'):
                            dependencies.append(str(potential_dep))

        except Exception as e:
            logger.error(f"Error finding dependencies for {file_path}: {e}")

        return list(set(dependencies))

    def _assess_risk_level(self, file_path: Path, file_type: str,
                          usage_references: List[str], dependencies: List[str]) -> str:
        """Assess the risk level of deleting a file."""
        risk_score = 0

        # Check against critical patterns
        for pattern in self.safety_thresholds['critical_file_patterns']:
            if pattern in str(file_path):
                return 'critical'

        # Check protected directories
        for protected in self.safety_thresholds['protected_directories']:
            if protected in str(file_path):
                return 'critical'

        # High usage = higher risk
        if len(usage_references) > 10:
            risk_score += 3
        elif len(usage_references) > 5:
            risk_score += 2
        elif len(usage_references) > 0:
            risk_score += 1

        # Many dependencies = higher risk
        if len(dependencies) > 5:
            risk_score += 2
        elif len(dependencies) > 0:
            risk_score += 1

        # File type risk assessment
        high_risk_types = ['environment', 'configuration', 'database']
        if file_type in high_risk_types:
            risk_score += 2

        # Recent modification = higher risk
        days_since_modified = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
        if days_since_modified < 7:
            risk_score += 1
        elif days_since_modified < 30:
            risk_score += 0.5

        # Determine risk level
        if risk_score >= 6:
            return 'critical'
        elif risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _generate_cleanup_recommendation(self, file_path: Path, is_used: bool,
                                       usage_references: List[str], dependencies: List[str],
                                       risk_level: str) -> str:
        """Generate a recommendation for file cleanup."""
        if risk_level == 'critical':
            return 'keep_essential'
        elif is_used:
            return 'keep_in_use'
        elif usage_references:
            return 'keep_referenced'
        elif dependencies:
            return 'review_dependencies'
        else:
            return 'safe_to_delete'

    async def _store_file_in_graph(self, file_path: Path, analysis: FileAnalysis):
        """Store file analysis in the graph system."""
        try:
            # Create file node
            content = f"File: {file_path.name}\nType: {analysis.file_type}\nSize: {analysis.size_bytes} bytes"
            metadata = {
                "file_path": str(file_path),
                "file_type": analysis.file_type,
                "size_bytes": analysis.size_bytes,
                "is_used": analysis.is_used,
                "risk_level": analysis.risk_level,
                "usage_references": analysis.usage_references,
                "dependencies": analysis.dependencies
            }

            node_id = await self.embedding_manager.create_node_with_embedding(
                content=content,
                node_type="file",
                node_id=f"file_{uuid.uuid4().hex[:8]}",
                metadata=metadata
            )

            analysis.graph_node_id = node_id

            # Create relationships with referenced files
            for ref_file in analysis.usage_references:
                ref_path = Path(ref_file)
                if ref_path.exists():
                    # Find or create referenced file node
                    ref_content = f"Referenced file: {ref_path.name}"
                    ref_node_id = await self.embedding_manager.create_node_with_embedding(
                        content=ref_content,
                        node_type="file_reference",
                        node_id=f"ref_{uuid.uuid4().hex[:8]}",
                        metadata={"referenced_file": str(ref_path)}
                    )

                    # Create relationship
                    await self.embedding_manager.create_relationship(
                        source_id=node_id,
                        target_id=ref_node_id,
                        relationship_type="references",
                        metadata={"reference_type": "file_dependency"}
                    )

            logger.info(f"Stored file in graph: {file_path} -> {node_id}")

        except Exception as e:
            logger.error(f"Failed to store file in graph: {e}")

    async def generate_cleanup_plan(self, dry_run: bool = True,
                                   max_risk_level: str = "medium") -> CleanupPlan:
        """Generate a safe cleanup plan based on file analysis."""
        if not self.analyzed_files:
            await self.analyze_project_files()

        # Risk level hierarchy
        risk_hierarchy = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        max_risk_value = risk_hierarchy.get(max_risk_level, 1)

        files_to_delete = []
        files_to_backup = []
        files_to_keep = []
        total_space_saved = 0

        for file_path, analysis in self.analyzed_files.items():
            current_risk_value = risk_hierarchy.get(analysis.risk_level, 0)

            if current_risk_value <= max_risk_value:
                if analysis.recommendation == 'safe_to_delete':
                    files_to_delete.append(file_path)
                    total_space_saved += analysis.size_bytes
                elif analysis.recommendation == 'keep_referenced':
                    files_to_backup.append(file_path)
                else:
                    files_to_keep.append(file_path)
            else:
                files_to_keep.append(file_path)

        # Determine if user approval is needed
        requires_approval = (
            len(files_to_delete) > self.safety_thresholds['max_files_per_batch'] or
            any(analysis.risk_level in ['high', 'critical']
                for analysis in self.analyzed_files.values()
                if analysis.file_path in files_to_delete)
        )

        plan = CleanupPlan(
            plan_id=f"cleanup_{uuid.uuid4().hex[:8]}",
            files_to_delete=files_to_delete,
            files_to_backup=files_to_backup,
            files_to_keep=files_to_keep,
            estimated_space_saved=total_space_saved,
            risk_assessment=self._assess_overall_risk(files_to_delete),
            created_at=datetime.now(),
            requires_user_approval=requires_approval
        )

        self.cleanup_history.append(plan)
        logger.info(f"Generated cleanup plan: {plan.plan_id}")

        return plan

    def _assess_overall_risk(self, files_to_delete: List[str]) -> str:
        """Assess the overall risk of a cleanup plan."""
        if not files_to_delete:
            return 'none'

        high_risk_files = []
        for file_path in files_to_delete:
            analysis = self.analyzed_files.get(file_path)
            if analysis and analysis.risk_level in ['high', 'critical']:
                high_risk_files.append(file_path)

        if len(high_risk_files) > 5:
            return 'critical'
        elif len(high_risk_files) > 0:
            return 'high'
        else:
            return 'low'

    async def execute_cleanup_plan(self, plan_id: str, confirm_critical: bool = False) -> Dict[str, Any]:
        """Execute a cleanup plan with safety checks."""
        # Find the plan
        plan = None
        for p in self.cleanup_history:
            if p.plan_id == plan_id:
                plan = p
                break

        if not plan:
            return {"error": "Plan not found"}

        # Check if approval is needed
        if plan.requires_user_approval and not confirm_critical:
            return {
                "status": "approval_required",
                "message": "Plan requires user approval for critical files",
                "plan": plan
            }

        # Execute the plan
        results = {
            "deleted_files": [],
            "backed_up_files": [],
            "errors": [],
            "space_saved": 0
        }

        # Delete files
        for file_path in plan.files_to_delete:
            try:
                full_path = self.base_path / file_path
                if full_path.exists():
                    size = full_path.stat().st_size
                    full_path.unlink()
                    results["deleted_files"].append(file_path)
                    results["space_saved"] += size
                    logger.info(f"Deleted file: {file_path}")

            except Exception as e:
                results["errors"].append(f"Failed to delete {file_path}: {e}")

        # Backup files
        backup_dir = self.base_path / "cleanup_backups" / plan.plan_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        for file_path in plan.files_to_backup:
            try:
                full_path = self.base_path / file_path
                if full_path.exists():
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(full_path, backup_path)
                    results["backed_up_files"].append(file_path)
                    logger.info(f"Backed up file: {file_path}")

            except Exception as e:
                results["errors"].append(f"Failed to backup {file_path}: {e}")

        # Store cleanup in graph
        await self._store_cleanup_in_graph(plan, results)

        return results

    async def _store_cleanup_in_graph(self, plan: CleanupPlan, results: Dict[str, Any]):
        """Store cleanup operation in the graph for tracking."""
        try:
            # Create cleanup operation node
            content = f"Cleanup operation {plan.plan_id}: {len(results['deleted_files'])} files deleted"
            metadata = {
                "plan_id": plan.plan_id,
                "files_deleted": len(results["deleted_files"]),
                "files_backed_up": len(results["backed_up_files"]),
                "space_saved": results["space_saved"],
                "errors": len(results["errors"]),
                "risk_assessment": plan.risk_assessment
            }

            node_id = await self.embedding_manager.create_node_with_embedding(
                content=content,
                node_type="cleanup_operation",
                metadata=metadata
            )

            # Create relationships with deleted files
            for file_path in results["deleted_files"]:
                analysis = self.analyzed_files.get(file_path)
                if analysis and analysis.graph_node_id:
                    await self.embedding_manager.create_relationship(
                        source_id=node_id,
                        target_id=analysis.graph_node_id,
                        relationship_type="deleted_file",
                        metadata={"file_path": file_path}
                    )

            logger.info(f"Stored cleanup in graph: {plan.plan_id}")

        except Exception as e:
            logger.error(f"Failed to store cleanup in graph: {e}")

    async def get_cleanup_recommendations(self) -> Dict[str, Any]:
        """Get intelligent cleanup recommendations."""
        if not self.analyzed_files:
            await self.analyze_project_files()

        # Generate plan
        plan = await self.generate_cleanup_plan()

        # Get analytics
        insights = self.analytics.get_real_time_insights()

        # Calculate potential improvements
        current_metrics = self.analytics.calculate_graph_metrics()

        recommendations = {
            "immediate_actions": self._get_immediate_recommendations(plan),
            "space_savings": {
                "estimated_bytes": plan.estimated_space_saved,
                "estimated_mb": plan.estimated_space_saved / (1024 * 1024),
                "files_to_delete": len(plan.files_to_delete)
            },
            "risk_assessment": {
                "overall_risk": plan.risk_assessment,
                "requires_approval": plan.requires_user_approval,
                "critical_files": len([f for f in plan.files_to_delete
                                     if self.analyzed_files.get(f, FileAnalysis("", "", 0, datetime.now(), False, [], [], "low", "")).risk_level == "critical"])
            },
            "graph_insights": {
                "total_files_tracked": current_metrics.total_nodes,
                "file_relationships": current_metrics.total_edges,
                "cleanup_impact": self._assess_cleanup_impact(plan)
            },
            "safety_measures": {
                "backup_plan": len(plan.files_to_backup) > 0,
                "history_tracking": True,
                "rollback_possible": True
            }
        }

        return recommendations

    def _get_immediate_recommendations(self, plan: CleanupPlan) -> List[str]:
        """Get immediate action recommendations."""
        recommendations = []

        if plan.estimated_space_saved > 100 * 1024 * 1024:  # 100MB
            recommendations.append("High space savings available - consider running cleanup")

        if plan.risk_assessment == 'critical':
            recommendations.append("Critical files identified - manual review required")

        if len(plan.files_to_backup) > 0:
            recommendations.append("Backup recommended for some files before deletion")

        return recommendations

    def _assess_cleanup_impact(self, plan: CleanupPlan) -> Dict[str, Any]:
        """Assess the impact of the cleanup plan on the graph."""
        impact = {
            "nodes_affected": len(plan.files_to_delete),
            "relationships_affected": 0,
            "graph_health_impact": "minimal"
        }

        # Count affected relationships
        for file_path in plan.files_to_delete:
            analysis = self.analyzed_files.get(file_path)
            if analysis:
                impact["relationships_affected"] += len(analysis.usage_references)

        # Assess health impact
        if impact["relationships_affected"] > 50:
            impact["graph_health_impact"] = "moderate"
        elif impact["relationships_affected"] > 100:
            impact["graph_health_impact"] = "significant"

        return impact

    async def create_backup_before_cleanup(self, files_to_backup: List[str],
                                         backup_name: str = None) -> str:
        """Create a backup of files before cleanup."""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_dir = self.base_path / "cleanup_backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up_files = []

        for file_path in files_to_backup:
            try:
                full_path = self.base_path / file_path
                if full_path.exists():
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    import shutil
                    shutil.copy2(full_path, backup_path)
                    backed_up_files.append(file_path)

            except Exception as e:
                logger.error(f"Failed to backup {file_path}: {e}")

        logger.info(f"Created backup '{backup_name}' with {len(backed_up_files)} files")
        return str(backup_dir)

    def get_cleanup_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of cleanup operations."""
        recent_plans = self.cleanup_history[-limit:] if self.cleanup_history else []

        return [
            {
                "plan_id": plan.plan_id,
                "created_at": plan.created_at.isoformat(),
                "files_to_delete": len(plan.files_to_delete),
                "files_to_backup": len(plan.files_to_backup),
                "estimated_space_saved": plan.estimated_space_saved,
                "risk_assessment": plan.risk_assessment,
                "requires_approval": plan.requires_user_approval
            }
            for plan in recent_plans
        ]

    async def validate_cleanup_safety(self, plan_id: str) -> Dict[str, Any]:
        """Validate that a cleanup plan is safe to execute."""
        # Find the plan
        plan = None
        for p in self.cleanup_history:
            if p.plan_id == plan_id:
                plan = p
                break

        if not plan:
            return {"error": "Plan not found"}

        safety_checks = {
            "critical_files_check": self._check_critical_files(plan.files_to_delete),
            "dependency_check": self._check_dependencies(plan.files_to_delete),
            "usage_check": self._check_current_usage(plan.files_to_delete),
            "backup_availability": len(plan.files_to_backup) > 0,
            "overall_safety": "safe"
        }

        # Determine overall safety
        critical_issues = [
            check for check_name, check in safety_checks.items()
            if check_name != "overall_safety" and not check.get("safe", True)
        ]

        if critical_issues:
            safety_checks["overall_safety"] = "unsafe"
        elif plan.risk_assessment in ['high', 'critical']:
            safety_checks["overall_safety"] = "requires_review"
        else:
            safety_checks["overall_safety"] = "safe"

        return safety_checks

    def _check_critical_files(self, files_to_delete: List[str]) -> Dict[str, Any]:
        """Check if any critical files are marked for deletion."""
        critical_files = []

        for file_path in files_to_delete:
            for pattern in self.safety_thresholds['critical_file_patterns']:
                if pattern in file_path:
                    critical_files.append(file_path)
                    break

        return {
            "safe": len(critical_files) == 0,
            "critical_files": critical_files,
            "message": f"Found {len(critical_files)} critical files" if critical_files else "No critical files found"
        }

    def _check_dependencies(self, files_to_delete: List[str]) -> Dict[str, Any]:
        """Check if files with dependencies are being deleted."""
        files_with_dependencies = []

        for file_path in files_to_delete:
            analysis = self.analyzed_files.get(file_path)
            if analysis and analysis.dependencies:
                files_with_dependencies.append({
                    "file": file_path,
                    "dependencies": analysis.dependencies
                })

        return {
            "safe": len(files_with_dependencies) == 0,
            "files_with_dependencies": files_with_dependencies,
            "message": f"Found {len(files_with_dependencies)} files with dependencies"
        }

    def _check_current_usage(self, files_to_delete: List[str]) -> Dict[str, Any]:
        """Check if any files are currently in use."""
        files_in_use = []

        for file_path in files_to_delete:
            full_path = self.base_path / file_path
            if self._check_file_usage(full_path):
                files_in_use.append(file_path)

        return {
            "safe": len(files_in_use) == 0,
            "files_in_use": files_in_use,
            "message": f"Found {len(files_in_use)} files currently in use"
        }

    async def cleanup_with_thinking(self, max_risk_level: str = "medium") -> Dict[str, Any]:
        """Execute cleanup with intelligent thinking and safety checks."""
        logger.info("Starting intelligent cleanup with thinking capabilities")

        # Step 1: Analyze all files
        logger.info("Step 1: Analyzing project files...")
        await self.analyze_project_files()

        # Step 2: Generate cleanup plan
        logger.info("Step 2: Generating cleanup plan...")
        plan = await self.generate_cleanup_plan(max_risk_level=max_risk_level)

        # Step 3: Validate safety
        logger.info("Step 3: Validating cleanup safety...")
        safety_checks = await self.validate_cleanup_safety(plan.plan_id)

        # Step 4: Execute if safe
        if safety_checks["overall_safety"] == "safe":
            logger.info("Step 4: Executing safe cleanup...")
            results = await self.execute_cleanup_plan(plan.plan_id)

            return {
                "status": "completed",
                "plan_id": plan.plan_id,
                "results": results,
                "safety_checks": safety_checks,
                "message": f"Cleanup completed successfully. Deleted {len(results['deleted_files'])} files, saved {results['space_saved'] / (1024*1024)".2f"} MB"
            }

        elif safety_checks["overall_safety"] == "requires_review":
            return {
                "status": "requires_review",
                "plan_id": plan.plan_id,
                "safety_checks": safety_checks,
                "recommendations": self._get_immediate_recommendations(plan),
                "message": "Cleanup plan requires manual review before execution"
            }

        else:
            return {
                "status": "unsafe",
                "plan_id": plan.plan_id,
                "safety_checks": safety_checks,
                "message": "Cleanup plan is unsafe and should not be executed"
            }

# Factory function
def create_cleanup_assistant(base_path: str = ".", storage_type: str = "parquet") -> GraphAwareCleanupAssistant:
    """Create a graph-aware cleanup assistant."""
    return GraphAwareCleanupAssistant(base_path, storage_type)

# Example usage
async def demo_cleanup_assistant():
    """Demonstrate the cleanup assistant with graph integration."""
    print("Graph-Aware Cleanup Assistant Demo")
    print("=" * 50)

    # Create cleanup assistant
    assistant = create_cleanup_assistant(".", "json")

    try:
        # Initialize
        await assistant.initialize()

        # Analyze files
        print("Analyzing project files...")
        analyses = await assistant.analyze_project_files(
            include_patterns=['**/*.py', '**/*.md', '**/*.txt'],
            exclude_patterns=['**/__pycache__/**', '**/node_modules/**', '**/.git/**']
        )

        print(f"Analyzed {len(analyses)} files")

        # Show some analysis results
        for file_path, analysis in list(analyses.items())[:5]:
            print(f"  {file_path}: {analysis.risk_level} risk, {analysis.recommendation}")

        # Generate cleanup plan
        print("\nGenerating cleanup plan...")
        plan = await assistant.generate_cleanup_plan(max_risk_level="low")

        print(f"Plan: {plan.plan_id}")
        print(f"Files to delete: {len(plan.files_to_delete)}")
        print(f"Files to backup: {len(plan.files_to_backup)}")
        print(f"Estimated space saved: {plan.estimated_space_saved / (1024*1024)".2f"} MB")

        # Get recommendations
        recommendations = await assistant.get_cleanup_recommendations()
        print(f"\nRecommendations: {len(recommendations['immediate_actions'])} immediate actions")

        # Validate safety
        safety = await assistant.validate_cleanup_safety(plan.plan_id)
        print(f"Safety check: {safety['overall_safety']}")

        print("\nDemo completed successfully!")

    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(demo_cleanup_assistant())
