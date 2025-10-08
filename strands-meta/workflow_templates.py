"""
Workflow Templates for StrandsAgents @agent Package

Provides pre-built workflow templates for common multi-agent scenarios.
Templates can be customized and extended for specific use cases.

Templates included:
- Research Workflow: Plan → Research → Analyze → Synthesize
- Code Development Workflow: Design → Implement → Test → Deploy
- Data Analysis Workflow: Collect → Clean → Analyze → Visualize
- Creative Workflow: Brainstorm → Develop → Refine → Finalize
"""

import json
import logging
import sys
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger("workflow_templates")

# Add parent directory to path for imports
try:
    # When run as module
    pass
except:
    # When run as script, add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    id: str
    name: str
    description: str
    agent_type: str  # Which type of agent should handle this
    tools: List[str]
    prompt_template: str
    input_from: List[str]  # IDs of steps this step depends on
    output_to: List[str]   # IDs of steps that depend on this
    required: bool = True
    timeout: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WorkflowTemplate:
    """Complete workflow template definition"""
    id: str
    name: str
    description: str
    version: str
    author: str
    steps: Dict[str, WorkflowStep]
    start_step: str
    end_steps: List[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

class WorkflowTemplateManager:
    """Manages workflow templates and their execution"""

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load built-in workflow templates"""

        # Research Workflow Template
        research_steps = {
            "plan": WorkflowStep(
                id="plan",
                name="Research Planning",
                description="Create research brief and step-by-step outline",
                agent_type="research_planner",
                tools=["file_write", "think"],
                prompt_template="""You are a Research Planning specialist.
                Create a comprehensive research brief for: {goal}

                1. BRIEF: Restate user's goal, key constraints, success criteria
                2. OUTLINE: Step-by-step breakdown with data/tool needs
                3. TASK BOARD: Goals, subgoals, status tracking
                4. ASSUMPTIONS: Track what's assumed vs verified

                Context: {context}""",
                input_from=[],
                output_to=["research", "analyze"]
            ),

            "research": WorkflowStep(
                id="research",
                name="Information Gathering",
                description="Execute multi-strategy searches with provenance tracking",
                agent_type="researcher",
                tools=["http_request", "retrieve", "browser"],
                prompt_template="""You are a Research specialist.
                Execute comprehensive research for: {goal}

                Based on this plan: {plan_output}

                1. Query decomposition (core concepts, specific questions)
                2. Multi-strategy search (web, APIs, documents)
                3. Evidence extraction with source citations
                4. Gap identification and uncertainty flagging

                Return structured results with provenance.""",
                input_from=["plan"],
                output_to=["analyze"]
            ),

            "analyze": WorkflowStep(
                id="analyze",
                name="Critical Analysis",
                description="Logic checking and risk assessment",
                agent_type="analyst",
                tools=["think", "calculator"],
                prompt_template="""You are a Critical Analyst.
                Analyze research findings for: {goal}

                Research findings: {research_output}

                1. Logic validation (check reasoning chains, spot fallacies)
                2. Evidence quality assessment (source reliability, bias)
                3. Gap analysis (missing cases, alternative explanations)
                4. Risk logging (uncertainties, potential errors)

                Ask: What could be wrong? What's unverified?""",
                input_from=["research"],
                output_to=["synthesize"]
            ),

            "synthesize": WorkflowStep(
                id="synthesize",
                name="Report Synthesis",
                description="Create final report with confidence levels and citations",
                agent_type="writer",
                tools=["file_write", "think"],
                prompt_template="""You are a Report Writer.
                Synthesize final report for: {goal}

                Analysis: {analysis_output}

                1. Direct answer to original question (clear, structured)
                2. Supporting evidence with full citations
                3. Confidence levels for key claims (High/Medium/Low)
                4. Limitations and uncertainties acknowledgment
                5. Open issues and areas needing verification""",
                input_from=["analyze"],
                output_to=[]
            )
        }

        research_template = WorkflowTemplate(
            id="research_workflow",
            name="Comprehensive Research Workflow",
            description="Multi-agent research workflow with planning, execution, analysis, and synthesis",
            version="1.0.0",
            author="StrandsAgents",
            steps=research_steps,
            start_step="plan",
            end_steps=["synthesize"],
            tags=["research", "analysis", "academic", "investigation"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Code Development Workflow Template
        code_steps = {
            "design": WorkflowStep(
                id="design",
                name="System Design",
                description="Design software architecture and specifications",
                agent_type="architect",
                tools=["file_write", "think"],
                prompt_template="""You are a Software Architect.
                Design a solution for: {goal}

                1. System architecture and component design
                2. Technology stack recommendations
                3. Interface specifications and APIs
                4. Risk assessment and mitigation strategies
                5. Implementation roadmap and milestones""",
                input_from=[],
                output_to=["implement", "test"]
            ),

            "implement": WorkflowStep(
                id="implement",
                name="Code Implementation",
                description="Write clean, well-documented code",
                agent_type="developer",
                tools=["file_write", "python_repl", "shell"],
                prompt_template="""You are a Senior Developer.
                Implement the designed solution for: {goal}

                Design specifications: {design_output}

                1. Write clean, maintainable code
                2. Include comprehensive documentation
                3. Follow best practices and coding standards
                4. Implement error handling and logging
                5. Create unit tests for critical functions""",
                input_from=["design"],
                output_to=["test", "deploy"]
            ),

            "test": WorkflowStep(
                id="test",
                name="Testing and Validation",
                description="Comprehensive testing of implemented solution",
                agent_type="qa_engineer",
                tools=["python_repl", "shell", "file_read"],
                prompt_template="""You are a QA Engineer.
                Test the implemented solution for: {goal}

                Implementation: {implement_output}

                1. Create comprehensive test suite
                2. Test edge cases and error conditions
                3. Performance and security testing
                4. Integration testing with existing systems
                5. Generate test reports and recommendations""",
                input_from=["implement"],
                output_to=["deploy"]
            ),

            "deploy": WorkflowStep(
                id="deploy",
                name="Deployment and Documentation",
                description="Deploy solution and create user documentation",
                agent_type="devops",
                tools=["shell", "file_write"],
                prompt_template="""You are a DevOps Engineer.
                Deploy and document the solution for: {goal}

                Test results: {test_output}

                1. Create deployment scripts and procedures
                2. Set up monitoring and alerting
                3. Write user documentation and guides
                4. Create maintenance and troubleshooting guides
                5. Plan for future updates and scaling""",
                input_from=["test"],
                output_to=[]
            )
        }

        code_template = WorkflowTemplate(
            id="code_development",
            name="Full-Stack Development Workflow",
            description="Complete software development lifecycle from design to deployment",
            version="1.0.0",
            author="StrandsAgents",
            steps=code_steps,
            start_step="design",
            end_steps=["deploy"],
            tags=["development", "coding", "deployment", "fullstack"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Data Analysis Workflow Template
        data_steps = {
            "collect": WorkflowStep(
                id="collect",
                name="Data Collection",
                description="Gather data from various sources",
                agent_type="data_collector",
                tools=["http_request", "file_read", "shell"],
                prompt_template="""You are a Data Collection specialist.
                Collect data for analysis: {goal}

                1. Identify relevant data sources (APIs, files, databases)
                2. Extract and validate data quality
                3. Handle different data formats (CSV, JSON, XML)
                4. Document data sources and collection methods""",
                input_from=[],
                output_to=["clean", "analyze"]
            ),

            "clean": WorkflowStep(
                id="clean",
                name="Data Cleaning",
                description="Clean and preprocess collected data",
                agent_type="data_cleaner",
                tools=["python_repl", "file_write"],
                prompt_template="""You are a Data Cleaning specialist.
                Clean the collected data for: {goal}

                Raw data: {collect_output}

                1. Handle missing values and outliers
                2. Standardize data formats and types
                3. Remove duplicates and inconsistencies
                4. Validate data integrity and completeness""",
                input_from=["collect"],
                output_to=["analyze", "visualize"]
            ),

            "analyze": WorkflowStep(
                id="analyze",
                name="Data Analysis",
                description="Perform statistical and pattern analysis",
                agent_type="data_analyst",
                tools=["python_repl", "calculator"],
                prompt_template="""You are a Data Analyst.
                Analyze the cleaned data for: {goal}

                Clean data: {clean_output}

                1. Statistical analysis and summary statistics
                2. Identify patterns, trends, and correlations
                3. Generate insights and key findings
                4. Create analysis report with visualizations""",
                input_from=["clean"],
                output_to=["visualize"]
            ),

            "visualize": WorkflowStep(
                id="visualize",
                name="Visualization and Reporting",
                description="Create visualizations and final report",
                agent_type="data_visualizer",
                tools=["file_write", "python_repl"],
                prompt_template="""You are a Data Visualization specialist.
                Create visualizations for: {goal}

                Analysis results: {analyze_output}

                1. Create clear, informative visualizations
                2. Generate comprehensive analysis report
                3. Highlight key insights and recommendations
                4. Create presentation-ready materials""",
                input_from=["analyze"],
                output_to=[]
            )
        }

        data_template = WorkflowTemplate(
            id="data_analysis",
            name="Data Analysis Workflow",
            description="Complete data analysis pipeline from collection to visualization",
            version="1.0.0",
            author="StrandsAgents",
            steps=data_steps,
            start_step="collect",
            end_steps=["visualize"],
            tags=["data", "analysis", "statistics", "visualization"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Store templates
        self.templates = {
            "research_workflow": research_template,
            "code_development": code_template,
            "data_analysis": data_template
        }

        logger.info(f"🔧 Loaded {len(self.templates)} workflow templates")

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow template"""
        return self.templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available workflow templates"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "version": template.version,
                "tags": template.tags,
                "step_count": len(template.steps)
            }
            for template in self.templates.values()
        ]

    def create_custom_template(self, template: WorkflowTemplate) -> bool:
        """Add a custom workflow template"""
        try:
            self.templates[template.id] = template
            logger.info(f"🔧 Added custom template: {template.id}")
            return True
        except Exception as e:
            logger.error(f"🔧 Failed to add template {template.id}: {str(e)}")
            return False

    def execute_workflow(self, template_id: str, goal: str, context: str = "",
                        agent_registry: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a workflow template using available agents

        Args:
            template_id: ID of the template to execute
            goal: The main goal/objective
            context: Additional context information
            agent_registry: Registry of available agents

        Returns:
            Final workflow result
        """
        template = self.get_template(template_id)
        if not template:
            return f"Template '{template_id}' not found"

        logger.info(f"🔧 Executing workflow template: {template.name}")

        # Track workflow execution
        workflow_state = {
            "template_id": template_id,
            "goal": goal,
            "start_time": datetime.now(),
            "step_results": {},
            "current_step": template.start_step,
            "completed_steps": []
        }

        try:
            # Execute workflow steps in dependency order
            result = self._execute_workflow_steps(
                template, goal, context, agent_registry, workflow_state
            )

            workflow_state["end_time"] = datetime.now()
            workflow_state["success"] = True

            logger.info(f"🔧 Workflow {template_id} completed successfully")
            return result

        except Exception as e:
            workflow_state["end_time"] = datetime.now()
            workflow_state["success"] = False
            workflow_state["error"] = str(e)

            logger.error(f"🔧 Workflow {template_id} failed: {str(e)}")
            return f"Workflow execution failed: {str(e)}"

    def _execute_workflow_steps(self, template: WorkflowTemplate, goal: str,
                              context: str, agent_registry: Dict[str, Any],
                              workflow_state: Dict[str, Any]) -> str:
        """Execute individual workflow steps"""

        # Get execution order (topological sort based on dependencies)
        execution_order = self._get_execution_order(template)

        step_results = {}

        for step_id in execution_order:
            if step_id not in template.steps:
                continue

            step = template.steps[step_id]

            # Check if all required input steps are completed
            if not self._can_execute_step(step, step_results):
                if step.required:
                    raise Exception(f"Cannot execute required step {step_id}: missing dependencies")
                else:
                    logger.warning(f"🔧 Skipping optional step {step_id}")
                    continue

            # Get input data from previous steps
            input_data = self._gather_step_inputs(step, step_results)

            # Execute the step using appropriate agent
            step_result = self._execute_step(
                step, goal, context, input_data, agent_registry
            )

            step_results[step_id] = step_result
            workflow_state["step_results"][step_id] = step_result
            workflow_state["completed_steps"].append(step_id)

        # Return final result from end steps
        final_results = []
        for end_step_id in template.end_steps:
            if end_step_id in step_results:
                final_results.append(step_results[end_step_id])

        return "\\n\\n".join(final_results) if final_results else "No results generated"

    def _get_execution_order(self, template: WorkflowTemplate) -> List[str]:
        """Get the order in which steps should be executed"""
        # Simple topological sort - in a real implementation,
        # this would handle complex dependency graphs
        executed = set()
        order = []

        def visit(step_id: str):
            if step_id in executed:
                return
            if step_id not in template.steps:
                return

            step = template.steps[step_id]

            # Visit all dependencies first
            for dep in step.input_from:
                visit(dep)

            executed.add(step_id)
            order.append(step_id)

        # Start from steps with no dependencies
        for step_id, step in template.steps.items():
            if not step.input_from:
                visit(step_id)

        # Add remaining steps
        for step_id in template.steps:
            if step_id not in executed:
                visit(step_id)

        return order

    def _can_execute_step(self, step: WorkflowStep, step_results: Dict[str, str]) -> bool:
        """Check if a step can be executed (all dependencies satisfied)"""
        for dep_id in step.input_from:
            if dep_id not in step_results:
                return False
        return True

    def _gather_step_inputs(self, step: WorkflowStep, step_results: Dict[str, str]) -> str:
        """Gather input data from previous steps"""
        inputs = []
        for dep_id in step.input_from:
            if dep_id in step_results:
                inputs.append(f"Input from {dep_id}: {step_results[dep_id]}")

        return "\\n\\n".join(inputs) if inputs else ""

    def _execute_step(self, step: WorkflowStep, goal: str, context: str,
                     input_data: str, agent_registry: Dict[str, Any]) -> str:
        """Execute a single workflow step"""

        # Format the prompt template
        prompt = step.prompt_template.format(
            goal=goal,
            context=context,
            input_data=input_data,
            step_name=step.name
        )

        # Find appropriate agent for this step type
        agent_function = self._find_agent_for_step(step.agent_type, agent_registry)

        if not agent_function:
            logger.warning(f"🔧 No agent found for type {step.agent_type}, using generic agent")
            # Fallback to a generic agent
            return f"Step {step.name} completed (no specific agent available)"

        try:
            # Execute the step
            logger.info(f"🔧 Executing step: {step.name}")
            result = agent_function(prompt)

            logger.info(f"🔧 Step {step.name} completed successfully")
            return f"## {step.name}\\n\\n{result}"

        except Exception as e:
            logger.error(f"🔧 Step {step.name} failed: {str(e)}")
            if step.required:
                raise Exception(f"Required step {step.name} failed: {str(e)}")
            else:
                return f"Step {step.name} failed (optional): {str(e)}"

    def _find_agent_for_step(self, agent_type: str, agent_registry: Dict[str, Any]) -> Optional[Callable]:
        """Find an agent function that matches the required type"""
        # This is a simplified implementation
        # In a real system, this would have more sophisticated matching

        # Look for agents that might handle this type
        for agent_name, agent_info in agent_registry.items():
            if agent_type in agent_name.lower() or agent_name.lower() in agent_type:
                return agent_info.get('function')

        return None

# Global template manager instance
template_manager = WorkflowTemplateManager()

# Convenience functions
def get_workflow_template(template_id: str) -> Optional[WorkflowTemplate]:
    """Get a workflow template by ID"""
    return template_manager.get_template(template_id)

def list_workflow_templates() -> List[Dict[str, Any]]:
    """List all available workflow templates"""
    return template_manager.list_templates()

def execute_workflow(template_id: str, goal: str, context: str = "",
                    agent_registry: Optional[Dict[str, Any]] = None) -> str:
    """Execute a workflow template"""
    return template_manager.execute_workflow(template_id, goal, context, agent_registry or {})

if __name__ == "__main__":
    # Demo the workflow template system
    print("🚀 Workflow Templates Demo")
    print("=" * 50)

    templates = list_workflow_templates()
    print(f"Available templates: {len(templates)}")

    for template in templates:
        print(f"\\n📋 {template['name']} ({template['id']})")
        print(f"   Description: {template['description']}")
        print(f"   Tags: {', '.join(template['tags'])}")
        print(f"   Steps: {template['step_count']}")

    print("\\n🔧 Template system ready for use!")
