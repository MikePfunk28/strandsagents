#!/usr/bin/env python3
"""
# Agentic Workflow: Research Assistant (Task-Board Orchestration)

This example demonstrates a multi-agent Strands workflow that mirrors a
planner -> researcher -> analyst -> writer loop. The agents share a lightweight
task board so they can keep track of constraints, assumptions, open questions,
and verified findings.

## Key Features
- Shared task board captures plan, findings, and risks for every agent
- Planner agent establishes constraints and sub-goals as structured JSON
- Researcher agent uses web tools and writes structured evidence with sources
- Analyst agent critiques findings, surfaces risks, and tracks validation
- Writer agent generates the final user-facing report from the shared state

## How to Run
1. Ensure an Ollama endpoint is running at http://localhost:11434 with `llama3.2`
2. From the project root, run: `python research_assistant.py`
3. Enter a research query or claim when prompted. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import http_request, handoff_to_user

# Create model once so the agents can share it
ollama_model = OllamaModel(host="http://localhost:11434", model_id="llama3.2")

# Agent dedicated to handing control back to the user
handoff_agent = Agent(tools=[handoff_to_user])

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def merge_unique(current: List[str] | None, updates: List[str] | None) -> List[str]:
    """Combine two string lists while preserving order and removing duplicates."""
    seen = set()
    merged: List[str] = []
    for item in (current or []) + (updates or []):
        if not item:
            continue
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def parse_structured_response(raw_output: Any) -> Dict[str, Any]:
    """Attempt to parse a JSON object from an agent response."""
    text = str(raw_output).strip()
    if not text:
        raise ValueError("Agent returned an empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_PATTERN.search(text)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse JSON from response:\n{text}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, received: {parsed!r}")
    return parsed


@dataclass
class TaskBoard:
    """Shared state that every agent can read from or write to."""

    goal: str
    constraints: List[str] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    iteration_log: List[Dict[str, str]] = field(default_factory=list)

    def record_plan(self, payload: Dict[str, Any]) -> None:
        self.constraints = merge_unique(self.constraints, payload.get("constraints"))
        plan = payload.get("plan")
        if isinstance(plan, list):
            self.plan = plan
        self.assumptions = merge_unique(self.assumptions, payload.get("assumptions"))
        self.open_questions = merge_unique(self.open_questions, payload.get("open_questions"))
        risks = payload.get("risks")
        if risks:
            self.analysis["risks"] = merge_unique(self.analysis.get("risks"), risks)

    def add_findings(self, payload: Dict[str, Any]) -> None:
        findings = payload.get("findings", [])
        if isinstance(findings, list):
            self.findings.extend(findings)
        self.gaps = merge_unique(self.gaps, payload.get("gaps"))
        next_actions = payload.get("recommended_next_actions")
        if next_actions:
            self.analysis["next_actions"] = merge_unique(self.analysis.get("next_actions"), next_actions)
        evidence_quality = payload.get("evidence_quality")
        if evidence_quality:
            self.analysis["evidence_quality"] = str(evidence_quality)

    def set_analysis(self, payload: Dict[str, Any]) -> None:
        insights = payload.get("insights")
        if insights:
            self.analysis["insights"] = insights
        summary = payload.get("summary")
        if summary:
            self.analysis["summary"] = summary
        risks = payload.get("risks")
        if risks:
            self.analysis["risks"] = merge_unique(self.analysis.get("risks"), risks)
        validation_checks = payload.get("validation_checks")
        if validation_checks:
            self.analysis["validation_checks"] = merge_unique(self.analysis.get("validation_checks"), validation_checks)
        verdict = payload.get("verdict")
        if verdict:
            self.analysis["verdict"] = verdict
        self.gaps = merge_unique(self.gaps, payload.get("gaps"))

    def record_iteration(self, role: str, summary: str) -> None:
        self.iteration_log.append({"role": role, "summary": summary.strip()})

    def brief(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.constraints:
            lines.append("Constraints: " + "; ".join(self.constraints))
        if self.plan:
            lines.append("Plan:")
            for idx, step in enumerate(self.plan, start=1):
                task = step.get("task") or step.get("description") or step.get("goal") or "unspecified task"
                owner = step.get("owner") or step.get("role") or "unassigned"
                status = step.get("status") or "pending"
                success = step.get("success_criteria") or step.get("metric")
                lines.append(f"  {idx}. [{owner} | {status}] {task}")
                if success:
                    lines.append(f"     success: {success}")
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        if self.open_questions:
            lines.append("Open questions: " + "; ".join(self.open_questions))
        if self.gaps:
            lines.append("Known gaps: " + "; ".join(self.gaps))
        return "\n".join(lines)

    def findings_overview(self) -> str:
        if not self.findings:
            return "No research findings yet."
        lines: List[str] = []
        for idx, item in enumerate(self.findings, start=1):
            summary = item.get("summary") or item.get("finding") or item.get("insight") or "(missing summary)"
            source = item.get("source") or item.get("url") or "unknown source"
            confidence = item.get("confidence")
            confidence_note = f" | confidence: {confidence}" if confidence else ""
            lines.append(f"{idx}. {summary} (source: {source}{confidence_note})")
            evidence = item.get("evidence")
            if evidence:
                lines.append(f"    evidence: {evidence}")
        return "\n".join(lines)

    def writer_brief(self) -> str:
        lines = [self.brief()]
        insights = self.analysis.get("insights")
        if insights:
            lines.append("Analyst insights: " + "; ".join(insights))
        risks = self.analysis.get("risks")
        if risks:
            lines.append("Tracked risks: " + "; ".join(risks))
        verdict = self.analysis.get("verdict")
        if verdict:
            lines.append(f"Analyst verdict: {verdict}")
        validation = self.analysis.get("validation_checks")
        if validation:
            lines.append("Validation to run: " + "; ".join(validation))
        next_actions = self.analysis.get("next_actions")
        if next_actions:
            lines.append("Next actions: " + "; ".join(next_actions))
        evidence_quality = self.analysis.get("evidence_quality")
        if evidence_quality:
            lines.append(f"Evidence quality: {evidence_quality}")
        if self.findings:
            lines.append("Findings summary:\n" + self.findings_overview())
        if self.gaps:
            lines.append("Unresolved gaps: " + "; ".join(self.gaps))
        return "\n".join(lines)


planner_agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are the planner and orchestrator for a multi-agent research team. "
        "Maintain a concise task board with goal, constraints, plan steps, assumptions, risks, and open questions. "
        "Always respond with strict JSON containing keys: constraints (list), plan (list of objects with id, owner, task, success_criteria), assumptions (list), open_questions (list), risks (list). "
        "Keep the plan to at most 4 steps. If information is missing, ask for it via open_questions. "
        "Do not include any explanation outside JSON."
    ),
)

researcher_agent = Agent(
    model=ollama_model,
    tools=[http_request],
    system_prompt=(
        "You are the researcher in a coordinated agent team. Read the task board, execute the plan steps assigned to you, "
        "and collect evidence from high-quality sources using the http_request tool when needed. "
        "Return strict JSON with keys: findings (list of {summary, source, evidence, confidence}), gaps (list), "
        "recommended_next_actions (list), evidence_quality (string). Limit yourself to two web calls and cite every source."
    ),
)

analyst_agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are the analyst and critic. Review the shared task board and incoming findings. "
        "Identify verified insights, highlight risks or contradictions, and note any validation still required. "
        "Respond with strict JSON containing: insights (list), risks (list), summary (string), validation_checks (list), gaps (list), verdict (string)."
    ),
)

writer_agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are the writer. Compose the final answer for the user using only validated information from the task board. "
        "Structure the report clearly, keep it under 500 words, cite sources inline, and acknowledge residual uncertainty."
    ),
)

def run_research_workflow(user_input: str) -> str:
    """Execute the planner -> researcher -> analyst -> writer workflow for the given goal."""

    print(f"\nProcessing: '{user_input}'")
    task_board = TaskBoard(goal=user_input)

    # Step 1: Planning / orchestration
    print("\nStep 1: Planner agent building the task board...")
    planner_prompt = (
        "Create or update the shared task board for the current user goal.\n\n"
        f"Goal: {user_input}\n"
        "Assume baseline constraints: cite sources, distinguish assumptions from facts, and flag missing information.\n"
        "Return only the JSON object in the schema from your system instructions."
    )
    planner_payload = parse_structured_response(planner_agent(planner_prompt))
    task_board.record_plan(planner_payload)
    task_board.record_iteration("planner", json.dumps(planner_payload, indent=2))
    print("Planner complete. Shared task board established.")

    # Step 2: Researcher executes web lookups
    print("\nStep 2: Researcher agent gathering evidence...")
    researcher_prompt = (
        "Task board summary:\n"
        + task_board.brief()
        + "\n\nFocus on the plan steps assigned to you (owner includes 'research' or unassigned)."
        + " Return JSON exactly as specified."
    )
    researcher_payload = parse_structured_response(researcher_agent(researcher_prompt))
    task_board.add_findings(researcher_payload)
    task_board.record_iteration("researcher", json.dumps(researcher_payload, indent=2))
    print("Research complete. Findings logged on the task board.")

    # Step 3: Analyst validates and critiques
    print("\nStep 3: Analyst agent reviewing findings...")
    analyst_prompt = (
        "Shared task board:\n"
        + task_board.brief()
        + "\n\nCurrent findings with sources:\n"
        + task_board.findings_overview()
        + "\n\nReturn the JSON schema you were instructed to use."
    )
    analyst_payload = parse_structured_response(analyst_agent(analyst_prompt))
    task_board.set_analysis(analyst_payload)
    task_board.record_iteration("analyst", json.dumps(analyst_payload, indent=2))
    print("Analysis complete. Risks and validation steps recorded.")

    # Step 4: Writer produces final report
    print("\nStep 4: Writer agent synthesizing the report...")
    writer_prompt = (
        "Use the following task board state to craft the final response:\n"
        + task_board.writer_brief()
        + "\n\nDeliver a polished answer for the user."
    )
    writer_output = writer_agent(writer_prompt)
    task_board.record_iteration("writer", "Final report delivered.")
    print("Report creation complete.")

    return str(writer_output)


if __name__ == "__main__":
    print("\nAgentic Workflow: Research Assistant (Task-Board Edition)\n")
    print("This demo shows how planner, researcher, analyst, and writer agents collaborate.")
    print("Enter a topic to research or a claim to fact-check. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            report = run_research_workflow(user_input)
            print("\n" + "=" * 72)
            print("FINAL REPORT")
            print("=" * 72)
            print(report)
            handoff_agent.tool.handoff_to_user(
                message="Research run complete. Please review the report above and confirm next steps.",
                breakout_of_loop=True
            )
        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as exc:  # pragma: no cover - runtime feedback for demo
            print(f"\nAn error occurred: {exc}")
            print("Please try a different request or re-run the workflow.")