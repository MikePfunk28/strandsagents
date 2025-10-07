from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
from datetime import datetime


class FileAssistant:
    """Utility for managing workflow artifacts and knowledge-base files."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        run_dir: str = "workflow_runs",
        context_dir: str = "knowledge_context",
        knowledge_log: str = "knowledge_base.txt",
        summary_file: str = "knowledge_summary.md",
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.run_dir = run_dir
        self.context_dir = context_dir
        self.knowledge_log = knowledge_log
        self.summary_file = summary_file

    @property
    def run_path(self) -> Path:
        return self.base_dir / self.run_dir

    @property
    def context_path(self) -> Path:
        return self.base_dir / self.context_dir

    @property
    def knowledge_log_path(self) -> Path:
        return self.base_dir / self.knowledge_log

    @property
    def summary_path(self) -> Path:
        return self.context_path / self.summary_file

    def ensure_directories(self) -> None:
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.context_path.mkdir(parents=True, exist_ok=True)

    def load_rolling_summary(self) -> str:
        if self.summary_path.exists():
            return self.summary_path.read_text(encoding="utf-8").strip()
        return ""

    def store_rolling_summary(self, summary_text: str) -> Path:
        self.ensure_directories()
        self.summary_path.write_text(summary_text.strip() + "\n", encoding="utf-8")
        return self.summary_path

    def persist_artifacts(
        self,
        workflow_state: Dict[str, Any],
        final_result: str,
        candidate_memories: Iterable[str],
        rolling_summary: Optional[str],
        store_summary: bool = True,
    ) -> Dict[str, Optional[str]]:
        """Persist full workflow, distilled context, and knowledge log entries."""
        self.ensure_directories()

        run_id = workflow_state.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")

        full_run_path = self.run_path / f"workflow_run_{run_id}.md"
        context_file_path = self.context_path / f"knowledge_context_{run_id}.md"

        candidate_list = list(candidate_memories)

        full_run_path.write_text(
            self._build_full_run_markdown(workflow_state, final_result),
            encoding="utf-8",
        )

        context_file_path.write_text(
            self._build_context_markdown(workflow_state, final_result, candidate_list, rolling_summary),
            encoding="utf-8",
        )

        knowledge_log_path: Optional[Path] = None
        if candidate_list:
            knowledge_log_path = self._append_knowledge_log(workflow_state, candidate_list, final_result)

        summary_path: Optional[Path] = None
        if store_summary and rolling_summary is not None:
            summary_path = self.store_rolling_summary(rolling_summary)

        return {
            "full_run": str(full_run_path),
            "context": str(context_file_path),
            "knowledge_log": str(knowledge_log_path) if knowledge_log_path else None,
            "rolling_summary": str(summary_path) if summary_path else None,
        }

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _build_full_run_markdown(self, workflow_state: Dict[str, Any], final_result: str) -> str:
        lines: List[str] = []
        run_id = workflow_state.get("run_id", "unknown")
        lines.append(f"# Workflow Run {run_id}")
        lines.append("")
        lines.append(f"- Query: {workflow_state.get('original_query', 'Unknown query')}")
        lines.append(f"- Started: {workflow_state.get('start_time', 'unknown')}")
        lines.append(f"- Finished: {workflow_state.get('end_time', 'unknown')}")
        lines.append(f"- Total steps: {workflow_state.get('total_steps', 0)}")
        lines.append("")

        thinking_results = workflow_state.get("thinking_results", {})
        lines.append("## Thinking Results")
        if thinking_results:
            for key, value in thinking_results.items():
                lines.append(f"### {key}")
                lines.append(self._stringify_value(value))
                lines.append("")
        else:
            lines.append("_No thinking results recorded._")
            lines.append("")

        steps = workflow_state.get("completed_steps", [])
        lines.append("## Completed Steps")
        if steps:
            for step in steps:
                step_index = step.get("step", "?")
                agent_name = step.get("agent", "Unknown agent")
                lines.append(f"### Step {step_index} - {agent_name}")
                lines.append(f"- Confidence: {step.get('confidence', 'unknown')}")
                lines.append(f"- Reasoning: {step.get('reasoning', 'No reasoning provided')}")
                lines.append(f"- Timestamp: {step.get('timestamp', 'Unknown')}")
                result_payload = step.get("result")
                if result_payload:
                    lines.append("")
                    lines.append("#### Result")
                    lines.append(self._stringify_value(result_payload))
                    lines.append("")
        else:
            lines.append("_No workflow steps were executed._")
            lines.append("")

        lines.append("## Final Result")
        lines.append(final_result.strip())
        lines.append("")

        return "\n".join(lines)

    def _build_context_markdown(
        self,
        workflow_state: Dict[str, Any],
        final_result: str,
        candidate_memories: List[str],
        rolling_summary: Optional[str],
    ) -> str:
        lines: List[str] = [
            f"# Query: {workflow_state.get('original_query', 'Unknown query')}",
            f"# Timestamp: {workflow_state.get('end_time', datetime.now().isoformat())}",
            "",
            "## Final Result",
            final_result.strip() or "_No final result captured._",
            "",
        ]

        lines.append("## Extracted Memories")
        if candidate_memories:
            for fact in candidate_memories:
                lines.append(f"- {fact}")
        else:
            lines.append("_No durable memories extracted from this run._")
        lines.append("")

        steps = workflow_state.get("completed_steps", [])
        lines.append("## Key Steps")
        if steps:
            for step in steps:
                agent_name = step.get("agent", "Unknown agent")
                result_payload = step.get("result")
                if not result_payload:
                    continue
                summary_text = self._stringify_value(result_payload).strip()
                if not summary_text:
                    continue
                formatted_summary = summary_text.replace("\n", "\n  ")
                lines.append(f"- **{agent_name}:** {formatted_summary}")
        else:
            lines.append("_No intermediate steps recorded._")
        lines.append("")

        if rolling_summary:
            lines.append("## Rolling Summary Snapshot")
            lines.append(rolling_summary.strip())
            lines.append("")

        return "\n".join(lines)

    def _append_knowledge_log(
        self,
        workflow_state: Dict[str, Any],
        candidate_memories: List[str],
        final_result: str,
    ) -> Path:
        self.ensure_directories()
        log_path = self.knowledge_log_path
        payload_lines = [
            f"# Query: {workflow_state.get('original_query', 'Unknown query')}",
            f"# Timestamp: {workflow_state.get('end_time', datetime.now().isoformat())}",
            "",
            "## Extracted Memories",
        ]
        for fact in candidate_memories:
            payload_lines.append(f"- {fact}")
        payload_lines.extend([
            "",
            "## Source Summary",
            final_result.strip(),
            "",
            "---",
        ])
        payload = "\n".join(payload_lines)

        with log_path.open("a", encoding="utf-8") as handle:
            if log_path.exists() and log_path.stat().st_size > 0:
                handle.write("\n\n")
            handle.write(payload)

        return log_path

