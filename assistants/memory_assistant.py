"""
Memory Assistant - A complete memory management system using Strands Agents

Replaces the mem0 dependency with local file-backed storage. Maintains
short-term and long-term memories, keeps a rolling long-term summary, and logs
full outputs for later inspection.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from strands import Agent
from strands_tools import use_llm


class MemoryAssistant:
    """A memory assistant that stores and retrieves information locally."""

    MEMORY_SYSTEM_PROMPT = (
        "You are a memory assistant that helps store and retrieve relevant information."
        " Capture important details, avoid duplicates, and keep helpful summaries."
    )

    ANSWER_SYSTEM_PROMPT = (
        "You are an assistant that creates helpful responses based on retrieved memories."
        " Use the provided memories to craft a natural, conversational answer."
    )

    SUMMARY_SYSTEM_PROMPT = (
        "You maintain a concise long-term memory summary. Blend the existing summary"
        " with the new information, keep it under 200 words, and surface durable facts."
    )

    STORE_ROOT = Path("memory_store")
    LONG_TERM_FILE = "long_term_memories.json"
    SHORT_TERM_FILE = "short_term_memories.json"
    SUMMARY_FILE = "long_term_summary.txt"
    FULL_OUTPUT_FILE = "memory_full_output.log"
    SHORT_TERM_LIMIT = 20

    def __init__(self, user_id: str = "demo_user") -> None:
        self.user_id = user_id
        self.base_dir = self.STORE_ROOT / self.user_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.long_term_path = self.base_dir / self.LONG_TERM_FILE
        self.short_term_path = self.base_dir / self.SHORT_TERM_FILE
        self.summary_path = self.base_dir / self.SUMMARY_FILE
        self.full_output_path = self.base_dir / self.FULL_OUTPUT_FILE

        self.memory_agent = Agent(
            system_prompt=self.MEMORY_SYSTEM_PROMPT,
            tools=[use_llm],
        )
        self.answer_agent = Agent(
            system_prompt=self.ANSWER_SYSTEM_PROMPT,
            tools=[use_llm],
        )
        self.summary_agent = Agent(
            system_prompt=self.SUMMARY_SYSTEM_PROMPT,
            tools=[use_llm],
        )

        self.long_term_memories: List[Dict[str, Any]] = self._load_json_list(self.long_term_path)
        short_term_seed = self._load_json_list(self.short_term_path)
        self.short_term_memories: Deque[Dict[str, Any]] = deque(short_term_seed, maxlen=self.SHORT_TERM_LIMIT)
        if not self.short_term_memories and self.long_term_memories:
            for entry in self.long_term_memories[-self.SHORT_TERM_LIMIT:]:
                self.short_term_memories.append(entry)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json_list(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _save_json_list(path: Path, payload: List[Dict[str, Any]]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_full_output(self, header: str, body: str) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        with self.full_output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n# Timestamp: {timestamp}\n")
            handle.write(f"# {header}\n")
            handle.write(body.strip() + "\n")

    # ------------------------------------------------------------------
    # Summaries and similarity
    # ------------------------------------------------------------------
    @staticmethod
    def _similarity_score(query: str, content: str) -> float:
        if not query or not content:
            return 0.0
        return SequenceMatcher(None, query.lower(), content.lower()).ratio()

    def _update_long_term_summary(self, new_entry: Dict[str, Any]) -> None:
        existing_summary = ""
        if self.summary_path.exists():
            existing_summary = self.summary_path.read_text(encoding="utf-8").strip()
        prompt = (
            f"Existing summary:\n{existing_summary or 'None'}\n\n"
            f"New memory:\n- {new_entry.get('content', '')}\n\n"
            "Update the summary to include the new fact while staying concise."
        )
        try:
            updated = self.summary_agent.tool.use_llm(prompt=prompt)
        except Exception:
            return
        if updated:
            summary_text = str(updated).strip()
            if summary_text:
                self.summary_path.write_text(summary_text + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = (content or "").strip()
        if not content:
            return {"error": "Empty content cannot be stored."}

        entry = {
            "id": uuid4().hex,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "content": content,
            "metadata": metadata or {},
        }

        self.long_term_memories.append(entry)
        self.short_term_memories.append(entry)

        self._save_json_list(self.long_term_path, self.long_term_memories)
        self._save_json_list(self.short_term_path, list(self.short_term_memories))
        self._update_long_term_summary(entry)
        self._append_full_output("Stored Memory", json.dumps(entry, indent=2))

        return {"status": "stored", "entry": entry}

    def retrieve_memories(
        self,
        query: str,
        min_score: float = 0.3,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.long_term_memories:
            return []
        if not query:
            return list(self.short_term_memories)

        scored: List[Dict[str, Any]] = []
        for entry in self.long_term_memories:
            score = self._similarity_score(query, entry.get("content", ""))
            if score >= min_score:
                with_score = dict(entry)
                with_score["score"] = round(score, 4)
                scored.append(with_score)
        scored.sort(key=lambda item: item.get("score", 0), reverse=True)
        return scored[:max_results]

    def list_all_memories(self) -> List[Dict[str, Any]]:
        return list(self.long_term_memories)

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        return self.retrieve_memories(query, min_score=0.0, max_results=10)

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        before = len(self.long_term_memories)
        self.long_term_memories = [m for m in self.long_term_memories if m.get("id") != memory_id]
        if len(self.long_term_memories) == before:
            return {"error": "Memory not found."}
        self.short_term_memories = deque(
            [m for m in self.short_term_memories if m.get("id") != memory_id],
            maxlen=self.SHORT_TERM_LIMIT,
        )
        self._save_json_list(self.long_term_path, self.long_term_memories)
        self._save_json_list(self.short_term_path, list(self.short_term_memories))
        self._append_full_output("Deleted Memory", memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    def clear_all_memories(self) -> Dict[str, Any]:
        self.long_term_memories.clear()
        self.short_term_memories.clear()
        self._save_json_list(self.long_term_path, [])
        self._save_json_list(self.short_term_path, [])
        if self.summary_path.exists():
            self.summary_path.unlink()
        self._append_full_output("Cleared Memories", "All memories removed.")
        return {"status": "cleared"}

    def generate_answer_from_memories(self, query: str, memories: List[Dict[str, Any]]) -> str:
        memories_text = "\n".join(
            f"- ({mem.get('timestamp', 'unknown')}) {mem.get('content', '')}" for mem in memories
        ) or "No relevant memories retrieved."
        prompt = (
            f"Question: {query}\n\n"
            f"Memories:\n{memories_text}\n\n"
            "If the memories are insufficient, acknowledge that limitation."
        )
        try:
            response = self.answer_agent.tool.use_llm(prompt=prompt)
        except Exception as exc:
            response = f"Unable to generate answer: {exc}"
        self._append_full_output(
            "Answer Generation",
            json.dumps(
                {
                    "query": query,
                    "memories": memories,
                    "response": str(response),
                },
                indent=2,
            ),
        )
        return str(response)

    def process_input(self, user_input: str) -> str:
        lower = (user_input or "").lower()
        if lower.startswith("store ") or lower.startswith("remember "):
            content = user_input.split(" ", 1)[1]
            result = self.store_memory(content)
            if "error" in result:
                return result["error"]
            return f"Stored: {result['entry']['content']}"

        if lower.startswith("list"):
            memories = self.list_all_memories()
            if not memories:
                return "No memories stored yet."
            lines = [f"{idx + 1}. {mem['content']}" for idx, mem in enumerate(memories[:10])]
            if len(memories) > 10:
                lines.append(f"...and {len(memories) - 10} more")
            return "\n".join(lines)

        if lower.startswith("clear"):
            return self.clear_all_memories().get("status", "cleared")

        if lower.startswith("search "):
            query = user_input.split(" ", 1)[1]
            memories = self.search_memories(query)
            if not memories:
                return "No matches found."
            return "\n".join(f"- {mem['content']} (score {mem.get('score', 0)})" for mem in memories)

        memories = self.retrieve_memories(user_input)
        return self.generate_answer_from_memories(user_input, memories)

    def get_memory_stats(self) -> Dict[str, Any]:
        total = len(self.long_term_memories)
        return {
            "total_memories": total,
            "user_id": self.user_id,
            "oldest_memory": self.long_term_memories[0] if total else None,
            "newest_memory": self.long_term_memories[-1] if total else None,
            "short_term_window": list(self.short_term_memories),
            "summary_path": str(self.summary_path),
            "full_output_path": str(self.full_output_path),
        }


def main() -> None:
    assistant = MemoryAssistant(user_id="demo_user")
    assistant.store_memory("I like to play tennis on weekends")
    assistant.store_memory("My favorite programming language is Python")
    assistant.store_memory("I work as a software engineer")
    print(assistant.process_input("What are my hobbies?"))
    print(assistant.process_input("list memories"))
    print(assistant.get_memory_stats())


if __name__ == "__main__":
    main()