from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.assistants.base import Task
from app.assistants.mentor import Mentor
from app.meta.meta_assistant import MetaAssistant
from app.orchestrator.router import list_available, set_runtime_model


def cli() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--switch", nargs=2, metavar=("TARGET", "MODEL_KEY"))
    args = parser.parse_args()

    handled = False
    if args.list_models:
        for key, model_id in list_available().items():
            print(f"{key:14s} -> {model_id}")
        handled = True
    if args.switch:
        target, key = args.switch
        set_runtime_model(target, key)
        print(f"[ok] {target} -> {key}")
        handled = True
    return handled


async def interactive_session() -> None:
    mentor = Mentor()
    meta = MetaAssistant()

    print("\nStrands Mentor ready. Describe the assistant you want to build.")
    print("Type 'build' when you want to execute the plan, or 'exit' to quit.\n")

    goal = input("Goal> ").strip()
    if not goal:
        print("No goal provided. Exiting.")
        return

    history: List[Dict[str, str]] = []
    clarifications: List[str] = []

    for _ in range(10):  # keep the loop bounded to avoid runaway conversations
        task_context: Dict[str, Any] = {
            "history": history,
            "clarifications": clarifications,
        }
        mentor_result = await mentor.run(Task(goal=goal, context=task_context, role="coordinator"))
        mentor_text = (mentor_result.artifacts.get("text") or "").strip()
        if mentor_text:
            print(f"\nMentor:\n{mentor_text}\n")
            history.append({"role": "mentor", "text": mentor_text})

        user_reply = input("You> ").strip()
        if not user_reply:
            continue

        lowered = user_reply.lower()
        if lowered in {"exit", "quit"}:
            print("Exiting without executing a plan.")
            return

        if lowered.startswith("goal:"):
            goal = user_reply.split(":", 1)[1].strip() or goal
            history.append({"role": "user", "text": user_reply})
            clarifications.append(user_reply)
            continue

        if lowered in {"build", "run", "execute"}:
            break

        history.append({"role": "user", "text": user_reply})
        clarifications.append(user_reply)
    else:
        print("Reached conversation limit; executing best-effort plan.\n")

    context_payload = {
        "goal": goal,
        "clarifications": clarifications,
        "history": history,
    }
    print("\n[mentor] Executing first plan step...\n")
    result = await meta.run_goal(goal, context=context_payload)
    print("Result artifacts:")
    print(result.artifacts)


if __name__ == "__main__":
    if not cli():
        asyncio.run(interactive_session())
