from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

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


async def demo() -> None:
    meta = MetaAssistant()
    result = await meta.run_goal("Create hello world app/main.py with a print statement")
    print(result.artifacts)


if __name__ == "__main__":
    if not cli():
        asyncio.run(demo())
