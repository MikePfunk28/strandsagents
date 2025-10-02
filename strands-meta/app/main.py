import asyncio, argparse
from app.meta.meta_assistant import MetaAssistant
from app.orchestrator.router import list_available, set_runtime_model

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--switch", nargs=2, metavar=("TARGET","MODEL_KEY"))
    args = p.parse_args()
    if args.list_models:
        for k, v in list_available().items():
            print(f"{k:14s} -> {v}")
        return
    if args.switch:
        target, key = args.switch
        set_runtime_model(target, key)
        print(f"[ok] {target} -> {key}")

async def demo():
    meta = MetaAssistant()
    # simple demo goal
    res = await meta.run_goal("Create hello world app/main.py with a print statement")
    print(res.artifacts)

if __name__ == "__main__":
    cli()
    asyncio.run(demo())
