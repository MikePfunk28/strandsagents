# builder/builder_cli.py
import argparse
import logging
import shutil
from pathlib import Path
from builder.decorators import dump_registry, agent
from builder.mcp_verify import verify_with_mcp
from builder.otel_setup import setup_otel
from jinja2 import Template
import json
import subprocess
import os

log = logging.getLogger(__name__)
tracer = setup_otel("agent-factory")

TEMPLATES = Path(__file__).parent / "templates"
GENERATED = Path(__file__).parent.parent / "generated"


def think(step: str, facts: dict):
    log.info("PLAN: %s :: %s", step, json.dumps(facts, default=str))


def verify(queries):
    v = verify_with_mcp(queries)
    log.info("VERIFY: %s", json.dumps(v))
    return v


def act_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    log.info("ACT: wrote %s (%d bytes)", path, len(content))


def render_template(name: str, **ctx) -> str:
    tpl = Template((TEMPLATES / name).read_text())
    return tpl.render(**ctx)


def build_interactive():
    name = input("Agent name: ").strip()
    goal = input("What should it do?: ").strip()
    provider = input(
        "Provider [bedrock|ollama] (default bedrock): ").strip() or "bedrock"

    tools = []
    add_tools = input(
        "Add built-in tools? comma-separated (or blank): ").strip()
    if add_tools:
        tools = [t.strip() for t in add_tools.split(",") if t.strip()]

    need_kb = input("Attach knowledge base? [y/N]: ").lower().startswith("y")
    need_guard = input("Enable guardrails? [y/N]: ").lower().startswith("y")

    think("Derive spec", {"name": name, "goal": goal, "provider": provider,
          "tools": tools, "kb": need_kb, "guard": need_guard})
    v = verify([
        "Bedrock Agents CreateAgent/ActionGroup/PrepareAgent",
        "CloudFormation AWS::Bedrock::Agent",
        "Lambda resource policy for Bedrock",
        "Ollama chat API"
    ])

    agent_slug = name.lower().replace(" ", "-")
    out_dir = GENERATED / agent_slug
    # === agent main ===
    act_write(out_dir / "agent" / "main.py",
              render_template("../snippets/agent_main.py.j2", provider=provider))

    # === deploy ===
    if provider == "bedrock":
        act_write(out_dir / "deploy" / "bedrock_agent.cfn.yaml",
                  (TEMPLATES / "bedrock_agent.cfn.yaml").read_text())
    else:
        act_write(out_dir / "deploy" / "ollama_stack.cfn.yaml",
                  (TEMPLATES / "ollama_stack.cfn.yaml").read_text())

    # minimal config
    cfg = {"provider": provider, "goal": goal, "ollama_model": "llama3.2"}
    act_write(out_dir / "agent" / "config.json", json.dumps(cfg, indent=2))
    print(f"\nGenerated at: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("cmd", choices=["build"])
    args = ap.parse_args()
    if args.interactive and args.cmd == "build":
        build_interactive()


if __name__ == "__main__":
    main()
