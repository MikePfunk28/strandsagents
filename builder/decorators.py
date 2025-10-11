# builder/decorators.py
from typing import Callable, List, Optional, Dict, Any
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path

REGISTRY: Dict[str, Any] = {"tools": {}, "agents": {}}


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    provider: str            # "bedrock" | "ollama"
    tools: List[str] = field(default_factory=list)
    custom_tools: Dict[str, str] = field(
        default_factory=dict)  # name->python code
    config: Dict[str, Any] = field(default_factory=dict)


def tool(name: Optional[str] = None, description: str = ""):
    def wrap(fn: Callable):
        tname = name or fn.__name__
        REGISTRY["tools"][tname] = ToolSpec(
            name=tname, fn=fn, description=description)
        return fn
    return wrap


def agent(name: str, system_prompt: str, provider: str, tools: Optional[List[str]] = None, **config):
    """Registers an agent spec. The builder_cli will compile this spec into code + IaC."""
    def wrap(cls_or_fn):
        REGISTRY["agents"][name] = AgentSpec(
            name=name,
            system_prompt=system_prompt.strip(),
            provider=provider,
            tools=tools or [],
            custom_tools=config.pop("custom_tools", {}),
            config=config,
        )
        return cls_or_fn
    return wrap


def dump_registry(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registry.json").write_text(
        json.dumps(
            {
                "tools": list(REGISTRY["tools"].keys()),
                "agents": {k: vars(v) for k, v in REGISTRY["agents"].items()},
            },
            indent=2,
        )
    )
