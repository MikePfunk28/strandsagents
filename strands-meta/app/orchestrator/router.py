from __future__ import annotations

import os
from typing import Dict, Optional

import yaml

from config.model_registry import ModelSpec, REGISTRY, get_spec

CFG_PATH = os.getenv("STRANDS_MODELS_YAML", "config/models.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    MODELS_CFG = yaml.safe_load(f) or {}

DEFAULTS: Dict[str, str] = MODELS_CFG.get("defaults", {})
OVERRIDES: Dict[str, str] = MODELS_CFG.get("overrides", {})
_RUNTIME: Dict[str, str] = {}


def list_available() -> Dict[str, str]:
    """Return registry keys mapped to their Ollama model identifiers."""
    return {k: spec.model_id for k, spec in REGISTRY.items()}


def _env_override(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        env_key = f"STRANDS_MODEL_{candidate.upper()}"
        value = os.getenv(env_key)
        if value:
            return value
    return os.getenv("STRANDS_MODEL_ID")


def _resolve_registry_key(role: str, assistant_name: Optional[str]) -> str:
    if assistant_name and assistant_name in _RUNTIME:
        return _RUNTIME[assistant_name]
    if role in _RUNTIME:
        return _RUNTIME[role]

    if assistant_name:
        env_override = _env_override(assistant_name)
        if env_override:
            return env_override
        if assistant_name in OVERRIDES:
            return OVERRIDES[assistant_name]

    env_override = _env_override(role)
    if env_override:
        return env_override

    if role in DEFAULTS:
        return DEFAULTS[role]

    return "gemma3_tiny"


def get_model_for(role: str, assistant_name: Optional[str] = None) -> ModelSpec:
    """Resolve the model spec for an assistant/role combination."""
    key = _resolve_registry_key(role, assistant_name)
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Update config/model_registry.py to register it.")
    return get_spec(key)


def set_runtime_model(target: str, registry_key: str) -> None:
    """Assign a runtime override for either a role or assistant name."""
    if registry_key not in REGISTRY:
        raise ValueError(f"Unknown model key: {registry_key}")
    _RUNTIME[target] = registry_key

