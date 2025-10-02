import os, yaml
from typing import Optional, Dict
from config.model_registry import REGISTRY

CFG_PATH = os.getenv("STRANDS_MODELS_YAML", "config/models.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    MODELS_CFG = yaml.safe_load(f) or {}

DEFAULTS: Dict[str, str] = MODELS_CFG.get("defaults", {})
OVERRIDES: Dict[str, str] = MODELS_CFG.get("overrides", {})
_RUNTIME: Dict[str, str] = {}

def list_available() -> Dict[str, str]:
    return {k: v["id"] for k, v in REGISTRY.items()}

def get_model_for(role: str, assistant_name: Optional[str] = None) -> Dict:
    if assistant_name and assistant_name in _RUNTIME:
        return REGISTRY[_RUNTIME[assistant_name]]
    if role in _RUNTIME:
        return REGISTRY[_RUNTIME[role]]
    if assistant_name and assistant_name in OVERRIDES:
        return REGISTRY[OVERRIDES[assistant_name]]
    if role in DEFAULTS:
        return REGISTRY[DEFAULTS[role]]
    env_key = f"STRANDS_MODEL_{role.upper()}"
    if os.getenv(env_key):
        return REGISTRY[os.getenv(env_key)]
    return REGISTRY["gemma3_tiny"]

def set_runtime_model(target: str, registry_key: str) -> None:
    if registry_key not in REGISTRY:
        raise ValueError(f"Unknown model key: {registry_key}")
    _RUNTIME[target] = registry_key
