from __future__ import annotations

import os
from typing import Any, Dict, Union

from strands.models.ollama import OllamaModel

from config.model_registry import ModelSpec, get_spec

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = value if not isinstance(value, dict) else _deep_merge(value, {})
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_tweaks(params: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(params)
    if (max_tokens := os.getenv("STRANDS_MAX_TOKENS")):
        updated["max_tokens"] = int(max_tokens)
    if (temperature := os.getenv("STRANDS_TEMPERATURE")):
        updated["temperature"] = float(temperature)
    if (top_p := os.getenv("STRANDS_TOP_P")):
        updated["top_p"] = float(top_p)
    if (keep_alive := os.getenv("STRANDS_KEEP_ALIVE")):
        updated["keep_alive"] = keep_alive
    if (num_ctx := os.getenv("STRANDS_NUM_CTX")):
        options = dict(updated.get("options") or {})
        options["num_ctx"] = int(num_ctx)
        updated["options"] = options
    return updated


def get_sa_model(spec: Union[ModelSpec, str], *, overrides: Dict[str, Any] | None = None) -> OllamaModel:
    """Translate a registry spec into an Ollama-backed Strands model."""
    model_spec = get_spec(spec) if isinstance(spec, str) else spec

    params = _deep_merge(model_spec.default_params, overrides or {})
    params = _apply_env_tweaks(params)

    host = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_HOST)
    model_id = os.getenv("STRANDS_MODEL_ID", model_spec.model_id)

    return OllamaModel(host, model_id=model_id, **params)

