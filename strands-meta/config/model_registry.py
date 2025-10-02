from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModelSpec:
    """Immutable model configuration entry."""

    key: str
    model_id: str
    default_params: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation for serialization/debugging."""
        return {"key": self.key, "id": self.model_id, "default_params": self.default_params}


_REGISTRY_ENTRIES: Dict[str, ModelSpec] = {
    "gemma3_tiny": ModelSpec(
        key="gemma3_tiny",
        model_id="gemma3:270m",
        default_params={"temperature": 0.2, "options": {"num_ctx": 2048}},
    ),
    "gemma3_1b": ModelSpec(
        key="gemma3_1b",
        model_id="gemma3:1b",
        default_params={"temperature": 0.2, "options": {"num_ctx": 4096}},
    ),
    "llama3_2_1b": ModelSpec(
        key="llama3_2_1b",
        model_id="llama3.2:1b",
        default_params={"temperature": 0.2, "options": {"num_ctx": 4096}},
    ),
    "llama3_2_3b": ModelSpec(
        key="llama3_2_3b",
        model_id="llama3.2",
        default_params={"temperature": 0.2, "options": {"num_ctx": 8192}},
    ),
}


def get_registry() -> Dict[str, ModelSpec]:
    """Return a copy of the model registry mapping."""
    return dict(_REGISTRY_ENTRIES)


def get_spec(key: str) -> ModelSpec:
    """Fetch a model specification by key.

    Raises:
        KeyError: If the key is not registered.
    """

    return _REGISTRY_ENTRIES[key]


REGISTRY = _REGISTRY_ENTRIES
