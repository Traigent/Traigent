"""Which configuration-space keys carry a model id.

One definition shared by the pre-run cost estimator, the provider-key
preflight and the unpriced-model coverage preflight, so a model placed under
a user-chosen knob (for example the ``expert_model`` arm of a cascade) is
priced and checked everywhere or nowhere (#2807).
"""

from __future__ import annotations

MODEL_CONFIG_KEYS = frozenset({"model", "model_name", "model_id", "engine"})
_MODEL_KEY_SUFFIXES = ("_model", "_model_name", "_model_id")


def is_model_config_key(key: object) -> bool:
    """Return True for ``model``-style keys and user knobs such as ``expert_model``."""
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    return normalized in MODEL_CONFIG_KEYS or normalized.endswith(_MODEL_KEY_SUFFIXES)
