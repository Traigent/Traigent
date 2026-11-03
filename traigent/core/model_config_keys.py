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
    """True for the fixed model keys (``model``, ``model_name``, ``model_id``, ``engine``)."""
    return isinstance(key, str) and key.strip().lower() in MODEL_CONFIG_KEYS


def is_model_suffix_key(key: object) -> bool:
    """True for a user knob such as ``expert_model`` (not one of the fixed keys).

    Such a key may hold a non-LLM model (an embedding, reranker or reward
    model), so only the cost *estimate* reads it unconditionally; the
    preflights read its values only when they are priced LLM ids.
    """
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    return normalized not in MODEL_CONFIG_KEYS and normalized.endswith(
        _MODEL_KEY_SUFFIXES
    )
