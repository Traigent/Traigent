"""Single shared inference of the model a captured LLM response reports.

Used by cost extraction and by both evaluator paths so a response's own model
identity (attribute, dict key, ``response_metadata`` or LangChain
``llm_output``) is priced before any configured-model fallback (#2443, #2411).
"""

from __future__ import annotations

from typing import Any

_MODEL_KEYS = ("model", "model_name")


def _from_mapping(mapping: Any) -> str | None:
    if isinstance(mapping, dict):
        for key in _MODEL_KEYS:
            val = mapping.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def infer_response_model(response: Any) -> str | None:
    """Return the model name a response carries, or None. Never raises."""
    try:
        for attr in _MODEL_KEYS:
            val = getattr(response, attr, None)
            if isinstance(val, str) and val:
                return val
        found = _from_mapping(response)
        if found:
            return found
        if isinstance(response, dict):
            for nested in ("response_metadata", "llm_output"):
                found = _from_mapping(response.get(nested))
                if found:
                    return found
        for nested in ("response_metadata", "llm_output"):
            found = _from_mapping(getattr(response, nested, None))
            if found:
                return found
        return None
    except Exception:
        return None
