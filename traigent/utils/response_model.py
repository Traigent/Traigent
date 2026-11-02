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


_OPENROUTER_PREFIX = "openrouter/"


def resolve_pricing_model(
    response_model: str | None, requested_model: str | None
) -> str | None:
    """The id to price a call at: the response's own model unless it cannot be priced.

    OpenRouter reports ``response.model`` WITHOUT the ``openrouter/`` prefix
    (``anthropic/claude-haiku-4.5``), and some of those bare ids have no price
    although the requested ``openrouter/...`` id does (#2854). For a call that
    was requested through OpenRouter, an unpriceable response id falls back to
    the same id behind the prefix, then to the requested id. When nothing
    resolves, the response id is returned so the unknown-model warning is real.
    """
    if not response_model:
        return requested_model or None
    requested_via_openrouter = isinstance(
        requested_model, str
    ) and requested_model.lower().startswith(_OPENROUTER_PREFIX)
    if not requested_via_openrouter:
        return response_model
    from traigent.utils.cost_calculator import model_pricing_resolves

    candidates = [
        response_model,
        f"{_OPENROUTER_PREFIX}{response_model}",
        str(requested_model),
    ]
    for candidate in dict.fromkeys(candidates):
        if model_pricing_resolves(candidate):
            return candidate
    return response_model
