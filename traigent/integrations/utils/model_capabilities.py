"""Reasoning-knob capability table and provider translation (#1542).

One Traigent knob maps to each provider's real parameter:

- ``reasoning_effort`` (``minimal|low|medium|high``) ->
  OpenAI ``reasoning_effort``; Anthropic ``thinking.budget_tokens`` (via
  :data:`EFFORT_TO_BUDGET_TOKENS`); Gemini 3 ``thinking_config.thinking_level``;
  Gemini 2.5 ``thinking_config.thinking_budget``.
- ``thinking_budget_tokens`` (int) -> Anthropic ``thinking.budget_tokens`` and
  Gemini 2.5 ``thinking_config.thinking_budget``. It has no OpenAI equivalent.

Parameter names were checked against the installed provider SDK types
(openai ``ReasoningEffort``, anthropic ``ThinkingConfigEnabledParam``,
google-genai ``ThinkingConfig``/``ThinkingLevel``).

Unknown model ids are never rewritten (the user's raw keys keep flowing to the
provider unchanged). Known non-reasoning models skip the knob with a
value-free warning, so a model x effort sweep does not fail on those trials.
Invalid knob values raise a value-free ``ValueError``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

Knob = Literal["effort", "budget_tokens", "level", "budget", "none"]

REASONING_EFFORT_KEY = "reasoning_effort"
THINKING_BUDGET_KEY = "thinking_budget_tokens"
EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")

# Full OpenAI SDK ReasoningEffort literal: raw values outside the factory's
# four levels (e.g. "xhigh") keep flowing through unchanged.
OPENAI_RAW_EFFORT_LEVELS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# Anthropic requires budget_tokens >= 1024 and < max_tokens.
ANTHROPIC_MIN_BUDGET_TOKENS = 1024
EFFORT_TO_BUDGET_TOKENS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 24576,
}
# Gemini 2.5 thinking_budget tokens.
EFFORT_TO_GEMINI_BUDGET: dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

# Exact ids only (all must be ids the SDK already knows, see
# traigent/providers/validation.py::_KNOWN_MODELS; a test enforces this so the
# table can never carry a retired id). Ids absent here are "unknown".
MODEL_REASONING_KNOB: dict[str, dict[str, Knob]] = {
    "openai": {
        "gpt-4o": "none",
        "gpt-4o-mini": "none",
        "gpt-4.1": "none",
        "gpt-4.1-mini": "none",
        "gpt-4.1-nano": "none",
        "gpt-5": "effort",
        "gpt-5-mini": "effort",
        "gpt-5-nano": "effort",
        "gpt-5.1": "effort",
        "gpt-5.2": "effort",
        "gpt-5.4": "effort",
        "gpt-5.4-mini": "effort",
        "gpt-5.4-nano": "effort",
        "gpt-5.5": "effort",
        "gpt-5.6": "effort",
        "gpt-5.6-luna": "effort",
        "gpt-5.6-sol": "effort",
        "gpt-5.6-terra": "effort",
        "gpt-6-astra": "effort",
        "o3": "effort",
        "o3-mini": "effort",
    },
    "anthropic": dict.fromkeys(
        (
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-5",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-opus-4-6-20260205",
            "claude-opus-4-7",
            "claude-opus-4-7-20260416",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ),
        "budget_tokens",
    ),
    "google": {
        "gemini-3.8-flash": "level",
        "gemini-3.5-flash-lite": "level",
        "gemini-2.0-flash": "none",
    },
}

# OpenAI o-series has no "minimal" level; it is clamped to "low".
_OPENAI_NO_MINIMAL_PREFIXES = ("o1", "o3", "o4")


def get_reasoning_knob(provider: str, model: str | None) -> Knob | None:
    """Return the knob a known model supports, or None if the model is unknown.

    Accepts ``models/gemini-*`` style ids. ``"none"`` means a known
    non-reasoning model.
    """
    if not model:
        return None
    mid = model.removeprefix("models/")
    return MODEL_REASONING_KNOB.get(provider, {}).get(mid)


def _validate_effort(value: Any) -> str:
    if not isinstance(value, str) or value not in EFFORT_LEVELS:
        raise ValueError(
            f"{REASONING_EFFORT_KEY} must be one of {', '.join(EFFORT_LEVELS)}"
        )
    return value


def _validate_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{THINKING_BUDGET_KEY} must be a non-negative integer")
    return value


def _skip(provider: str, reason: str) -> None:
    logger.warning("Skipping reasoning knob for %s: %s", provider, reason)


def translate_openai(model: str | None, effort: Any, budget: Any) -> dict[str, Any]:
    """Return OpenAI kwargs for the knob (``{}`` means leave kwargs untouched)."""
    knob = get_reasoning_knob("openai", model)
    if budget is not None:
        _validate_budget(budget)
        _skip("openai", "thinking_budget_tokens has no OpenAI equivalent")
    if effort is None:
        return {}
    if effort not in OPENAI_RAW_EFFORT_LEVELS:
        raise ValueError(
            f"{REASONING_EFFORT_KEY} must be one of "
            f"{', '.join(OPENAI_RAW_EFFORT_LEVELS)}"
        )
    if knob == "none":
        _skip("openai", "model does not support reasoning_effort")
        return {}
    mid = (model or "").lower()
    if effort == "minimal" and mid.startswith(_OPENAI_NO_MINIMAL_PREFIXES):
        effort = "low"
    return {"reasoning_effort": effort}


def translate_anthropic(
    model: str | None, effort: Any, budget: Any, *, max_tokens: int | None = None
) -> dict[str, Any]:
    """Return Anthropic kwargs (``thinking={...}``); ``{}`` means no change."""
    if effort is None and budget is None:
        return {}
    if effort is not None:
        effort = _validate_effort(effort)
    if budget is not None:
        budget = _validate_budget(budget)
    if get_reasoning_knob("anthropic", model) == "none":
        _skip("anthropic", "model does not support extended thinking")
        return {}
    tokens = budget if budget is not None else EFFORT_TO_BUDGET_TOKENS[effort]
    if tokens < ANTHROPIC_MIN_BUDGET_TOKENS:
        raise ValueError(
            f"thinking budget must be at least {ANTHROPIC_MIN_BUDGET_TOKENS} tokens"
        )
    if max_tokens is not None and tokens >= max_tokens:
        raise ValueError("thinking budget must be smaller than max_tokens")
    return {"thinking": {"type": "enabled", "budget_tokens": tokens}}


def translate_gemini(model: str | None, effort: Any, budget: Any) -> dict[str, Any]:
    """Return a Gemini ``thinking_config`` dict; ``{}`` means no change."""
    if effort is None and budget is None:
        return {}
    if effort is not None:
        effort = _validate_effort(effort)
    if budget is not None:
        budget = _validate_budget(budget)
    knob = get_reasoning_knob("google", model)
    if knob is None:
        mid = (model or "").removeprefix("models/")
        knob = "level" if mid.startswith("gemini-3") else "budget"
    if knob == "none":
        _skip("gemini", "model does not support thinking")
        return {}
    if knob == "level":
        if effort is None:
            _skip("gemini", "model uses thinking_level; budget tokens ignored")
            return {}
        return {"thinking_config": {"thinking_level": effort.upper()}}
    tokens = budget if budget is not None else EFFORT_TO_GEMINI_BUDGET[effort]
    return {"thinking_config": {"thinking_budget": tokens}}


__all__ = [
    "ANTHROPIC_MIN_BUDGET_TOKENS",
    "EFFORT_LEVELS",
    "EFFORT_TO_BUDGET_TOKENS",
    "EFFORT_TO_GEMINI_BUDGET",
    "MODEL_REASONING_KNOB",
    "OPENAI_RAW_EFFORT_LEVELS",
    "REASONING_EFFORT_KEY",
    "THINKING_BUDGET_KEY",
    "get_reasoning_knob",
    "translate_anthropic",
    "translate_gemini",
    "translate_openai",
]
