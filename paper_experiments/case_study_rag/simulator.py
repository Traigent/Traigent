"""Simulator for HotpotQA case study mock evaluations."""

from __future__ import annotations

from typing import Any, Mapping

from . import pipeline
from .dataset import _RAW_EXAMPLES

__all__ = ["generate_case_study_answer"]


_KNOWN_MODELS = {"gpt-4o", "gpt-4o-mini", "haiku-3.5"}
_ALIASES = {
    "gpt-4.1-nano": "haiku-3.5",
    "gpt-4.1-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4o",
}

# Build answer lookup from the dataset itself so it stays in sync automatically.
_ANSWER_MAP: dict[str, str] = {ex.question: ex.answer for ex in _RAW_EXAMPLES}


def _ensure_known_model(model: str) -> str:
    """Normalize user-friendly aliases to concrete back-end models."""
    normalized = (model or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return "gpt-4o-mini"

    if lowered in _KNOWN_MODELS:
        for candidate in _KNOWN_MODELS:
            if candidate.lower() == lowered:
                return candidate

    for alias, resolved in _ALIASES.items():
        if lowered.startswith(alias):
            return resolved

    # Return as-is if not recognized
    return normalized


def generate_case_study_answer(
    question: str,
    config: Mapping[str, Any] | dict[str, Any],
) -> str:
    """Return a deterministic answer that mirrors evaluator output.

    This function is called by the HotpotQA demo when running in mock mode.
    It generates a realistic-looking response without making actual API calls.

    Args:
        question: The question being asked
        config: Configuration dict with model, temperature, retriever_k, etc.

    Returns:
        A mock answer string in the expected format
    """
    config = dict(config or {})
    model = _ensure_known_model(config.get("model", "gpt-4o-mini"))

    # Look up the expected answer from the dataset directly.
    answer = _ANSWER_MAP.get(question, "I don't know")

    # Build mock response in the expected format
    return pipeline.build_mock_response(
        question=question,
        answer=answer,
        context=[],  # Empty in simulator - context would come from retriever
        config=config,
    )
