"""Simple simulator used by the FEVER unit tests."""

from __future__ import annotations

from typing import Any, Mapping

from traigent.evaluators.base import EvaluationExample

from . import pipeline

__all__ = ["_ensure_known_model", "generate_case_study_answer"]


_KNOWN_MODELS = {"gpt-4o", "gpt-4o-mini", "haiku-3.5"}
_ALIASES = {
    "gpt-4.1-nano": "haiku-3.5",
    "gpt-4.1-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4o",
}


def _ensure_known_model(model: str) -> str:
    """Normalize user-friendly aliases to concrete back-end models."""

    normalized = (model or "").strip()
    lowered = normalized.lower()
    if not normalized:
        raise ValueError("Model cannot be empty.")

    if lowered in _KNOWN_MODELS:
        # Preserve user casing for friendly logging.
        for candidate in _KNOWN_MODELS:
            if candidate.lower() == lowered:
                return candidate

    for alias, resolved in _ALIASES.items():
        if lowered.startswith(alias):
            return resolved

    raise ValueError(f"Unknown model '{model}'. Supported models: {sorted(_KNOWN_MODELS)}")


def generate_case_study_answer(
    claim: str,
    example: EvaluationExample,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic verdict that mirrors evaluator output."""

    config = dict(config or {})
    model = _ensure_known_model(config.get("model", "gpt-4o"))
    verdict = str(example.expected_output or "NOT ENOUGH INFO").strip() or "NOT ENOUGH INFO"
    disposition = "supports" if verdict == "SUPPORTS" else "refutes"

    evidence_items = [
        {
            "page": example.metadata.get("page", "Unknown"),
            "line": example.metadata.get("line", 0),
            "text": example.metadata.get("evidence_text", claim),
        }
    ]

    reasoning = (
        f"{model} cites [{evidence_items[0]['page']}] line {evidence_items[0]['line']} "
        f"which {disposition} the claim."
    )
    return pipeline.build_mock_response(verdict=verdict, reasoning=reasoning, evidence_items=evidence_items)
