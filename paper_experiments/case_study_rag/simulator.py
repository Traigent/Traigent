"""Simulator for HotpotQA case study mock evaluations."""

from __future__ import annotations

from typing import Any, Mapping

from . import pipeline

__all__ = ["generate_case_study_answer"]


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

    # Map questions to expected answers (deterministic mock responses)
    # These match the custom demo questions in hotpotqa_dev_subset.jsonl
    answer_map = {
        # Q1: What is the capital of the country where the Eiffel Tower is located?
        "eiffel tower": "Paris",
        "capital of the country": "Paris",
        # Q2: Who founded the company that created the iPhone?
        "iphone": "Steve Jobs",
        "founded the company": "Steve Jobs",
        # Q3: What is the nationality of the author who wrote 'A Tale of Two Cities'?
        "tale of two cities": "English",
        "nationality of the author": "English",
        # Q4: What river flows through the city where Mozart was born?
        "mozart": "Salzach",
        "river flows through": "Salzach",
        # Q5: What is the population of the country where Mount Fuji is located?
        "mount fuji": "125 million",
        "population of the country": "125 million",
        # Q6: Who wrote the national anthem of the country where the Amazon River originates?
        "amazon river": "Jose de la Torre Ugarte",
        "national anthem": "Jose de la Torre Ugarte",
        # Q7: What sport is played at the stadium where the 1966 World Cup final was held?
        "1966 world cup": "Football",
        "wembley": "Football",
        # Q8: What language is primarily spoken in the country where the Nobel Prize ceremony is held?
        "nobel prize": "Swedish",
        "language is primarily spoken": "Swedish",
    }

    # Find matching answer based on question keywords
    q_lower = question.lower()
    answer = "I don't know"
    for keyword, expected_answer in answer_map.items():
        if keyword in q_lower:
            answer = expected_answer
            break

    # Build mock response in the expected format
    return pipeline.build_mock_response(
        question=question,
        answer=answer,
        context=[],  # Empty in simulator - context would come from retriever
        config=config,
    )
