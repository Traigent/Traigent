"""Metric helpers shared by the FEVER case study tests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from traigent.evaluators.base import EvaluationExample

__all__ = [
    "_mock_example_metrics",
    "_real_example_metrics",
    "build_fever_metric_functions",
]


def _mock_example_metrics(
    *,
    claim: str,
    config: Mapping[str, Any],
    llm_metrics: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return deterministic metrics for mock-mode evaluations."""

    alias = str(config.get("model", "")).lower()
    # Encourage stable yet non-trivial floating-point output across models.
    base_value = 1.0 if "mini" not in alias else 0.9

    label_accuracy = float(llm_metrics.get("label_accuracy", base_value)) if llm_metrics else base_value
    return {"label_accuracy": label_accuracy, "accuracy": label_accuracy}


def _real_example_metrics(
    *,
    output: Mapping[str, Any],
    expected: Any,
    example: EvaluationExample,
    config: Mapping[str, Any],
    llm_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map pipeline outputs to FEVER-style metrics."""

    verdict = str(output.get("verdict", "")).strip().upper()
    expected_verdict = str(expected).strip().upper()

    accuracy = 1.0 if verdict and verdict == expected_verdict else 0.0

    predicted_evidence = _extract_predicted_evidence(output.get("evidence", []))
    gold_evidence = _example_gold_evidence(example)

    metrics = {
        "label_accuracy": accuracy,
        "accuracy": accuracy,
        "predicted_evidence": predicted_evidence,
        "gold_evidence": gold_evidence,
    }

    if llm_metrics:
        metrics.update(llm_metrics)

    return metrics


def _extract_predicted_evidence(items: Iterable[Mapping[str, Any]]) -> list[list[Any]]:
    evidence: list[list[Any]] = []
    for item in items:
        page = str(item.get("page", "")).strip()
        line = item.get("line", 0)
        try:
            line_number = int(line)
        except (TypeError, ValueError):
            continue
        if not page:
            continue
        evidence.append([page, line_number])
    return evidence


def _example_gold_evidence(example: EvaluationExample) -> list[list[Any]]:
    metadata = example.metadata or {}
    page = str(metadata.get("page", "")).strip()
    line = metadata.get("line", 0)
    if not page:
        return []
    try:
        line_number = int(line)
    except (TypeError, ValueError):
        return []
    return [[page, line_number]]


def build_fever_metric_functions(mock_mode: bool = False) -> dict[str, Any]:
    """Return callables wired up according to the execution mode."""

    if mock_mode:
        return {"per_example": _mock_example_metrics}

    return {"per_example": _real_example_metrics}
