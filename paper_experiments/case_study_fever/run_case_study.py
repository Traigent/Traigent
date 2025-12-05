"""Orchestrates evaluation for the FEVER regression fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from traigent.evaluators.base import EvaluationExample

from .dataset import load_case_study_dataset
from .metrics import _mock_example_metrics, _real_example_metrics
from .simulator import generate_case_study_answer

__all__ = ["evaluate_configuration"]


@dataclass
class _ExampleResult:
    claim: str
    output: dict[str, Any]
    metrics: dict[str, Any]


def evaluate_configuration(
    config: Mapping[str, Any] | None,
    *,
    mock_mode: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Evaluate a configuration against the bundled FEVER dataset."""

    dataset = load_case_study_dataset()
    resolved_config = dict(config or {})

    limit = len(dataset.examples)
    if max_examples is not None:
        limit = min(limit, max(0, int(max_examples)))

    selected_examples = dataset.examples[:limit]

    per_example: list[_ExampleResult] = []
    accuracies: list[float] = []

    for example in selected_examples:
        claim = str(example.input_data.get("claim", ""))
        output = _produce_output(example, resolved_config, mock_mode=mock_mode)
        metrics = _compute_metrics(example, output, resolved_config, mock_mode=mock_mode)
        per_example.append(_ExampleResult(claim=claim, output=output, metrics=metrics))
        accuracies.append(float(metrics.get("accuracy", 0.0)))

    aggregated_metrics = {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "sample_count": len(per_example),
    }

    return {
        "config": resolved_config,
        "mock_mode": mock_mode,
        "sample_count": len(per_example),
        "per_example": [result.__dict__ for result in per_example],
        "aggregated_metrics": aggregated_metrics,
    }


def _produce_output(
    example: EvaluationExample,
    config: Mapping[str, Any],
    *,
    mock_mode: bool,
) -> dict[str, Any]:
    if mock_mode:
        claim = str(example.input_data.get("claim", ""))
        return generate_case_study_answer(claim, example, config=config)

    # Real pipelines would call out to tooling / LLMs; for unit tests we reuse the simulator.
    return generate_case_study_answer(str(example.input_data.get("claim", "")), example, config=config)


def _compute_metrics(
    example: EvaluationExample,
    output: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    mock_mode: bool,
) -> dict[str, Any]:
    if mock_mode:
        accuracy = 1.0 if str(output.get("verdict")).upper() == str(example.expected_output).upper() else 0.0
        return _mock_example_metrics(claim=str(example.input_data.get("claim", "")), config=config, llm_metrics={"label_accuracy": accuracy})

    return _real_example_metrics(
        output=output,
        expected=example.expected_output,
        example=example,
        config=config,
        llm_metrics=None,
    )
