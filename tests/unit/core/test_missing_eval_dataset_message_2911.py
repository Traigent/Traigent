"""Omitting eval_dataset names the parameter and the next step (#2911)."""

from __future__ import annotations

import pytest

from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


def _fn(eval_dataset):
    def agent(text: str) -> str:
        return text

    return OptimizedFunction(
        func=agent,
        configuration_space={"model": ["a", "b"]},
        objectives=["accuracy"],
        eval_dataset=eval_dataset,
    )


def test_missing_eval_dataset_names_parameter_and_alternative() -> None:
    with pytest.raises(ConfigurationError) as exc:
        _fn(None)._load_dataset()
    msg = str(exc.value)
    assert "eval_dataset" in msg
    assert "EvaluationOptions" in msg
    assert ".jsonl" in msg
    assert "not provided" in msg or "omitted" in msg


def test_wrong_type_names_received_type_and_accepted_forms() -> None:
    with pytest.raises(ConfigurationError) as exc:
        _fn(12345)._load_dataset()
    msg = str(exc.value)
    assert "int" in msg
    assert "JSONL" in msg


def test_missing_eval_dataset_names_only_remedies_that_avoid_the_error() -> None:
    """A custom evaluator does not avoid this error, so it must not be suggested."""
    with pytest.raises(ConfigurationError) as exc:
        _fn(None)._load_dataset()
    msg = str(exc.value)
    assert "custom evaluator" not in msg.lower()
    assert 'eval_dataset="data.jsonl"' in msg
    assert "EvaluationOptions(eval_dataset=...)" in msg
