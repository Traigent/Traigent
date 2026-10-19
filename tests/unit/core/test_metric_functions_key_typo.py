"""A misspelt ``metric_functions`` key must fail fast (Traigent#2659)."""

from __future__ import annotations

from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.core.optimization_pipeline import validate_metric_function_keys
from traigent.utils.exceptions import ValidationError


def _mine(prediction, expected, input_data=None, **kw):  # noqa: ANN001, ANN201
    return 0.123


def test_near_miss_key_raises_with_did_you_mean() -> None:
    with pytest.raises(ValidationError, match="Did you mean 'accuracy'"):
        validate_metric_function_keys({"acuracy": _mine}, ["accuracy"])


def test_wrong_case_key_raises() -> None:
    with pytest.raises(ValidationError, match="Did you mean 'accuracy'"):
        validate_metric_function_keys({"Accuracy": _mine}, ["accuracy"])


def test_correct_key_and_auxiliary_key_are_accepted() -> None:
    validate_metric_function_keys({"accuracy": _mine}, ["accuracy"])
    validate_metric_function_keys(
        {"accuracy": _mine, "latency_ms": _mine}, ["accuracy"]
    )
    validate_metric_function_keys(None, ["accuracy"])


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> float | None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"aa"}\n{"input":{"q":"2"},"output":"aa"}\n'
    )

    @traigent.optimize(
        configuration_space={"v": ["aa"]},
        objectives=["accuracy"],
        offline=True,
        max_trials=1,
        algorithm="grid",
        evaluation=EvaluationOptions(
            eval_dataset="d.jsonl", metric_functions={key: _mine}
        ),
    )
    def agent(q: str) -> str:
        return traigent.get_config()["v"]

    return agent.optimize_sync().best_score


def test_end_to_end_typo_fails_before_trials_and_correct_key_uses_user_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, monkeypatch, "accuracy") == pytest.approx(0.123)
    with pytest.raises(ValidationError, match="acuracy"):
        _run(tmp_path, monkeypatch, "acuracy")
