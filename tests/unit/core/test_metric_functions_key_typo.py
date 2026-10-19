"""A misspelt ``metric_functions`` key must fail fast (Traigent#2659)."""

from __future__ import annotations

from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.core.optimization_pipeline import validate_metric_function_keys


def _mine(prediction, expected, input_data=None, **kw):  # noqa: ANN001, ANN201
    return 0.123


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_near_miss_key_warns_with_did_you_mean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        validate_metric_function_keys({"acuracy": _mine}, ["accuracy"])
    msgs = _warnings(caplog)
    assert any("acuracy" in m and "Did you mean 'accuracy'" in m for m in msgs)


def test_wrong_case_key_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        validate_metric_function_keys({"Accuracy": _mine}, ["accuracy"])
    assert any("Did you mean 'accuracy'" in m for m in _warnings(caplog))


def test_mapping_returning_scorer_key_passes_without_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``{"scores": fn}`` (fn returns a mapping) must never be rejected."""
    with caplog.at_level("WARNING"):
        validate_metric_function_keys({"scores": _mine}, ["score", "precision"])
    # At most an advisory naming the key and the suggestion, no values.
    for m in _warnings(caplog):
        assert "0.123" not in m


def test_correct_key_and_auxiliary_key_are_accepted() -> None:
    validate_metric_function_keys({"accuracy": _mine}, ["accuracy"])
    validate_metric_function_keys(
        {"accuracy": _mine, "latency_ms": _mine}, ["accuracy"]
    )
    validate_metric_function_keys(None, ["accuracy"])


def test_derived_auxiliary_names_are_not_typos(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        validate_metric_function_keys({"latency_ms": _mine}, ["latency"])
        validate_metric_function_keys({"accuracy_raw": _mine}, ["accuracy"])
    assert not _warnings(caplog)


def test_key_for_objective_covered_by_scoring_function_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        validate_metric_function_keys(
            {"acuracy": _mine}, ["accuracy"], scoring_function=_mine
        )
    assert not _warnings(caplog)


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


def test_end_to_end_typo_warns_and_correct_key_uses_user_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert _run(tmp_path, monkeypatch, "accuracy") == pytest.approx(0.123)
    with caplog.at_level("WARNING"):
        _run(tmp_path, monkeypatch, "acuracy")
    assert any("acuracy" in m for m in _warnings(caplog))
