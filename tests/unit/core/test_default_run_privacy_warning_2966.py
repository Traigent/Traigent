"""A default optimize run must not raise the privacy_enabled deprecation (#2966).

The SDK forwards its own internal privacy default to TraigentConfig. Because
TraigentConfig treats any supplied privacy_enabled value as user-supplied, the
internal default produced a DeprecationWarning attributed to the user's
decorator line. Under warnings-as-errors that aborts a default run.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.config.types import TraigentConfig
from traigent.core.optimization_pipeline import create_traigent_config


def _privacy_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "privacy_enabled" in str(w.message)
    ]


def test_create_traigent_config_without_privacy_emits_no_deprecation() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = create_traigent_config(
            execution_mode="local",
            local_storage_path=None,
            minimal_logging=True,
            privacy_enabled=None,
        )

    assert _privacy_warnings(caught) == []
    assert config.privacy_enabled is False


def test_explicit_privacy_setting_still_warns() -> None:
    """A value the user actually supplied keeps its deprecation notice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TraigentConfig(privacy_enabled=False)

    assert _privacy_warnings(caught)


def test_default_optimize_run_survives_privacy_deprecation_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"a"}\n{"input":{"q":"2"},"output":"a"}\n'
    )

    @traigent.optimize(
        configuration_space={"v": ["a", "b"]},
        objectives=["accuracy"],
        offline=True,
        max_trials=2,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def agent(q: str) -> str:
        return traigent.get_config()["v"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="privacy_enabled is deprecated",
            category=DeprecationWarning,
        )
        result = agent.optimize_sync(max_trials=2)

    assert len(result.trials) == 2
