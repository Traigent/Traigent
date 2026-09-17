"""Focused tests covering apply_best_config across injection modes."""

from __future__ import annotations

from datetime import datetime

import pytest

import traigent
from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult
from traigent.core.optimized_function import OptimizedFunction


class DummyLLM:
    """Minimal stand-in for an LLM client to test literal arguments."""

    def __init__(self, model: str) -> None:
        self.model = model

    def invoke(self, prompt: str) -> str:
        return f"{self.model}:{prompt}"


@pytest.fixture
def sample_optimization_result() -> OptimizationResult:
    trials = [
        TrialResult(
            trial_id="trial_1",
            config={"model": "gpt-3.5", "temperature": 0.2},
            metrics={"accuracy": 0.80},
            status=OptimizationStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_2",
            config={"model": "gpt-4", "temperature": 0.1},
            metrics={"accuracy": 0.92},
            status=OptimizationStatus.COMPLETED,
            duration=1.2,
            timestamp=datetime.now(),
            metadata={},
        ),
    ]

    return OptimizationResult(
        trials=trials,
        best_config={"model": "gpt-4", "temperature": 0.1},
        best_score=0.92,
        optimization_id="opt_123",
        duration=2.2,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
        metadata={},
    )


def test_apply_best_config_seamless_signature_defaults(
    sample_optimization_result,
) -> None:
    """Applying best config updates signature defaults via runtime shim."""

    @traigent.optimize(
        configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.2]},
        injection_mode="seamless",
    )
    def fn(
        question: str, model: str = "gpt-3.5", temperature: float = 0.2
    ) -> tuple[str, float]:
        return model, temperature

    opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
    opt_fn._optimization_results = sample_optimization_result

    opt_fn.apply_best_config()

    assert fn("hi") == ("gpt-4", 0.1)


def test_apply_best_config_parameter_mode(sample_optimization_result) -> None:
    """Parameter injection mode continues to work after apply_best_config."""

    @traigent.optimize(
        configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.2]},
        injection_mode="parameter",
        config_param="cfg",
    )
    def fn(question: str, cfg: dict) -> tuple[str, float | None]:
        return cfg.get("model", "unset"), cfg.get("temperature")

    opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
    opt_fn._optimization_results = sample_optimization_result

    opt_fn.apply_best_config()

    assert fn("hi") == ("gpt-4", 0.1)


def test_apply_best_config_seamless_literal_call_fails_closed(
    sample_optimization_result,
) -> None:
    """Issue #2298: a literal call-site argument (no local assignment or
    matching parameter) means the configuration_space has zero injectable
    targets. Decoration still succeeds (a function that is only ever called
    directly is unaffected), but once a config is applied every call raises
    -- on the first call and on every later one (C1: the unvaried function is
    never cached) -- and the body never runs."""
    from traigent.config.providers import SeamlessNoInjectableTargetsError

    calls: list[str] = []

    @traigent.optimize(
        configuration_space={"model": ["gpt-3.5", "gpt-4"]},
        injection_mode="seamless",
    )
    def fn(prompt: str) -> str:
        calls.append(prompt)
        return DummyLLM(model="gpt-3.5").invoke(prompt)

    opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
    opt_fn._optimization_results = sample_optimization_result
    opt_fn.apply_best_config()

    for _ in range(3):
        with pytest.raises(
            SeamlessNoInjectableTargetsError, match="no injectable target"
        ):
            fn("hi")
    assert calls == []


def test_optimize_seamless_literal_call_aborts_before_first_trial(
    tmp_path, monkeypatch
) -> None:
    """Issue #2298 direction 1: with zero injectable targets the run aborts
    before the first trial -- no trial is created and the body never runs."""
    import json

    from traigent.config.providers import SeamlessNoInjectableTargetsError

    monkeypatch.chdir(tmp_path)  # dataset paths must sit under the working directory
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"input": {"prompt": f"q{i}"}, "expected_output": "a"})
            for i in range(3)
        )
        + "\n"
    )
    calls: list[str] = []

    @traigent.optimize(
        eval_dataset="ds.jsonl",
        configuration_space={"model": ["gpt-3.5", "gpt-4"]},
        objectives=["accuracy"],
        injection_mode="seamless",
        offline=True,
    )
    def fn(prompt: str) -> str:
        calls.append(prompt)
        return DummyLLM(model="gpt-3.5").invoke(prompt)

    with pytest.raises(SeamlessNoInjectableTargetsError, match="no injectable target"):
        fn.optimize_sync(algorithm="grid", max_trials=2)
    assert calls == []
