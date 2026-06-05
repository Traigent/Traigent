"""Regression for #1109: injection_mode="parameter" must deliver PER-TRIAL
configs, not the wrap-time default.

Before the fix, ParameterBasedProvider captured ``TraigentConfig.from_dict``
once at wrap time and injected that frozen object on every call — every
trial of a parameter-mode optimization evaluated the DEFAULT configuration
while the recorded ``trial.config`` showed the suggested values, silently
making the results meaningless. The pre-existing integration tests only
asserted ``len(result.trials) > 0`` and passed regardless.
"""

from __future__ import annotations

import pytest

import traigent
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample


def _dataset(size: int = 2) -> Dataset:
    return Dataset(
        [EvaluationExample({"text": f"q{i}"}, "fast") for i in range(size)],
        name="param_injection_regression",
    )


def _make_wrapped():
    function_saw: list[dict] = []

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"variant": ["slow", "fast"]},
        default_config={"variant": "slow"},
        injection_mode="parameter",
    )
    def answer(text: str, config: TraigentConfig) -> str:
        function_saw.append(dict(config.custom_params))
        return str(config.custom_params.get("variant"))

    return answer, function_saw


@pytest.mark.asyncio
@pytest.mark.usefixtures("cost_preflight_approved")
async def test_parameter_mode_delivers_per_trial_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate .traigent/ storage per test
    wrapped, function_saw = _make_wrapped()

    result = await wrapped.optimize(algorithm="grid", max_trials=3)

    # every recorded trial config value must have been SEEN by the function
    recorded = {trial.config.get("variant") for trial in result.trials}
    seen = {call.get("variant") for call in function_saw}
    assert recorded == {"slow", "fast"}
    assert seen == {"slow", "fast"}, (
        f"function saw only {seen} — wrap-time default injected (#1109)"
    )

    # and the metrics must reflect what the function actually evaluated:
    # variant=fast answers 'fast' (accuracy 1.0), variant=slow answers 'slow'
    by_variant = {
        trial.config.get("variant"): trial.metrics.get("accuracy")
        for trial in result.trials
    }
    assert by_variant.get("fast") == 1.0, by_variant
    assert by_variant.get("slow") == 0.0, by_variant


def test_direct_call_outside_optimization_keeps_wrap_time_default(
    tmp_path, monkeypatch
):
    """Preserved behavior: calling the wrapped function directly (no active
    configuration context) injects the wrap-time default config."""
    monkeypatch.chdir(tmp_path)  # isolate .traigent/ storage per test
    wrapped, function_saw = _make_wrapped()

    out = wrapped("hello")

    assert out == "slow"
    assert function_saw[-1].get("variant") == "slow"


def test_explicit_config_kwarg_still_wins(tmp_path, monkeypatch):
    """Preserved behavior: a caller-supplied config param is never overridden."""
    monkeypatch.chdir(tmp_path)  # isolate .traigent/ storage per test
    wrapped, function_saw = _make_wrapped()

    out = wrapped("hello", config=TraigentConfig.from_dict({"variant": "fast"}))

    assert out == "fast"
