"""Regression coverage for #1109 parameter-mode trial injection."""

from __future__ import annotations

from typing import Any

import pytest

import traigent
from traigent.config.context import get_config
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "case-0"}, "fast"),
            EvaluationExample({"text": "case-1"}, "fast"),
        ],
        name="parameter_injection_per_trial",
    )


def _config_value(config: TraigentConfig | dict[str, Any], key: str) -> Any:
    if isinstance(config, TraigentConfig):
        return config.get(key)
    return config.get(key)


def _recorded_variants(result: Any) -> set[str]:
    return {trial.config.get("variant") for trial in result.trials}


def _assert_function_saw_recorded_trials(
    seen: list[dict[str, str]],
    result: Any,
) -> None:
    recorded = _recorded_variants(result)
    param_seen = {entry["param_variant"] for entry in seen}
    context_seen = {entry["context_variant"] for entry in seen}

    assert recorded == {"slow", "fast"}
    assert param_seen == recorded
    assert context_seen == recorded
    assert all(entry["param_variant"] == entry["context_variant"] for entry in seen)


@pytest.mark.asyncio
async def test_parameter_mode_sync_function_uses_per_trial_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    seen: list[dict[str, str]] = []

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"variant": ["slow", "fast"]},
        default_config={"variant": "slow"},
        injection_mode="parameter",
    )
    def answer(text: str, config: TraigentConfig) -> str:
        context_config = get_config()
        param_variant = str(config.get("variant"))
        context_variant = str(_config_value(context_config, "variant"))
        seen.append(
            {
                "text": text,
                "param_variant": param_variant,
                "context_variant": context_variant,
            }
        )
        return param_variant

    result = await answer.optimize(algorithm="grid", max_trials=3)

    _assert_function_saw_recorded_trials(seen, result)


@pytest.mark.asyncio
async def test_parameter_mode_async_function_uses_per_trial_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    seen: list[dict[str, str]] = []

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"variant": ["slow", "fast"]},
        default_config={"variant": "slow"},
        injection_mode="parameter",
    )
    async def answer(text: str, config: TraigentConfig) -> str:
        context_config = get_config()
        param_variant = str(config.get("variant"))
        context_variant = str(_config_value(context_config, "variant"))
        seen.append(
            {
                "text": text,
                "param_variant": param_variant,
                "context_variant": context_variant,
            }
        )
        return param_variant

    result = await answer.optimize(algorithm="grid", max_trials=3)

    _assert_function_saw_recorded_trials(seen, result)
