"""Tests for the SeamlessOptunaAdapter."""

from __future__ import annotations

import asyncio

import pytest

from traigent.config.context import TrialContext, get_trial_context
from traigent.config.seamless_optuna_adapter import SeamlessOptunaAdapter


def test_adapter_injects_configuration_and_context():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": 42,
        "model": "gpt-4",
        "temperature": 0.2,
    }

    captured: dict[str, float] = {}

    def target(model: str, temperature: float) -> float:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == 42
        captured[model] = temperature
        return temperature

    wrapped = adapter.inject(target, trial_config)
    result = wrapped()

    assert result == 0.2
    assert captured == {"gpt-4": 0.2}


def test_adapter_does_not_mutate_trial_config():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": 99,
        "model": "gpt-4o",
        "temperature": 0.5,
    }
    snapshot_before = dict(trial_config)

    def target(model: str, temperature: float) -> float:
        assert model == "gpt-4o"
        assert temperature == 0.5
        return temperature

    wrapped = adapter.inject(target, trial_config)
    wrapped()

    assert trial_config == snapshot_before


@pytest.mark.asyncio
async def test_adapter_supports_async_functions():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": "async-1",
        "model": "claude",
        "temperature": 0.6,
    }

    async def async_target(model: str, temperature: float) -> str:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == "async-1"
        await asyncio.sleep(0)
        return f"{model}:{temperature}"

    wrapped = adapter.inject(async_target, trial_config)
    result = await wrapped()
    assert result == "claude:0.6"


@pytest.mark.asyncio
async def test_adapter_is_safe_for_parallel_invocations():
    adapter = SeamlessOptunaAdapter()
    trial_config = {
        "_optuna_trial_id": "parallel",
        "param": 1,
    }

    async def async_target(param: int) -> int:
        ctx = get_trial_context()
        assert ctx and ctx["trial_id"] == "parallel"
        snapshot = ctx["config_snapshot"]
        snapshot["param"] += 1
        await asyncio.sleep(0)
        return snapshot["param"]

    wrapped = adapter.inject(async_target, trial_config)
    results = await asyncio.gather(wrapped(), wrapped())

    assert results == [2, 2]
    assert trial_config["param"] == 1


def test_adapter_telemetry_hook_invoked():
    events: list[dict[str, object]] = []

    def telemetry(payload: dict[str, object]) -> None:
        events.append(payload)

    adapter = SeamlessOptunaAdapter(telemetry_hook=telemetry)
    trial_config = {
        "_optuna_trial_id": 7,
        "model": "cohere",
    }

    def target(model: str) -> str:
        return model

    wrapped = adapter.inject(target, trial_config)
    wrapped()

    assert events[0]["event"] == "trial_call_started"
    assert events[-1]["event"] == "trial_call_completed"
    assert all(event["trial_id"] == 7 for event in events)


def test_adapter_telemetry_hook_errors_are_suppressed(caplog):
    def telemetry(_: dict[str, object]) -> None:
        raise RuntimeError("telemetry boom")

    adapter = SeamlessOptunaAdapter(telemetry_hook=telemetry)
    trial_config = {"_optuna_trial_id": 8}

    def target() -> str:
        return "ok"

    wrapped = adapter.inject(target, trial_config)
    with caplog.at_level("WARNING"):
        assert wrapped() == "ok"
    assert any("telemetry boom" in record.message for record in caplog.records)


def test_adapter_requires_trial_id():
    adapter = SeamlessOptunaAdapter()
    with pytest.raises(ValueError):
        adapter.inject(lambda: None, {})


def test_trial_context_manager_manual_usage():
    payload = {"extra": "value"}
    with TrialContext(trial_id="manual", metadata=payload) as ctx:
        assert ctx["trial_id"] == "manual"
        assert ctx["extra"] == "value"
        assert get_trial_context()["trial_id"] == "manual"
    assert get_trial_context() is None
