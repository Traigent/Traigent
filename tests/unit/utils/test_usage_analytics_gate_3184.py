"""#3184: every usage-analytics send honours one gate.

Zero backend requests when no_egress, TRAIGENT_DISABLE_TELEMETRY or offline is
set, on the orchestrator path, collect_and_submit_analytics, LocalAnalytics and
the incentive/CLI callers; a control test proves the default still sends.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.local_analytics import (
    LocalAnalytics,
    collect_and_submit_analytics,
)

KEY = "tg_" + "k" * 61  # synthetic


class _Recorder:
    calls: list[str] = []

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def create_hybrid_session(self, **kwargs: Any):
        _Recorder.calls.append("create_hybrid_session")
        return "sid", "tok", "http://127.0.0.1:1"

    async def submit_privacy_trial_results(self, **kwargs: Any) -> bool:
        _Recorder.calls.append("submit")
        return True


@pytest.fixture(autouse=True)
def transport(monkeypatch, tmp_path):
    _Recorder.calls = []
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "TRAIGENT_DISABLE_TELEMETRY",
        "TRAIGENT_OFFLINE_MODE",
        "TRAIGENT_OFFLINE",
        "TRAIGENT_MOCK_LLM",
    ):
        monkeypatch.delenv(name, raising=False)
    # The in-code mock flag can leak from earlier tests in a shared process.
    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: False)
    monkeypatch.setattr(
        "traigent.cloud.backend_client.get_backend_client",
        lambda **k: _Recorder(),
    )
    monkeypatch.setattr(
        "traigent.config.backend_config.BackendConfig.get_api_key",
        staticmethod(lambda: KEY),
    )
    monkeypatch.setattr(
        LocalAnalytics,
        "collect_usage_stats",
        lambda self: {"total_sessions": 3, "total_trials": 9, "sdk_version": "x"},
    )
    return tmp_path


def _config(tmp_path, **kw: Any) -> TraigentConfig:
    return TraigentConfig(
        execution_mode=ExecutionMode.LOCAL.value,
        local_storage_path=str(tmp_path / "store"),
        **kw,
    )


CASES = {
    "no_egress": ({"no_egress": True}, {}),
    "disable_telemetry": ({}, {"TRAIGENT_DISABLE_TELEMETRY": "true"}),
    "offline_env": ({}, {"TRAIGENT_OFFLINE_MODE": "true"}),
    "offline_config": ({"offline": True}, {}),
    "analytics_off": ({"enable_usage_analytics": False}, {}),
}


def _entry_direct(config):
    return asyncio.run(LocalAnalytics(config).submit_usage_stats(force=True))


async def _entry_collect(config):
    collect_and_submit_analytics(config)
    await asyncio.sleep(0.3)


async def _entry_orchestrator(config):
    from traigent.core.orchestrator import OptimizationOrchestrator

    stub = SimpleNamespace(traigent_config=config)
    await OptimizationOrchestrator._submit_usage_analytics(stub)  # type: ignore[arg-type]


async def _entry_incentives(config):
    from traigent.utils.incentives import IncentiveManager

    manager = IncentiveManager(config)
    manager.update_usage_stats()
    await asyncio.sleep(0.3)


async def _entry_cli(config, monkeypatch, tmp_path):
    from traigent.cli import local_commands

    monkeypatch.setattr(
        local_commands.TraigentConfig, "from_environment", staticmethod(lambda: config)
    )
    monkeypatch.setattr(
        local_commands.LocalStorageManager,
        "list_sessions",
        lambda self, status=None: [SimpleNamespace(completed_trials=1)],
    )
    monkeypatch.setattr(local_commands, "_echo_sessions_table", lambda *a: None)
    local_commands.list_sessions.callback(None, 10, "table")  # type: ignore[attr-defined]
    await asyncio.sleep(0.3)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize(
    "entry", ["direct", "collect", "orchestrator", "incentives", "cli"]
)
@pytest.mark.asyncio
async def test_gate_blocks_every_entry_point(case, entry, transport, monkeypatch):
    cfg_kw, env = CASES[case]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config = _config(transport, **cfg_kw)
    if entry == "direct":
        await asyncio.to_thread(_entry_direct, config)
    elif entry == "collect":
        await _entry_collect(config)
    elif entry == "orchestrator":
        await _entry_orchestrator(config)
    elif entry == "incentives":
        await _entry_incentives(config)
    else:
        await _entry_cli(config, monkeypatch, transport)
    assert _Recorder.calls == []
    from traigent.utils.local_analytics import usage_analytics_allowed

    assert usage_analytics_allowed(config) is False


@pytest.mark.parametrize(
    "entry", ["direct", "collect", "orchestrator", "incentives", "cli"]
)
@pytest.mark.asyncio
async def test_default_config_with_key_still_sends(entry, transport, monkeypatch):
    config = _config(transport)
    assert config.enable_usage_analytics is True
    if entry == "direct":
        await asyncio.to_thread(_entry_direct, config)
    elif entry == "collect":
        await _entry_collect(config)
    elif entry == "orchestrator":
        await _entry_orchestrator(config)
    elif entry == "incentives":
        await _entry_incentives(config)
    else:
        await _entry_cli(config, monkeypatch, transport)
    assert "create_hybrid_session" in _Recorder.calls


@pytest.mark.parametrize("entry", ["collect", "orchestrator"])
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.asyncio
async def test_entry_guard_stops_before_building_analytics(
    case, entry, transport, monkeypatch
):
    """The entry point itself refuses: it never even constructs LocalAnalytics."""
    cfg_kw, env = CASES[case]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    built: list[str] = []

    class _Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            built.append("built")

    monkeypatch.setattr("traigent.utils.local_analytics.LocalAnalytics", _Boom)
    config = _config(transport, **cfg_kw)
    if entry == "collect":
        await _entry_collect(config)
    else:
        await _entry_orchestrator(config)
    assert built == []
