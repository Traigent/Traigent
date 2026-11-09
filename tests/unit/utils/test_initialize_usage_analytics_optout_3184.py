"""#3184: traigent.initialize(enable_usage_analytics=False) must reach the run config.

Mocked transport; only sessions whose search_space type is "analytics" count.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import traigent
from traigent import Choices
from traigent.api.functions import _GLOBAL_CONFIG
from traigent.config.types import TraigentConfig
from traigent.utils.local_analytics import LocalAnalytics

KEY = "tg_" + "k" * 61  # synthetic

pytestmark = pytest.mark.backend_online


class _Recorder:
    analytics_posts: list[dict[str, Any]] = []

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def create_hybrid_session(self, **kwargs: Any):
        space = kwargs.get("search_space") or {}
        if isinstance(space, dict) and space.get("type") == "analytics":
            _Recorder.analytics_posts.append(kwargs)
        raise RuntimeError("mock transport: no backend")

    async def submit_privacy_trial_results(self, **kwargs: Any) -> bool:
        return True


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    _Recorder.analytics_posts = []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # No key in the environment: the run itself never contacts a backend.
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_ALLOW_UNTRACKED", "1")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: False)
    monkeypatch.setattr(
        "traigent.cloud.backend_client.get_backend_client", lambda **k: _Recorder()
    )
    monkeypatch.setattr(
        "traigent.utils.local_analytics._analytics_api_key_error", lambda key: None
    )
    monkeypatch.setattr(
        LocalAnalytics,
        "collect_usage_stats",
        lambda self: {"total_sessions": 3, "total_trials": 9, "sdk_version": "x"},
    )
    saved = dict(_GLOBAL_CONFIG)
    _GLOBAL_CONFIG.clear()
    yield tmp_path
    _GLOBAL_CONFIG.clear()
    _GLOBAL_CONFIG.update(saved)


def _run_local_optimize(tmp_path) -> None:
    data = tmp_path / "ds.jsonl"
    data.write_text(json.dumps({"input": {"q": "hi"}, "output": "x"}) + "\n")

    @traigent.optimize(
        configuration_space={"m": Choices(["a", "b"])},
        eval_dataset=str(data),
        objectives=["accuracy"],
        scoring_function=lambda o, e, *a, **k: 1.0,
        execution_mode="local",
        local_storage_path=str(tmp_path / "store"),
    )
    def f(q: str) -> str:
        return traigent.get_config()["m"]

    f.optimize_sync(max_trials=2, algorithm="grid")


def _run_and_capture_flag(env, monkeypatch) -> list[bool]:
    """Run a local optimize; return the usage-analytics flag the run config carried."""
    from traigent.core.orchestrator import OptimizationOrchestrator

    seen: list[bool] = []
    orig = OptimizationOrchestrator._submit_usage_analytics

    async def spy(self):
        seen.append(self.traigent_config.enable_usage_analytics)
        return await orig(self)

    monkeypatch.setattr(OptimizationOrchestrator, "_submit_usage_analytics", spy)
    _run_local_optimize(env)
    return seen


def test_initialize_kwarg_opt_out_reaches_run(env, monkeypatch) -> None:
    traigent.initialize(enable_usage_analytics=False)
    assert _run_and_capture_flag(env, monkeypatch) == [False]
    assert _Recorder.analytics_posts == []


def test_initialize_config_opt_out_reaches_run(env, monkeypatch) -> None:
    traigent.initialize(config=TraigentConfig(enable_usage_analytics=False))
    assert _run_and_capture_flag(env, monkeypatch) == [False]
    assert _Recorder.analytics_posts == []


def test_default_run_keeps_analytics_enabled(env, monkeypatch) -> None:
    """Control: no initialize opt-out leaves the (default-on) flag True."""
    assert _run_and_capture_flag(env, monkeypatch) == [True]


def test_run_config_picks_up_global_opt_out(env) -> None:
    from traigent.core.optimization_pipeline import create_traigent_config

    traigent.initialize(enable_usage_analytics=False)
    cfg = create_traigent_config("local", None, True, False)
    assert cfg.enable_usage_analytics is False
    # explicit per-run value wins over the global setting, both directions
    assert (
        create_traigent_config(
            "local", None, True, False, enable_usage_analytics=True
        ).enable_usage_analytics
        is True
    )
    _GLOBAL_CONFIG.pop("enable_usage_analytics", None)
    traigent.initialize(enable_usage_analytics=True)
    assert (
        create_traigent_config(
            "local", None, True, False, enable_usage_analytics=False
        ).enable_usage_analytics
        is False
    )
