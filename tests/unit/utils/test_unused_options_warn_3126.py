"""#3126: options that no code path reads must warn, not silently do nothing."""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

from traigent.cloud.backend_components import BackendClientConfig
from traigent.cloud.billing import CostTrackingConfig
from traigent.generation.options import DatasetGrowthOptions, PromptRewriteOptions
from traigent.optimizers.batch_optimizers import BatchOptimizationConfig
from traigent.security.rate_limiter import SecureRateLimitConfig
from traigent.utils.batch_processing import process_with_retry_and_recovery

CASES = [
    (BatchOptimizationConfig, {"distributed_workers": 4}, "distributed_workers"),
    (BatchOptimizationConfig, {"enable_checkpointing": False}, "enable_checkpointing"),
    (SecureRateLimitConfig, {"global_rate_limit": 5}, "global_rate_limit"),
    (CostTrackingConfig, {"enable_client_tracking": False}, "enable_client_tracking"),
    (BackendClientConfig, {"use_mcp": True}, "use_mcp"),
    (PromptRewriteOptions, {"rewrite_model": "m"}, "rewrite_model"),
    (DatasetGrowthOptions, {"gen_model": "m"}, "gen_model"),
]
DEFAULTS = [
    (BatchOptimizationConfig, {"distributed_workers": 1, "enable_checkpointing": True}),
    (SecureRateLimitConfig, {"global_rate_limit": 1000}),
    (CostTrackingConfig, {"enable_client_tracking": True}),
    (BackendClientConfig, {"use_mcp": False}),
    (PromptRewriteOptions, {"rewrite_model": None}),
    (DatasetGrowthOptions, {"gen_model": None}),
    (BatchOptimizationConfig, {}),
]


def _warned(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "no effect" in r.getMessage()]


@pytest.mark.parametrize(("cls", "kwargs", "name"), CASES)
def test_non_default_value_warns_3126(cls, kwargs, name, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cls(**kwargs)
    msgs = _warned(caplog)
    assert len(msgs) == 1 and name in msgs[0]


@pytest.mark.parametrize(("cls", "kwargs"), DEFAULTS)
def test_explicit_default_does_not_warn_3126(cls, kwargs, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cls(**kwargs)
    assert _warned(caplog) == []


def test_checkpoint_interval_warns_only_when_non_default_3126(caplog) -> None:
    async def proc(batch):  # pragma: no cover - never reached with no items
        return []

    with caplog.at_level(logging.WARNING):
        asyncio.run(process_with_retry_and_recovery([], proc, checkpoint_interval=10))
        assert _warned(caplog) == []
        asyncio.run(process_with_retry_and_recovery([], proc, checkpoint_interval=3))
    msgs = _warned(caplog)
    assert len(msgs) == 1 and "checkpoint_interval" in msgs[0]


def test_no_browser_flag_reaches_streamlit_3126(monkeypatch) -> None:
    plugin = Path(__file__).resolve().parents[3] / "plugins" / "traigent-ui"
    monkeypatch.syspath_prepend(str(plugin))
    seen: list[list[str]] = []
    cli_mod = types.ModuleType("streamlit.web.cli")
    cli_mod.main = lambda: seen.append(list(sys.argv)) or 0  # type: ignore[attr-defined]
    for name, mod in (
        ("streamlit", types.ModuleType("streamlit")),
        ("streamlit.web", types.ModuleType("streamlit.web")),
        ("streamlit.web.cli", cli_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    ctrl = types.ModuleType("traigent_ui.traigent_control_center")
    ctrl.__file__ = "app.py"
    monkeypatch.setitem(sys.modules, "traigent_ui.traigent_control_center", ctrl)
    monkeypatch.setattr(sys, "argv", ["x"])
    from traigent_ui import launch_playground

    with pytest.raises(SystemExit):
        launch_playground(port=1, open_browser=False)
    with pytest.raises(SystemExit):
        launch_playground(port=1)
    assert "--server.headless" in seen[0]
    assert "--server.headless" not in seen[1]
