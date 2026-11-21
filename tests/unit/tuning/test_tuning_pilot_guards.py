"""Pilot guards: no search-space values leave the machine, schedule-only, no fcntl at start."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.tuning.controller import (
    E_CONNECTED_TUNING_REFUSED,
    E_LOCK_UNSUPPORTED_PLATFORM,
    traigent_optimize_fn,
)
from traigent.tuning.errors import E_UNSUPPORTED_IN_RELEASE, TuningError

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    NEW,
    OLD,
    FakeOptimizer,
    _holdout,
    make,
    module,
    state,
)

SPACE = {"model": [OLD["model"], NEW["model"]]}


def _connected_function(**extra: Any) -> Any:
    """A function in the SDK's DEFAULT (connected-capable) execution mode."""

    @traigent.optimize(
        configuration_space=SPACE,
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        max_trials=2,
        default_config=dict(OLD),
        **extra,
    )
    def live(q: str) -> str:
        return "a"

    return live


@pytest.fixture
def backend_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every construction / session call of the backend cloud client."""
    from traigent.cloud.client import TraigentCloudClient

    calls: list[str] = []
    original_init = TraigentCloudClient.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        calls.append("__init__")
        original_init(self, *args, **kwargs)

    async def recording_session(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("create_optimization_session")
        raise RuntimeError("no backend in tests")

    monkeypatch.setattr(TraigentCloudClient, "__init__", recording_init)
    monkeypatch.setattr(
        TraigentCloudClient, "create_optimization_session", recording_session
    )
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key_for_unit_tests_only")
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE", "TRAIGENT_REQUIRE_CLOUD"):
        monkeypatch.delenv(name, raising=False)
    return calls


# -- (c) privacy: tuning never runs connected ---------------------------------------
def test_a_tuning_run_makes_no_backend_call_carrying_search_space_values(
    backend_calls: list[str],
) -> None:
    live = _connected_function()
    result = traigent_optimize_fn(live, algorithm="grid", max_trials=2)(SPACE)
    assert result.trials  # the optimization really ran, locally
    assert backend_calls == []


def test_the_default_algorithm_is_pinned_local_too(backend_calls: list[str]) -> None:
    live = _connected_function()
    result = traigent_optimize_fn(live, max_trials=2)(SPACE)  # no algorithm: "auto"
    assert result.trials
    assert backend_calls == []


def test_a_smart_algorithm_would_run_connected_so_it_is_refused(
    backend_calls: list[str],
) -> None:
    live = _connected_function()
    with pytest.raises(TuningError) as err:
        traigent_optimize_fn(live, algorithm="bayesian", max_trials=2)(SPACE)
    assert err.value.code == E_CONNECTED_TUNING_REFUSED
    live_smart = _connected_function(algorithm="bayesian")
    with pytest.raises(TuningError) as err:
        traigent_optimize_fn(live_smart, max_trials=2)(SPACE)
    assert err.value.code == E_CONNECTED_TUNING_REFUSED
    assert backend_calls == []


def test_a_cloud_required_function_is_refused_not_pinned_offline(
    backend_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_REQUIRE_CLOUD", "1")  # resolves require_cloud=True
    live = _connected_function()
    assert live.execution_policy.require_cloud
    for kwargs in ({}, {"algorithm": "auto"}, {"algorithm": "grid"}):
        with pytest.raises(TuningError) as err:
            traigent_optimize_fn(live, max_trials=2, **kwargs)(SPACE)
        assert err.value.code == E_CONNECTED_TUNING_REFUSED
    assert backend_calls == []


def test_a_hybrid_api_function_is_refused(backend_calls: list[str]) -> None:
    live = _connected_function()
    live.external_service_evaluator = object()  # an external (connected) evaluator
    with pytest.raises(TuningError) as err:
        traigent_optimize_fn(live, algorithm="grid", max_trials=2)(SPACE)
    assert err.value.code == E_CONNECTED_TUNING_REFUSED
    assert backend_calls == []


def test_the_live_function_keeps_its_own_execution_policy(
    backend_calls: list[str],
) -> None:
    live = _connected_function()
    before = (live.execution_policy, live.offline, live.execution_mode)
    traigent_optimize_fn(live, algorithm="grid", max_trials=2)(SPACE)
    assert (live.execution_policy, live.offline, live.execution_mode) == before


# -- (f) schedule-only ----------------------------------------------------------------
@pytest.mark.parametrize(
    "trigger",
    [
        {"monitor": "quality_prod", "event": "episode_opened"},
        {"monitor": "quality_prod", "event": "recovery_overdue"},
        {"external": "model_released"},
        {"schedule": {"every_days": 30}},
    ],
)
def test_every_non_schedule_trigger_is_refused_at_construction(
    tmp_path: Path, trigger: dict[str, Any]
) -> None:
    st = state()
    st.tvl = module()
    st.tvl["responses"][0]["rules"].append(
        {
            "id": "extra",
            "trigger": trigger,
            "do": "optimize",
            "approval": "human",
            "deploy": "if_promoted",
            "protects": ["quality_floor"],
        }
    )
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "store", FakeOptimizer())
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE
    assert not (tmp_path / "store").exists()


# -- (a) Windows: fcntl is only needed when the lock is used ---------------------------
_NO_FCNTL = textwrap.dedent(
    """
    import sys
    sys.modules["fcntl"] = None  # as on a platform without fcntl
    import traigent.cli.main  # CLI start must not need fcntl
    import traigent.tuning
    from traigent.tuning.controller import Controller, E_LOCK_UNSUPPORTED_PLATFORM
    from traigent.tuning.errors import TuningError
    c = Controller.__new__(Controller)
    c.dir = __import__("pathlib").Path(sys.argv[1])
    try:
        with c._locked():
            raise SystemExit("lock acquired without fcntl")
    except TuningError as exc:
        assert exc.code == E_LOCK_UNSUPPORTED_PLATFORM, exc.code
        print("closed:" + exc.code)
    """
)


def test_cli_and_tuning_import_without_fcntl_and_the_lock_refuses_cleanly(
    tmp_path: Path,
) -> None:
    out = subprocess.run(  # noqa: S603 - fixed argv, a test-owned script
        [sys.executable, "-c", _NO_FCNTL, str(tmp_path / "store")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"closed:{E_LOCK_UNSUPPORTED_PLATFORM}" in out.stdout
