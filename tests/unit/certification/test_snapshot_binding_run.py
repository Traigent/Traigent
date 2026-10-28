"""``ExecutionOptions.snapshot_store``: S0 before dispatch, S1 from the terminal result."""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import traigent
from tests.unit.certification.fakes import FakeG1Backend
from traigent.api.functions import with_usage
from traigent.api.types import OptimizationResult
from traigent.certification import snapshot_binding
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.snapshot.backends import LocalDirStore
from traigent.utils.exceptions import OptimizationError, SnapshotBindingError
from traigent.utils.optimization_result_persistence import (
    RESULT_RESET,
    encode_result_fields,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    for name in ("TRAIGENT_REQUIRE_CLOUD", "TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)

    def _no_backend() -> Any:
        raise AssertionError("the Backend must not be contacted")

    monkeypatch.setattr(snapshot_binding, "default_transport", _no_backend)


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


def _dataset(payload: Any = "q") -> Dataset:
    return Dataset([EvaluationExample({"text": payload}, "YES")], name="g1")


def _agent(
    store: Any,
    *,
    calls: list[str] | None = None,
    dataset: Dataset | None = None,
    execution: dict[str, Any] | None = None,
    mutate: Any = None,
):
    options: dict[str, Any] = {"offline": True}
    if store is not None:
        options["snapshot_store"] = store
    options.update(execution or {})

    @traigent.optimize(
        eval_dataset=dataset or _dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.2]},
        scoring_function=_scorer,
        execution=options,
    )
    def agent(text: str) -> str:
        if calls is not None:
            calls.append(text)
        if mutate is not None:
            mutate()
        return with_usage("YES", total_cost=0.0)

    return agent


@pytest.fixture
def store(tmp_path: Path) -> LocalDirStore:
    return LocalDirStore(tmp_path / "store")


def test_offline_run_captures_s0_s1_and_attaches_local_handles(
    store: LocalDirStore, tmp_path: Path
) -> None:
    result = _agent(store).optimize_sync(max_trials=2)
    assert result.snapshot_ref is not None
    assert result.snapshot_ref.manifest_digest.startswith("sha256:")
    path = Path(result.client_evidence_path)
    assert path.is_file() and path.is_relative_to(tmp_path / "home")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_result_fields_are_local_only() -> None:
    assert {"snapshot_ref", "client_evidence_path"} <= RESULT_RESET
    result = OptimizationResult(
        trials=[],
        best_config={"a": 1},
        best_score=1.0,
        optimization_id="x",
        duration=0.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="random",
        timestamp=__import__("datetime").datetime.now(),
    )
    result.snapshot_ref = "sha256:" + "a" * 64
    result.client_evidence_path = "/tmp/evidence"
    dumped = str(encode_result_fields(result))
    assert "client_evidence_path" not in dumped.replace("RESET", "")
    assert "/tmp/evidence" not in dumped
    assert "sha256:" + "a" * 64 not in dumped


def test_s0_refuses_incomplete_input_before_any_trial(store: LocalDirStore) -> None:
    calls: list[str] = []
    agent = _agent(store, calls=calls, dataset=_dataset(payload=object()))
    with pytest.raises(SnapshotBindingError) as caught:
        agent.optimize_sync(max_trials=2)
    assert caught.value.code == "SNAPSHOT_INCOMPLETE"
    assert caught.value.result is None
    assert calls == []


def test_online_run_refuses_closed_before_spend_when_not_ready(
    store: LocalDirStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeG1Backend()
    fake.readiness = "unavailable"
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)
    calls: list[str] = []
    agent = _agent(store, calls=calls, execution={"offline": False})
    with pytest.raises(SnapshotBindingError) as caught:
        agent.optimize_sync(max_trials=2)
    assert caught.value.code == "READINESS_UNAVAILABLE"
    assert calls == []
    assert [m for m, _, _ in fake.requests] == ["GET"]


def test_readiness_probe_is_skipped_for_offline_capture(store: LocalDirStore) -> None:
    # The autouse fixture makes default_transport raise if it were used.
    assert _agent(store).optimize_sync(max_trials=1).snapshot_ref is not None


def test_s1_runs_on_the_cloud_completion_path(
    store: LocalDirStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.core.optimized_function import OptimizedFunction

    fake = FakeG1Backend()
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)

    async def _cloud(self: Any, *args: Any, **kwargs: Any) -> OptimizationResult:
        return OptimizationResult(
            trials=[],
            best_config={"temperature": 0.2},
            best_score=1.0,
            optimization_id="cloud",
            duration=0.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="remote",
            timestamp=__import__("datetime").datetime.now(),
        )

    monkeypatch.setattr(OptimizedFunction, "_try_cloud_execution", _cloud)
    calls: list[str] = []
    result = _agent(store, calls=calls, execution={"offline": False}).optimize_sync(
        max_trials=1
    )
    assert calls == []  # the cloud path returned before any local trial
    assert [m for m, _, _ in fake.requests] == ["GET"]  # readiness before spend
    assert result.snapshot_ref is not None
    assert Path(result.client_evidence_path).is_file()


def test_s1_drift_raises_typed_error_with_result_on_sync_and_async(
    store: LocalDirStore,
) -> None:
    for runner in ("sync", "async"):
        payload = {"n": 0}
        dataset = _dataset(payload=payload)

        def mutate(payload: dict[str, int] = payload) -> None:
            payload["n"] += 1

        agent = _agent(store, dataset=dataset, mutate=mutate)
        with pytest.raises(SnapshotBindingError) as caught:
            if runner == "sync":
                agent.optimize_sync(max_trials=1)
            else:
                asyncio.run(agent.optimize(max_trials=1))
        error = caught.value
        assert isinstance(error, OptimizationError)
        assert error.code == "SNAPSHOT_INPUT_DRIFT"
        assert isinstance(error.result, OptimizationResult)
        assert error.result.best_config is not None
        assert error.result.snapshot_ref is None


def test_no_eligible_winner_refuses_with_result(store: LocalDirStore) -> None:
    run = snapshot_binding.SnapshotRun(store)
    result = OptimizationResult(
        trials=[],
        best_config=None,
        best_score=None,
        optimization_id="x",
        duration=0.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="random",
        timestamp=__import__("datetime").datetime.now(),
    )
    run._s0_dataset = ("present", "x")
    with pytest.raises(SnapshotBindingError) as caught:
        run.capture_s1(result)
    assert caught.value.code == "NO_ELIGIBLE_WINNER"
    assert caught.value.result is result


def test_snapshot_binding_error_is_public_and_closed() -> None:
    assert traigent.SnapshotBindingError is SnapshotBindingError
    with pytest.raises(ValueError):
        SnapshotBindingError("SOMETHING_ELSE")
    assert "q" not in str(SnapshotBindingError("SNAPSHOT_INCOMPLETE"))


def test_default_off_changes_nothing(tmp_path: Path) -> None:
    result = _agent(None).optimize_sync(max_trials=1)
    assert result.snapshot_ref is None
    assert result.client_evidence_path is None
    assert not (tmp_path / "home" / ".traigent" / "certification").exists()
    script = textwrap.dedent(
        """
        import sys
        import traigent
        from traigent.evaluators.base import Dataset, EvaluationExample

        @traigent.optimize(
            eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g"),
            objectives=["accuracy"],
            configuration_space={"temperature": [0.1]},
            scoring_function=lambda p, e: 1.0,
            execution={"offline": True},
        )
        def agent(text):
            from traigent.api.functions import with_usage
            return with_usage("YES", total_cost=0.0)

        agent.optimize_sync(max_trials=1)
        print(sorted(m for m in sys.modules if m.startswith(
            ("traigent.certification", "traigent.snapshot"))))
        """
    )
    env = dict(
        os.environ, TRAIGENT_COST_APPROVED="true", TRAIGENT_RUN_COST_LIMIT="100.0"
    )
    done = subprocess.run(  # nosec B603 - fixed argv, test-only
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.stdout.strip().splitlines()[-1] == "[]", done.stdout + done.stderr
