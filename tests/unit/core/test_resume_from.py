"""``optimize(resume_from=...)`` reuses matching finished trials (Traigent#1697)."""

from __future__ import annotations

import copy
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.resume import (
    REUSED_TRIAL_KEY,
    load_resume_source,
    plan_resume,
)
from traigent.utils.persistence import PersistenceManager

_FPS = {
    "dataset": "fp1:d",
    "evaluator": "fp1:e",
    "agent": "fp1:a",
    "config_space": "fp1:c",
}


def _trial(
    tid: str, status: TrialStatus = TrialStatus.COMPLETED, **metrics: float
) -> TrialResult:
    return TrialResult(
        trial_id=tid,
        config={"v": tid},
        metrics={"accuracy": 1.0, "total_cost": 0.5, **metrics},
        status=status,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def _prior(trials: list[TrialResult], fps: dict[str, Any] | None = None):
    return OptimizationResult(
        trials=trials,
        best_config=None,
        best_score=None,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={"artifact_fingerprints": dict(_FPS if fps is None else fps)},
    )


# ---- plan_resume (unit) -----------------------------------------------------


def test_matching_completed_trials_are_reused_and_marked() -> None:
    plan = plan_resume(_prior([_trial("a"), _trial("b")]), _FPS)
    assert [t.metadata[REUSED_TRIAL_KEY] for t in plan.reusable] == [True, True]
    assert plan.mismatched_components == ()


@pytest.mark.parametrize("component", ["dataset", "evaluator", "agent", "config_space"])
def test_any_identity_mismatch_reuses_nothing_and_warns(
    component: str, caplog: pytest.LogCaptureFixture
) -> None:
    current = {**_FPS, component: "fp1:other"}
    with caplog.at_level("WARNING"):
        plan = plan_resume(_prior([_trial("a")]), current)
        from traigent.core.resume import warn_if_not_reusable

        warn_if_not_reusable(plan)
    assert plan.reusable == ()
    assert plan.mismatched_components == (component,)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(msgs) == 1 and component in msgs[0]
    assert "fp1:" not in msgs[0]  # value-free


def test_missing_identity_fails_closed() -> None:
    assert plan_resume(_prior([_trial("a")], fps={}), _FPS).reusable == ()
    assert plan_resume(_prior([_trial("a")]), None).identity_unavailable


def test_in_flight_and_unknown_status_trials_are_not_reused() -> None:
    trials = [
        _trial("done"),
        _trial("run", TrialStatus.RUNNING),
        _trial("pend", TrialStatus.PENDING),
        _trial("fail", TrialStatus.FAILED),
    ]
    plan = plan_resume(_prior(trials), _FPS)
    assert [t.metadata["resumed_original_trial_id"] for t in plan.reusable] == ["done"]
    assert plan.not_reusable_unfinished == 2
    assert plan.not_reusable_other == 1


def test_completed_trial_with_no_cost_information_is_not_reused() -> None:
    bare = _trial("x")
    del bare.metrics["total_cost"]
    assert plan_resume(_prior([bare]), _FPS).reusable == ()


def test_unknown_cost_trial_stays_unknown() -> None:
    t = _trial("u", cost_unmeasured=1.0)
    del t.metrics["total_cost"]
    (reused,) = plan_resume(_prior([t]), _FPS).reusable
    assert reused.metrics["cost_unmeasured"] == 1.0
    assert "total_cost" not in reused.metrics


def test_reuse_is_capped_by_max_trials() -> None:
    plan = plan_resume(_prior([_trial(str(i)) for i in range(5)]), _FPS, max_trials=3)
    assert len(plan.reusable) == 3
    assert plan.dropped_over_max_trials == 2


def test_source_is_not_mutated() -> None:
    prior = _prior([_trial("a")])
    before = copy.deepcopy(prior.trials[0].metadata)
    plan_resume(prior, _FPS)
    assert prior.trials[0].metadata == before


# ---- load_resume_source -----------------------------------------------------


def test_wrong_type_raises_type_error() -> None:
    with pytest.raises(TypeError):
        load_resume_source(123)


def test_missing_saved_result_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_resume_source(str(tmp_path / "nope"))


# ---- end to end (mocked LLM, offline) ---------------------------------------


@pytest.fixture
def _local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"a"}\n{"input":{"q":"2"},"output":"a"}\n'
    )
    return tmp_path


def _agent(space: list[str], seen: list[str], dataset: str = "d.jsonl"):
    @traigent.optimize(
        configuration_space={"v": space},
        objectives=["accuracy"],
        offline=True,
        max_trials=4,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset=dataset),
    )
    def agent(q: str) -> str:
        value = traigent.get_config()["v"]
        seen.append(value)
        return value

    return agent


def test_resume_runs_only_the_remainder(_local: Path) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=2)
    assert len(first.trials) == 2
    seen.clear()

    resumed = agent.optimize_sync(max_trials=4, resume_from=first)

    assert set(seen) == {"c", "d"}  # a, b were reused, not re-run
    assert len(resumed.trials) == 4  # reused trials count toward max_trials
    summary = resumed.metadata["resume"]
    assert summary["reused_trials"] == 2
    assert summary["cost_charged_to_this_run_limit"] is False
    reused = [t for t in resumed.trials if t.metadata.get(REUSED_TRIAL_KEY)]
    assert len(reused) == 2
    # Unknown cost stays unknown (mock runs do not measure cost).
    assert all(t.metrics.get("cost_unmeasured") for t in reused)


def test_resume_does_not_exceed_max_trials(_local: Path) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=3)
    seen.clear()
    resumed = agent.optimize_sync(max_trials=3, resume_from=first)
    assert seen == []  # everything was already done
    assert len(resumed.trials) == 3


def test_space_mismatch_reuses_nothing_and_warns(
    _local: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _agent(["a", "b", "c", "d"], []).optimize_sync(max_trials=2)
    seen: list[str] = []
    other = _agent(["a", "b", "c"], seen)
    with caplog.at_level("WARNING"):
        resumed = other.optimize_sync(max_trials=3, resume_from=first)
    assert resumed.metadata["resume"]["reused_trials"] == 0
    assert resumed.metadata["resume"]["identity_mismatch"] == ["config_space"]
    assert set(seen) == {"a", "b", "c"}
    assert sum("identity components" in r.getMessage() for r in caplog.records) == 1


def test_saved_result_handle_round_trips(_local: Path) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=2)
    saved = PersistenceManager(base_dir="saved").save_result(first, "first")
    seen.clear()
    resumed = agent.optimize_sync(max_trials=4, resume_from=saved)
    assert set(seen) == {"c", "d"}
    assert resumed.metadata["resume"]["reused_trials"] == 2


def test_tampered_saved_result_raises_value_error(_local: Path) -> None:
    agent = _agent(["a", "b", "c", "d"], [])
    first = agent.optimize_sync(max_trials=2)
    saved = Path(PersistenceManager(base_dir="saved").save_result(first, "first"))
    trials_file = saved / "trials.json.gz"
    with gzip.open(trials_file, "rt") as fh:
        payload = json.load(fh)
    payload[0]["metrics"]["accuracy"] = 0.999
    with gzip.open(trials_file, "wt") as fh:
        json.dump(payload, fh)

    with pytest.raises(ValueError, match="integrity"):
        agent.optimize_sync(max_trials=4, resume_from=str(saved))


def test_cloud_run_with_resume_from_raises(
    _local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.core import optimized_function as of
    from traigent.core.execution_policy_runtime import SOURCE_CLOUD_BRAIN

    seen: list[str] = []
    agent = _agent(["a", "b"], seen)
    first = agent.optimize_sync(max_trials=1)
    seen.clear()
    # The unit-test environment forces offline mode; make this run resolve to
    # the managed cloud route instead.
    monkeypatch.setattr(of, "initial_result_source", lambda *a, **k: SOURCE_CLOUD_BRAIN)
    with pytest.raises(ValueError, match="local runs only"):
        agent.optimize_sync(resume_from=first)
    assert seen == []  # refused before any trial ran


def test_external_evaluator_run_with_resume_from_raises(_local: Path) -> None:
    agent = _agent(["a", "b"], [])
    first = agent.optimize_sync(max_trials=1)
    agent.execution_mode = "hybrid_api"
    with pytest.raises(ValueError, match="local runs only"):
        agent.optimize_sync(resume_from=first)


def test_grid_optimizer_skips_marked_configs() -> None:
    from traigent.optimizers.grid import GridSearchOptimizer

    opt = GridSearchOptimizer({"v": ["a", "b", "c"]}, ["accuracy"])
    opt.mark_config_evaluated({"v": "b", "ignored_extra": 1})
    assert [opt.suggest_next_trial([])["v"] for _ in range(2)] == ["a", "c"]


# ---- reviewer follow-ups (#1697): execution-level -----------------------------


def test_in_flight_prior_trial_config_is_never_run(_local: Path) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=3)  # a, b, c finished
    first.trials[2].status = TrialStatus.RUNNING  # c: outcome unknown
    seen.clear()

    resumed = agent.optimize_sync(max_trials=4, resume_from=first)

    assert "c" not in seen  # not retried
    assert set(seen) == {"d"}
    assert resumed.metadata["resume"]["unfinished_prior_trials_not_reused"] == 1
    assert [t.config["v"] for t in resumed.trials].count("c") == 0


def test_reused_baseline_does_not_spend_another_trial(_local: Path) -> None:
    seen: list[str] = []

    @traigent.optimize(
        configuration_space={"v": ["a", "b", "c", "d"]},
        default_config={"v": "a"},
        objectives=["accuracy"],
        offline=True,
        max_trials=4,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def agent(q: str) -> str:
        seen.append(traigent.get_config()["v"])
        return "a"

    first = agent.optimize_sync(max_trials=2)
    assert first.trials[0].config["v"] == "a"
    seen.clear()
    resumed = agent.optimize_sync(max_trials=4, resume_from=first)
    assert "a" not in seen  # neither baseline nor grid point re-run
    assert set(seen) == {"b", "c"}
    assert len(resumed.trials) == 4


def test_reused_measured_cost_is_in_study_total_and_not_charged(
    _local: Path,
) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=2)
    for t in first.trials:  # measured cost under a non-canonical alias
        t.metrics.pop("cost_unmeasured", None)
        t.metrics.pop("total_cost", None)
        t.metrics["cost"] = 0.25
    resumed = agent.optimize_sync(max_trials=4, resume_from=first)
    assert resumed.total_cost == pytest.approx(0.5)
    summary = resumed.metadata["resume"]
    assert summary["reused_known_cost"] == pytest.approx(0.5)
    assert summary["cost_charged_to_this_run_limit"] is False
    # New (unmeasured) trials stay unknown; reused measured cost is canonical.
    reused = [t for t in resumed.trials if t.metadata.get(REUSED_TRIAL_KEY)]
    assert all(t.metrics["total_cost"] == pytest.approx(0.25) for t in reused)


def test_reused_cost_does_not_trip_a_small_cost_limit(_local: Path) -> None:
    seen: list[str] = []

    @traigent.optimize(
        configuration_space={"v": ["a", "b", "c", "d"]},
        objectives=["accuracy"],
        offline=True,
        max_trials=4,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def agent(q: str) -> str:
        value = traigent.get_config()["v"]
        seen.append(value)
        return traigent.with_usage(value, total_cost=0.01)

    first = agent.optimize_sync(max_trials=2)
    for t in first.trials:
        t.metrics["total_cost"] = 5.0  # far above the new run's limit
    seen.clear()
    resumed = agent.optimize_sync(
        max_trials=4, resume_from=first, cost_limit=1.0, cost_approved=True
    )
    assert resumed.stop_reason != "cost_limit"
    assert set(seen) == {"c", "d"}
    assert resumed.total_cost == pytest.approx(10.0 + 4 * 0.01 * 2, abs=0.2)


def test_legacy_result_without_manifest_is_refused_unless_opted_in(
    _local: Path,
) -> None:
    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=2)
    saved = Path(PersistenceManager(base_dir="saved").save_result(first, "first"))
    (saved / "integrity.sha256.json").unlink()

    with pytest.raises(ValueError, match="integrity manifest"):
        agent.optimize_sync(max_trials=4, resume_from=str(saved))
    seen.clear()
    resumed = agent.optimize_sync(
        max_trials=4, resume_from=str(saved), allow_unverified_resume=True
    )
    assert resumed.metadata["resume"]["reused_trials"] == 2


def test_in_memory_result_needs_no_manifest(_local: Path) -> None:
    first = _agent(["a", "b"], []).optimize_sync(max_trials=1)
    assert load_resume_source(first) is first


def test_cloud_route_is_refused_before_any_trial(
    _local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    first = _agent(["a", "b"], seen).optimize_sync(max_trials=1)
    seen.clear()
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.setenv(name, "false")

    @traigent.optimize(
        configuration_space={"v": ["a", "b"]},
        objectives=["accuracy"],
        algorithm="auto",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def cloud_agent(q: str) -> str:
        seen.append("ran")
        return traigent.get_config()["v"]

    with pytest.raises(ValueError, match="local runs only"):
        cloud_agent.optimize_sync(resume_from=first)
    assert seen == []


def _discrete_batch_optimizers():
    from traigent.optimizers.batch_optimizers import (
        AdaptiveBatchOptimizer,
        BatchOptimizationConfig,
        MultiObjectiveBatchOptimizer,
        ParallelBatchOptimizer,
    )
    from traigent.optimizers.random import RandomSearchOptimizer

    space = {"v": ["a", "b", "c"]}
    cfg = BatchOptimizationConfig(batch_size=2)

    def base():
        return RandomSearchOptimizer(space, ["accuracy"], max_trials=10)

    return space, [
        ParallelBatchOptimizer(base_optimizer=base(), batch_config=cfg)
        if "base_optimizer" in ParallelBatchOptimizer.__init__.__code__.co_varnames
        else ParallelBatchOptimizer(space, ["accuracy"], batch_config=cfg),
        AdaptiveBatchOptimizer(base_optimizer=base(), batch_config=cfg),
        MultiObjectiveBatchOptimizer(space, ["accuracy"], batch_config=cfg),
    ]


def test_batch_optimizers_forward_marks_to_inner_optimizer() -> None:
    space, optimizers = _discrete_batch_optimizers()
    for opt in optimizers:
        opt.mark_config_evaluated({"v": "a"})
        opt.mark_config_evaluated({"v": "b"})
        got = {opt.suggest_next_trial([])["v"] for _ in range(1)}
        assert got == {"c"}, type(opt).__name__


def test_continuous_random_same_seed_does_not_replay_marked_configs() -> None:
    from traigent.optimizers.random import RandomSearchOptimizer

    space = {"x": (0.0, 1.0)}
    first = RandomSearchOptimizer(space, ["accuracy"], max_trials=50, random_seed=7)
    done = [first.suggest_next_trial([]) for _ in range(5)]

    again = RandomSearchOptimizer(space, ["accuracy"], max_trials=50, random_seed=7)
    for config in done:
        again.mark_config_evaluated(config)
    fresh = [again.suggest_next_trial([]) for _ in range(5)]
    assert not any(c in done for c in fresh)


def test_in_flight_baseline_is_not_executed(_local: Path) -> None:
    seen: list[str] = []

    @traigent.optimize(
        configuration_space={"v": ["a", "b", "c"]},
        default_config={"v": "a"},
        objectives=["accuracy"],
        offline=True,
        max_trials=3,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def agent(q: str) -> str:
        seen.append(traigent.get_config()["v"])
        return "a"

    first = agent.optimize_sync(max_trials=2)
    for t in first.trials:
        t.status = TrialStatus.RUNNING  # outcome unknown, baseline included
    seen.clear()
    resumed = agent.optimize_sync(max_trials=3, resume_from=first)
    assert "a" not in seen
    assert resumed.metadata["resume"]["reused_trials"] == 0


def test_concurrent_resume_is_refused_and_does_not_mix_state(_local: Path) -> None:
    import threading

    from traigent.utils.exceptions import OverlappingOptimizationError

    started, release = threading.Event(), threading.Event()
    seen: list[str] = []

    @traigent.optimize(
        configuration_space={"v": ["a", "b", "c", "d"]},
        objectives=["accuracy"],
        offline=True,
        max_trials=4,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
    )
    def agent(q: str) -> str:
        started.set()
        release.wait(timeout=30)
        seen.append(traigent.get_config()["v"])
        return "a"

    release.set()
    first = agent.optimize_sync(max_trials=2)
    release.clear()
    started.clear()
    seen.clear()
    box: dict[str, Any] = {}

    def run() -> None:
        box["r"] = agent.optimize_sync(max_trials=4, resume_from=first)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(timeout=30)
        with pytest.raises(OverlappingOptimizationError):
            agent.optimize_sync(max_trials=4, resume_from=first)
    finally:
        release.set()
        worker.join(timeout=60)
    resumed = box["r"]
    assert resumed.metadata["resume"]["reused_trials"] == 2
    assert set(seen) == {"c", "d"}
    assert len(resumed.trials) == 4


def test_resumed_local_run_makes_no_network_connection(
    _local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    seen: list[str] = []
    agent = _agent(["a", "b", "c", "d"], seen)
    first = agent.optimize_sync(max_trials=2)
    attempts: list[Any] = []

    def refuse(self: Any, address: Any, *a: Any, **k: Any) -> None:
        attempts.append(address)
        raise OSError("network disabled in test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    resumed = agent.optimize_sync(max_trials=4, resume_from=first)
    assert attempts == []
    assert resumed.metadata.get("offline_mode") is True
    assert resumed.metadata["resume"]["reused_trials"] == 2
