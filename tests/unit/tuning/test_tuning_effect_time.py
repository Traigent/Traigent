"""Effect-time checks, environment, loaded-dataset identity, directory durability,
draining attempts and path components."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.code_version import CODE_VERSION_ENV, resolve_code_version
from traigent.tuning.controller import Controller
from traigent.tuning.errors import TuningError

from .conftest import tvl_free
from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    KIT,
    NEW,
    T0,
    FakeOptimizer,
    _deploy_awaiting,
    _holdout,
    at,
    controller,
    e_ref,
    head,
    make,
    state,
)


def _scope_end() -> Any:
    """When the first schedule occurrence of ``E1`` ends: 30 days after its tick."""
    return T0 + timedelta(hours=9, days=30)


def _clock(monkeypatch: pytest.MonkeyPatch, start: Any) -> list[Any]:
    clock = [start]
    real_utc = controller_module._utc
    monkeypatch.setattr(
        controller_module,
        "_utc",
        lambda value: real_utc(value) if value is not None else clock[0],
    )
    return clock


# -- the resolved callable itself needs readable source -----------------------------
@tvl_free
def test_generated_functions_without_source_do_not_share_the_module_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    # both claim a READABLE module (this test file) but have no source of their own
    namespace: dict[str, Any] = {"__name__": __name__}
    exec("def one(q):\n    return 1\n", namespace)  # noqa: S102
    exec("def two(q):\n    return 2\n", namespace)  # noqa: S102
    for fn in (namespace["one"], namespace["two"]):
        with pytest.raises(TuningError) as err:
            resolve_code_version(None, fn)
        assert err.value.code == "E_NO_CODE_VERSION"
    assert resolve_code_version("explicit-1", namespace["one"]) == (
        "tag:" + hashlib.sha256(b"explicit-1").hexdigest()
    )


# -- effect-time checks come AFTER authentication and replay ------------------------
def test_a_scope_that_ends_during_authentication_does_not_start_the_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    calls = [0]

    def slow_hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] > 1:  # the use-time re-check at the spend is slow
            clock[0] = _scope_end() + timedelta(seconds=1)
        return True

    c = make(
        state(),
        tmp_path / "store",
        opt,
        now=at(0),
        authority=ApproverAuthority(
            frozenset(APPROVERS), mode="platform", authenticator=slow_hook
        ),
    )
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    calls[0] = 0
    c.step()
    assert opt.calls == []


# -- the environment travels with the attached tuning state -------------------------
def test_the_attached_state_and_the_cli_carry_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from traigent.cli.tuning_commands import _build_controller
    from traigent.observability import observe
    from traigent.tuning.policy import Tuning

    from .test_controller import module

    @observe(
        client=MagicMock(),
        agent="env-agent",
        environment="staging",
        tune=Tuning.from_tvl(module()),
    )
    def classify(x: int) -> str:
        return "a"

    assert classify.tuning.environment == "staging"
    target = tmp_path / "target_mod.py"
    target.write_text(
        "from unittest.mock import MagicMock\n"
        "from traigent.observability import observe\n"
        "from traigent.tuning.policy import Tuning\n"
        "import yaml\n"
        f"_doc = yaml.safe_load(open({str(tmp_path / 'm.yml')!r}))\n"
        "@observe(client=MagicMock(), agent='env-agent', environment='staging',\n"
        "         tune=Tuning.from_tvl(_doc))\n"
        "def classify(x):\n    return 'a'\n",
        encoding="utf-8",
    )
    import yaml

    (tmp_path / "m.yml").write_text(yaml.safe_dump(module()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    ctl = _build_controller(None, None, f"{target}:classify", None)
    assert ctl.meta["environment"] == "staging"
    assert "staging" in ctl.dir.parts


# -- identity of the dataset the run ACTUALLY loaded --------------------------------
def test_a_per_call_dataset_equal_to_the_holdout_is_refused(tmp_path: Path) -> None:
    from traigent.tuning.evidence import _dataset_rows, examples_digest

    holdout_rows = [
        json.loads(ln) for ln in KIT["holdout"].read_text().splitlines() if ln.strip()
    ]

    class Optimized:
        eval_dataset = [{"input": {"x": 999}, "output": "billing"}]  # NOT the holdout

        def __call__(self, x: int) -> str:
            return "billing"

        def _dataset_from_source(self, source: Any) -> Any:
            from traigent.core.optimized_function import OptimizedFunction

            return OptimizedFunction._dataset_from_source(OptimizedFunction, source)

    class RunsOnHoldout(FakeOptimizer):
        def __call__(self, space: dict[str, Any]) -> Any:
            # the run was given the holdout as its per-call dataset
            result = super().__call__(space)
            result["loaded_dataset_digests"] = [examples_digest(holdout_rows)]
            return result

    assert _dataset_rows is not None
    c = controller(tmp_path, RunsOnHoldout())
    c.state.optimized = Optimized()
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert any(
        f.get("reason") == "holdout_is_optimization_eval_set"
        for f in c.controller_facts()
    )


@tvl_free
def test_the_real_adapter_records_the_digest_of_every_dataset_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import traigent
    from traigent.tuning.controller import traigent_optimize_fn
    from traigent.tuning.evidence import examples_digest

    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    rows = [{"input": {"q": "per-call"}, "output": "per-call"}]

    @traigent.optimize(
        configuration_space={"model": ["anthropic/claude-haiku-4-5", "x/none"]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=1,
        default_config={"model": "anthropic/claude-haiku-4-5"},
    )
    def agent(q: str) -> str:
        return q

    run = traigent_optimize_fn(agent, eval_dataset=rows)
    outcome = run(
        {"model": ["anthropic/claude-haiku-4-5"]}, candidate_dir=tmp_path / "c"
    )
    assert examples_digest(rows) in outcome.loaded_dataset_digests
    assert not hasattr(run, "loaded_dataset_digests")  # nothing shared across runs


# -- a newly created directory entry is durable before the optimizer starts ---------
@tvl_free
def test_a_new_directory_is_fsynced_into_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[str] = []
    real = os.fsync

    def spy(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    controller_module._write_json(tmp_path, tmp_path / "attempts" / "m.json", {"a": 1})
    assert str(tmp_path / "attempts") in synced  # the rename inside it
    assert str(tmp_path) in synced  # the new "attempts" entry itself


# -- a closed (draining) attempt still records its saved finish ---------------------
def test_a_closed_attempt_with_a_saved_finish_marker_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = Controller._append

    def crash(self: Controller, category: str, record: Any, *a: Any, **k: Any) -> None:
        if record and record.get("fact") == "finished":
            raise RuntimeError("crash before recording")
        real(self, category, record, *a, **k)

    monkeypatch.setattr(Controller, "_append", crash)
    with pytest.raises(RuntimeError):
        c.step(at(10))
    monkeypatch.undo()
    restarted = controller(tmp_path, opt)
    restarted.step(at(24 * 60))  # long after the scope closed
    delivered = [
        ln
        for ln in restarted._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]
    assert len(delivered) == 1  # the saved finish reached the relation, exactly once


# -- reserved path components -----------------------------------------------------
@pytest.mark.parametrize("bad", [".", "..", ""])
@tvl_free
def test_dot_components_are_refused_in_the_default_store_path(bad: str) -> None:
    from traigent.tuning.controller import default_store_dir

    with pytest.raises(TuningError):
        default_store_dir(bad, "production")
    if bad:
        with pytest.raises(TuningError):
            default_store_dir("agent-1", bad)


# -- the actual start of an optimization is recorded ---------------
def started_lines(c: Controller) -> list[Any]:
    return [
        ln
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "started"
    ]


def test_the_start_of_an_optimization_is_recorded_once_before_it_runs(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    started = [
        ln
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "started"
    ]
    assert len(started) == 1
    c.step(at(11))
    assert len(started_lines(c)) == 1


# -- no history replay after the final effect-time check ---------------------------------
def _replay_advances(
    monkeypatch: pytest.MonkeyPatch, clock: list[Any], by: timedelta
) -> None:
    real = Controller._history

    def slow(self: Controller, *args: Any, **kwargs: Any) -> Any:
        clock[0] = clock[0] + by  # every history replay takes this long
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Controller, "_history", slow)


def test_a_scope_that_ends_during_the_final_scope_replay_does_not_start_the_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    real = Controller._scope_requires_intervention
    calls = [0]

    def slow(self: Controller, occ: Any, instant: int) -> bool:
        calls[0] += 1
        verdict = real(self, occ, instant)
        if calls[0] >= 2:  # the replay made right before the spend is slow
            clock[0] = _scope_end() + timedelta(seconds=1)
        return verdict

    monkeypatch.setattr(Controller, "_scope_requires_intervention", slow)
    report = c.step()
    assert opt.calls == []
    assert report.refused and report.refused[0].endswith(":scope_expired")


def test_a_scope_that_ends_while_the_start_marker_is_written_does_not_start_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The very last check is made after the start marker is written, not before."""
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    real = controller_module._write_json

    def delayed(path: Path, value: Any) -> None:
        real(path, value)
        if value.get("state") == "started":  # the marker write was slow
            clock[0] = _scope_end() + timedelta(seconds=1)

    monkeypatch.setattr(controller_module, "_write_json", delayed)
    report = c.step()
    assert opt.calls == []
    assert report.refused and report.refused[0].endswith(":scope_expired")
    finished = [
        ln["record"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]
    assert [r["completion"] for r in finished] == ["cancelled"]  # never started
    assert started_lines(c) == []


def test_a_revoked_approver_at_the_start_marker_does_not_start_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = controller_module._write_json

    def revoke(path: Path, value: Any) -> None:
        real(path, value)
        if value.get("state") == "started":
            c.authority = ApproverAuthority(frozenset({"someone-else"}))

    monkeypatch.setattr(controller_module, "_write_json", revoke)
    c.step(at(10))
    assert opt.calls == []
    finished = [
        ln["record"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]
    assert [r["completion"] for r in finished] == ["cancelled"]  # never started
    assert started_lines(c) == []


def test_replay_that_takes_a_second_each_never_defers_a_valid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    _replay_advances(monkeypatch, clock, timedelta(seconds=1))
    for _ in range(3):  # consecutive steps: the first one already runs it
        c.step()
    assert len(opt.calls) == 1


# -- every optimizer path returns the digests of what ITS run loaded ---------------------
def test_an_injected_optimizer_that_reports_no_dataset_identity_cannot_promote(
    tmp_path: Path,
) -> None:
    class Silent:
        def __call__(self, space: dict[str, Any]) -> Any:
            # the run may have used the holdout; it does not say what it loaded
            return {"candidates": [{"config": {"model": space["model"][-1]}}]}

    c = controller(tmp_path, Silent())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert any(
        f.get("reason") == "eval_set_identity_unknown" for f in c.controller_facts()
    )
    assert c.status()["pending_deploys"] == []


def _holdout_digest_variants() -> list[Any]:
    from traigent.tuning.evidence import holdout_digest

    digest = holdout_digest(KIT["holdout"])
    return [
        digest,  # a bare string is not a collection of digests
        None,
        7,
        {"digest": digest},
        [digest, 3],  # a malformed entry
        ["not-a-digest"],
        [digest.upper()],
        [digest + "\n"],  # a trailing newline is not a digest
    ]


@pytest.mark.parametrize("index", range(8))
def test_a_malformed_loaded_digest_collection_is_an_unknown_identity(
    tmp_path: Path, index: int
) -> None:
    bad = _holdout_digest_variants()[index]

    class Reports(FakeOptimizer):
        def __call__(self, space: dict[str, Any]) -> Any:
            result = super().__call__(space)
            result["loaded_dataset_digests"] = bad
            return result

    c = controller(tmp_path, Reports())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert any(
        f.get("reason") == "eval_set_identity_unknown" for f in c.controller_facts()
    )
    assert c.status()["pending_deploys"] == []


def test_a_single_string_digest_equal_to_the_holdout_never_reaches_a_deploy(
    tmp_path: Path,
) -> None:
    from traigent.tuning.evidence import holdout_digest

    class Reports(FakeOptimizer):
        def __call__(self, space: dict[str, Any]) -> Any:
            result = super().__call__(space)
            result["loaded_dataset_digests"] = holdout_digest(KIT["holdout"])
            return result

    c = controller(tmp_path, Reports())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert c.status()["pending_deploys"] == []
    assert any(
        f.get("reason") == "eval_set_identity_unknown" for f in c.controller_facts()
    )


# -- local-trusted authority needs a recognised non-production store environment --------
@pytest.mark.parametrize("recorded", [None, "", "prod-ish", "Production"])
def test_local_trusted_authority_is_refused_for_an_unrecognised_store_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: Any
) -> None:
    from traigent.tuning.errors import TuningError as _Err

    monkeypatch.setenv("ENVIRONMENT", "development")
    c = controller(tmp_path, FakeOptimizer())
    assert c.require_authority() is c.authority  # a recorded development store
    c.meta["environment"] = recorded
    with pytest.raises(_Err) as err:
        c.require_authority()
    assert err.value.code == "E_LOCAL_AUTHORITY_IN_PRODUCTION"
    assert c.authority.authenticated("approver-1", {}, recorded) is False
    assert c._on_list_now("approver-1") is False


# -- the start is recorded at actual invocation, journaled, recoverable ----------------
def test_a_crash_between_the_start_marker_and_the_start_fact_is_abandonable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = controller_module._prepare_append

    def crashing_prepare(path: Path) -> Any:
        append = real(path)

        def crash(value: Any) -> None:
            if (value.get("record") or {}).get("fact") == "started":
                raise RuntimeError("crash after the marker, before the start fact")
            append(value)

        return crash

    monkeypatch.setattr(controller_module, "_prepare_append", crashing_prepare)
    with pytest.raises(RuntimeError):
        c.step(at(10))
    monkeypatch.undo()
    assert opt.calls == []
    restarted = controller(tmp_path, opt, now=at(0))
    with pytest.raises(TuningError):  # an interrupted start is not silently re-run
        restarted.step(at(10))
    restarted.step(at(10), abandon_interrupted=True)
    restarted.step(at(10))
    assert opt.calls == []  # cancelled, never invoked


# -- status lists only open attempts --------------------------------------------------
def test_status_does_not_list_a_closed_attempt_as_open(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    assert [a["attempt"] for a in c.status()["open_attempts"]] == ["A1"]
    c.step(_scope_end() + timedelta(hours=1))  # the relation's clock passes the scope
    states = [a["attempt"] for a in c.status()["open_attempts"]]
    assert "A1" not in states
