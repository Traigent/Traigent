"""Delta-review items: code source, effect-time checks, environment, loaded-dataset
identity, directory durability, ingestion close, draining attempts, path components."""

from __future__ import annotations

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

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    KIT,
    NEW,
    FakeOptimizer,
    _deploy_awaiting,
    _holdout,
    _standing_state,
    at,
    controller,
    e_ref,
    event,
    head,
    make,
    state,
)


def _clock(monkeypatch: pytest.MonkeyPatch, start: Any) -> list[Any]:
    clock = [start]
    real_utc = controller_module._utc
    monkeypatch.setattr(
        controller_module,
        "_utc",
        lambda value: real_utc(value) if value is not None else clock[0],
    )
    return clock


# -- D1. the resolved callable itself needs readable source -----------------------------
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
    assert resolve_code_version("explicit-1", namespace["one"]) == "explicit-1"


# -- D2. effect-time checks come AFTER authentication and replay ------------------------
def test_a_grant_that_expires_during_authentication_does_not_start_the_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    calls = [0]

    def slow_hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] > 1:  # the use-time re-check at the spend is slow
            clock[0] = clock[0] + timedelta(hours=3)
        return True

    c = make(
        _standing_state(),
        tmp_path / "store",
        opt,
        now=at(0),
        authority=ApproverAuthority(
            frozenset(APPROVERS), mode="platform", authenticator=slow_hook
        ),
    )
    c.record_event(event("E1", NEW["model"], 39), at(40))
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(40) + timedelta(hours=2), "issuer-1", at(40)
    )
    calls[0] = 0
    c.step(ingestion_closed=True)
    assert opt.calls == []


def test_a_release_whose_second_authentication_crosses_expiry_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    clock = _clock(monkeypatch, at(32))
    calls = [0]

    def hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] == 2:  # the check made at the physical head write
            clock[0] = clock[0] + timedelta(days=30)
        return True

    c.authority = ApproverAuthority(
        frozenset(APPROVERS), mode="platform", authenticator=hook
    )
    outcome = c.submit_acceptance(record)
    assert outcome.result != "written"
    assert head(tmp_path)["generation"] == 0


# -- D3. the environment travels with the attached tuning state -------------------------
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


# -- D4. identity of the dataset the run ACTUALLY loaded --------------------------------
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
            self.loaded_dataset_digests = {examples_digest(holdout_rows)}
            return super().__call__(space)

    assert _dataset_rows is not None
    c = controller(tmp_path, RunsOnHoldout())
    c.state.optimized = Optimized()
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert any(
        f.get("reason") == "holdout_is_optimization_eval_set"
        for f in c.controller_facts()
    )


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
    run({"model": ["anthropic/claude-haiku-4-5"]}, candidate_dir=tmp_path / "c")
    assert examples_digest(rows) in run.loaded_dataset_digests  # type: ignore[attr-defined]


# -- D5. a newly created directory entry is durable before the optimizer starts ---------
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
    controller_module._write_json(tmp_path / "attempts" / "m.json", {"a": 1})
    assert str(tmp_path / "attempts") in synced  # the rename inside it
    assert str(tmp_path) in synced  # the new "attempts" entry itself


# -- D8. no spend on the second events are still arriving in ----------------------------
def test_no_spend_in_the_second_events_were_ingested_until_it_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(10))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step()  # live clock, still inside the event's second
    assert opt.calls == []
    clock[0] = at(10) + timedelta(seconds=1)  # the second has closed
    c.step()
    assert len(opt.calls) == 1


# -- D9. a closed (draining) attempt still records its saved finish ---------------------
def test_a_closed_attempt_with_a_saved_finish_marker_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
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


# -- D12. reserved path components -----------------------------------------------------
@pytest.mark.parametrize("bad", [".", "..", ""])
def test_dot_components_are_refused_in_the_default_store_path(bad: str) -> None:
    from traigent.tuning.controller import default_store_dir

    with pytest.raises(TuningError):
        default_store_dir(bad, "production")
    if bad:
        with pytest.raises(TuningError):
            default_store_dir("agent-1", bad)


# -- D9 (relation input). the actual start of an optimization is recorded ---------------
def test_the_start_of_an_optimization_is_recorded_once_before_it_runs(
    tmp_path: Path,
) -> None:
    c = _standing_controller(tmp_path)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(days=1), "issuer-1", at(10)
    )
    c.step(at(10))
    started = [
        ln
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "started"
    ]
    assert len(started) == 1
    c.step(at(11))
    assert len([ln for ln in c._lines() if ln["record"].get("fact") == "started"]) == 1


def _standing_controller(tmp_path: Path) -> Controller:
    return make(_standing_state(), tmp_path / "store", FakeOptimizer())
