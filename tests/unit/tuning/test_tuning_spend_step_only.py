"""The local controller runs the spend step only.

Release acceptance is planned, not built: every release entry point
refuses before touching the store.  The spend step's final gate scans no journal after
its clock sample, cancels an approved attempt it refuses (so it never holds a later
schedule), and hands the final check only to the adapter the SDK built.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.controller import Controller, traigent_optimize_fn
from traigent.tuning.errors import (
    E_UNSUPPORTED_IN_RELEASE,
    RELEASE_REFUSAL,
    TuningError,
)

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    FakeOptimizer,
    _cli_fixture,
    _deploy_awaiting,
    _holdout,
    at,
    controller,
    e_ref,
)
from .test_tuning_effect_time import _clock, _scope_end
from .test_tuning_gated_effect import _FakeSdkFunction


def _platform(hook: Any) -> ApproverAuthority:
    return ApproverAuthority(frozenset(APPROVERS), mode="platform", authenticator=hook)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` (relative path -> bytes); empty when it is missing."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# -- release is refused before any I/O ---------------------------------------------------
@pytest.mark.parametrize("entry", ["build_acceptance", "submit_acceptance", "accept"])
def test_every_python_release_entry_refuses_and_writes_nothing(
    tmp_path: Path, entry: str
) -> None:
    c = _deploy_awaiting(tmp_path)
    before = _tree(tmp_path / "store")
    calls = {
        "build_acceptance": lambda: c.build_acceptance(e_ref("E1"), "approver-1", 0),
        "submit_acceptance": lambda: c.submit_acceptance({}, at(31)),
        "accept": lambda: c.accept(e_ref("E1"), "approver-1", 0, now=at(31)),
    }
    with pytest.raises(TuningError) as err:
        calls[entry]()
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE
    assert err.value.detail == RELEASE_REFUSAL
    assert _tree(tmp_path / "store") == before


def test_a_release_entry_refuses_even_without_an_authority_or_a_validated_policy(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer(), authorized_approvers=None)
    c.state.validated = False
    for call in (
        lambda: c.accept(e_ref("E1"), "approver-1", 0),
        lambda: c.submit_acceptance({}),
    ):
        with pytest.raises(TuningError) as err:
            call()
        assert err.value.code == E_UNSUPPORTED_IN_RELEASE


def test_the_cli_accept_command_refuses_and_creates_no_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth, common = _cli_fixture(tmp_path, monkeypatch)
    store = tmp_path / "s"
    assert not store.exists()
    for extra in (["--authority", str(auth)], []):
        result = CliRunner().invoke(
            cli,
            ["tune", "accept", e_ref("E1"), "--generation", "0", "--approver", "a"]
            + common
            + extra,
        )
        assert result.exit_code != 0
        assert E_UNSUPPORTED_IN_RELEASE in result.output
    assert not store.exists()  # nothing was read, nothing was created


def test_the_cli_accept_command_refuses_first_whatever_its_options_name(
    tmp_path: Path,
) -> None:
    store = tmp_path / "s"
    for args in (
        [],
        ["--authority", "/missing/authority.yml"],
        ["--tvl", "/missing/p.tvl.yml", "--store", str(store), "--generation", "x"],
        ["D1", "--generation", "not-a-number", "--reject", "--unknown-flag"],
    ):
        refusal = CliRunner().invoke(cli, ["tune", "event", "/x"])
        result = CliRunner().invoke(cli, ["tune", "accept", *args])
        assert result.exit_code == refusal.exit_code != 0
        assert E_UNSUPPORTED_IN_RELEASE in result.output
        assert "does not exist" not in result.output
    assert not store.exists()


def test_the_cli_accept_command_leaves_an_existing_store_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    store = tmp_path / "store"
    before = _tree(store)
    result = CliRunner().invoke(
        cli,
        [
            "tune",
            "accept",
            e_ref("E1"),
            "--generation",
            "0",
            "--approver",
            "approver-1",
            "--store",
            str(store),
            "--agent",
            "agent-1",
        ],
    )
    assert result.exit_code != 0 and E_UNSUPPORTED_IN_RELEASE in result.output
    assert _tree(store) == before
    assert c.status()["head"]["generation"] == 0


def test_the_controller_never_moves_the_head_or_writes_head_journals(
    tmp_path: Path,
) -> None:
    c = _deploy_awaiting(tmp_path)
    head_bytes = (tmp_path / "store" / "head.json").read_bytes()
    c.step(at(40))
    c.step(at(50))
    assert (tmp_path / "store" / "head.json").read_bytes() == head_bytes
    names = {p.name for p in (tmp_path / "store").iterdir()}
    assert "heads.jsonl" not in names and "commits.jsonl" not in names


# -- the final gate scans no journal after its clock sample -------------------------------
class _GateSpy:
    """Fails when a journal is read or repaired between the final sample and the run."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, on_final_auth: Any = None):
        self.authenticated = False
        self.sampled = False
        self.scans_after_sample: list[str] = []
        real_gated = Controller._gated_effect
        real_scan = controller_module._scan_lines
        real_repair = controller_module._repair_tail
        spy = self

        def gated(ctl: Controller, *, clock: Any, authenticate: Any, **kw: Any) -> Any:
            def sampled_clock() -> Any:
                value = clock()
                if spy.authenticated:
                    spy.sampled = True
                return value

            def final_authenticate() -> bool:
                if on_final_auth is not None:
                    on_final_auth()
                answer = authenticate()
                spy.authenticated = True
                return bool(answer)

            try:
                return real_gated(
                    ctl,
                    clock=sampled_clock,
                    authenticate=final_authenticate,
                    **kw,
                )
            finally:
                spy.authenticated = spy.sampled = False

        def scan(path: Path) -> Any:
            if spy.sampled:
                spy.scans_after_sample.append(Path(path).name)
            return real_scan(path)

        def repair(path: Path) -> None:
            if spy.sampled:
                spy.scans_after_sample.append(Path(path).name)
            real_repair(path)

        monkeypatch.setattr(Controller, "_gated_effect", gated)
        monkeypatch.setattr(controller_module, "_scan_lines", scan)
        monkeypatch.setattr(controller_module, "_repair_tail", repair)


class _CountingOptimizer(FakeOptimizer):
    """Clears the spy the moment the optimizer is invoked (the gate is over)."""

    spy: _GateSpy | None = None

    def __call__(self, space: dict[str, Any]) -> Any:
        if self.spy is not None:
            self.spy.sampled = False
        return super().__call__(space)


def test_no_journal_is_read_or_repaired_after_the_final_clock_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _GateSpy(monkeypatch)
    opt = _CountingOptimizer()
    opt.spy = spy
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    report = c.step(at(10))
    assert len(opt.calls) == 1 and report.started
    assert spy.scans_after_sample == []
    started = [
        ln
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "started"
    ]
    assert len(started) == 1  # the start fact still landed, once


def test_a_revocation_during_the_tail_repair_is_seen_by_the_final_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revoked = [False]
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, authority=_platform(lambda who, ctx: not revoked[0]))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = controller_module._prepare_append

    def prepare(path: Path) -> Any:
        appender = real(path)  # the tail repair happens here
        approved = any(
            ln["category"] == "optimizations" and ln["record"].get("fact") == "approval"
            for ln in controller_module._read_lines(path)
        )
        if approved:  # the preparation of the spend, not the approval's own append
            revoked[0] = True
        return appender

    monkeypatch.setattr(controller_module, "_prepare_append", prepare)
    report = c.step(at(10))
    assert opt.calls == []
    assert report.refused and report.refused[0].endswith(":approver_not_authorized")


# -- an approved attempt the final gate refuses is cancelled -------------------------------
def _later_schedule_is_not_held(c: Controller, now: Any) -> None:
    c.step(now)
    facts = c._derive()["facts"]
    held = {f["occurrence"]["rule"] for f in facts if f["fact"] == "held"}
    opened = {f["occurrence"]["rule"] for f in facts if f["fact"] == "attempt_opened"}
    assert "on_E6" not in held, held  # E6 waits for approval; E7 waits behind E6
    assert "on_E6" in opened


def test_an_approver_revoked_at_the_final_gate_cancels_the_attempt_and_frees_later_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revoked = [False]
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, authority=_platform(lambda who, ctx: not revoked[0]))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    _GateSpy(monkeypatch, on_final_auth=lambda: revoked.__setitem__(0, True))
    report = c.step(at(10))
    assert opt.calls == []
    assert report.refused[0].endswith(":approver_not_authorized")
    completions = [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]
    assert completions == ["cancelled"]
    revoked[0] = False
    _later_schedule_is_not_held(c, at(41))
    assert opt.calls == []


def test_a_scope_that_expires_at_the_final_gate_cancels_the_approved_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(10))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))

    def expire() -> None:
        clock[0] = _scope_end() + timedelta(seconds=1)

    _GateSpy(monkeypatch, on_final_auth=expire)
    report = c.step()
    assert opt.calls == []
    assert report.refused[0].endswith(":scope_expired")
    assert [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ] == ["cancelled"]
    clock[0] = _scope_end() + timedelta(days=2)
    c.step()
    facts = c._derive()["facts"]
    (first,) = [o for o in c._open_attempts(facts) if o["attempt"] == "A1"]
    # the relation sees the approved attempt as ended (cancelled, or closed with its scope)
    assert first["finished"] or first["closed"]
    assert opt.calls == []


def test_a_persisted_instant_ahead_of_the_clock_cancels_the_approved_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    _GateSpy(monkeypatch, on_final_auth=lambda: clock.__setitem__(0, at(39)))
    report = c.step()
    assert opt.calls == []
    assert report.refused[0].endswith(":persisted_time_ahead")
    assert [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ] == ["cancelled"]


# -- only the adapter the SDK built receives the final check -----------------------------
class _ThirdParty:
    """An injected optimizer that advertises the candidate-directory protocol."""

    accepts_candidate_dir = True

    def __init__(self) -> None:
        self.calls = 0
        self.received_final_check: list[Any] = []

    def __call__(
        self,
        space: dict[str, Any],
        *,
        candidate_dir: Path | None = None,
        admitted_models: Any = (),
        final_check: Any = None,
    ) -> Any:
        self.calls += 1
        self.received_final_check.append(final_check)
        return {"loaded_dataset_digests": [], "candidates": []}


def test_a_third_party_optimizer_advertising_the_candidate_dir_is_gated_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revoked = [False]
    third = _ThirdParty()
    c = controller(
        tmp_path, third, authority=_platform(lambda who, ctx: not revoked[0])
    )
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    _GateSpy(monkeypatch, on_final_auth=lambda: revoked.__setitem__(0, True))
    report = c.step(at(10))
    assert third.calls == 0  # never invoked: the gate ran first and refused
    assert report.refused[0].endswith(":approver_not_authorized")


def test_a_third_party_optimizer_is_invoked_once_without_the_final_check(
    tmp_path: Path,
) -> None:
    third = _ThirdParty()
    c = controller(tmp_path, third)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert third.calls == 1 and third.received_final_check == [None]


def test_an_optimizer_that_copies_the_sdk_adapters_attributes_is_not_the_sdk_adapter(
    tmp_path: Path,
) -> None:
    adapter = traigent_optimize_fn(_FakeSdkFunction([]))
    third = _ThirdParty()
    third.__dict__.update(adapter.__dict__)  # accepts_candidate_dir, spend_limit
    assert not controller_module._is_sdk_adapter(third)
    assert controller_module._is_sdk_adapter(adapter)
    c = controller(tmp_path, third)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert third.calls == 1 and third.received_final_check == [None]


def test_the_sdk_adapter_runs_the_final_check_inside_its_artifact_lock(
    tmp_path: Path,
) -> None:
    started: list[int] = []
    c = controller(tmp_path, traigent_optimize_fn(_FakeSdkFunction(started)))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    report = c.step(at(10))
    assert started == [1] and report.started
    assert (
        json.loads(next((tmp_path / "store" / "attempts").glob("*.json")).read_text())[
            "finished_record"
        ]["fact"]
        == "finished"
    )


# -- a grant followed by an expired scope cancels, never holds ------------------------------
def test_a_scope_that_expires_right_after_a_grant_cancels_the_attempt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(10))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real_append = c._append

    def append_then_expire(category: str, record: Any, instant: Any) -> Any:
        out = real_append(category, record, instant)
        if isinstance(record, dict) and record.get("fact") == "approval":
            clock[0] = _scope_end() + timedelta(seconds=1)  # lapses right after
        return out

    monkeypatch.setattr(c, "_append", append_then_expire)

    def finishes() -> list[str]:
        return [
            ln["record"]["completion"]
            for ln in c._lines()
            if ln["category"] == "optimizations"
            and ln["record"].get("fact") == "finished"
        ]

    c.step()
    assert finishes() == ["cancelled"]  # already by the first step
    c.step()
    assert opt.calls == []
    completions = [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]
    assert completions == ["cancelled"]
    clock[0] = at(41)
    c.step(at(41))
    (first,) = [
        o for o in c._open_attempts(c._derive()["facts"]) if o["attempt"] == "A1"
    ]
    assert first["closed"]  # the relation sees the granted attempt as ended
    assert [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ] == ["cancelled"]  # still recorded exactly once
    assert opt.calls == []


# -- the SDK adapter is recognised by identity, not equality ----------------------------------
def test_an_optimizer_equal_and_hash_alike_to_the_sdk_adapter_is_not_the_sdk_adapter(
    tmp_path: Path,
) -> None:
    adapter = traigent_optimize_fn(_FakeSdkFunction([]))

    class _Lookalike(_ThirdParty):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(adapter)

    third = _Lookalike()
    assert not controller_module._is_sdk_adapter(third)
    assert controller_module._is_sdk_adapter(adapter)
    c = controller(tmp_path, third)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert third.calls == 1 and third.received_final_check == [None]


# -- a head ahead of the history is refused before any spend -----------------------------------
def test_a_head_ahead_of_the_history_raises_before_any_spend_or_journal_write(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    (tmp_path / "store" / "head.json").write_text(
        json.dumps({"generation": 1}), encoding="utf-8"
    )
    before = _tree(tmp_path / "store")
    with pytest.raises(TuningError) as err:
        c.step(at(10))
    assert err.value.code == "E_HEAD_DIVERGED"
    assert opt.calls == []
    assert _tree(tmp_path / "store") == before


def test_a_grant_that_survived_a_crash_before_its_scope_closed_is_cancelled_on_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(10))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real_append = c._append

    def append_then_expire(category: str, record: Any, instant: Any) -> Any:
        out = real_append(category, record, instant)
        if isinstance(record, dict) and record.get("fact") == "approval":
            clock[0] = _scope_end() + timedelta(seconds=1)
        return out

    def crash(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("process stopped")

    monkeypatch.setattr(c, "_append", append_then_expire)
    with monkeypatch.context() as m:
        m.setattr(c, "_cancel_refused", crash)
        with pytest.raises(RuntimeError):
            c.step()
    clock[0] = _scope_end() + timedelta(days=2)  # the relation now closes the scope
    c.step()
    assert opt.calls == []
    assert [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ] == ["cancelled"]
