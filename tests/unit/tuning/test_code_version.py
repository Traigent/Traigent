"""RFC 0012: an acceptance binds the agent code version (measurement == accept ==
serving); a code change between measurement and accept refuses the write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.tuning import Tuning
from traigent.tuning.code_version import (
    CODE_VERSION_ENV,
    E_NO_CODE_VERSION,
    resolve_code_version,
)
from traigent.tuning.errors import TuningError

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    NEW,
    OLD,
    _deploy_awaiting,
    _holdout,
    at,
    e_ref,
    head,
)


def test_the_acceptance_carries_the_code_version_the_gate_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-abc123")
    c = _deploy_awaiting(tmp_path)
    (marker,) = (tmp_path / "store" / "attempts").glob("*.json")
    (gate,) = json.loads(marker.read_text())["gates"].values()
    assert gate["code_version"] == "rev-abc123"  # recorded at measurement
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    assert record["agent_code_version"] == "rev-abc123"
    view = c.status()["pending_deploys"][0]["accept_release"]
    assert view["code_version"] == "rev-abc123"
    assert c.submit_acceptance(record, at(31)).result == "written"
    assert (
        head(tmp_path)["code_version"] == "rev-abc123"
    )  # the head says what it is for
    assert head(tmp_path)["config"] == NEW


def test_code_changed_between_measurement_and_accept_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-measured")
    c = _deploy_awaiting(tmp_path)
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-edited-after")  # the code moved on
    outcome = c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert (outcome.result, outcome.reason) == ("refused", "code_version_mismatch")
    assert head(tmp_path)["config"] == OLD  # head unchanged
    assert head(tmp_path)["generation"] == 0
    lines = tmp_path / "store" / "commits.jsonl"
    assert not lines.exists() or "commit_intent" not in lines.read_text()
    # the relation finished the occurrence as deploy_refused(code_version_mismatch)
    facts = c._derive()["facts"]
    assert any(
        f["fact"] == "outcome"
        and f["event"] == "deploy_refused"
        and f["reason"] == "code_version_mismatch"
        for f in facts
    )
    # the journal and history agree after a reload
    assert c.status()["head"]["generation"] == 0


def test_a_record_for_other_code_than_the_gate_measured_does_not_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-measured")
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    record["agent_code_version"] = "rev-other"
    outcome = c.submit_acceptance(record, at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "binding_mismatch")
    assert head(tmp_path)["generation"] == 0


def test_the_env_override_wins_over_the_explicit_value_and_the_source_digest() -> None:
    import os

    def fn() -> None: ...

    old = os.environ.pop(CODE_VERSION_ENV, None)
    try:
        assert resolve_code_version("explicit-1", fn) == "explicit-1"
        derived = resolve_code_version(None, fn)
        assert derived.startswith("src:") and len(derived) == 4 + 64
        os.environ[CODE_VERSION_ENV] = "from-env"
        assert resolve_code_version("explicit-1", fn) == "from-env"
    finally:
        os.environ.pop(CODE_VERSION_ENV, None)
        if old is not None:
            os.environ[CODE_VERSION_ENV] = old


def test_no_code_version_is_refused_with_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    with pytest.raises(TuningError) as err:
        resolve_code_version(None, None)  # nothing to digest, nothing given
    assert err.value.code == E_NO_CODE_VERSION
    with pytest.raises(TuningError):
        Tuning(code_version="has spaces and é")


def test_without_a_code_version_no_candidate_has_promotion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    from tests.unit.tuning import test_controller as tc

    c = tc.controller(tmp_path, tc.FakeOptimizer())
    c.state.source = None  # nothing to digest
    c.record_event(tc.event("E1", NEW["model"], 9), at(10))
    tc.to_deploy_awaiting(c, e_ref("E1"), 10)
    assert (
        c.status()["pending_deploys"] == []
    )  # never promoted, never awaiting a deploy


def test_followed_head_traces_carry_the_code_version_the_head_was_accepted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from traigent.observability.decorators import observe

    monkeypatch.setenv(CODE_VERSION_ENV, "rev-abc123")
    c = _deploy_awaiting(tmp_path)
    assert c.accept(e_ref("E1"), "approver-1", 0, now=at(31)).result == "written"
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-newer")  # serving is NOT blocked
    client = MagicMock()

    @observe(
        "classify",
        client=client,
        agent="agent-1",
        follow="head",
        tuning_store=tmp_path / "store",
    )
    def classify() -> str:
        return "ok"

    assert classify() == "ok"
    metadata = client.start_trace.call_args.kwargs["metadata"]
    assert metadata["head_code_version"] == "rev-abc123"
