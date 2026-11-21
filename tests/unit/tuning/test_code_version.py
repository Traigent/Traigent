"""The agent code version: how it is resolved, and that the gate measures and records it."""

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
from traigent.tuning.controller import config_fingerprint
from traigent.tuning.errors import TuningError

from .conftest import tvl_free
from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    NEW,
    OLD,
    _deploy_awaiting,
    _holdout,
    at,
    e_ref,
    head,
)


@tvl_free
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


@tvl_free
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
    tc.to_deploy_awaiting(c, e_ref("E1"), 10)
    assert (
        c.status()["pending_deploys"] == []
    )  # never promoted, never awaiting a deploy


def test_the_gate_records_the_code_version_it_measured_and_status_shows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODE_VERSION_ENV, "rev-abc123")
    c = _deploy_awaiting(tmp_path)
    (marker,) = (tmp_path / "store" / "attempts").glob("*.json")
    (gate,) = json.loads(marker.read_text())["gates"].values()
    assert gate["code_version"] == "rev-abc123"  # recorded at measurement
    view = c.status()["pending_deploys"][0]["release_evidence"]
    assert view["code_version"] == "rev-abc123"
    assert head(tmp_path)["generation"] == 0 and head(tmp_path)["config"] == OLD


def test_followed_head_traces_stamp_no_code_version_until_a_signed_head_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from traigent.observability.decorators import observe

    _deploy_awaiting(tmp_path)
    (tmp_path / "store" / "head.json").write_text(
        json.dumps(
            {
                "fingerprint": config_fingerprint(NEW),
                "config": NEW,
                "generation": 1,
                "code_version": "rev-abc123",
            }
        ),
        encoding="utf-8",
    )
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
    assert "head_code_version" not in metadata  # a head file is not a signed release
    assert metadata["head_state"] == "unreleased"
