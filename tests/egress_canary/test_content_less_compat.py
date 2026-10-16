"""Content-less compatibility matrix, SDK side (phase 3 group 1).

A content-less SDK against a Backend that does not advertise ``content_less_v1`` refuses
locally with ``backend_upgrade_required``: no plaintext fallback, no legacy-shape retry,
nothing more sent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .canaries import CANARIES
from .scenario import run_scenario

pytestmark = pytest.mark.backend_online

IDENTITY_KINDS = (
    "agent_name",
    "evaluation_set",
    "dataset_identity",
    "evaluator_id",
    "dataset_file_name",
)


def _creates(result):
    return [
        r
        for r in result.requests
        if r.method == "POST" and r.path.endswith("/sessions")
    ]


def test_backend_without_capability_gets_one_create_and_nothing_more(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    result = run_scenario(
        Path(tmp_path_factory.mktemp("cl-old-backend")),
        direct_paths=False,
        content_less=True,
        advertise_content_less=False,
    )
    assert result.optimization_error is not None
    assert "ContentLessRefusal" in result.optimization_error
    assert "upgrade" in result.optimization_error.lower()
    creates = _creates(result)
    assert len(creates) == 1, "no legacy-shape retry after backend_upgrade_required"
    later = [
        r
        for r in result.requests
        if any(v in r.path for v in ("next-trial", "results", "finalize"))
    ]
    assert later == [], "requests followed the local refusal"
    body = creates[0].body.decode("utf-8")
    assert '"content_less_v1"' in body
    for kind in IDENTITY_KINDS:
        assert CANARIES[kind] not in body


def test_capable_backend_run_sends_capability_environment_and_no_identity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    result = run_scenario(
        Path(tmp_path_factory.mktemp("cl-capable")),
        direct_paths=False,
        content_less=True,
    )
    assert result.optimization_error is None, result.optimization_error
    creates = _creates(result)
    assert len(creates) == 1
    import json

    body = json.loads(creates[0].body)
    assert body["capabilities"] == ["content_less_v1"]
    assert body["environment"] == "development"  # TRAIGENT_ENVIRONMENT of the scenario
    everything = b"\n".join(r.body or b"" for r in result.requests).decode(
        "utf-8", errors="replace"
    )
    for kind in IDENTITY_KINDS:
        assert CANARIES[kind] not in everything, kind
    # never the agents.id the fake Backend used to return
    assert "agent-canary" not in everything
