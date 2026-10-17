"""The best-config exercisers publish REAL specs and actually reach the fetch (round F1e).

The client now verifies the identity of a publish / fetch answer
(``BackendIntegratedClient._verify_published_best_config_identity``), so an exerciser that
sends a raw dict as the "spec" - and a fake Backend that returns it unchanged - fails
BEFORE any fetch, silently shrinking the canary coverage. These tests pin that the two
exercisers run clean, that the fetch really happens (1 request; 3 when the injected
faults force retries), and that every canary-bearing value still reaches the wire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import direct_paths
from .canaries import CANARIES
from .harness import CapturedRequest, EgressCapture
from .scenario import scenario_env


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str):
    monkeypatch.setattr(
        direct_paths, "EXERCISERS", {name: direct_paths.EXERCISERS[name]}
    )
    with scenario_env(tmp_path / "results", tmp_path / "datasets"):
        with EgressCapture() as cap:
            outcomes = direct_paths.run_exercisers(cap)
    return cap, outcomes


def _best_config_requests(cap: EgressCapture, method: str) -> list[CapturedRequest]:
    return [r for r in cap.requests if r.method == method and "/best-configs" in r.path]


@pytest.mark.parametrize(
    ("name", "publishes", "fetches"),
    [("best_config_sync", 1, 1), ("best_config_retry", 3, 3)],
)
def test_the_exerciser_runs_clean_and_reaches_the_fetch(
    monkeypatch, tmp_path, name, publishes, fetches
) -> None:
    cap, outcomes = _run(monkeypatch, tmp_path, name)
    assert outcomes == {name: None}, outcomes
    assert len(_best_config_requests(cap, "POST")) == publishes
    assert len(_best_config_requests(cap, "GET")) == fetches


@pytest.mark.parametrize("name", ["best_config_sync", "best_config_retry"])
def test_the_published_body_is_identified_and_every_canary_stays_where_it_is_pinned(
    monkeypatch, tmp_path, name
) -> None:
    """``config_id`` is new; the canaries keep the ratchet-pinned paths and values."""

    c = CANARIES
    cap, _ = _run(monkeypatch, tmp_path, name)
    post = _best_config_requests(cap, "POST")[-1]
    body = json.loads(post.body)
    spec = body["spec"]
    assert spec["config_id"] == direct_paths.BEST_CONFIG_ID
    assert spec[c["knob_name"]] == c["choice_value"]  # $.spec.<knob>
    assert spec["metric"] == c["metric_name"]  # $.spec.metric
    assert body["environment"] == c["agent_name"]  # $.environment
    get = _best_config_requests(cap, "GET")[-1]
    assert c["agent_name"] in get.query  # environment,function_ref
