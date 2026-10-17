"""The best-config exercisers send an identified RAW input and reach the fetch (round F1e/F1f).

The client now verifies the identity of a publish / fetch answer
(``BackendIntegratedClient._verify_published_best_config_identity``), so an exerciser that
sends a raw dict as the "spec" - and a fake Backend that returns it unchanged - fails
BEFORE any fetch, silently shrinking the canary coverage. These tests pin that the two
registered exercisers run clean, that the fetch really happens (1 request; 3 when the
injected faults force retries), and that every canary-bearing value still reaches the
wire at its ratchet-pinned path.

What this module does NOT claim: the registered exercisers publish a raw-input dict
(``config_id`` plus the canary knob and metric at the top level), not a contract-valid
``traigent.best_config.v1`` spec, and the fake Backend binds a ``function_ref``-less
stored spec to the requested function (``tests/egress_canary/harness.py``). They are a
raw-input outbound-traffic ratchet. The canonical shape - ``spec.config``, a
``function_ref`` the stored spec itself carries, no harness repair - is exercised
separately, outside the ratchet registry, by the ``canonical`` tests below.
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


# --- canonical fixture: a contract-valid spec, no harness repair (round F1f) -----------

CANONICAL_CONFIG_ID = "best-canary-canonical"
CANONICAL_FUNCTION_REF = f"canary_pkg.agents:{CANARIES['agent_name']}"


def _canonical_spec() -> dict:
    """A contract-valid ``traigent.best_config.v1`` spec bound to its function.

    Unlike the raw-input fixture the knob lives under ``spec.config`` and ``function_ref``
    is part of the stored spec itself, so the fake Backend's "bind a function_ref-less
    spec to the requested function" repair never runs.
    """

    c = CANARIES
    return {
        "schema_version": "traigent.best_config.v1",
        "config_id": CANONICAL_CONFIG_ID,
        "function_ref": CANONICAL_FUNCTION_REF,
        "config": {c["knob_name"]: c["choice_value"]},
        "validation": {"metric": c["metric_name"]},
    }


def _run_canonical(tmp_path: Path):
    c = CANARIES
    spec = _canonical_spec()
    with scenario_env(tmp_path / "results", tmp_path / "datasets"):
        with EgressCapture() as cap:
            client = direct_paths._client()
            published = client.publish_best_config_sync(spec, environment="prod")
            fetched = client.fetch_best_config_sync(
                CANONICAL_CONFIG_ID,
                environment="prod",
                function_ref=CANONICAL_FUNCTION_REF,
            )
    assert c["agent_name"] in CANONICAL_FUNCTION_REF
    return cap, spec, published, fetched


def test_canonical_spec_is_published_and_fetched_without_harness_repair(
    tmp_path,
) -> None:
    cap, spec, published, fetched = _run_canonical(tmp_path)
    assert len(_best_config_requests(cap, "POST")) == 1
    assert len(_best_config_requests(cap, "GET")) == 1
    # The Backend stored function_ref itself: the harness's binding repair (it only
    # fires for a stored spec WITHOUT function_ref) had nothing to repair.
    stored = cap.backend.best_configs[CANONICAL_CONFIG_ID]
    assert stored["function_ref"] == CANONICAL_FUNCTION_REF
    assert published["spec"]["function_ref"] == CANONICAL_FUNCTION_REF
    assert fetched["spec"] == {**spec, "environment": "prod"}


def test_canonical_spec_carries_every_canary_under_spec_config_on_the_wire(
    tmp_path,
) -> None:
    c = CANARIES
    cap, _spec, _published, _fetched = _run_canonical(tmp_path)
    body = json.loads(_best_config_requests(cap, "POST")[-1].body)
    assert body["spec"]["config"] == {c["knob_name"]: c["choice_value"]}
    assert body["spec"]["validation"] == {"metric": c["metric_name"]}
    assert body["spec"]["function_ref"] == CANONICAL_FUNCTION_REF
    assert body["environment"] == "prod"
