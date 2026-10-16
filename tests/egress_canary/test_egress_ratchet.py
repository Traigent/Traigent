"""Ratchet on content leaking over SDK optimization traffic (Phase 1: measure).

Leak identities ``(route, location, field_path, canary kind)`` are pinned in
``leak_manifest.json``.  An identity not in the register fails (additions are
never allowed); an identity that disappears fails until it is removed from the
register and given a disposition (reason + reference) in
``leak_dispositions.json``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .canaries import distinct_identities, distinct_pairs
from .pins import (
    BASELINE,
    PRE_EXPANSION_PAIRS,
    REGISTER,
    load_anchor,
    load_dispositions,
    ratchet_problems,
    reference_problem,
)
from .scenario import ScenarioResult, run_scenario

# The run is mocked at every transport; the marker only selects connected mode.
pytestmark = pytest.mark.backend_online

REQUIRED_ROUTES = (
    "POST /api/v1/sessions",
    "POST /api/v1/sessions/{id}/next-trial",
    "POST /api/v1/sessions/{id}/results",
    "POST /api/v1/sessions/{id}/finalize",
)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> ScenarioResult:
    # Phase 3 group 1: the main optimization runs content-less (opt-in). The direct-path
    # exercisers and the offline replay keep the legacy shape and still pin their leaks.
    return run_scenario(Path(tmp_path_factory.mktemp("egress-canary")), content_less=True)


@pytest.fixture(scope="module")
def legacy_scenario(tmp_path_factory: pytest.TempPathFactory) -> ScenarioResult:
    """Default (opt-in off) main optimization: the legacy request shape."""
    return run_scenario(
        Path(tmp_path_factory.mktemp("egress-canary-legacy")), direct_paths=False
    )


#: Canary kinds that are group-1 identity on POST /api/v1/sessions (CHECKLIST S1 rows).
G1_IDENTITY_KINDS = frozenset(
    {
        "agent_name",
        "evaluation_set",
        "dataset_identity",
        "evaluator_id",
        "function_name",
        "dataset_file_name",
    }
)


def _fmt(ids: set[tuple[str, str, str, str]]) -> str:
    return "\n".join(
        f"  {route}  {location} {field_path!r} [{kind}]"
        for route, location, field_path, kind in sorted(ids)
    )


def test_canary_run_exercised_the_optimization_routes(scenario: ScenarioResult) -> None:
    """A ratchet over an empty capture would pass vacuously; prove coverage."""

    assert scenario.optimization_error is None, scenario.optimization_error
    assert scenario.result_source == "cloud_brain"
    missing = [r for r in REQUIRED_ROUTES if r not in scenario.routes]
    assert not missing, f"routes not captured: {missing}; got {scenario.routes}"
    assert scenario.guard_violations == []
    failed = {k: v for k, v in scenario.exerciser_outcomes.items() if v is not None}
    assert not failed, f"direct-path exercisers failed (coverage shrank): {failed}"
    assert len(scenario.exerciser_outcomes) >= 13
    assert "sync_manager_replay" in scenario.exerciser_outcomes
    assert scenario.unhandled_routes == [], scenario.unhandled_routes

    assert scenario.offline_error is None, scenario.offline_error
    assert scenario.offline_optimize_requests == 0, "offline optimize() made requests"
    assert scenario.offline_synced_sessions >= 2


def test_failure_paths_ran_and_every_retry_body_was_captured(
    scenario: ScenarioResult,
) -> None:
    """Injected 408/429/5xx/connection errors fired and the resends hit the wire.

    Every captured request (original and retry) goes through ``find_leaks``, so
    asserting the resend was captured is asserting it was scanned.
    """

    assert scenario.faults, "no faults were injected"
    kinds = {o for f in scenario.faults for o in f.outcomes}
    assert {408, 429, 500, 503, "connection"} <= kinds, kinds
    for fault in scenario.faults:
        label = f"{fault.owner}: {fault.method} {fault.path_regex}"
        assert fault.consumed == len(fault.outcomes), f"fault not exhausted: {label}"
        if fault.expects_retry:
            assert fault.matches > len(fault.outcomes), (
                f"no resend captured after the failure: {label}"
            )
    retried_owners = {f.owner for f in scenario.faults if f.expects_retry}
    assert {
        "best_config_retry",
        "aiohttp_retry_optimize",
        "optimizer_batch_retry",
    } <= retried_owners
    import re

    for fault in scenario.faults:
        if not fault.expects_retry:
            continue
        bodies = [
            r.body
            for r in scenario.requests
            if r.method == fault.method and re.search(fault.path_regex, r.path)
        ]
        if fault.method != "GET":
            # every attempt, original and resends, carries a body that was scanned
            assert sum(1 for b in bodies if b) > len(fault.outcomes), fault.owner
    transports = {r.transport for r in scenario.requests}
    assert {"aiohttp", "requests"} <= transports


def test_no_leak_identity_outside_the_register(scenario: ScenarioResult) -> None:
    """Additions never pass, and a leak moving field/location is an addition."""

    new = distinct_identities(scenario.leaks) - REGISTER
    assert not new, f"new leak identities outside the manifest:\n{_fmt(new)}"


def test_removed_leaks_need_a_disposition(scenario: ScenarioResult) -> None:
    gone = REGISTER - distinct_identities(scenario.leaks)
    assert not gone, (
        "leaks no longer found: remove them from `register` in leak_manifest.json "
        "and add a disposition (reason + reference) to leak_dispositions.json.\n"
        f"{_fmt(gone)}"
    )


def test_manifest_and_dispositions_are_consistent() -> None:
    assert REGISTER <= BASELINE, "register may only shrink"
    dispositions = load_dispositions()
    disposed = set()
    for entry in dispositions:
        assert entry.get("reason", "").strip(), f"disposition without reason: {entry}"
        assert entry.get("reference", "").strip(), (
            f"disposition without reference: {entry}"
        )
        disposed.add(
            (entry["route"], entry["location"], entry["field_path"], entry["kind"])
        )
    assert disposed == BASELINE - REGISTER, (
        "every identity removed from the register needs exactly one disposition; "
        f"missing: {sorted((BASELINE - REGISTER) - disposed)}, "
        f"stale: {sorted(disposed - (BASELINE - REGISTER))}"
    )


def test_original_43_pairs_are_still_pinned() -> None:
    """The pre-expansion baseline is a subset of the manifest (nothing hidden)."""

    pairs = {(route, kind) for route, _loc, _fp, kind in REGISTER}
    assert len(PRE_EXPANSION_PAIRS) == 43
    missing = set(PRE_EXPANSION_PAIRS) - pairs
    disposed = {(d["route"], d["kind"]) for d in load_dispositions()}
    assert missing <= disposed, f"pairs vanished without disposition: {sorted(missing)}"


def test_new_pairs_vs_pre_expansion_baseline_are_known(
    scenario: ScenarioResult,
) -> None:
    new_pairs = distinct_pairs(scenario.leaks) - set(PRE_EXPANSION_PAIRS)
    assert new_pairs <= {(route, kind) for route, _l, _f, kind in REGISTER}


def test_live_leaks_hold_against_origin_develop_anchor(
    scenario: ScenarioResult,
) -> None:
    """The authority is origin/develop, not the manifest this PR may have edited.

    Fails (never skips) when git or the ref is unreadable.
    """

    problems = ratchet_problems(
        distinct_identities(scenario.leaks),
        load_anchor(),
        REGISTER,
        load_dispositions(),
    )
    assert not problems, "\n".join(problems)


def test_disposition_references_are_validated() -> None:
    for entry in load_dispositions():
        identity = (
            entry["route"],
            entry["location"],
            entry["field_path"],
            entry["kind"],
        )
        assert (
            reference_problem(entry.get("reference", ""), identity=identity) is None
        ), entry


def test_content_less_create_carries_no_group1_identity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A content-less run sends no agent/dataset/evaluation-set/evaluator identity, anywhere."""

    result = run_scenario(
        Path(tmp_path_factory.mktemp("egress-canary-cl")),
        direct_paths=False,
        content_less=True,
    )
    assert result.optimization_error is None, result.optimization_error
    g1 = sorted(
        leak.identity for leak in result.leaks if leak.kind in G1_IDENTITY_KINDS
    )
    assert not g1, f"group-1 identity on the wire in content-less mode:\n{_fmt(set(g1))}"
    # later groups stay explicitly open: still recorded in the register, never reclassified
    open_ids = distinct_identities(result.leaks)
    assert open_ids, "later-group leaks vanished from the capture (vacuous run?)"
    assert open_ids <= REGISTER, f"new identities:\n{_fmt(open_ids - REGISTER)}"
    assert {kind for _r, _l, _f, kind in open_ids} >= {"knob_name", "metric_name"}


def test_legacy_default_run_leaks_stay_inside_the_baseline(
    legacy_scenario: ScenarioResult,
) -> None:
    """Opt-in off keeps the legacy request: its leaks are all known (baseline) identities."""

    assert legacy_scenario.optimization_error is None
    unknown = distinct_identities(legacy_scenario.leaks) - BASELINE
    assert not unknown, f"legacy shape grew leaks:\n{_fmt(unknown)}"
    # and it still carries the G1 fields the content-less path drops
    kinds = {leak.kind for leak in legacy_scenario.leaks}
    assert {"agent_name", "evaluation_set", "evaluator_id"} <= kinds
