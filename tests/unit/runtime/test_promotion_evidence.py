"""Promotion-gate evidence: fingerprint, TVL join, gate digest, closed candidate facts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "tvl.promotion_join",
    reason="tvl-spec (the runtime-tvl extra, Python >= 3.12) is not installed",
)

from tvl.promotion_join import join_promotion_decisions  # noqa: E402

from traigent.runtime.model import (  # noqa: E402
    PromotionMonitor,
    RuntimeRequirement,
    build_rules,
)
from traigent.runtime.promotion_evidence import (  # noqa: E402
    CandidateEvidence,
    candidate_extractor,
    candidate_fingerprint,
    evaluate_candidate,
    latency_counts,
)
from traigent.runtime.respond import respond_hosted  # noqa: E402

from .conftest import CANARY, Reactive  # noqa: E402
from .test_respond_hosted import Fn  # noqa: E402

CD = "a" * 64
CONFIG = {"temperature": 0.2, "model": "m-small"}


@pytest.fixture
def promo(reactive: Reactive) -> Reactive:
    reactive.requirement = RuntimeRequirement(
        name="answer_latency",
        latency_threshold_ms=2000,
        environment="production",
        max_violation_rate=0.05,
        on_breach="fix",
        max_fix_cost_usd=5.0,
        agent_id="agent-1",
        rules=build_rules("fix"),
        promotion=PromotionMonitor("answer_promo", 0.95, 0.05),
    )
    from traigent.runtime import manifest

    manifest.ensure_tokens(reactive.state, [("monitor", "answer_promo")])
    return reactive


def candidate(
    promo: Reactive,
    legacy: str = "Promote",
    counts: Any = "default",
    config: Any = CONFIG,
) -> dict[str, Any]:
    if counts == "default":
        counts = {"answer_promo": {"n": 400, "k": 2}}
    return evaluate_candidate(
        promo.requirement,
        CandidateEvidence(config, legacy, counts),
        contract_digest=CD,
        tokens=promo.tokens,
    )


def test_fingerprint_is_the_canonical_config_digest() -> None:
    a = candidate_fingerprint({"b": 1, "a": [1, 2.5], "c": {"y": 1, "x": None}})
    b = candidate_fingerprint({"c": {"x": None, "y": 1}, "a": [1, 2.5], "b": 1})
    assert a == b and len(a) == 64
    assert a != candidate_fingerprint({"b": 2, "a": [1, 2.5], "c": {"y": 1, "x": None}})
    for bad in ({}, {"x": float("nan")}, {"x": object()}):
        with pytest.raises(Exception):  # noqa: B017 - closed RuntimeRequirementError
            candidate_fingerprint(bad)


def test_promotable_candidate_carries_the_join_inputs_and_digests(
    promo: Reactive,
) -> None:
    fact = candidate(promo)
    assert fact["evidence_valid"] is True and fact["legacy_decision"] == "Promote"
    assert fact["monitor_verdicts"] == ["MET"]
    assert fact["fingerprint"] == candidate_fingerprint(CONFIG)
    assert (
        join_promotion_decisions(fact["legacy_decision"], fact["monitor_verdicts"])
        == "Promote"
    )
    assert set(fact) == {
        "evidence_valid",
        "legacy_decision",
        "monitor_verdicts",
        "fingerprint",
        "gate_result_digest",
    }
    assert candidate(promo) == fact  # deterministic


@pytest.mark.parametrize(
    ("legacy", "counts", "verdicts", "joined"),
    [
        ("Promote", {"answer_promo": {"n": 400, "k": 60}}, ["BREACHED"], "Reject"),
        ("Reject", {"answer_promo": {"n": 400, "k": 2}}, ["MET"], "Reject"),
        ("NoDecision", {"answer_promo": {"n": 400, "k": 2}}, ["MET"], "NoDecision"),
        (
            "Promote",
            {"answer_promo": {"n": 400, "k": 16}},
            ["UNKNOWN(inconclusive)"],
            "NoDecision",
        ),
        (
            "Promote",
            {},
            ["UNKNOWN(insufficient)"],
            "NoDecision",
        ),  # absence never promotes
        (
            "promote",
            {"answer_promo": {"n": 400, "k": 2}},
            ["MET"],
            "Promote",
        ),  # SDK gate spelling
    ],
    ids=[
        "breached-rejects",
        "legacy-reject",
        "legacy-nodecision",
        "inconclusive",
        "absent-counts",
        "sdk-spelling",
    ],
)
def test_join_matches_the_tvl_reference(
    promo: Reactive, legacy: str, counts: Any, verdicts: list[str], joined: str
) -> None:
    fact = candidate(promo, legacy, counts)
    assert fact["monitor_verdicts"] == verdicts
    assert (
        join_promotion_decisions(fact["legacy_decision"], fact["monitor_verdicts"])
        == joined
    )


def test_unresolved_units_are_unknown_not_met(promo: Reactive) -> None:
    counts = {
        "answer_promo": latency_counts(
            [100.0, 3000.0, None, float("nan")] + [90.0] * 400, 2000
        )
    }
    assert counts["answer_promo"] == {"n": 402, "k": 1, "u": 2}
    fact = candidate(promo, counts=counts)
    assert fact["monitor_verdicts"] == [
        "UNKNOWN"
    ]  # engine_required is not a listed reason


@pytest.mark.parametrize(
    "bad",
    [
        {"answer_promo": {"n": 5, "k": 9}},  # k > n violates conservation
        {"answer_promo": {"n": -1, "k": 0}},
        {"answer_promo": {"k": 1}},
        {"answer_promo": {"n": 1.5, "k": 0}},
    ],
    ids=["k-gt-n", "negative", "missing-n", "float-n"],
)
def test_evidence_that_cannot_be_established_is_invalid_with_nothing_else(
    promo: Reactive, bad: Any
) -> None:
    assert candidate(promo, counts=bad) == {"evidence_valid": False}


def test_unusable_inputs_are_invalid(promo: Reactive) -> None:
    assert candidate(promo, legacy="Maybe") == {"evidence_valid": False}
    assert candidate(promo, config={}) == {"evidence_valid": False}
    assert candidate(promo, config={"x": float("inf")}) == {"evidence_valid": False}


def test_gate_digest_binds_contract_candidate_decision_and_verdicts(
    promo: Reactive,
) -> None:
    base = candidate(promo)
    other_contract = evaluate_candidate(
        promo.requirement,
        CandidateEvidence(CONFIG, "Promote", {"answer_promo": {"n": 400, "k": 2}}),
        contract_digest="b" * 64,
        tokens=promo.tokens,
    )
    assert other_contract["gate_result_digest"] != base["gate_result_digest"]
    assert other_contract["fingerprint"] == base["fingerprint"]
    assert (
        candidate(promo, "NoDecision")["gate_result_digest"]
        != base["gate_result_digest"]
    )
    assert (
        candidate(promo, config={**CONFIG, "temperature": 0.3})["gate_result_digest"]
        != base["gate_result_digest"]
    )
    assert (
        candidate(promo, counts={"answer_promo": {"n": 400, "k": 60}})[
            "gate_result_digest"
        ]
        != base["gate_result_digest"]
    )


def test_monitor_without_a_token_fails_closed(promo: Reactive) -> None:
    (promo.state / "tvl-token-map.json").unlink()
    with pytest.raises(Exception) as err:  # noqa: B017
        candidate(promo)
    assert getattr(err.value, "code", "") == "local_state_invalid"


def test_reported_through_respond_and_accepted_by_the_strict_contract(
    promo: Reactive,
) -> None:
    ref = promo.approved_optimize()
    extractor = candidate_extractor(
        promo.requirement,
        promo.tokens,
        lambda result: [
            CandidateEvidence(
                result.best_config, "Promote", {"answer_promo": {"n": 400, "k": 2}}
            ),
            CandidateEvidence({"x": 1}, "Promote", {"answer_promo": {"n": 2, "k": 9}}),
        ],
    )

    async def body() -> Any:
        return SimpleNamespace(
            optimization_id="run-1",
            best_config=CONFIG,
            prompts=[CANARY],
            outputs=[CANARY],
            stop_reason="max_trials_reached",
        )

    grant_digest = promo.backend.grants[promo.backend.attempts[ref]["grant_ref"]][
        "contract_digest"
    ]
    asyncio.run(
        respond_hosted(
            Fn(promo, body),
            promo.requirement,
            agent_ref=promo.agent_ref,
            client=promo.client,
            state_dir=promo.state,
            clock=promo.clock,
            retry_delay=0.0,
            candidates=extractor,
        )
    )
    (report,) = promo.backend.results
    good, bad = report["candidates"]
    assert good["fingerprint"] == candidate_fingerprint(CONFIG)
    assert bad == {"evidence_valid": False}
    assert promo.backend.attempts[ref]["optimize_outcome"] == "promoted"
    expected = evaluate_candidate(
        promo.requirement,
        CandidateEvidence(CONFIG, "Promote", {"answer_promo": {"n": 400, "k": 2}}),
        contract_digest=grant_digest,
        tokens=promo.tokens,
    )
    assert good == expected  # the digest is under the GRANT's current contract digest
    for sent in promo.backend.log:
        assert CANARY not in json.dumps(sent) and "m-small" not in json.dumps(sent)


def _optimize_promoted(promo: Reactive) -> tuple[Any, Any, str]:
    from traigent.runtime.promotion_evidence import candidate_extractor

    ref = promo.approved_optimize()
    reporter = candidate_extractor(
        promo.requirement,
        promo.tokens,
        lambda result: [
            CandidateEvidence(
                result.best_config, "Promote", {"answer_promo": {"n": 400, "k": 2}}
            ),
            CandidateEvidence(
                {"x": 2}, "Promote", {"answer_promo": {"n": 400, "k": 60}}
            ),
        ],
    )

    async def body() -> Any:
        return SimpleNamespace(
            optimization_id="run-1", best_config=CONFIG, prompts=[CANARY]
        )

    result = asyncio.run(
        respond_hosted(
            Fn(promo, body),
            promo.requirement,
            agent_ref=promo.agent_ref,
            client=promo.client,
            state_dir=promo.state,
            clock=promo.clock,
            retry_delay=0.0,
            candidates=reporter,
        )
    )
    return result, reporter, ref


def test_optimize_to_deploy_end_to_end_through_the_strict_fake(promo: Reactive) -> None:
    from traigent.runtime.deploy import DEPLOYED, DeployController
    from traigent.runtime.promotion_evidence import promoted

    b = promo.backend
    result, reporter, ref = _optimize_promoted(promo)
    assert b.attempts[ref]["optimize_outcome"] == "promoted"
    (winner,) = promoted(result.candidates)  # the BREACHED one is not promotable
    # A person approves the deploy attempt, binding what was reported + the head generation.
    deploy_ref = promo.attempt("deploy")
    b.grant(
        deploy_ref,
        deploy={
            "candidate_fingerprint": winner["fingerprint"],
            "gate_result_digest": winner["gate_result_digest"],
            "expected_head_generation": b.head.generation,
        },
    )

    b.add_best_config("answer", 1, CONFIG)

    async def sleep(_s: float) -> None:
        return None

    outcome = asyncio.run(
        DeployController(
            promo.client, state_dir=promo.state, clock=promo.clock, sleep=sleep
        ).run_promoted(
            deploy_ref,
            result,
            configs=reporter.configs,
            grant_ref=b.attempts[deploy_ref]["grant_ref"],
            best_config_id="answer",
            version=1,
        )
    )
    assert outcome.state == DEPLOYED and outcome.result_recorded
    assert b.head.spec["config"] == CONFIG and b.head.generation == 4
    last = b.results[-1]
    assert (last["fact"], last["write"], last["retries_exhausted"]) == (
        "deploy_result",
        "succeeded",
        False,
    )
    assert b.attempts[deploy_ref]["deploy_outcome"] == "deployed"
    (pub,) = b.requests_of("publishTvlDeployResult")
    assert winner["fingerprint"] in pub["path"].rsplit("/", 1)[-1]


def test_deploy_refuses_a_grant_bound_to_a_candidate_that_was_not_promoted(
    promo: Reactive,
) -> None:
    from traigent.runtime.deploy import DeployController

    b = promo.backend
    result, reporter, _ref = _optimize_promoted(promo)
    deploy_ref = promo.attempt("deploy")
    b.grant(
        deploy_ref,
        deploy={
            "candidate_fingerprint": candidate_fingerprint(
                {"x": 2}
            ),  # the BREACHED candidate
            "gate_result_digest": "1" * 64,
            "expected_head_generation": b.head.generation,
        },
    )
    with pytest.raises(Exception) as err:  # noqa: B017
        asyncio.run(
            DeployController(
                promo.client, state_dir=promo.state, clock=promo.clock
            ).run_promoted(
                deploy_ref,
                result,
                candidate_fingerprint=candidate_fingerprint({"x": 2}),
                configs=reporter.configs,
                grant_ref=b.attempts[deploy_ref]["grant_ref"],
                best_config_id="answer",
                version=1,
            )
        )
    assert getattr(err.value, "code", "") == "deploy_binding_mismatch"
    assert [
        r for r in b.requests_of("reserveTvlReactiveAttempt") if deploy_ref in r["path"]
    ] == []
    assert b.head.generation == 3


VECTORS = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "runtime"
    / "promotion_digest_vectors.json"
)


def test_digest_vectors_are_pinned_and_every_case_recomputes() -> None:
    import hashlib

    from tvl.canonical import canonical_json_digest

    from traigent.runtime.promotion_evidence import gate_result_digest

    from .reactive_fake import canonical_digest

    raw = VECTORS.read_bytes()
    pinned = (VECTORS.parent / "SOURCE.md").read_text()
    assert (
        f"promotion_digest_vectors.json sha256: {hashlib.sha256(raw).hexdigest()}"
        in pinned
    )
    doc = json.loads(raw)
    assert len(doc["fingerprints"]) >= 10 and len(doc["gates"]) >= 5
    for case in doc["fingerprints"]:
        assert candidate_fingerprint(case["config"]) == case["fingerprint"], case[
            "name"
        ]
        assert (
            canonical_json_digest(case["config"]) == case["fingerprint"]
        )  # TVL is the authority
        assert (
            canonical_digest(case["config"]) == case["fingerprint"]
        )  # independent re-derivation
    by_name = {c["name"]: c["fingerprint"] for c in doc["fingerprints"]}
    assert by_name["key-order-a"] == by_name["key-order-b"]
    assert len(set(by_name.values())) == len(by_name) - 1
    for case in doc["gates"]:
        got = gate_result_digest(
            contract_digest=case["contract_digest"],
            fingerprint=case["fingerprint"],
            legacy_decision=case["legacy_decision"],
            verdicts_by_monitor_token=case["monitors"],
            decision=case["decision"],
        )
        assert got == case["gate_result_digest"], case["name"]
        document = {
            "format": doc["gate_document"]["format"],
            **{
                k: case[k]
                for k in (
                    "contract_digest",
                    "fingerprint",
                    "legacy_decision",
                    "monitors",
                    "decision",
                )
            },
        }
        assert canonical_digest(document) == case["gate_result_digest"]


def test_the_fake_backend_refuses_a_publication_whose_spec_does_not_match_the_fingerprint(
    promo: Reactive,
) -> None:
    from traigent.runtime.deploy import DeployController

    b = promo.backend
    result, reporter, _ref = _optimize_promoted(promo)
    from traigent.runtime.promotion_evidence import promoted

    (winner,) = promoted(result.candidates)
    deploy_ref = promo.attempt("deploy")
    b.grant(
        deploy_ref,
        deploy={
            "candidate_fingerprint": winner["fingerprint"],
            "gate_result_digest": winner["gate_result_digest"],
            "expected_head_generation": b.head.generation,
        },
    )
    tampered = {**reporter.configs, winner["fingerprint"]: {**CONFIG, "model": "other"}}
    with pytest.raises(Exception) as err:  # noqa: B017
        asyncio.run(
            DeployController(
                promo.client, state_dir=promo.state, clock=promo.clock
            ).run_promoted(
                deploy_ref,
                result,
                configs=tampered,
                grant_ref=b.attempts[deploy_ref]["grant_ref"],
                best_config_id="answer",
                version=1,
            )
        )
    assert (
        getattr(err.value, "code", "") == "deploy_binding_mismatch"
    )  # caught before any write
    assert b.requests_of("publishTvlDeployResult") == [] and b.head.generation == 3
    # Bypassing the client-side check, the Backend still recomputes from the stored
    # best-config version it holds (the publish names it by id+version) and refuses.
    from traigent.runtime.deploy import DeployHold

    b.add_best_config("answer", 1, tampered[winner["fingerprint"]])
    with pytest.raises(DeployHold) as hold:
        asyncio.run(
            DeployController(
                promo.client, state_dir=promo.state, clock=promo.clock
            ).run(
                deploy_ref,
                grant_ref=b.attempts[deploy_ref]["grant_ref"],
                best_config_id="answer",
                version=1,
                candidate_fingerprint=winner["fingerprint"],
                gate_result_digest=winner["gate_result_digest"],
            )
        )
    assert hold.value.path == "idempotency_conflict" and b.head.generation == 3


# --- the optimizer's own decision and trial timings feed the evidence ----------------------


def _real_result(decision: str | None, latencies_s: list[float | None] | None) -> Any:
    from datetime import UTC, datetime

    from traigent.api.types import (
        OptimizationResult,
        OptimizationStatus,
        TrialResult,
        TrialStatus,
    )
    from traigent.utils.hashing import generate_config_hash

    entries = (
        [
            {"execution_time": t, "metrics": {}} if t is not None else {"metrics": {}}
            for t in latencies_s
        ]
        if latencies_s is not None
        else []
    )
    trial = TrialResult(
        trial_id="t1",
        config=dict(CONFIG),
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={"example_results": entries},
    )
    return OptimizationResult(
        trials=[trial],
        best_config=dict(CONFIG),
        best_score=0.9,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="mock",
        timestamp=datetime.now(UTC),
        promotion_decisions=(
            {generate_config_hash(dict(CONFIG)): decision} if decision else {}
        ),
    )


def _respond_default(promo: Reactive, result: Any) -> Any:
    ref = promo.approved_optimize()

    async def body() -> Any:
        return result

    out = asyncio.run(
        respond_hosted(
            Fn(promo, body),
            promo.requirement,
            agent_ref=promo.agent_ref,
            client=promo.client,
            state_dir=promo.state,
            clock=promo.clock,
            retry_delay=0.0,
        )
    )
    return out, promo.backend.attempts[ref]["optimize_outcome"]


def test_gate_promote_and_measured_latency_report_a_promotable_candidate(
    promo: Reactive,
) -> None:
    out, outcome = _respond_default(promo, _real_result("promote", [0.1] * 400))
    (fact,) = out.candidates
    assert fact["legacy_decision"] == "Promote" and fact["monitor_verdicts"] == ["MET"]
    assert fact["fingerprint"] == candidate_fingerprint(CONFIG)
    assert out.configs[fact["fingerprint"]] == CONFIG
    assert outcome == "promoted"
    assert "m-small" not in json.dumps(
        promo.backend.log
    )  # the config never travels here


@pytest.mark.parametrize(
    ("decision", "latencies", "legacy", "verdicts"),
    [
        (
            "no_incumbent",
            [0.1] * 400,
            "NoDecision",
            ["MET"],
        ),  # first candidate: explicit, not Promote
        (None, [0.1] * 400, "NoDecision", ["MET"]),  # no record at all
        ("reject", [0.1] * 400, "Reject", ["MET"]),
        ("promote", None, "Promote", ["UNKNOWN(insufficient)"]),  # nothing measured
        (
            "promote",
            [0.1] * 400 + [None],
            "Promote",
            ["UNKNOWN"],
        ),  # one unresolved example
        (
            "promote",
            [0.1] * 300 + [3.0] * 100,
            "Promote",
            ["BREACHED"],
        ),  # 100 of 400 above 2000 ms
    ],
    ids=[
        "first-candidate",
        "no-record",
        "gate-reject",
        "unmeasured",
        "unresolved",
        "breached",
    ],
)
def test_default_evidence_never_promotes_without_a_gate_answer_and_measurements(
    promo: Reactive, decision: Any, latencies: Any, legacy: str, verdicts: list[str]
) -> None:
    out, outcome = _respond_default(promo, _real_result(decision, latencies))
    (fact,) = out.candidates
    assert (fact["legacy_decision"], fact["monitor_verdicts"]) == (legacy, verdicts)
    assert outcome != "promoted"
