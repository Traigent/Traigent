"""Promotion-gate evidence: fingerprint, TVL join, gate digest, closed candidate facts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("tvl.promotion_join")

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
