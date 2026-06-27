"""Composite EXECUTION runtime tests — post-cascade only (RFC 0002 §3.2/§3.2.1).

This packet executes the ``placement: post`` cascade kind by ADAPTING the IR
into the shipped ``CascadePolicy`` (it never reimplements gate math). The tests
prove:

- ``binary_cascade`` executes: the cheap arm is accepted when its vote margin
  ``≥ θ`` (no escalation); it escalates to the expert arm below ``θ``; the
  §3.10 telemetry is correct (deterministic stub stages — NO LLM calls);
- a nested-composite arm raises the explicit ``NotImplementedError`` (deferred,
  honest); ensemble/loop roots likewise raise loudly;
- fail-closed: a missing stage callable or a missing/non-finite calibrated
  threshold yields an ``error`` result — never a silent stage pick;
- the GOVERNANCE NO-CHANGE proof: a ``ConfigSpace`` built from a
  ``binary_cascade(...).members`` with the gate threshold Calibrated produces
  the SAME ``tvl_governance`` via the EXISTING
  ``traigent.cloud.governance.build_tvl_governance`` and the SAME certified-
  report behavior via the EXISTING ``build_certified_selection`` — with ZERO
  modifications to ``traigent/core`` or ``traigent/cloud``.
"""

from __future__ import annotations
from typing import Any

import math

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import Choices
from traigent.cloud.governance import build_certified_selection, build_tvl_governance
from traigent.knobs import (
    Calibrated,
    CertificateDecision,
    FreshnessContext,
    Knob,
    Ref,
    SignalSpec,
    TargetProperty,
    Tuned,
    canonical_hash,
    issue_certificate,
)
from traigent.knobs.composites import (
    AcceptDecl,
    AggregateDecl,
    AggregateKind,
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    EnsembleBody,
    GateDecl,
    GateKind,
    LoopBody,
    Placement,
    SignalUse,
    StageArm,
    StatKind,
    StopDecl,
    StopKind,
)
from traigent.knobs.patterns import (
    best_of_n,
    binary_cascade,
    n_cascade,
    self_consistency,
    self_debug,
    self_refine,
)
from traigent.knobs.runtime import (
    CompositeRunResult,
    LoopBodyResult,
    LoopBodyRunner,
    ResultKind,
    StageRunner,
    StopReason,
    execute_composite,
    execute_kchain,
)

GATE = "router_margin_threshold"


# --------------------------------------------------------------------------- #
# Deterministic stub stages (no LLM calls)                                    #
# --------------------------------------------------------------------------- #


def _stage(outputs: list[str]) -> StageRunner:
    """A voting stage runner over a fixed output multiset (identity keys)."""
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


def _binary(threshold: str = GATE) -> CompositeNode:
    return binary_cascade(
        "answerer",
        base_stage="cheap",
        expert_stage="strong",
        threshold=threshold,
    ).structure


# --------------------------------------------------------------------------- #
# binary_cascade execution (§3.2 post-cascade)                                #
# --------------------------------------------------------------------------- #


class TestBinaryCascadeExecution:
    def test_cheap_accepted_when_margin_at_or_above_theta(self):
        # cheap unanimous: margin = 3/3 = 1.0 >= theta 0.6 -> NO escalate
        stages = {"cheap": _stage(["A", "A", "A"]), "strong": _stage(["STRONG"])}
        result = execute_composite(
            _binary(),
            stages,
            config={},
            calibrated_values={GATE: 0.6},
        )
        assert isinstance(result, CompositeRunResult)
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"  # the cheap arm's representative
        assert result.measures["stage_selected"] == 0
        assert result.measures["escalation_rate"] == 0.0
        assert result.measures["gate_margin_pass_rate"] == {0: 1.0}

    def test_escalates_below_theta(self):
        # cheap split 2/3 -> margin 0.666... < theta 0.9 -> escalate to strong
        stages = {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])}
        result = execute_composite(
            _binary(),
            stages,
            config={},
            calibrated_values={GATE: 0.9},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "STRONG"  # the escalated expert arm
        assert result.measures["stage_selected"] == 1
        assert result.measures["escalation_rate"] == 1.0
        assert result.measures["gate_margin_pass_rate"] == {0: 0.0}

    def test_threshold_is_read_live_not_snapshotted(self):
        # The SAME structure escalates or not purely as a function of the LIVE
        # calibrated_values — the threshold_ref is a live read (cascade.py).
        stages = {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])}
        node = _binary()
        kept = execute_composite(node, stages, config={}, calibrated_values={GATE: 0.5})
        escalated = execute_composite(
            node, stages, config={}, calibrated_values={GATE: 0.9}
        )
        assert kept.measures["stage_selected"] == 0  # 0.666 >= 0.5
        assert escalated.measures["stage_selected"] == 1  # 0.666 < 0.9

    def test_bare_callable_stage_is_wrapped_single_sample(self):
        # A degenerate m=1 cascade (no gates) over a bare callable executes.
        node = CompositeNode(
            name="solo",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("only"),), gates=(), placement=Placement.POST
            ),
        )
        result = execute_composite(
            node,
            {"only": lambda _item: "OUT"},
            config={},
            calibrated_values={},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "OUT"
        assert result.measures["stage_selected"] == 0
        assert result.measures["gate_margin_pass_rate"] == {}


# --------------------------------------------------------------------------- #
# Surviving deferrals raise loudly (honest, never faked)                      #
# --------------------------------------------------------------------------- #


class TestSurvivingDeferrals:
    """The one genuinely-deferred surface (a nested-composite LOOP body) raises
    NotImplementedError naming the gap.

    Every OTHER former deferral (ensemble/loop roots, nested-composite cascade
    AND ensemble arms, the K-chain unroll, AND the pre-cascade dispatch
    executor) is now executed end-to-end — proven by the execution test classes
    below (pre-cascade dispatch: ``TestPreCascadeDispatch``).
    """

    def test_nested_composite_loop_body_raises_not_implemented(self):
        # A loop whose BODY is a nested composite has no v1 state-producing
        # convention -> deferred (stage-bodied loops execute fully).
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("a"),), gates=(), placement=Placement.POST),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("inner"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                state_keys=(),
                max_iters=2,
            ),
        )
        with pytest.raises(NotImplementedError, match="nested-composite loop"):
            execute_composite(
                outer,
                {"a": lambda _i: "x"},
                config={},
                calibrated_values={},
                registry={"inner": inner, "outer": outer},
            )


# --------------------------------------------------------------------------- #
# Fail-closed (§3.2.1 error is absorbing) — never a silent stage pick         #
# --------------------------------------------------------------------------- #


class TestFailClosed:
    def test_missing_stage_callable_is_error(self):
        result = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "A"])},  # 'strong' absent
            config={},
            calibrated_values={GATE: 0.6},
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert "missing stage callable" in (result.error or "")
        assert "strong" in (result.error or "")

    def test_missing_calibrated_threshold_is_error(self):
        result = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={},  # threshold absent -> Gate fails closed
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert result.measures == {}

    def test_non_finite_threshold_is_error(self):
        result = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={GATE: math.nan},  # non-finite -> fail closed
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None

    def test_stage_exception_propagates_as_error(self):
        def explode(_item):
            raise RuntimeError("stage blew up")

        stages = {
            "cheap": StageRunner(run=explode, key_fn=lambda x: x, samples=1),
            "strong": _stage(["STRONG"]),
        }
        result = execute_composite(
            _binary(), stages, config={}, calibrated_values={GATE: 0.6}
        )
        assert result.result_kind is ResultKind.ERROR
        assert "RuntimeError" in (result.error or "")
        # never a silent degradation to a stage output
        assert result.output is None


# --------------------------------------------------------------------------- #
# Governance NO-CHANGE proof (the key architectural test)                     #
# --------------------------------------------------------------------------- #


def _signal() -> SignalSpec:
    return SignalSpec(
        name="vote_margin",
        version="1",
        score_function="exact_match",
        score_function_version="1",
        comparator="key_eq",
        comparator_version="1",
    )


def _target() -> TargetProperty:
    return TargetProperty(name="margin_floor", mode="require_calibration")


def _ctx() -> FreshnessContext:
    return FreshnessContext(
        cvar_name=GATE,
        tuned_parent_values=(("model", "a"),),
        calibration_source_id="pool_a",
        signal_spec_hash=_signal().spec_hash(),
        calibrator_id="budget_threshold",
        calibrator_version="1",
        calibrator_params_hash=canonical_hash({}),
        dataset_hash="ds_v1",
        evidence_n=20,
        calibration_split="cal",
        eval_split="eval",
        target=_target(),
    )


def _composite_space() -> ConfigSpace:
    """A ConfigSpace built from binary_cascade(...).members with the gate
    threshold declared as a GOVERNED Calibrated CVAR (and its tuned parent)."""
    bc = binary_cascade(
        "answerer",
        base_stage="cheap",
        expert_stage="strong",
        threshold=GATE,
        base_tuned_params=("model",),
        members={
            "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"]))),
            GATE: Knob(
                name=GATE,
                binding=Calibrated(
                    signal=_signal(),
                    target=_target(),
                    depends_on=(Ref(knob="model"),),
                    require_calibration=True,
                ),
            ),
        },
    )
    return ConfigSpace(knobs=bc.members)


class TestGovernanceNoChangeProof:
    """The composite's gate CVAR rides the EXISTING Phase-8 machinery with ZERO
    modifications to traigent/core or traigent/cloud."""

    def test_gate_cvar_is_governed_by_existing_builder(self):
        # EXISTING builder, imported verbatim — no composite-aware code path.
        governance = build_tvl_governance(_composite_space())
        assert governance is not None
        names = {c["name"] for c in governance["cvars"]}
        assert GATE in names
        entry = next(c for c in governance["cvars"] if c["name"] == GATE)
        assert entry["governed"] is True
        assert entry["type"] == "float"

    def test_governance_matches_a_handwritten_equivalent_space(self):
        """The SAME tvl_governance as a hand-built equivalent space — the
        composite path adds nothing and changes nothing on the wire."""
        composite_gov = build_tvl_governance(_composite_space())
        handwritten = ConfigSpace(
            knobs={
                "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"]))),
                GATE: Knob(
                    name=GATE,
                    binding=Calibrated(
                        signal=_signal(),
                        target=_target(),
                        depends_on=(Ref(knob="model"),),
                        require_calibration=True,
                    ),
                ),
            }
        )
        assert composite_gov == build_tvl_governance(handwritten)

    def test_certified_report_includes_gate_cvar_with_certificate(self):
        # EXISTING report builder; an issued certificate -> the report includes
        # the composite's gate CVAR (Phase 8 machinery unchanged).
        cert = issue_certificate(GATE, "float", 0.5, _ctx())
        report = build_certified_selection("trial1", {GATE: cert})
        assert report is not None
        assert report["certificates"][0]["cvar_name"] == GATE
        assert report["certificates"][0]["decision"] == "CERTIFIED_SELECTION"
        assert report["attestation"] == "sdk_client_attested"

    def test_certified_report_withheld_without_certificate_fail_closed(self):
        # NO certified decision -> the EXISTING builder WITHHOLDS the report
        # (fail-closed, the honest no-winner; partial reports are bugs).
        no_decision = issue_certificate(
            GATE, "float", 0.5, _ctx(), decision=CertificateDecision.NO_DECISION
        )
        assert build_certified_selection("trial1", {GATE: no_decision}) is None
        # and an entirely empty certificate set is likewise withheld.
        assert build_certified_selection("trial1", {}) is None


class TestPerGateTelemetryIndexed:
    def test_duplicate_threshold_refs_keep_per_gate_values(self):
        """Codex runtime-round blocker: keying per-gate telemetry on the
        threshold NAME collapses duplicated refs (legal in n_cascade) — a
        3-arm cascade reusing 'theta' lost the first gate's 0.0. Per-gate
        measures are keyed by GATE INDEX (§3.10: per-gate)."""

        knob = n_cascade(
            "tri",
            stages=("a", "b", "c"),
            thresholds=("theta", "theta"),
        )
        stages = {
            # split 1/3 -> margin ~0.33 < theta 0.5 -> escalate past gate 0
            "a": _stage(["x", "y", "z"]),
            # unanimous -> margin 1.0 >= 0.5 -> gate 1 stops here
            "b": _stage(["k", "k", "k"]),
            "c": _stage(["k", "k", "k"]),
        }
        result = execute_composite(
            knob.structure, stages, config={}, calibrated_values={"theta": 0.5}
        )
        assert result.result_kind is ResultKind.OUTPUT
        per_gate = result.measures["gate_margin_pass_rate"]
        assert per_gate == {0: 0.0, 1: 1.0}  # BOTH gates preserved by INDEX


# --------------------------------------------------------------------------- #
# Ensemble execution (§3.2 sampling/committee; majority_vote/judge_max)       #
# --------------------------------------------------------------------------- #


class TestEnsembleSamplingMajorityVote:
    """Sampling form: one arm runs k times, majority_vote over keys (§3.2).

    Reuses the shipped cascade.vote_over (RFC 0001 tie/abstain rules) — never a
    reimplementation; telemetry is the §3.10 ensemble dict.
    """

    def _sc(self, accept_threshold=None):
        return self_consistency(
            "sc", stage="a", cardinality="k", accept_threshold=accept_threshold
        ).structure

    def test_majority_winner_and_telemetry(self):
        # k=3: A,A,B -> winner A, margin 2/3, agreement 3/3
        node = self._sc()
        stages = {"a": _stage(["A", "A", "B"])}
        result = execute_composite(node, stages, config={"k": 3}, calibrated_values={})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"
        assert result.measures["candidates_evaluated"] == 3
        assert result.measures["candidates_excluded"] == 0
        assert result.measures["vote_margin"] == pytest.approx(2 / 3)
        assert result.measures["vote_agreement"] == pytest.approx(1.0)

    def test_accept_gate_passes_when_margin_at_or_above_theta(self):
        node = self._sc(accept_threshold="acc")
        stages = {"a": _stage(["A", "A", "A"])}  # unanimous -> margin 1.0
        result = execute_composite(
            node, stages, config={"k": 3}, calibrated_values={"acc": 0.6}
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"

    def test_accept_gate_failure_is_no_accept_not_error(self):
        # margin 2/3 < theta 0.9 -> HONEST no_accept (§3.2.1), telemetry kept.
        node = self._sc(accept_threshold="acc")
        stages = {"a": _stage(["A", "A", "B"])}
        result = execute_composite(
            node, stages, config={"k": 3}, calibrated_values={"acc": 0.9}
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.output is None
        assert result.measures["vote_margin"] == pytest.approx(2 / 3)

    def test_cardinality_resolves_from_calibrated_when_absent_in_config(self):
        node = self._sc()
        stages = {"a": _stage(["A", "A"])}
        result = execute_composite(node, stages, config={}, calibrated_values={"k": 2})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.measures["candidates_evaluated"] == 2

    def test_cardinality_below_one_is_error_r9(self):
        node = self._sc()
        result = execute_composite(
            node, {"a": _stage([])}, config={"k": 0}, calibrated_values={}
        )
        assert result.result_kind is ResultKind.ERROR
        assert "R9" in (result.error or "")

    def test_missing_cardinality_value_is_error(self):
        node = self._sc()
        result = execute_composite(
            node, {"a": _stage(["A"])}, config={}, calibrated_values={}
        )
        assert result.result_kind is ResultKind.ERROR
        assert "unresolved" in (result.error or "")

    def test_sample_count_mismatch_is_error(self):
        # arm declared k=3 but produced 2 -> the effectuation did not happen.
        node = self._sc()
        result = execute_composite(
            node, {"a": _stage(["A", "B"])}, config={"k": 3}, calibrated_values={}
        )
        assert result.result_kind is ResultKind.ERROR
        assert "k=3" in (result.error or "")


class TestEnsembleJudgeMax:
    """judge_max: judge scores each candidate; max wins; the judge output
    contract (finite numeric score) gates exclusion vs. all-excluded failure."""

    def _bon(self):
        return best_of_n("bon", stage="a", judge_stage="j", cardinality="k").structure

    def test_judge_max_selects_highest_score(self):
        node = self._bon()
        scores = {"x": 0.2, "y": 0.9, "z": 0.5}
        stages = {
            "a": _stage(["x", "y", "z"]),
            "j": StageRunner(run=lambda cand: [scores[cand]], samples=1),
        }
        result = execute_composite(node, stages, config={"k": 3}, calibrated_values={})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "y"  # the max-scored candidate
        assert result.measures["candidates_evaluated"] == 3
        assert result.measures["candidates_excluded"] == 0

    def test_nan_score_candidate_is_excluded(self):
        node = self._bon()
        scores = {"x": float("nan"), "y": 0.4}
        stages = {
            "a": _stage(["x", "y"]),
            "j": StageRunner(run=lambda cand: [scores[cand]], samples=1),
        }
        result = execute_composite(node, stages, config={"k": 2}, calibrated_values={})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "y"  # x excluded by NaN
        assert result.measures["candidates_excluded"] == 1

    def test_judge_exception_excludes_candidate(self):
        node = self._bon()

        def judge(cand):
            if cand == "x":
                raise RuntimeError("judge blew up on x")
            return [0.7]

        stages = {
            "a": _stage(["x", "y"]),
            "j": StageRunner(run=judge, samples=1),
        }
        result = execute_composite(node, stages, config={"k": 2}, calibrated_values={})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "y"
        assert result.measures["candidates_excluded"] == 1

    def test_all_excluded_by_error_fails(self):
        # ALL candidates excluded by judge-contract violation -> error (§3.2.1),
        # DISTINCT from a committee all-excluded-by-no_accept.
        node = self._bon()
        stages = {
            "a": _stage(["x", "y"]),
            "j": StageRunner(run=lambda _c: [float("inf")], samples=1),
        }
        result = execute_composite(node, stages, config={"k": 2}, calibrated_values={})
        assert result.result_kind is ResultKind.ERROR
        assert "all candidates excluded" in (result.error or "")

    def test_score_tie_breaks_by_deterministic_serialized_key_order(self):
        # x and y tie at 0.5; the SMALLEST serialized key ('x') wins (§3.2).
        node = self._bon()
        stages = {
            "a": _stage(["y", "x"]),  # note order: y first
            "j": StageRunner(run=lambda _c: [0.5], samples=1),
        }
        result = execute_composite(node, stages, config={"k": 2}, calibrated_values={})
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "x"  # deterministic, order-independent


class TestEnsembleCommittee:
    """Committee form (|arms| > 1): each arm runs once; a committee arm
    yielding no_accept is excluded; ALL-excluded-by-no_accept -> no_accept."""

    def _committee(self, accept_threshold=None):
        accept = (
            AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold=accept_threshold)
            if accept_threshold is not None
            else None
        )
        return CompositeNode(
            name="committee",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(CompositeArm("m1"), CompositeArm("m2"), CompositeArm("m3")),
                aggregate=AggregateDecl(
                    kind=AggregateKind.MAJORITY_VOTE, accept=accept
                ),
            ),
        )

    def _member(self, name, outputs):
        return CompositeNode(
            name=name,
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("s"),), gates=(), placement=Placement.POST),
        )

    def test_committee_majority_over_member_outputs(self):
        committee = self._committee()
        # Three single-stage cascade members, each producing one output.
        registry = {
            "committee": committee,
            "m1": CompositeNode(
                name="m1",
                kind=CompositeKind.CASCADE,
                body=CascadeBody(arms=(StageArm("s1"),), gates=()),
            ),
            "m2": CompositeNode(
                name="m2",
                kind=CompositeKind.CASCADE,
                body=CascadeBody(arms=(StageArm("s2"),), gates=()),
            ),
            "m3": CompositeNode(
                name="m3",
                kind=CompositeKind.CASCADE,
                body=CascadeBody(arms=(StageArm("s3"),), gates=()),
            ),
        }
        stages = {
            "s1": lambda _i: "P",
            "s2": lambda _i: "P",
            "s3": lambda _i: "Q",
        }
        result = execute_composite(
            committee, stages, config={}, calibrated_values={}, registry=registry
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "P"  # 2/3 majority
        assert result.measures["candidates_evaluated"] == 3

    def test_all_committee_arms_no_accept_yields_no_accept(self):
        # Each member is a self_consistency with an accept gate that ALL fail,
        # so every committee candidate is excluded by no_accept -> no_accept.
        committee = self._committee()

        def member(name):
            return CompositeNode(
                name=name,
                kind=CompositeKind.ENSEMBLE,
                body=EnsembleBody(
                    arms=(StageArm(f"{name}_s"),),
                    aggregate=AggregateDecl(
                        kind=AggregateKind.MAJORITY_VOTE,
                        accept=AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold="acc"),
                    ),
                    cardinality="k",
                ),
            )

        registry = {
            "committee": committee,
            "m1": member("m1"),
            "m2": member("m2"),
            "m3": member("m3"),
        }
        stages = {
            "m1_s": _stage(["a", "b"]),  # margin 0.5 < acc 0.9 -> no_accept
            "m2_s": _stage(["a", "b"]),
            "m3_s": _stage(["a", "b"]),
        }
        result = execute_composite(
            committee,
            stages,
            config={"k": 2},
            calibrated_values={"acc": 0.9},
            registry=registry,
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.measures["candidates_excluded"] == 3


# --------------------------------------------------------------------------- #
# Loop execution (§3.2 signal_accept / external_accept / exhausted)           #
# --------------------------------------------------------------------------- #


def _stateful_body(seq, signal_key="draft"):
    """A loop body whose successive outputs are `seq`, threading them on
    `signal_key`. Iteration i returns LoopBodyResult(output=seq[i],
    state={signal_key: seq[i]})."""
    counter = {"i": 0}

    def run(_item, _state):
        i = counter["i"]
        counter["i"] += 1
        value = seq[min(i, len(seq) - 1)]
        return LoopBodyResult(output=value, state={signal_key: value})

    return LoopBodyRunner(run=run)


class TestLoopSignalAccept:
    """signal_accept loop: σ(state) >= θ accepts; stop_reason/iterations_used."""

    def _sr(self, max_iters):
        return self_refine(
            "sr",
            stage="body",
            signal="quality",
            threshold="theta",
            max_iters=max_iters,
        ).structure

    def test_accepts_when_signal_reaches_theta(self):
        node = self._sr(max_iters=4)
        # quality rises 0.2, 0.5, 0.95; theta 0.9 -> accepts on iteration 3.
        body = _stateful_body([0.2, 0.5, 0.95])

        def quality(state):
            return state["draft"]

        result = execute_composite(
            node,
            {"body": body},
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": quality},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == 0.95
        assert result.measures["iterations_used"] == 3
        assert result.measures["stop_reason"] == StopReason.SIGNAL_ACCEPT.value

    def test_exhausts_without_acceptance_yields_no_accept(self):
        node = self._sr(max_iters=2)
        body = _stateful_body([0.1, 0.2])

        def quality(state):
            return state["draft"]

        result = execute_composite(
            node,
            {"body": body},
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": quality},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.measures["iterations_used"] == 2
        assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value

    def test_missing_signal_is_error_fail_closed(self):
        node = self._sr(max_iters=2)
        result = execute_composite(
            node,
            {"body": _stateful_body([0.5])},
            config={},
            calibrated_values={"theta": 0.9},
            signals={},  # signal absent -> fail closed
        )
        assert result.result_kind is ResultKind.ERROR
        assert "not provided" in (result.error or "")

    def test_missing_threshold_is_error_fail_closed(self):
        node = self._sr(max_iters=2)
        result = execute_composite(
            node,
            {"body": _stateful_body([0.5])},
            config={},
            calibrated_values={},  # threshold absent
            signals={"quality": lambda s: s["draft"]},
        )
        assert result.result_kind is ResultKind.ERROR

    def test_body_exception_is_error(self):
        node = self._sr(max_iters=2)

        def boom(_item, _state):
            raise RuntimeError("body blew up")

        result = execute_composite(
            node,
            {"body": LoopBodyRunner(run=boom)},
            config={},
            calibrated_values={"theta": 0.5},
            signals={"quality": lambda s: 1.0},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "RuntimeError" in (result.error or "")


class TestLoopExternalAccept:
    """external_accept loop: opaque predicate over state decides acceptance."""

    def _sd(self, max_iters):
        return self_debug(
            "sd", stage="body", predicate="tests_pass", max_iters=max_iters
        ).structure

    def test_predicate_accepts(self):
        node = self._sd(max_iters=3)
        calls = {"n": 0}

        def body(_item, _state):
            calls["n"] += 1
            return LoopBodyResult(output=f"attempt{calls['n']}", state={})

        def predicate(_state):
            return calls["n"] >= 2  # accepts on the 2nd attempt

        result = execute_composite(
            node,
            {"body": LoopBodyRunner(run=body)},
            config={},
            calibrated_values={},
            predicates={"tests_pass": predicate},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "attempt2"
        assert result.measures["iterations_used"] == 2
        assert result.measures["stop_reason"] == StopReason.EXTERNAL_ACCEPT.value

    def test_missing_predicate_is_error_fail_closed(self):
        node = self._sd(max_iters=2)
        result = execute_composite(
            node,
            {"body": lambda _i: "x"},
            config={},
            calibrated_values={},
            predicates={},  # predicate absent -> fail closed
        )
        assert result.result_kind is ResultKind.ERROR
        assert "not provided" in (result.error or "")

    def test_never_accepts_yields_no_accept(self):
        node = self._sd(max_iters=2)
        result = execute_composite(
            node,
            {"body": lambda _i: "x"},
            config={},
            calibrated_values={},
            predicates={"tests_pass": lambda _s: False},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value


class TestLoopExhausted:
    """exhausted loop: always runs max_iters; final produced output is the
    result (no acceptance predicate)."""

    def _exhausted_loop(self, max_iters):
        return CompositeNode(
            name="ex",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=StageArm("body"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                state_keys=(),
                max_iters=max_iters,
            ),
        )

    def test_runs_all_iters_and_returns_final_output(self):
        node = self._exhausted_loop(max_iters=3)
        seq = ["r1", "r2", "r3"]
        counter = {"i": 0}

        def body(_item, _state):
            v = seq[counter["i"]]
            counter["i"] += 1
            return LoopBodyResult(output=v, state={})

        result = execute_composite(
            node,
            {"body": LoopBodyRunner(run=body)},
            config={},
            calibrated_values={},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "r3"  # final produced output
        assert result.measures["iterations_used"] == 3
        assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value


class TestLoopStateThreading:
    """State threads over declared state_keys; undeclared keys are dropped."""

    def test_undeclared_state_keys_invisible_to_stop(self):
        # state_keys=[draft]; the body also produces 'scratch' -> dropped, so
        # the signal (reading 'draft') only ever sees 'draft'.
        node = self_refine(
            "sr",
            stage="body",
            signal="q",
            threshold="theta",
            max_iters=2,
            state_keys=("draft",),
        ).structure
        seen_keys = []

        def body(_item, state):
            seen_keys.append(set(state.keys()))
            return LoopBodyResult(
                output="x", state={"draft": 1.0, "scratch": "INVISIBLE"}
            )

        def signal(state):
            # 'scratch' must NOT be present (dropped: not in state_keys).
            assert "scratch" not in state
            return state.get("draft", 0.0)

        result = execute_composite(
            node,
            {"body": LoopBodyRunner(run=body)},
            config={},
            calibrated_values={"theta": 0.9},
            signals={"q": signal},
        )
        assert result.result_kind is ResultKind.OUTPUT
        # iteration 1 sees empty state; the body's 'scratch' never threads.
        assert seen_keys[0] == set()


# --------------------------------------------------------------------------- #
# Nested-composite arm execution (§3.2.1 propagation through nesting)          #
# --------------------------------------------------------------------------- #


class TestNestedCompositeArms:
    def test_nested_cascade_arm_output_lifts_onto_escalation_line(self):
        # outer cascade: [stage(a), composite(inner)] — a escalates (margin
        # below theta) to the nested inner composite, whose output is the result.
        inner = self_consistency("inner", stage="i", cardinality="k").structure
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        registry = {"outer": outer, "inner": inner}
        stages = {
            "a": _stage(["x", "y"]),  # margin 0.5 < t 0.9 -> escalate
            "i": _stage(["W", "W"]),  # nested unanimous -> output W
        }
        result = execute_composite(
            outer,
            stages,
            config={"k": 2},
            calibrated_values={"t": 0.9},
            registry=registry,
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "W"  # nested composite's output
        assert result.measures["stage_selected"] == 1

    def test_nested_error_propagates_to_parent_error(self):
        # The nested inner stage raises -> parent yields error (absorbing).
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("i"),), gates=()),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )

        def boom(_i):
            raise RuntimeError("nested stage exploded")

        stages = {
            "a": _stage(["x", "y"]),  # escalate
            "i": StageRunner(run=boom, samples=1),
        }
        result = execute_composite(
            outer,
            stages,
            config={},
            calibrated_values={"t": 0.9},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "nested composite 'inner' failed" in (result.error or "")

    def test_nested_no_accept_at_last_arm_yields_cascade_no_accept(self):
        # outer = [stage(a), composite(inner)]; a escalates; inner is a
        # self_consistency whose accept gate fails -> nested no_accept; at the
        # LAST arm a no_accept yields cascade no_accept (§3.2.1).
        inner = self_consistency(
            "inner", stage="i", cardinality="k", accept_threshold="acc"
        ).structure
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        stages = {
            "a": _stage(["x", "y"]),  # escalate
            "i": _stage(["p", "q"]),  # margin 0.5 < acc 0.9 -> inner no_accept
        }
        result = execute_composite(
            outer,
            stages,
            config={"k": 2},
            calibrated_values={"t": 0.9, "acc": 0.9},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT

    def test_missing_nested_ref_is_error_fail_closed(self):
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("ghost")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        result = execute_composite(
            outer,
            {"a": _stage(["x"])},
            config={},
            calibrated_values={"t": 0.5},
            registry={"outer": outer},  # 'ghost' absent
        )
        assert result.result_kind is ResultKind.ERROR
        assert "missing composite ref" in (result.error or "")
        assert "ghost" in (result.error or "")


# --------------------------------------------------------------------------- #
# F1: nested composite arms in GATED cascade positions (§3.2.1)               #
# --------------------------------------------------------------------------- #


def _mv_ensemble(name, *, stage_name, accept_threshold=None):
    """A nested majority_vote ensemble (the ONLY legal gated nested arm, §3.2)."""
    accept = (
        AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold=accept_threshold)
        if accept_threshold is not None
        else None
    )
    return CompositeNode(
        name=name,
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=(StageArm(stage_name),),
            aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE, accept=accept),
            cardinality="k",
        ),
    )


class TestGatedNestedCompositeArm:
    """F1: a nested majority_vote ensemble sitting in a GATED cascade position.

    (a) Its OWN vote stats must feed the cascade gate (margin surfaced), so it
        no longer errors with 'feeds a gate but has no key_fn'.
    (b) A gated arm's no_accept ESCALATES (i < m); only the terminal arm's
        no_accept becomes the cascade's no_accept (§3.2.1).
    """

    def _outer(self):
        return CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(CompositeArm("inner"), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )

    def test_gated_nested_ensemble_high_margin_stops_and_returns_its_output(self):
        # inner unanimous: vote margin 1.0 >= gate t 0.5 -> NO escalate; the
        # cascade selects the nested ensemble arm and returns ITS output.
        inner = _mv_ensemble("inner", stage_name="i")
        outer = self._outer()
        stages = {"i": _stage(["A", "A", "A"]), "b": _stage(["B"])}
        result = execute_composite(
            outer,
            stages,
            config={"k": 3},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"  # the nested ensemble's selected output
        assert result.measures["stage_selected"] == 0
        assert result.measures["escalation_rate"] == 0.0

    def test_gated_nested_ensemble_low_margin_escalates(self):
        # inner split 1/3: vote margin ~0.33 < gate t 0.9 -> ESCALATE to b.
        inner = _mv_ensemble("inner", stage_name="i")
        outer = self._outer()
        stages = {"i": _stage(["x", "y", "z"]), "b": _stage(["B"])}
        result = execute_composite(
            outer,
            stages,
            config={"k": 3},
            calibrated_values={"t": 0.9},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "B"  # escalated to the expert stage arm
        assert result.measures["stage_selected"] == 1
        assert result.measures["escalation_rate"] == 1.0

    def test_gated_nested_no_accept_escalates_to_next_stage(self):
        # F1(b) PROBE: inner has an accept gate that FAILS (margin 0.5 < acc 0.9)
        # -> nested no_accept. At a GATED position (i < m) that no_accept MUST
        # escalate to the next stage, NOT become the cascade's no_accept.
        inner = _mv_ensemble("inner", stage_name="i", accept_threshold="acc")
        outer = self._outer()
        stages = {"i": _stage(["p", "q"]), "b": _stage(["B"])}
        result = execute_composite(
            outer,
            stages,
            config={"k": 2},
            calibrated_values={"t": 0.5, "acc": 0.9},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "B"  # escalated past the no_accept gated arm
        assert result.measures["stage_selected"] == 1
        assert result.measures["escalation_rate"] == 1.0

    def test_terminal_nested_no_accept_still_yields_cascade_no_accept(self):
        # Contrast: a no_accept at the TERMINAL arm DOES become cascade
        # no_accept (the existing §3.2.1 propagation must be preserved).
        inner = _mv_ensemble("inner", stage_name="i", accept_threshold="acc")
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        stages = {"a": _stage(["x", "y"]), "i": _stage(["p", "q"])}
        result = execute_composite(
            outer,
            stages,
            config={"k": 2},
            calibrated_values={"t": 0.9, "acc": 0.9},  # a escalates; inner no_accept
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT

    def test_gated_nested_error_propagates_as_error(self):
        # A nested ERROR is absorbing even at a gated position.
        inner = _mv_ensemble("inner", stage_name="i")
        outer = self._outer()

        def boom(_i):
            raise RuntimeError("nested ensemble stage exploded")

        stages = {
            "i": StageRunner(run=boom, key_fn=lambda x: x, samples=1),
            "b": _stage(["B"]),
        }
        result = execute_composite(
            outer,
            stages,
            config={"k": 1},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR


class TestDriverSampleCardinalityParity:
    """Codex round-2 blocker: the manual driver must keep the engine's

    ``StageSpec.samples >= 1`` validation. A ``samples=0`` stage arm behind the
    new gated-nested driver path must fail CLOSED (§3.2.1 error), never return
    ``OUTPUT None``.
    """

    def _outer(self):
        return CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(CompositeArm("inner"), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )

    def test_zero_sample_terminal_stage_arm_fails_closed(self):
        # RED PROBE (pre-fix: OUTPUT None): split inner escalates to a
        # samples=0 terminal arm -> must be an ERROR, not a silent None output.
        inner = _mv_ensemble("inner", stage_name="i")
        empty = StageRunner(run=lambda _i: [], key_fn=lambda x: x, samples=0)
        outer = self._outer()
        result = execute_composite(
            outer,
            {"i": _stage(["x", "y", "z"]), "b": empty},
            config={"k": 3},
            calibrated_values={"t": 0.9},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "samples" in (result.error or "")

    def test_zero_sample_gated_stage_arm_fails_closed(self):
        # Twin at the GATED position: stage arm 'g' (samples=0) feeds a gate.
        inner = _mv_ensemble("inner", stage_name="i")
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("g"), CompositeArm("inner"), StageArm("b")),
                gates=(
                    GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),
                    GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),
                ),
                placement=Placement.POST,
            ),
        )
        empty = StageRunner(run=lambda _i: [], key_fn=lambda x: x, samples=0)
        result = execute_composite(
            outer,
            {"g": empty, "i": _stage(["A", "A", "A"]), "b": _stage(["B"])},
            config={"k": 3},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "samples" in (result.error or "")


# --------------------------------------------------------------------------- #
# F2: single-arm SAMPLING ensemble over a CompositeArm body                   #
# --------------------------------------------------------------------------- #


class TestSamplingEnsembleNestedComposite:
    def test_sampling_runs_nested_composite_k_times_keyed_by_output(self):
        # F2 PROBE: a single-arm (sampling) ensemble whose ONE arm is a
        # CompositeArm must run the nested composite k times (each run = one
        # candidate, keyed by its output) — NOT KeyError on stages[ref].
        leaf = CompositeNode(
            name="leaf",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("s"),), gates=()),
        )
        samp = CompositeNode(
            name="samp",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(CompositeArm("leaf"),),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
                cardinality="k",
            ),
        )
        # leaf produces a value that depends on a per-call counter so the k runs
        # are observable as k distinct candidate evaluations.
        calls = {"n": 0}

        def s(_item):
            calls["n"] += 1
            # first two runs -> 'A', third -> 'B' (majority A, margin 2/3).
            return "A" if calls["n"] <= 2 else "B"

        result = execute_composite(
            samp,
            {"s": s},
            config={"k": 3},
            calibrated_values={},
            registry={"samp": samp, "leaf": leaf},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"  # 2/3 majority over the nested runs
        assert result.measures["candidates_evaluated"] == 3
        assert result.measures["vote_margin"] == pytest.approx(2 / 3)
        assert calls["n"] == 3  # the nested composite ran EXACTLY k times


# --------------------------------------------------------------------------- #
# F3: committee arms contribute EXACTLY ONE candidate (the selected output)   #
# --------------------------------------------------------------------------- #


class TestCommitteeOneCandidatePerArm:
    def test_multisample_committee_stage_arm_contributes_one_candidate(self):
        # F3 PROBE: a samples=3 member must NOT get triple vote weight. With
        # a=[A,A,A] and b=[B] the committee has EXACTLY 2 candidates (A, B),
        # vote_margin 0.5 — NOT 4 candidates / 0.75 (which could falsely pass a
        # stat_at_least accept gate).
        comm = CompositeNode(
            name="comm",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a"), StageArm("b")),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
            ),
        )
        stages = {"a": _stage(["A", "A", "A"]), "b": _stage(["B"])}
        result = execute_composite(
            comm, stages, config={}, calibrated_values={}, registry={"comm": comm}
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.measures["candidates_evaluated"] == 2  # one per arm
        assert result.measures["vote_margin"] == pytest.approx(0.5)

    def test_committee_arm_selected_output_is_its_majority_winner(self):
        # Each multisample committee arm contributes its OWN majority winner.
        # a=[A,A,B] -> winner A; b=[C,C,D] -> winner C; committee = {A, C},
        # a 2-way tie at margin 0.5 (deterministic representative).
        comm = CompositeNode(
            name="comm",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a"), StageArm("b")),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
            ),
        )
        stages = {"a": _stage(["A", "A", "B"]), "b": _stage(["C", "C", "D"])}
        result = execute_composite(
            comm, stages, config={}, calibrated_values={}, registry={"comm": comm}
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.measures["candidates_evaluated"] == 2
        assert result.measures["vote_margin"] == pytest.approx(0.5)
        assert result.output in {"A", "C"}  # one arm's selected winner

    def test_inflated_weight_does_not_falsely_pass_accept_gate(self):
        # The bug's HARM: with the old 4-candidate / 0.75 margin a 0.7 accept
        # threshold would FALSELY pass. The correct 0.5 margin must FAIL it.
        comm = CompositeNode(
            name="comm",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a"), StageArm("b")),
                aggregate=AggregateDecl(
                    kind=AggregateKind.MAJORITY_VOTE,
                    accept=AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold="acc"),
                ),
            ),
        )
        stages = {"a": _stage(["A", "A", "A"]), "b": _stage(["B"])}
        result = execute_composite(
            comm,
            stages,
            config={},
            calibrated_values={"acc": 0.7},
            registry={"comm": comm},
        )
        # correct margin 0.5 < 0.7 -> honest no_accept (NOT a false OUTPUT)
        assert result.result_kind is ResultKind.NO_ACCEPT


# --------------------------------------------------------------------------- #
# F5: deep fail-closed preflight over ALL reachable composites                #
# --------------------------------------------------------------------------- #


class TestDeepPreflight:
    def test_missing_stage_in_unselected_nested_arm_is_error_upfront(self):
        # F5 PROBE: 'a' is unanimous so the gate STOPS at it and the nested
        # 'inner' arm is never selected — yet 'inner' references stage 'z' which
        # is not provided. The deep preflight must fail closed BEFORE any arm
        # runs, never return the root's silent output.
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("z"),), gates=()),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        result = execute_composite(
            outer,
            {"a": _stage(["A", "A", "A"])},  # 'z' (inside inner) absent
            config={},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "missing stage callable" in (result.error or "")
        assert "z" in (result.error or "")

    def test_missing_signal_in_unreached_nested_loop_is_error_upfront(self):
        # A nested signal_accept loop arm whose signal is not provided fails
        # closed upfront even if the loop arm is never selected.
        innerloop = CompositeNode(
            name="innerloop",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=StageArm("lb"),
                stop=StopDecl(
                    kind=StopKind.SIGNAL_ACCEPT,
                    threshold="lt",
                    signal=SignalUse(signal="needed_sig", inputs=()),
                ),
                state_keys=(),
                max_iters=2,
            ),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("innerloop")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        result = execute_composite(
            outer,
            {"a": _stage(["A", "A", "A"]), "lb": lambda _i: "x"},
            config={},
            calibrated_values={"t": 0.5, "lt": 0.5},
            registry={"outer": outer, "innerloop": innerloop},
            signals={},  # 'needed_sig' absent
        )
        assert result.result_kind is ResultKind.ERROR
        assert "not provided" in (result.error or "")
        assert "needed_sig" in (result.error or "")

    def test_missing_predicate_in_unreached_nested_loop_is_error_upfront(self):
        innerloop = CompositeNode(
            name="innerloop",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=StageArm("lb"),
                stop=StopDecl(kind=StopKind.EXTERNAL_ACCEPT, predicate="needed_pred"),
                state_keys=(),
                max_iters=2,
            ),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("innerloop")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        result = execute_composite(
            outer,
            {"a": _stage(["A", "A", "A"]), "lb": lambda _i: "x"},
            config={},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "innerloop": innerloop},
            predicates={},  # 'needed_pred' absent
        )
        assert result.result_kind is ResultKind.ERROR
        assert "not provided" in (result.error or "")
        assert "needed_pred" in (result.error or "")

    def test_all_present_deep_chain_executes_normally(self):
        # Sanity: when every reachable leaf/signal/predicate IS provided the
        # preflight is transparent (no spurious error).
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("z"),), gates=()),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        result = execute_composite(
            outer,
            {"a": _stage(["A", "A", "A"]), "z": lambda _i: "Z"},
            config={},
            calibrated_values={"t": 0.5},
            registry={"outer": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A"  # gate stops at the unanimous first arm


# --------------------------------------------------------------------------- #
# K-chain unroll execution (§3.8) + trace-equality property + teeth twin      #
# --------------------------------------------------------------------------- #

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from traigent.knobs.patterns import self_refine as _self_refine  # noqa: E402


def _refine_knob(max_iters):
    return _self_refine(
        "sr",
        stage="body",
        signal="quality",
        threshold="theta",
        max_iters=max_iters,
        state_keys=("draft",),
    )


def _seq_body(values):
    """A deterministic loop/chain body whose i-th invocation outputs values[i]
    and threads it on 'draft'. SAME body object convention for both the live
    loop and the K-chain so traces are comparable."""
    counter = {"i": 0}

    def run(_item, _state):
        i = counter["i"]
        counter["i"] += 1
        v = values[min(i, len(values) - 1)]
        return LoopBodyResult(output=v, state={"draft": v})

    return LoopBodyRunner(run=run)


class TestKChainExecutionBasics:
    def test_kchain_accepts_at_first_stage_meeting_theta(self):
        knob = _refine_knob(max_iters=4)
        chain = knob.unroll(4)
        body = _seq_body([0.2, 0.6, 0.95])
        result = execute_kchain(
            chain,
            body,
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": lambda s: s["draft"]},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == 0.95
        assert result.measures["iterations_used"] == 3
        assert result.measures["stop_reason"] == StopReason.SIGNAL_ACCEPT.value

    def test_kchain_exhausts_to_no_accept(self):
        knob = _refine_knob(max_iters=2)
        chain = knob.unroll(2)
        body = _seq_body([0.1, 0.2])
        result = execute_kchain(
            chain,
            body,
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": lambda s: s["draft"]},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value

    def test_kchain_missing_signal_fail_closed(self):
        chain = _refine_knob(max_iters=2).unroll(2)
        result = execute_kchain(
            chain,
            _seq_body([0.5]),
            config={},
            calibrated_values={"theta": 0.9},
            signals={},
        )
        assert result.result_kind is ResultKind.ERROR

    def test_kchain_body_error_is_error_parity_with_live_loop(self):
        # F4 PROBE: a body that reports an ERROR result-kind must ABSORB to an
        # error in BOTH executors. The live loop already does; the K-chain used
        # to treat the ERROR cell's .output as a produced output ('output bad')
        # — fail-closed parity broken. After the fix both yield ERROR.
        knob = _refine_knob(max_iters=4)

        def err_body():
            def run(_item, _state):
                # ERROR result whose state WOULD cross theta if (wrongly) read
                # as an output — exposing the K-chain's 'output bad' divergence.
                return LoopBodyResult(
                    output="bad", state={"draft": 1.0}, result_kind=ResultKind.ERROR
                )

            return LoopBodyRunner(run=run)

        def signal(s):
            return s["draft"]

        loop = execute_composite(
            knob.structure,
            {"body": err_body()},
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": signal},
        )
        chain = execute_kchain(
            knob.unroll(4),
            err_body(),
            config={},
            calibrated_values={"theta": 0.9},
            signals={"quality": signal},
        )
        assert loop.result_kind is ResultKind.ERROR
        assert chain.result_kind is ResultKind.ERROR
        assert chain.output is None  # never a produced 'output bad'


@settings(max_examples=120, deadline=None)
@given(
    k=st.integers(min_value=1, max_value=4),
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ),
    theta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_kchain_trace_equals_live_loop_under_conditions_1_3(k, values, theta):
    """§3.8 claim C5 (SDK property level): under conditions 1–3 (state flows
    exclusively through declared state_keys; the stop is a deterministic
    function of that state — structural for signal_accept; bounded by
    max_iters), the K-chain execution TRACE equals the live loop's: SAME
    result_kind, SAME selected output, SAME iterations_used, SAME stop_reason.

    The signal is a PURE deterministic function of the threaded 'draft' state
    (condition 2), the body threads state ONLY through 'draft' (condition 1),
    and both are bounded by K = max_iters (condition 3)."""
    knob = _refine_knob(max_iters=k)
    node = knob.structure
    chain = knob.unroll(k)

    def signal(s):
        return s["draft"]

    loop = execute_composite(
        node,
        {"body": _seq_body(values)},
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": signal},
    )
    chain_result = execute_kchain(
        chain,
        _seq_body(values),  # fresh body with the SAME deterministic sequence
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": signal},
    )

    assert loop.result_kind is chain_result.result_kind
    assert loop.output == chain_result.output
    assert loop.measures.get("iterations_used") == chain_result.measures.get(
        "iterations_used"
    )
    assert loop.measures.get("stop_reason") == chain_result.measures.get("stop_reason")


def _seq_body_with_error(values, error_at):
    """A deterministic body that reports an ERROR result-kind at iteration
    ``error_at`` (1-based), otherwise behaving like :func:`_seq_body`.

    The ERROR cell is the §3.2.1 absorbing error a body honestly reports — the
    live loop and the K-chain MUST absorb it identically (F4 parity). ``None``
    error_at means no error cell (a pure value body)."""
    counter = {"i": 0}

    def run(_item, _state):
        counter["i"] += 1
        i = counter["i"]  # 1-based iteration index
        v = values[min(i - 1, len(values) - 1)]
        if error_at is not None and i == error_at:
            return LoopBodyResult(
                output=v, state={"draft": v}, result_kind=ResultKind.ERROR
            )
        return LoopBodyResult(output=v, state={"draft": v})

    return LoopBodyRunner(run=run)


@settings(max_examples=120, deadline=None)
@given(
    k=st.integers(min_value=1, max_value=4),
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ),
    theta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    error_at=st.integers(min_value=1, max_value=4),
)
def test_kchain_trace_equals_live_loop_with_error_result_kind_bodies(
    k, values, theta, error_at
):
    """§3.8 claim C5 EXTENDED (reviewer non-blocker, F4): the trace-equality
    holds even when the body reports an ERROR result-kind (the §3.2.1 absorbing
    error). Whether the error cell is reached before acceptance or not, BOTH
    executors produce the SAME trace — proving fail-closed parity over the full
    result-kind codomain, not just OUTPUT/no_accept."""
    knob = _refine_knob(max_iters=k)
    node = knob.structure
    chain = knob.unroll(k)

    def signal(s):
        return s["draft"]

    loop = execute_composite(
        node,
        {"body": _seq_body_with_error(values, error_at)},
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": signal},
    )
    chain_result = execute_kchain(
        chain,
        _seq_body_with_error(values, error_at),
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": signal},
    )

    assert loop.result_kind is chain_result.result_kind
    assert loop.output == chain_result.output
    assert loop.measures.get("iterations_used") == chain_result.measures.get(
        "iterations_used"
    )
    assert loop.measures.get("stop_reason") == chain_result.measures.get("stop_reason")


def _trace_of(result):
    """The observable trace tuple the §3.8 C5 property compares (result kind,
    selected output, iterations_used, stop_reason)."""
    return (
        result.result_kind,
        result.output,
        result.measures.get("iterations_used"),
        result.measures.get("stop_reason"),
    )


def test_kchain_teeth_condition1_violation_diverges_detectably():
    """TEETH twin (model-checking discipline at runtime level).

    The trace-equality property above could be vacuous — passing because the two
    executors are SO aligned that NO body could ever make them differ. This twin
    proves it has teeth by constructing a body that VIOLATES §3.8 condition 1
    (state flows ONLY through declared state_keys) and showing the very same
    trace comparator the property uses DETECTS the divergence.

    The violation: the body's acceptance behavior depends on AMBIENT mutable
    state (a hidden counter) NOT threaded through state_keys, and — critically —
    each executor walks the body a different number of effective times before the
    ambient counter trips the signal, because the live loop and the chain
    initialize the ambient state at different points. A correct contract
    (conditions 1–3) forbids exactly this; under it the property holds. Here we
    force the violation deterministically and ASSERT the comparator fires —
    mirroring the model checker's 'add a violating transition and require it be
    SAT' discipline (the UNSAT-only proof is not enough)."""
    theta = 0.5

    # Two bodies that share ONE ambient counter (ambient mutation — the cond-1
    # violation). The signal reads the ambient counter, NOT the threaded draft.
    # Because the live loop's body and the chain's body increment the SAME shared
    # ambient across BOTH runs (no reset between them), the chain — run SECOND —
    # starts from a higher ambient value and accepts at an earlier stage, so the
    # traces diverge. This is precisely the non-determinism cond-1 rules out.
    ambient = {"hidden": 0}

    def shared_body():
        def run(_item, _state):
            ambient["hidden"] += 1
            return LoopBodyResult(output=ambient["hidden"], state={"draft": 0})

        return LoopBodyRunner(run=run)

    def ambient_signal(_restricted_state):
        # Reads ambient (undeclared) state — a cond-1 violation. >= 3 -> accept.
        return float(ambient["hidden"]) / 1.0 - 2.5  # crosses theta=0.5 at hidden=3

    knob = _refine_knob(max_iters=4)

    # NOTE: NO reset of `ambient` between the two runs — the shared ambient
    # mutation is the contract violation under test.
    loop = execute_composite(
        knob.structure,
        {"body": shared_body()},
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": ambient_signal},
    )
    chain = execute_kchain(
        knob.unroll(4),
        shared_body(),
        config={},
        calibrated_values={"theta": theta},
        signals={"quality": ambient_signal},
    )

    # The comparator the C5 property relies on MUST detect the divergence. If
    # the traces were equal here, the property would be vacuous.
    assert _trace_of(loop) != _trace_of(chain), (
        "teeth twin saw NO divergence under a condition-1 violation; the "
        "trace-equality property may be vacuous — investigate before trusting C5"
    )
    # Concretely: the live loop accepts later (lower ambient start) than the
    # chain (which begins after the loop already advanced the shared counter).
    assert loop.measures["iterations_used"] != chain.measures["iterations_used"]


# --------------------------------------------------------------------------- #
# Pre-cascade DISPATCH execution (§3.2 / §3.2.1 / §3.10) — the router path     #
# --------------------------------------------------------------------------- #


def _pre_cascade(
    arms,
    gates_signals,
    *,
    name="dispatch",
):
    """A ``placement: pre`` cascade IR node.

    ``arms`` is a tuple of stage names (or ``CompositeArm`` objects); the LAST
    is the terminal fallback (no gate). ``gates_signals`` is a tuple of
    ``(threshold, signal_name)`` pairs, one per non-terminal arm
    (``|gates| = |arms| - 1``), each a ``signal_below`` PRE gate.
    """
    arm_objs = tuple(StageArm(a) if isinstance(a, str) else a for a in arms)
    gate_objs = tuple(
        GateDecl(
            kind=GateKind.SIGNAL_BELOW,
            threshold=threshold,
            signal=SignalUse(signal=sig),
        )
        for threshold, sig in gates_signals
    )
    return CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(arms=arm_objs, gates=gate_objs, placement=Placement.PRE),
    )


class _SpyStage:
    """A stage runner that records whether it was called (route-isolation proof)."""

    def __init__(self, value):
        self.value = value
        self.called = False

    def runner(self):
        def run(_item):
            self.called = True
            return [self.value]

        return StageRunner(run=run, key_fn=lambda x: x, samples=1)


class TestPreCascadeDispatch:
    """The §3.2 pre-cascade (dispatch) law: route(x) = arm_j where
    j = min({i < m : σ_i(x) >= θ_i} ∪ {m}). FIRST-MATCH-WINS, exactly ONE arm
    executes; the routed arm is TERMINAL (its no_accept/error is final, no
    re-route)."""

    def test_routes_to_first_adequate_arm_only(self):
        # σ_0 = 0.9 >= θ_0 0.5 -> arm 0 selected; arms 1, 2 must NOT run.
        a0, a1, a2 = _SpyStage("A0"), _SpyStage("A1"), _SpyStage("A2")
        node = _pre_cascade(("a0", "a1", "a2"), (("t0", "s0"), ("t1", "s1")))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner(), "a2": a2.runner()},
            config={},
            calibrated_values={"t0": 0.5, "t1": 0.5},
            signals={"s0": lambda _x: 0.9, "s1": lambda _x: 0.9},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A0"
        assert a0.called is True
        assert a1.called is False  # first-match-wins: no other arm runs
        assert a2.called is False

    def test_fall_through_to_terminal_when_all_gates_inadequate(self):
        # σ_0, σ_1 both < θ -> terminal arm 2 selected.
        a0, a1, a2 = _SpyStage("A0"), _SpyStage("A1"), _SpyStage("A2")
        node = _pre_cascade(("a0", "a1", "a2"), (("t0", "s0"), ("t1", "s1")))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner(), "a2": a2.runner()},
            config={},
            calibrated_values={"t0": 0.5, "t1": 0.5},
            signals={"s0": lambda _x: 0.1, "s1": lambda _x: 0.1},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A2"  # terminal fallback
        assert a0.called is False
        assert a1.called is False
        assert a2.called is True
        assert result.measures["route_selected"] == 2
        # terminal fall-through: NO dispatch_signal_margin (no gated selection).
        assert "dispatch_signal_margin" not in result.measures

    def test_second_gate_adequate_when_first_inadequate(self):
        # σ_0 < θ (inadequate), σ_1 >= θ -> arm 1 selected; arm 0 never runs.
        a0, a1, a2 = _SpyStage("A0"), _SpyStage("A1"), _SpyStage("A2")
        node = _pre_cascade(("a0", "a1", "a2"), (("t0", "s0"), ("t1", "s1")))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner(), "a2": a2.runner()},
            config={},
            calibrated_values={"t0": 0.5, "t1": 0.5},
            signals={"s0": lambda _x: 0.1, "s1": lambda _x: 0.9},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A1"
        assert a0.called is False  # arm 0 never runs
        assert a1.called is True
        assert a2.called is False
        assert result.measures["route_selected"] == 1

    def test_absent_value_signal_is_inadequate_continues(self):
        # codex MINOR coverage: σ_0 returns None -> ABSTENTION. This is §3.2's
        # explicit license ("absent/abstaining signal value is INADEQUATE"):
        # route onward to arm 1, NEVER an error. Contrast the malformed-score
        # cases above (bool/non-numeric/non-finite), which DO error — None is the
        # ONLY value that legitimately abstains.
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: None},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A1"  # fell through to terminal
        assert result.measures["route_selected"] == 1
        # the gate was evaluated and recorded inadequate (abstention), not skipped.
        assert result.measures["gate_signal_adequate"] == {0: 0}
        assert a0.called is False
        assert a1.called is True

    def test_non_numeric_signal_value_is_absorbing_error(self):
        # codex B2 (REAL): a non-numeric σ value (str, dict, ...) is a MALFORMED
        # score — a defect, NOT an abstention. RFC 0002 §3.2.1 mandates finite
        # numeric comparison ("never a silent comparison"); the LOOP signal_accept
        # convention (runtime.py ~1587) RAISES on non-numeric. The PRE-gate is now
        # at parity: a non-numeric σ is an absorbing ERROR, never a silent route.
        # (Earlier this test pinned route-onward under the §3.2 "absent/abstaining
        #  value is INADEQUATE" reading; that reading mis-classified a malformed
        #  score as an abstention and is now rejected — flipped to ERROR.)
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: "not-a-number"},
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert result.measures == {}
        assert a0.called is False  # malformed-signal defect: no arm runs
        assert a1.called is False  # never silently routes onward

    def test_bool_signal_value_is_absorbing_error(self):
        # codex B2 (REAL): bool is NOT a number (parity with the §3.10 measure
        # rule AND the loop signal_accept rule at runtime.py ~1587). A bool σ is a
        # malformed score -> absorbing ERROR, never a silent route onward.
        # (Earlier the PRE-gate let a bool route onward as "non-numeric ->
        #  inadequate"; that interpretation is now rejected.)
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: True},  # bool is not a finite score
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert result.measures == {}
        assert a0.called is False
        assert a1.called is False

    @pytest.mark.parametrize(
        "bad_sigma",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_non_finite_signal_value_is_absorbing_error(self, bad_sigma):
        # codex B2 (REAL): a NON-FINITE σ (NaN/±inf) cannot satisfy the §3.2.1
        # finite-comparison law ("comparisons are over finite numbers ... never a
        # silent comparison") and the loop signal_accept convention RAISES on
        # non-finite (runtime.py ~1592). The PRE-gate now RAISES too: a non-finite
        # σ is an absorbing ERROR, NOT a silent fail-toward-the-stronger-arm route.
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: bad_sigma},
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert result.measures == {}
        assert a0.called is False  # non-finite score: no arm runs
        assert a1.called is False  # never silently routes onward

    def test_signal_that_raises_is_absorbing_error(self):
        # §3.2: a signal that RAISES fails the item's evaluation (never silently
        # routes). ERROR is absorbing; no arm executes.
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")

        def boom(_x):
            raise RuntimeError("signal blew up")

        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": boom},
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert "RuntimeError" in (result.error or "")
        assert a0.called is False  # no arm executed
        assert a1.called is False

    def test_routed_arm_no_accept_is_cascade_no_accept_no_reroute(self):
        # The routed arm yields no_accept -> the cascade's result is no_accept;
        # routing selects the arm, it does NOT re-route on non-acceptance.
        a1 = _SpyStage("A1")  # the later arm must NOT run

        def no_accept_arm(_item, _state=None):  # pragma: no cover - via runner
            return None

        # arm 0 is a nested composite that yields no_accept (an ensemble whose
        # accept gate fails); arm 1 is the terminal fallback that must NOT run.
        inner = self_consistency(
            "inner", stage="ia", cardinality="k", accept_threshold="acc"
        ).structure
        node = _pre_cascade((CompositeArm("inner"), "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {
                "ia": _stage(["X", "Y"]),  # split -> margin 0 < acc -> no_accept
                "a1": a1.runner(),
            },
            config={"k": 2},
            calibrated_values={"t0": 0.5, "acc": 0.9},
            signals={"s0": lambda _x: 0.9},  # routes to arm 0
            registry={"dispatch": node, "inner": inner},
        )
        assert result.result_kind is ResultKind.NO_ACCEPT
        assert a1.called is False  # NO re-route to the later arm
        assert result.measures["route_selected"] == 0

    def test_routed_arm_error_is_cascade_error(self):
        # The routed arm raises -> absorbing error (§3.2.1).
        def explode(_item):
            raise RuntimeError("routed arm blew up")

        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {
                "a0": StageRunner(run=explode, key_fn=lambda x: x, samples=1),
                "a1": _stage(["A1"]),
            },
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: 0.9},  # routes to arm 0
        )
        assert result.result_kind is ResultKind.ERROR
        assert "RuntimeError" in (result.error or "")
        assert result.output is None

    def test_threshold_read_live_at_decide_time(self):
        # The SAME structure routes differently purely as a function of the LIVE
        # calibrated_values — θ is read at decide time, never snapshotted.
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        stages = {"a0": _stage(["A0"]), "a1": _stage(["A1"])}
        sigs = {"s0": lambda _x: 0.6}
        adequate = execute_composite(
            node, stages, config={}, calibrated_values={"t0": 0.5}, signals=sigs
        )
        inadequate = execute_composite(
            node, stages, config={}, calibrated_values={"t0": 0.9}, signals=sigs
        )
        assert adequate.output == "A0"  # 0.6 >= 0.5 -> arm 0
        assert inadequate.output == "A1"  # 0.6 < 0.9 -> terminal arm 1

    def test_missing_calibrated_threshold_for_reached_gate_is_error(self):
        # A reached gate whose threshold CVAR is unset fails CLOSED (calibration
        # defect), exactly as POST does — never silently inadequate.
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": _stage(["A0"]), "a1": _stage(["A1"])},
            config={},
            calibrated_values={},  # 't0' absent
            signals={"s0": lambda _x: 0.9},
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None

    def test_preflight_missing_dispatch_signal_is_error_before_execution(self):
        # FAIL-CLOSED preflight: a pre-gate signal not in the registry is an
        # error BEFORE any arm/signal runs (mirrors signal_accept loop preflight).
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": 0.5},
            signals={},  # 's0' absent -> fail closed up front
        )
        assert result.result_kind is ResultKind.ERROR
        assert "not provided" in (result.error or "")
        assert "s0" in (result.error or "")
        assert a0.called is False  # nothing ran
        assert a1.called is False

    def test_preflight_missing_signal_in_unreached_nested_pre_arm_is_error(self):
        # The deep preflight reaches a nested PRE-cascade arm too: a missing
        # dispatch signal inside an unselected nested arm fails closed up front.
        inner = _pre_cascade(("ia0", "ia1"), (("it0", "needed_sig"),), name="inner")
        outer = _pre_cascade((CompositeArm("inner"), "a1"), (("t0", "s0"),))
        result = execute_composite(
            outer,
            {
                "ia0": _stage(["IA0"]),
                "ia1": _stage(["IA1"]),
                "a1": _stage(["A1"]),
            },
            config={},
            calibrated_values={"t0": 0.5, "it0": 0.5},
            signals={"s0": lambda _x: 0.1},  # 'needed_sig' absent
            registry={"dispatch": outer, "inner": inner},
        )
        assert result.result_kind is ResultKind.ERROR
        assert "needed_sig" in (result.error or "")

    def test_gated_selection_emits_dispatch_signal_margin(self):
        # A GATED selection (not terminal) carries dispatch_signal_margin =
        # σ_sel − θ_sel; gate_signal_adequate is an index-keyed adequacy map.
        node = _pre_cascade(("a0", "a1", "a2"), (("t0", "s0"), ("t1", "s1")))
        result = execute_composite(
            node,
            {"a0": _stage(["A0"]), "a1": _stage(["A1"]), "a2": _stage(["A2"])},
            config={},
            calibrated_values={"t0": 0.5, "t1": 0.5},
            signals={"s0": lambda _x: 0.2, "s1": lambda _x: 0.8},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.measures["route_selected"] == 1
        assert result.measures["dispatch_signal_margin"] == pytest.approx(0.3)
        # gate 0 evaluated inadequate (0), gate 1 evaluated adequate (1).
        assert result.measures["gate_signal_adequate"] == {0: 0, 1: 1}

    def test_telemetry_flattens_through_composite_measures_no_collision(self):
        from traigent.knobs.telemetry import (
            composite_measures,
            merge_composite_measures,
        )

        node = _pre_cascade(("a0", "a1", "a2"), (("t0", "s0"), ("t1", "s1")))
        result = execute_composite(
            node,
            {"a0": _stage(["A0"]), "a1": _stage(["A1"]), "a2": _stage(["A2"])},
            config={},
            calibrated_values={"t0": 0.5, "t1": 0.5},
            signals={"s0": lambda _x: 0.2, "s1": lambda _x: 0.8},
        )
        flat = composite_measures(result)
        assert flat["composite_route_selected"] == 1
        assert flat["composite_dispatch_signal_margin"] == pytest.approx(0.3)
        # the index-keyed gate_signal_adequate map flattens per gate index.
        assert flat["composite_gate_0_signal_adequate"] == 0
        assert flat["composite_gate_1_signal_adequate"] == 1
        # merges into a user metrics dict without collision.
        merged = merge_composite_measures({"accuracy": 0.9}, result)
        assert merged["accuracy"] == 0.9
        assert merged["composite_route_selected"] == 1

    def test_terminal_telemetry_omits_dispatch_margin_and_flattens(self):
        from traigent.knobs.telemetry import composite_measures

        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": _stage(["A0"]), "a1": _stage(["A1"])},
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: 0.1},  # inadequate -> terminal
        )
        flat = composite_measures(result)
        assert flat["composite_route_selected"] == 1
        assert "composite_dispatch_signal_margin" not in flat
        # the single evaluated gate is recorded inadequate.
        assert flat["composite_gate_0_signal_adequate"] == 0

    def test_non_finite_threshold_on_reached_gate_is_error(self):
        # codex MINOR coverage (existing behavior, previously untested): a REACHED
        # gate whose calibrated θ is NON-FINITE (NaN) is a calibration defect ->
        # absorbing ERROR, exactly as POST (_resolve_threshold fails closed). The
        # signal is fine; the defect is the threshold. Distinct from the σ-side
        # malformed-value cases — both σ and θ must be finite to compare.
        a0, a1 = _SpyStage("A0"), _SpyStage("A1")
        node = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        result = execute_composite(
            node,
            {"a0": a0.runner(), "a1": a1.runner()},
            config={},
            calibrated_values={"t0": math.nan},  # reached-gate θ non-finite
            signals={"s0": lambda _x: 0.9},  # σ is a fine finite score
        )
        assert result.result_kind is ResultKind.ERROR
        assert result.output is None
        assert result.measures == {}
        assert a0.called is False  # no arm runs on a calibration defect
        assert a1.called is False

    # ----------------------------------------------------------------------- #
    # Adjudicated codex findings — PINNED with the adjudication rationale.     #
    # The captain verified each by direct probe against POST cascades/loops    #
    # and confirmed the PRE executor matches the shipped system-wide           #
    # convention; these tests freeze that parity so a future change to one     #
    # side must consciously touch the other.                                   #
    # ----------------------------------------------------------------------- #

    def test_pre_gate_signal_sees_full_config_not_declared_inputs_subset(self):
        """codex B1 (ADJUDICATED): ``SignalUse.inputs`` is a freshness-bound
        DECLARATION, not a runtime projection.

        A pre-gate signal receives the FULL dispatch ``config`` even when its
        ``SignalUse.inputs`` declares a strict subset. This is the SAME convention
        as the loop ``signal_accept`` stop (runtime.py ~1563-1569): the signal sees
        the full restricted state; ``signal.inputs`` is a §3.5 ctx_ext
        calibration/freshness COVERAGE binding (§3.2 item 11 ``signal_inputs``),
        NOT a runtime input filter, so it never narrows what σ observes.

        ADJUDICATION: codex read RFC §3.2's prose as implying a runtime projection
        of inputs onto the signal. The captain confirmed (a) the loop does NOT
        project, and (b) projecting here would diverge from the loop and break
        §3.5 freshness binding. The RFC §3.2 wording clarification (make the
        "declaration not projection" reading explicit) is recorded as a
        documentation follow-up, NOT a runtime change. Here we pin that the signal
        reads a key OUTSIDE its declared ``inputs=("x",)`` and routing still works.
        """
        node = CompositeNode(
            name="dispatch",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a0"), StageArm("a1")),
                gates=(
                    GateDecl(
                        kind=GateKind.SIGNAL_BELOW,
                        threshold="t0",
                        # declares ONLY "x"; the signal below reads "y" too.
                        signal=SignalUse(signal="s0", inputs=("x",)),
                    ),
                ),
                placement=Placement.PRE,
            ),
        )
        seen: dict[str, Any] = {}

        def reads_undeclared_key(cfg):
            # The signal observes the FULL config, including "y" which is NOT in
            # its declared inputs=("x",). If inputs were a runtime filter, "y"
            # would be absent and this would KeyError (-> absorbing error).
            seen.update(cfg)
            return float(cfg["x"] + cfg["y"])  # 0.4 + 0.4 = 0.8 >= θ 0.5

        result = execute_composite(
            node,
            {"a0": _stage(["A0"]), "a1": _stage(["A1"])},
            config={"x": 0.4, "y": 0.4},
            calibrated_values={"t0": 0.5},
            signals={"s0": reads_undeclared_key},
        )
        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "A0"  # routed on σ=0.8 >= θ 0.5
        # PROOF the signal saw the FULL config, not the declared subset:
        assert seen == {"x": 0.4, "y": 0.4}
        assert "y" in seen  # the undeclared key was visible

    def test_routed_arm_error_measures_empty_parity_pre_and_post(self):
        """codex M3 (ADJUDICATED): a routed/escalated arm ERROR returns
        ``measures == {}`` — and this is PARITY with the POST cascade.

        Captain probe: a POST cascade whose escalated arm RAISES also returns an
        ``error`` result with ``measures == {}`` (``_error`` carries no telemetry,
        runtime.py ~201). The PRE executor does the SAME on a routed-arm error.

        ADJUDICATION: codex flagged "no telemetry on the error path" as a defect.
        The captain confirmed it is the shipped system-wide convention (errors are
        absorbing and content-free). We pin BOTH PRE and POST in ONE test so any
        future "emit telemetry on error" change is forced to touch — and re-justify
        — both placements together rather than silently diverging them.
        """

        def explode(_item):
            raise RuntimeError("arm blew up")

        # --- PRE: route to a raising arm 0. ---
        pre = _pre_cascade(("a0", "a1"), (("t0", "s0"),))
        pre_result = execute_composite(
            pre,
            {
                "a0": StageRunner(run=explode, key_fn=lambda x: x, samples=1),
                "a1": _stage(["A1"]),
            },
            config={},
            calibrated_values={"t0": 0.5},
            signals={"s0": lambda _x: 0.9},  # routes to arm 0 (the raiser)
        )
        assert pre_result.result_kind is ResultKind.ERROR
        assert pre_result.measures == {}  # no route_selected, no adequacy map

        # --- POST: escalate into a raising arm 1 (the captain's parity probe). ---
        post = CompositeNode(
            name="post",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        post_result = execute_composite(
            post,
            {
                "a": _stage(["x", "y"]),  # margin 0.5 < t 0.9 -> escalate to b
                "b": StageRunner(run=explode, key_fn=lambda x: x, samples=1),
            },
            config={},
            calibrated_values={"t": 0.9},
        )
        assert post_result.result_kind is ResultKind.ERROR
        assert post_result.measures == {}  # PARITY: POST also drops telemetry

    def test_nested_composite_measures_do_not_propagate_parity_pre_and_post(self):
        """codex M4 (ADJUDICATED): a nested composite's measures do NOT propagate
        to the outer run — and this is PARITY with the POST cascade.

        Captain probe: a POST cascade that escalates into a nested ensemble arm
        emits only the OUTER cascade measures (``stage_selected`` etc.); the inner
        ensemble's ``vote_margin``/``samples_used`` never surface on the outer
        result (``_cascade_measures`` builds outer-only telemetry, runtime.py
        ~738). The PRE executor matches: an outer PRE cascade routing to a NESTED
        PRE cascade arm emits only the OUTER ``route_selected`` — the inner's
        ``route_selected`` / ``gate_signal_adequate`` are absent.

        ADJUDICATION: codex flagged the missing inner telemetry as lost
        observability. The captain confirmed outer-only measures are the shipped
        convention for nested composites across PRE and POST (telemetry is
        per-decided-composite, not recursively merged). We pin BOTH placements so
        a future "bubble up nested telemetry" change must touch both.
        """
        # --- PRE: outer routes to a NESTED PRE-cascade arm. ---
        inner = _pre_cascade(("ia0", "ia1"), (("it0", "is0"),), name="inner")
        outer = _pre_cascade((CompositeArm("inner"), "a1"), (("t0", "s0"),))
        pre_result = execute_composite(
            outer,
            {
                "ia0": _stage(["IA0"]),
                "ia1": _stage(["IA1"]),
                "a1": _stage(["A1"]),
            },
            config={},
            calibrated_values={"t0": 0.5, "it0": 0.5},
            # outer s0 routes to arm 0 (inner); inner is0 routes to its arm 0.
            signals={"s0": lambda _x: 0.9, "is0": lambda _x: 0.9},
            registry={"dispatch": outer, "inner": inner},
        )
        assert pre_result.result_kind is ResultKind.OUTPUT
        assert pre_result.output == "IA0"  # inner's routed arm output
        # OUTER telemetry only: the outer route_selected is present...
        assert pre_result.measures["route_selected"] == 0
        # ...and the inner's measures do NOT bleed up. The inner ALSO selected
        # route 0, so route_selected==0 is ambiguous; instead assert the inner's
        # gate adequacy map is exactly the OUTER's single-gate map (inner had its
        # own gate too — if inner telemetry merged, the map would differ/collide).
        assert pre_result.measures["gate_signal_adequate"] == {0: 1}
        assert "dispatch_signal_margin" in pre_result.measures  # outer gated route

        # --- POST: outer escalates into a nested ensemble arm (parity probe). ---
        inner_ens = self_consistency("inner_ens", stage="ie", cardinality="k").structure
        post = CompositeNode(
            name="post",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner_ens")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        post_result = execute_composite(
            post,
            {
                "a": _stage(["x", "y"]),  # margin 0.5 < t 0.9 -> escalate
                "ie": _stage(["W", "W"]),  # nested unanimous -> output W
            },
            config={"k": 2},
            calibrated_values={"t": 0.9},
            registry={"post": post, "inner_ens": inner_ens},
        )
        assert post_result.result_kind is ResultKind.OUTPUT
        assert post_result.output == "W"  # nested ensemble's output
        # OUTER-only telemetry: cascade keys present, NO ensemble keys (the inner
        # majority_vote's vote_margin / samples_used never surface).
        assert post_result.measures["stage_selected"] == 1
        assert "vote_margin" not in post_result.measures
        assert "samples_used" not in post_result.measures
