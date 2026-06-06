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
    """The two genuinely-deferred surfaces (pre-cascade dispatch + a
    nested-composite LOOP body) raise NotImplementedError naming the gap.

    Every OTHER former deferral (ensemble/loop roots, nested-composite cascade
    AND ensemble arms, the K-chain unroll) is now executed end-to-end — proven
    by the execution test classes below.
    """

    def test_pre_cascade_dispatch_raises_not_implemented(self):
        node = CompositeNode(
            name="dispatch",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), StageArm("b")),
                gates=(
                    GateDecl(
                        kind=GateKind.SIGNAL_BELOW,
                        threshold="t",
                        signal=SignalUse(signal="router"),
                    ),
                ),
                placement=Placement.PRE,
            ),
        )
        with pytest.raises(NotImplementedError, match="pre-cascade"):
            execute_composite(
                node,
                {"a": _stage(["x"]), "b": _stage(["y"])},
                config={},
                calibrated_values={"t": 0.5},
            )

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
        from traigent.knobs.patterns import n_cascade

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
        quality = lambda state: state["draft"]
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
        quality = lambda state: state["draft"]
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
    signal = lambda s: s["draft"]

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
