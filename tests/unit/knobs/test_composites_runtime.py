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
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    GateDecl,
    GateKind,
    Placement,
    SignalUse,
    StageArm,
)
from traigent.knobs.patterns import binary_cascade, self_consistency, self_debug
from traigent.knobs.runtime import (
    CompositeRunResult,
    ResultKind,
    StageRunner,
    execute_composite,
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
        assert result.measures["gate_margin_pass_rate"] == {GATE: 1.0}

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
        assert result.measures["gate_margin_pass_rate"] == {GATE: 0.0}

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
# Deferred kinds raise loudly (honest, never faked)                           #
# --------------------------------------------------------------------------- #


class TestDeferredKinds:
    def test_nested_composite_arm_raises_not_implemented(self):
        node = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), CompositeArm("inner")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
                placement=Placement.POST,
            ),
        )
        with pytest.raises(NotImplementedError, match="nested-composite"):
            execute_composite(
                node,
                {"a": _stage(["x"])},
                config={},
                calibrated_values={"t": 0.5},
            )

    def test_ensemble_root_raises_not_implemented(self):
        node = self_consistency(
            "sc", stage="a", cardinality="k", accept_threshold="acc"
        ).structure
        with pytest.raises(NotImplementedError, match="ensemble"):
            execute_composite(node, {}, config={}, calibrated_values={})

    def test_loop_root_raises_not_implemented(self):
        node = self_debug("sd", stage="a", predicate="tests", max_iters=2).structure
        with pytest.raises(NotImplementedError, match="loop"):
            execute_composite(node, {}, config={}, calibrated_values={})

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
