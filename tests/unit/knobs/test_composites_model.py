"""Composite IR model tests (RFC 0002 P5, §3.2/§3.5/§3.6/§3.11).

Red-first: every structural rejection is asserted by its EXACT §3.11
error-code message prefix (``code: detail``), so the test pins the diagnostic
contract, never just "some ValueError". The derived functions ``leaf_tvars``,
``required_parents``, ``cal_fold``, and ``roots`` are pinned against the §3.5 /
§3.6 definitions including the sampling-ensemble tuned-cardinality term.
"""

from __future__ import annotations

import pytest

from traigent.knobs.composites import (
    AggregateDecl,
    AggregateKind,
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    CompositeProgram,
    EnsembleBody,
    GateDecl,
    GateKind,
    LoopBody,
    Placement,
    Scope,
    SignalUse,
    StageArm,
    StopDecl,
    StopKind,
    cal_fold,
    leaf_tvars,
    required_parents,
    roots,
    validate_program,
)


def _code(excinfo: pytest.ExceptionInfo[ValueError]) -> str:
    """The §3.11 code prefix of a rejection message (text before the colon)."""
    return str(excinfo.value).split(":", 1)[0]


def _post_cascade(threshold: str = "theta") -> CompositeNode:
    return CompositeNode(
        name="casc",
        kind=CompositeKind.CASCADE,
        body=CascadeBody(
            arms=(StageArm("a"), StageArm("b")),
            gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold=threshold),),
            placement=Placement.POST,
        ),
    )


def _program(node: CompositeNode, **ns) -> CompositeProgram:
    base = {
        "composites": {node.name: node},
        "tvars": frozenset(),
        "cvars": frozenset({"theta"}),
        "cvar_types": {"theta": "float"},
        "cvar_signals": {"theta": "vote_margin"},
        "cvar_depends_on": {"theta": frozenset()},
        "cvar_signal_inputs": {},
    }
    base.update(ns)
    return CompositeProgram(**base)


# --------------------------------------------------------------------------- #
# Node-local rejections (checked on construction)                            #
# --------------------------------------------------------------------------- #


class TestNodeLocalRejections:
    def test_unknown_composite_kind(self):
        with pytest.raises(ValueError) as e:
            CompositeNode(name="x", kind="quantum", body=CascadeBody(arms=(StageArm("a"),)))  # type: ignore[arg-type]
        assert _code(e) == "unknown_composite_kind"

    def test_unknown_composite_field_kind_body_mismatch(self):
        with pytest.raises(ValueError) as e:
            CompositeNode(
                name="x",
                kind=CompositeKind.LOOP,
                body=CascadeBody(arms=(StageArm("a"),)),
            )
        assert _code(e) == "unknown_composite_field"

    def test_cascade_arity(self):
        with pytest.raises(ValueError) as e:
            CascadeBody(arms=(StageArm("a"), StageArm("b")), gates=())
        assert _code(e) == "cascade_arity"

    def test_empty_arms(self):
        with pytest.raises(ValueError) as e:
            CascadeBody(arms=())
        assert _code(e) == "empty_arms"

    def test_empty_arms_ensemble(self):
        with pytest.raises(ValueError) as e:
            EnsembleBody(
                arms=(), aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE)
            )
        assert _code(e) == "empty_arms"

    def test_duplicate_stage(self):
        with pytest.raises(ValueError) as e:
            CascadeBody(
                arms=(StageArm("a"), StageArm("a")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
            )
        assert _code(e) == "duplicate_stage"

    def test_duplicate_stage_includes_ensemble_judge(self):
        # F4b (§3.2 item 6): a judge stage that duplicates a committee arm is a
        # duplicate WITHIN one composite — the judge is in the stage scope.
        with pytest.raises(ValueError) as e:
            EnsembleBody(
                arms=(StageArm("a"), StageArm("b")),
                aggregate=AggregateDecl(
                    kind=AggregateKind.JUDGE_MAX, judge=StageArm("a")
                ),
            )
        assert _code(e) == "duplicate_stage"

    def test_sampling_ensemble_judge_duplicates_sole_arm(self):
        # The single-arm (sampling) form, too: judge == the arm is a duplicate.
        with pytest.raises(ValueError) as e:
            EnsembleBody(
                arms=(StageArm("a"),),
                aggregate=AggregateDecl(
                    kind=AggregateKind.JUDGE_MAX, judge=StageArm("a")
                ),
                cardinality="k",
            )
        assert _code(e) == "duplicate_stage"

    def test_distinct_judge_stage_is_accepted(self):
        # A judge with a DISTINCT name is well-formed (the happy path).
        body = EnsembleBody(
            arms=(StageArm("a"),),
            aggregate=AggregateDecl(
                kind=AggregateKind.JUDGE_MAX, judge=StageArm("judge")
            ),
            cardinality="k",
        )
        assert body.aggregate.judge == StageArm("judge")

    def test_empty_external_accept_predicate_rejects(self):
        # F5 (§3.11 missing_stop_predicate): an empty predicate identifier is a
        # missing predicate, not a valid one (previously only None was caught).
        with pytest.raises(ValueError) as e:
            StopDecl(kind=StopKind.EXTERNAL_ACCEPT, predicate="")
        assert _code(e) == "missing_stop_predicate"

    def test_empty_signal_accept_threshold_rejects(self):
        # F5 sweep: the same empty-string hole on the signal_accept threshold
        # Ident — an empty threshold is a missing threshold.
        with pytest.raises(ValueError) as e:
            StopDecl(
                kind=StopKind.SIGNAL_ACCEPT,
                threshold="",
                signal=SignalUse(signal="q"),
            )
        assert _code(e) == "missing_stop_threshold"

    def test_unknown_gate_kind(self):
        with pytest.raises(ValueError) as e:
            GateDecl(kind="signal_above", threshold="theta")  # type: ignore[arg-type]
        assert _code(e) == "unknown_gate_kind"

    def test_missing_gate_signal(self):
        with pytest.raises(ValueError) as e:
            GateDecl(kind=GateKind.SIGNAL_BELOW, threshold="theta")
        assert _code(e) == "missing_gate_signal"

    def test_gate_kind_placement_mismatch_margin_in_pre(self):
        with pytest.raises(ValueError) as e:
            CascadeBody(
                arms=(StageArm("a"), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
                placement=Placement.PRE,
            )
        assert _code(e) == "gate_kind_placement_mismatch"

    def test_gate_kind_placement_mismatch_signal_in_post(self):
        with pytest.raises(ValueError) as e:
            CascadeBody(
                arms=(StageArm("a"), StageArm("b")),
                gates=(
                    GateDecl(
                        kind=GateKind.SIGNAL_BELOW,
                        threshold="theta",
                        signal=SignalUse(signal="router_sig"),
                    ),
                ),
                placement=Placement.POST,
            )
        assert _code(e) == "gate_kind_placement_mismatch"

    def test_cardinality_arity_mismatch_missing_on_single_arm(self):
        with pytest.raises(ValueError) as e:
            EnsembleBody(
                arms=(StageArm("a"),),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
            )
        assert _code(e) == "cardinality_arity_mismatch"

    def test_cardinality_arity_mismatch_present_on_committee(self):
        with pytest.raises(ValueError) as e:
            EnsembleBody(
                arms=(StageArm("a"), StageArm("b")),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
                cardinality="k",
            )
        assert _code(e) == "cardinality_arity_mismatch"

    def test_invalid_max_iters(self):
        with pytest.raises(ValueError) as e:
            LoopBody(
                body=StageArm("a"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                max_iters=0,
            )
        assert _code(e) == "invalid_max_iters"

    def test_stop_signal_outside_state(self):
        with pytest.raises(ValueError) as e:
            LoopBody(
                body=StageArm("a"),
                stop=StopDecl(
                    kind=StopKind.SIGNAL_ACCEPT,
                    threshold="theta",
                    signal=SignalUse(signal="q", inputs=("not_a_state_key",)),
                ),
                state_keys=("draft",),
                max_iters=2,
            )
        assert _code(e) == "stop_signal_outside_state"

    def test_missing_stop_threshold(self):
        with pytest.raises(ValueError) as e:
            StopDecl(kind=StopKind.SIGNAL_ACCEPT, signal=SignalUse(signal="q"))
        assert _code(e) == "missing_stop_threshold"

    def test_missing_stop_signal(self):
        with pytest.raises(ValueError) as e:
            StopDecl(kind=StopKind.SIGNAL_ACCEPT, threshold="theta")
        assert _code(e) == "missing_stop_signal"

    def test_missing_stop_predicate(self):
        with pytest.raises(ValueError) as e:
            StopDecl(kind=StopKind.EXTERNAL_ACCEPT)
        assert _code(e) == "missing_stop_predicate"

    def test_unknown_stop_kind(self):
        with pytest.raises(ValueError) as e:
            StopDecl(kind="give_up")  # type: ignore[arg-type]
        assert _code(e) == "unknown_stop_kind"

    def test_missing_judge(self):
        with pytest.raises(ValueError) as e:
            AggregateDecl(kind=AggregateKind.JUDGE_MAX)
        assert _code(e) == "missing_judge"

    def test_unknown_aggregate_kind(self):
        with pytest.raises(ValueError) as e:
            AggregateDecl(kind="weighted_mean")  # type: ignore[arg-type]
        assert _code(e) == "unknown_aggregate_kind"

    def test_invalid_signal_use_missing_signal(self):
        with pytest.raises(ValueError) as e:
            SignalUse(signal="")
        assert _code(e) == "invalid_signal_use"

    def test_invalid_signal_use_non_tuple_inputs(self):
        with pytest.raises(ValueError) as e:
            SignalUse(signal="q", inputs=["a"])  # type: ignore[arg-type]
        assert _code(e) == "invalid_signal_use"

    def test_invalid_arm_shape_bad_tuned_params(self):
        with pytest.raises(ValueError) as e:
            StageArm("a", tuned_params=("ok", ""))
        assert _code(e) == "invalid_arm_shape"

    def test_gate_arm_incompatible_stage_case_is_accepted(self):
        # A stage arm IS margin-bearing — post + margin_below over a stage is OK.
        node = _post_cascade()
        assert node.kind is CompositeKind.CASCADE


class TestScopeCarryOver:
    """F3 (§3.2 / §4): a composite carries the RFC 0001 scope_spec verbatim."""

    def test_scope_optional_default_none(self):
        node = _post_cascade()
        assert node.scope is None

    def test_scope_carried_verbatim(self):
        # node/agent/workflow ownership fields, same shape as PolicyDecl.scope.
        scope = Scope(node="n1", agent="agentA", workflow="wfX")
        node = CompositeNode(
            name="casc",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a"), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
            ),
            scope=scope,
        )
        assert node.scope is scope
        assert node.scope.node == "n1"
        assert node.scope.agent == "agentA"
        assert node.scope.workflow == "wfX"
        # A program with a scoped composite still validates (scope is metadata).
        validate_program(_program(node))

    def test_scope_partial_fields(self):
        # agent-only / workflow-only are expressible (RFC 0001 alternation).
        assert Scope(agent="a").agent == "a"
        assert Scope(workflow="w").node is None

    def test_scope_empty_field_rejects(self):
        with pytest.raises(ValueError) as e:
            Scope(node="")
        assert _code(e) == "invalid_arm_shape"

    def test_non_scope_object_rejects(self):
        with pytest.raises(ValueError) as e:
            CompositeNode(
                name="casc",
                kind=CompositeKind.CASCADE,
                body=CascadeBody(arms=(StageArm("a"),)),
                scope="agentA",  # type: ignore[arg-type]
            )
        assert _code(e) == "unknown_composite_field"


# --------------------------------------------------------------------------- #
# Program-level (cross-composite) rejections                                  #
# --------------------------------------------------------------------------- #


class TestProgramRejections:
    def test_missing_composite_ref(self):
        loop = CompositeNode(
            name="L",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("ghost"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                max_iters=2,
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(CompositeProgram(composites={"L": loop}))
        assert _code(e) == "missing_composite_ref"

    def test_composite_cycle(self):
        a = CompositeNode(
            name="A",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("B"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                max_iters=2,
            ),
        )
        b = CompositeNode(
            name="B",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("A"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                max_iters=2,
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(CompositeProgram(composites={"A": a, "B": b}))
        assert _code(e) == "composite_cycle"

    def test_invalid_tuned_param(self):
        node = CompositeNode(
            name="casc",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a", ("ghost_tvar",)), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(_program(node))  # tvars is empty -> ghost_tvar unknown
        assert _code(e) == "invalid_tuned_param"

    def test_missing_ref_threshold_not_a_cvar(self):
        node = _post_cascade(threshold="not_declared")
        with pytest.raises(ValueError) as e:
            validate_program(_program(node))
        assert _code(e) == "missing_ref"

    def test_invalid_threshold_type(self):
        node = _post_cascade()
        with pytest.raises(ValueError) as e:
            validate_program(_program(node, cvar_types={"theta": "string"}))
        assert _code(e) == "invalid_threshold_type"

    def _ensemble_cardinality(self) -> CompositeNode:
        return CompositeNode(
            name="ens",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a"),),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
                cardinality="k",
            ),
        )

    def test_invalid_cardinality_type_cvar_float(self):
        # §3.2 item 7: a CALIBRATED (CVAR) cardinality declared float, not int.
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    self._ensemble_cardinality(),
                    cvars=frozenset({"k"}),
                    cvar_types={"k": "float"},  # not int
                    cvar_signals={},
                    cvar_depends_on={},
                )
            )
        assert _code(e) == "invalid_cardinality_type"

    def test_invalid_cardinality_type_tvar_float(self):
        # F2 (§3.2 item 7): a TUNED (TVAR) cardinality must ALSO be int-typed;
        # a float TVAR cardinality rejects invalid_cardinality_type (previously
        # any name in program.tvars was accepted unchecked).
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    self._ensemble_cardinality(),
                    tvars=frozenset({"k"}),
                    tvar_types={"k": "float"},  # not int
                    cvars=frozenset(),
                    cvar_types={},
                    cvar_signals={},
                    cvar_depends_on={},
                )
            )
        assert _code(e) == "invalid_cardinality_type"

    def test_int_typed_tvar_cardinality_passes(self):
        # An int-typed TVAR cardinality is well-formed (the happy path).
        validate_program(
            _program(
                self._ensemble_cardinality(),
                tvars=frozenset({"k"}),
                tvar_types={"k": "int"},
                cvars=frozenset(),
                cvar_types={},
                cvar_signals={},
                cvar_depends_on={},
            )
        )

    def test_missing_calibration_signal(self):
        node = _post_cascade()
        with pytest.raises(ValueError) as e:
            validate_program(_program(node, cvar_signals={"theta": None}))
        assert _code(e) == "missing_calibration_signal"

    def test_signal_mismatch(self):
        node = _post_cascade()
        with pytest.raises(ValueError) as e:
            # margin_below determines sig=vote_margin; declaring a different one rejects.
            validate_program(_program(node, cvar_signals={"theta": "other_signal"}))
        assert _code(e) == "signal_mismatch"

    def test_unbound_signal_inputs(self):
        node = CompositeNode(
            name="L",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=StageArm("a"),
                stop=StopDecl(
                    kind=StopKind.SIGNAL_ACCEPT,
                    threshold="theta",
                    signal=SignalUse(signal="q", inputs=("draft",)),
                ),
                state_keys=("draft",),
                max_iters=2,
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    node,
                    cvar_signals={"theta": "q"},
                    cvar_signal_inputs={},  # inputs not covered
                )
            )
        assert _code(e) == "unbound_signal_inputs"

    def test_gate_arm_incompatible_nested_pre_cascade(self):
        # A margin_below gate gating a nested PRE-cascade composite (yields a
        # route, not a vote margin) is incompatible (item 5).
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("p"), StageArm("q")),
                gates=(
                    GateDecl(
                        kind=GateKind.SIGNAL_BELOW,
                        threshold="route_t",
                        signal=SignalUse(signal="route_sig"),
                    ),
                ),
                placement=Placement.PRE,
            ),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(CompositeArm("inner"), StageArm("z")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
                placement=Placement.POST,
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(
                CompositeProgram(
                    composites={"inner": inner, "outer": outer},
                    cvars=frozenset({"theta", "route_t"}),
                    cvar_types={"theta": "float", "route_t": "float"},
                    cvar_signals={"theta": "vote_margin", "route_t": "route_sig"},
                    cvar_depends_on={"theta": frozenset(), "route_t": frozenset()},
                )
            )
        assert _code(e) == "gate_arm_incompatible"

    def test_missing_composite_parent(self):
        # A stage declares tuned_params [m]; the gate over it must depend_on m.
        node = CompositeNode(
            name="casc",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a", ("m",)), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
            ),
        )
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    node,
                    tvars=frozenset({"m"}),
                    cvar_depends_on={"theta": frozenset()},  # missing m
                )
            )
        assert _code(e) == "missing_composite_parent"

    def test_parent_coverage_satisfied_passes(self):
        node = CompositeNode(
            name="casc",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a", ("m",)), StageArm("b")),
                gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold="theta"),),
            ),
        )
        validate_program(
            _program(
                node,
                tvars=frozenset({"m"}),
                cvar_depends_on={"theta": frozenset({"m"})},
            )
        )


# --------------------------------------------------------------------------- #
# Derived functions (§3.5 leafT / required_parents, §3.6 Cal/roots)           #
# --------------------------------------------------------------------------- #


class TestDerivedFunctions:
    def test_leaf_tvars_of_stage(self):
        prog = CompositeProgram(composites={}, tvars=frozenset({"m", "t"}))
        assert leaf_tvars(prog, StageArm("a", ("m", "t"))) == frozenset({"m", "t"})

    def test_leaf_tvars_folds_nested_sampling_cardinality(self):
        # §3.5: a sampling ensemble whose TUNED cardinality is k folds k into
        # leafT, recursively through nesting.
        inner = CompositeNode(
            name="ens",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a", ("m",)),),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
                cardinality="k",
            ),
        )
        prog = CompositeProgram(composites={"ens": inner}, tvars=frozenset({"m", "k"}))
        leaves = leaf_tvars(prog, CompositeArm("ens"))
        assert leaves == frozenset({"m", "k"})

    def test_leaf_tvars_public_never_returns_a_calibrated_cardinality(self):
        # F1 (§3.5 / C6 / P5 boundary): when a sampling ensemble's cardinality
        # is bound CALIBRATED (a CVAR, not a TVAR), the PUBLIC leaf_tvars must
        # NOT fold it in — it is a Cal member, never a TVAR leaf. The stage's
        # own tuned param stays.
        inner = CompositeNode(
            name="ens",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a", ("m",)),),
                aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE),
                cardinality="kc",  # a CALIBRATED cardinality (in N_C, not N_T)
            ),
        )
        prog = CompositeProgram(
            composites={"ens": inner},
            tvars=frozenset({"m"}),  # kc is NOT a TVAR
            cvars=frozenset({"kc"}),
        )
        # The calibrated cardinality kc is intersected out of the leaf set ...
        assert leaf_tvars(prog, CompositeArm("ens")) == frozenset({"m"})
        # ... but it IS a Cal-fold member (still certificate-covered, §3.6).
        assert "kc" in cal_fold(prog)

    def test_required_parents_post_cascade_prefix_union(self):
        node = CompositeNode(
            name="casc",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(
                arms=(StageArm("a", ("m1",)), StageArm("b", ("m2",)), StageArm("c")),
                gates=(
                    GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t1"),
                    GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t2"),
                ),
            ),
        )
        prog = CompositeProgram(
            composites={"casc": node}, tvars=frozenset({"m1", "m2"})
        )
        rp = required_parents(prog, node)
        # θ_1 covers leafT(a_1); θ_2 covers leafT(a_1) ∪ leafT(a_2).
        assert rp["t1"] == frozenset({"m1"})
        assert rp["t2"] == frozenset({"m1", "m2"})

    def test_cal_fold_collects_gate_thresholds_over_roots(self):
        casc = _post_cascade()
        prog = _program(casc)
        assert cal_fold(prog) == frozenset({"theta"})

    def test_cal_fold_descends_into_nested_judge_and_accept(self):
        inner = CompositeNode(
            name="best",
            kind=CompositeKind.ENSEMBLE,
            body=EnsembleBody(
                arms=(StageArm("a"),),
                aggregate=AggregateDecl(
                    kind=AggregateKind.JUDGE_MAX, judge=StageArm("judge")
                ),
                cardinality="kc",  # a CALIBRATED cardinality -> a Cal member
            ),
        )
        outer = CompositeNode(
            name="loop",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("best"),
                stop=StopDecl(
                    kind=StopKind.SIGNAL_ACCEPT,
                    threshold="stop_t",
                    signal=SignalUse(signal="q"),
                ),
                max_iters=2,
            ),
        )
        prog = CompositeProgram(
            composites={"best": inner, "loop": outer},
            cvars=frozenset({"kc", "stop_t"}),
        )
        # roots = {loop} (best is nested); Cal folds the loop stop threshold +
        # the nested ensemble's calibrated cardinality.
        assert roots(prog) == frozenset({"loop"})
        assert cal_fold(prog) == frozenset({"kc", "stop_t"})

    def test_roots_excludes_nested_composites(self):
        inner = CompositeNode(
            name="inner",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=StageArm("a"), stop=StopDecl(kind=StopKind.EXHAUSTED), max_iters=1
            ),
        )
        outer = CompositeNode(
            name="outer",
            kind=CompositeKind.LOOP,
            body=LoopBody(
                body=CompositeArm("inner"),
                stop=StopDecl(kind=StopKind.EXHAUSTED),
                max_iters=1,
            ),
        )
        prog = CompositeProgram(composites={"inner": inner, "outer": outer})
        assert roots(prog) == frozenset({"outer"})

    def test_m1_degenerate_cascade_no_gates(self):
        node = CompositeNode(
            name="solo",
            kind=CompositeKind.CASCADE,
            body=CascadeBody(arms=(StageArm("only"),), gates=()),
        )
        prog = CompositeProgram(composites={"solo": node})
        validate_program(prog)
        assert cal_fold(prog) == frozenset()
