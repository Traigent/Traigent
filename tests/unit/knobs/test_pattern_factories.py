"""Pattern catalog v1 tests (RFC 0002 P5, §3.7/§3.8/§3.10).

Four admission-contract obligations are exercised here:

(a) GOLDEN EXPANSION — a byte-stable canonical serialization (``_node_to_dict``
    + ``canonical_hash``) of each factory's expansion, pinned as a known-answer
    constant; the same hash must reproduce across runs (determinism, §3.7
    admission item 5);
(b) the P8 CANARY — a sentinel string passed as a pattern param must NEVER
    appear in the serialized provenance/IR (only ``param_hash`` rides; §3.7,
    §7);
(c) FAIL-CLOSED coverage — a composite whose gate-threshold CVAR lacks a
    CERTIFIED certificate yields a non-empty coverage gap from a ``cal_fold``
    walk; issuing a real certificate (the resolver-test fixture style) closes
    it (§3.6 fold, admission item 6);
(d) UNROLL — ``self_refine.unroll(3)`` golden K-chain; ``self_debug.unroll``
    raises (§3.8).
"""

from __future__ import annotations

import json

import pytest

from traigent.api.parameter_ranges import Choices
from traigent.knobs import (
    Calibrated,
    CertificateDecision,
    Fixed,
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
    CompositeProgram,
    EnsembleBody,
    LoopBody,
    StageArm,
    cal_fold,
)
from traigent.knobs.patterns import (
    best_of_n,
    binary_cascade,
    n_cascade,
    self_consistency,
    self_debug,
    self_refine,
)

# --------------------------------------------------------------------------- #
# Byte-stable serializer (the golden-expansion oracle)                        #
# --------------------------------------------------------------------------- #


def _arm_to_dict(arm):
    if isinstance(arm, StageArm):
        return {
            "tag": "stage",
            "name": arm.name,
            "tuned_params": list(arm.tuned_params),
        }
    if isinstance(arm, CompositeArm):
        return {"tag": "composite", "ref": arm.ref}
    raise TypeError(f"unknown arm type: {type(arm).__name__}")


def _sig_to_dict(sig):
    if sig is None:
        return None
    return {"signal": sig.signal, "inputs": list(sig.inputs)}


def _prov_to_dict(prov):
    if prov is None:
        return None
    return {
        "pattern": prov.pattern,
        "pattern_version": prov.pattern_version,
        "param_hash": prov.param_hash,
        "node_path": prov.node_path,
    }


def _body_to_dict(body):
    if isinstance(body, CascadeBody):
        return {
            "type": "cascade",
            "placement": body.placement.value,
            "arms": [_arm_to_dict(a) for a in body.arms],
            "gates": [
                {
                    "kind": g.kind.value,
                    "threshold": g.threshold,
                    "signal": _sig_to_dict(g.signal),
                }
                for g in body.gates
            ],
        }
    if isinstance(body, EnsembleBody):
        agg = body.aggregate
        return {
            "type": "ensemble",
            "cardinality": body.cardinality,
            "arms": [_arm_to_dict(a) for a in body.arms],
            "aggregate": {
                "kind": agg.kind.value,
                "judge": _arm_to_dict(agg.judge) if agg.judge is not None else None,
                "accept": (
                    {
                        "kind": agg.accept.kind,
                        "stat": agg.accept.stat.value,
                        "threshold": agg.accept.threshold,
                    }
                    if agg.accept is not None
                    else None
                ),
            },
        }
    if isinstance(body, LoopBody):
        return {
            "type": "loop",
            "body": _arm_to_dict(body.body),
            "state_keys": list(body.state_keys),
            "max_iters": body.max_iters,
            "stop": {
                "kind": body.stop.kind.value,
                "threshold": body.stop.threshold,
                "signal": _sig_to_dict(body.stop.signal),
                "predicate": body.stop.predicate,
            },
        }
    raise TypeError(f"unknown body type: {type(body).__name__}")


def _node_to_dict(node):
    return {
        "name": node.name,
        "kind": node.kind.value,
        "body": _body_to_dict(node.body),
        "provenance": _prov_to_dict(node.provenance),
    }


def _golden_hash(node) -> str:
    return canonical_hash(_node_to_dict(node))


# Canonical factory invocations used by the golden tests below.
def _bc():
    return binary_cascade(
        "answerer",
        base_stage="cheap",
        expert_stage="strong",
        threshold="router.margin",
        base_tuned_params=("cheap_model", "temperature"),
        expert_tuned_params=("strong_model",),
    )


def _nc():
    return n_cascade(
        "chain",
        stages=("a", "b", "c"),
        thresholds=("t1", "t2"),
        tuned_params=(("m1",), ("m2",), ()),
    )


def _sc():
    return self_consistency(
        "sc",
        stage="gen",
        cardinality="k",
        accept_threshold="acc_t",
        stage_tuned_params=("m",),
    )


def _bon():
    return best_of_n("bon", stage="gen", judge_stage="judge", cardinality="k")


def _sd():
    return self_debug("coder", stage="gen", predicate="run_tests", max_iters=3)


def _sr():
    return self_refine(
        "refiner",
        stage="gen",
        signal="quality",
        threshold="stop_t",
        max_iters=4,
        state_keys=("draft",),
    )


# --------------------------------------------------------------------------- #
# (a) Golden expansion known-answers                                          #
# --------------------------------------------------------------------------- #

# NOTE: these are deterministic SHA-256 known-answer fixtures over the IR
# expansion (the §3.7 golden tests), NOT secrets — detect-secrets flags them as
# high-entropy hex, so each carries an allowlist pragma.
GOLDEN = {
    "binary_cascade": "353abb8c42b5fab1f32738ec4f0fca76215f4fc7122e6dd9363e11d1389cd055",  # pragma: allowlist secret
    "n_cascade": "52c8ace634c47ad28768e67ddb4b05d7a054d7febd82f0c3c678b1b766e42849",  # pragma: allowlist secret
    "self_consistency": "18f0c814758112bed8ad3a9817c864420aa5a0d6d4f2d474a5529ae18f7b513d",  # pragma: allowlist secret
    "best_of_n": "a0e959620ca6c58099f2fa53762530f565446348ba3045edf23cdbf9b8b7fbae",  # pragma: allowlist secret
    "self_debug": "c816be0383cdf46b81aa6ca4bdea27070503d062580cf46c7832685e0c94e979",  # pragma: allowlist secret
    "self_refine": "00e7ef9438a34f6347b7d894d900a147e5cd0ff95e45ae1779bf60c754512fa4",  # pragma: allowlist secret
}


class TestGoldenExpansion:
    @pytest.mark.parametrize(
        "name,factory",
        [
            ("binary_cascade", _bc),
            ("n_cascade", _nc),
            ("self_consistency", _sc),
            ("best_of_n", _bon),
            ("self_debug", _sd),
            ("self_refine", _sr),
        ],
    )
    def test_golden_hash_pinned(self, name, factory):
        assert _golden_hash(factory().structure) == GOLDEN[name]

    def test_expansion_is_deterministic_across_two_builds(self):
        # §3.7 admission item 5: expand is a pure, total function of validated
        # params — two builds yield byte-identical expansions.
        assert _golden_hash(_bc().structure) == _golden_hash(_bc().structure)
        assert _golden_hash(_sr().structure) == _golden_hash(_sr().structure)

    def test_telemetry_names_per_kind(self):
        assert _bc().telemetry_names == (
            "escalation_rate",
            "stage_selected",
            "gate_margin_pass_rate",
        )
        assert _bon().telemetry_names == (
            "vote_agreement",
            "vote_margin",
            "candidates_evaluated",
            "candidates_excluded",
        )
        assert _sr().telemetry_names == ("iterations_used", "stop_reason")

    def test_provenance_pattern_name_stamped(self):
        assert _bc().structure.provenance.pattern == "binary_cascade"
        assert _sr().structure.provenance.pattern == "self_refine"


# --------------------------------------------------------------------------- #
# (b) The P8 canary                                                           #
# --------------------------------------------------------------------------- #

_SENTINEL = "SECRET-CANARY-d4f1c0de-leak-detector"


class TestP8Canary:
    def test_sentinel_param_never_appears_in_serialized_provenance(self):
        # The sentinel rides in a pattern param (a stage name + a tuned param);
        # only its canonical HASH may appear in provenance, never the raw value.
        ck = binary_cascade(
            "answerer",
            base_stage=_SENTINEL,
            expert_stage="strong",
            threshold=_SENTINEL + "_cvar",
            base_tuned_params=(_SENTINEL + "_tvar",),
        )
        # The serialized PROVENANCE blob must not contain the sentinel string;
        # the closed shape carries only identifiers + the param hash.
        prov_blob = json.dumps(_prov_to_dict(ck.structure.provenance), sort_keys=True)
        assert _SENTINEL not in prov_blob
        # The full IR node serialization is allowed to carry structural
        # identifiers (stage names are content-free ids); but PROVENANCE never
        # carries the raw param — param_hash is present and is a hash.
        assert ck.structure.provenance.param_hash is not None
        assert _SENTINEL not in ck.structure.provenance.param_hash
        # canonical_hash of the provenance succeeds (closed, hashable shape).
        assert canonical_hash(_prov_to_dict(ck.structure.provenance))

    def test_param_hash_changes_with_params_but_stays_content_free(self):
        a = binary_cascade(
            "x", base_stage="cheap", expert_stage="strong", threshold="t"
        ).structure.provenance.param_hash
        b = binary_cascade(
            "x", base_stage="OTHER", expert_stage="strong", threshold="t"
        ).structure.provenance.param_hash
        assert a != b  # the hash is sensitive to params
        assert "cheap" not in a and "OTHER" not in b  # but never reveals them


# --------------------------------------------------------------------------- #
# (c) Fail-closed coverage under the §3.6 fold                                #
# --------------------------------------------------------------------------- #


def _signal_spec() -> SignalSpec:
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


def _ctx(cvar: str) -> FreshnessContext:
    return FreshnessContext(
        cvar_name=cvar,
        tuned_parent_values=(),
        calibration_source_id="pool_a",
        signal_spec_hash=_signal_spec().spec_hash(),
        calibrator_id="budget_threshold",
        calibrator_version="1",
        calibrator_params_hash=canonical_hash({}),
        dataset_hash="ds_v1",
        evidence_n=20,
        calibration_split="cal",
        eval_split="eval",
        target=_target(),
    )


def _coverage_gap(
    program: CompositeProgram, certified: dict[str, object]
) -> frozenset[str]:
    """The §3.6 fold made operational: the member CVARs whose certificate is
    absent or not CERTIFIED. A non-empty result is a fail-closed gap (no
    certified selection)."""
    needed = cal_fold(program)
    covered = {
        cvar
        for cvar, cert in certified.items()
        if cert is not None and cert.valid_for(cvar, "float", 0.5, _ctx(cvar))
    }
    return frozenset(needed - covered)


class TestFailClosedCoverage:
    def test_uncertified_gate_threshold_yields_coverage_gap(self):
        ck = binary_cascade(
            "answerer", base_stage="cheap", expert_stage="strong", threshold="theta"
        )
        program = CompositeProgram(
            composites={"answerer": ck.structure},
            cvars=frozenset({"theta"}),
            cvar_types={"theta": "float"},
            cvar_signals={"theta": "vote_margin"},
            cvar_depends_on={"theta": frozenset()},
        )
        # No certificate issued -> the fold reports theta as an uncertified gap.
        gap = _coverage_gap(program, certified={"theta": None})
        assert gap == frozenset({"theta"})  # fail-closed: non-empty gap

    def test_certified_gate_threshold_closes_the_gap(self):
        ck = binary_cascade(
            "answerer", base_stage="cheap", expert_stage="strong", threshold="theta"
        )
        program = CompositeProgram(
            composites={"answerer": ck.structure},
            cvars=frozenset({"theta"}),
            cvar_types={"theta": "float"},
            cvar_signals={"theta": "vote_margin"},
            cvar_depends_on={"theta": frozenset()},
        )
        # Issue a REAL certificate for theta (resolver-test fixture style).
        cert = issue_certificate("theta", "float", 0.5, _ctx("theta"))
        assert cert.decision is CertificateDecision.CERTIFIED
        gap = _coverage_gap(program, certified={"theta": cert})
        assert gap == frozenset()  # positive case: fully covered

    def test_members_dict_carries_real_bindings_not_composites(self):
        # The catalog returns DECLARATIONS: members are Tuned/Calibrated/Fixed
        # Knobs the caller spreads in — a composite never binds a value.
        theta = Knob(
            name="theta",
            binding=Calibrated(
                signal=_signal_spec(),
                target=_target(),
                depends_on=(Ref(knob="cheap_model"),),
            ),
        )
        cheap_model = Knob(name="cheap_model", binding=Tuned(range=Choices(["a", "b"])))
        seed = Knob(name="seed", binding=Fixed(value=7))
        ck = binary_cascade(
            "answerer",
            base_stage="cheap",
            expert_stage="strong",
            threshold="theta",
            base_tuned_params=("cheap_model",),
            members={"theta": theta, "cheap_model": cheap_model, "seed": seed},
        )
        assert ck.members["theta"].is_calibrated()
        assert ck.members["cheap_model"].is_tuned()
        assert ck.members["seed"].is_fixed()


# --------------------------------------------------------------------------- #
# (d) Unroll (§3.8)                                                           #
# --------------------------------------------------------------------------- #


class TestUnroll:
    def test_self_refine_unroll_golden(self):
        chain = _sr().unroll(3)
        assert chain.signal == "quality"
        assert chain.threshold == "stop_t"
        assert chain.state_keys == ("draft",)
        assert len(chain.stages) == 3
        # Each stage escalates on the acceptance-direction complement σ < θ.
        assert [s.index for s in chain.stages] == [1, 2, 3]
        assert chain.stages[0].escalate_when == "quality(state_1) < stop_t"
        assert chain.stages[2].escalate_when == "quality(state_3) < stop_t"
        # Byte-stable golden over the chain stage descriptors (known-answer).
        chain_blob = canonical_hash(
            [
                {
                    "index": s.index,
                    "escalate_when": s.escalate_when,
                    "body": s.body.name,
                }
                for s in chain.stages
            ]
        )
        # Deterministic known-answer over the K-chain stage descriptors.
        assert (
            chain_blob
            == "be1d95f953848df99bbd6972a5ab85b08f13b306782f67732f9d4ae8bbc16908"  # pragma: allowlist secret
        )

    def test_self_refine_unroll_is_deterministic(self):
        def blob():
            return canonical_hash(
                [
                    {"index": s.index, "escalate_when": s.escalate_when}
                    for s in _sr().unroll(3).stages
                ]
            )

        assert blob() == blob()

    def test_self_debug_unroll_raises(self):
        with pytest.raises(ValueError, match="signal_accept"):
            _sd().unroll(3)

    def test_non_loop_unroll_raises(self):
        with pytest.raises(ValueError, match="loop composites"):
            _bc().unroll(2)

    def test_unroll_k_must_be_positive(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            _sr().unroll(0)
