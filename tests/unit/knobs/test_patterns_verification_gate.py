"""Verification-gate catalog tests.

This file is deliberately self-contained: it replicates the local golden-hash
convention from ``test_pattern_factories.py`` without changing that fixture.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields

import pytest

from traigent.knobs import canonical_hash
from traigent.knobs.composites import (
    CascadeBody,
    CompositeArm,
    CompositeProgram,
    EnsembleBody,
    LoopBody,
    SignalUse,
    StageArm,
    StopDecl,
    StopKind,
    validate_program,
)
from traigent.knobs.patterns import self_refine, verification_gate
from traigent.knobs.runtime import (
    LoopBodyResult,
    LoopBodyRunner,
    ResultKind,
    StopReason,
    execute_composite,
)

_THRESHOLD = "verifier_pass_threshold"
_SIGNAL = "verifier_pass_score"
_STAGE = "generate_verify_revise"
_STATE_KEYS = (
    "draft",
    "verification_questions",
    "verification_answers",
    "verifier_pass_score",
    "contradiction_score",
    "revision",
    "independent_context",
)
_SIGNAL_INPUTS = ("draft", "verification_answers", "independent_context")
_TVARS = frozenset(
    {
        "verification_style",
        "verification_question_count",
        "verifier_model",
        "independent_context",
        "revision_policy",
    }
)


# Byte-stable serializer (the golden-expansion oracle) ---------------------- #


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


def _scope_to_dict(scope):
    if scope is None:
        return None
    return {"node": scope.node, "agent": scope.agent, "workflow": scope.workflow}


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
        "scope": _scope_to_dict(node.scope),
        "provenance": _prov_to_dict(node.provenance),
    }


def _golden_hash(node) -> str:
    return canonical_hash(_node_to_dict(node))


def _vg(**overrides):
    params = {
        "name": "qa_verified",
        "stage": _STAGE,
        "verifier_signal": _SIGNAL,
        "verifier_pass_threshold": _THRESHOLD,
        "verification_style": "verification_style",
        "verification_question_count": "verification_question_count",
        "verifier_model": "verifier_model",
        "independent_context": "independent_context",
        "revision_policy": "revision_policy",
        "max_iters": 2,
    }
    params.update(overrides)
    return verification_gate(**params)


def _program(**overrides) -> CompositeProgram:
    ck = _vg()
    params = {
        "composites": {ck.name: ck.structure},
        "tvars": _TVARS,
        "cvars": frozenset({_THRESHOLD}),
        "cvar_types": {_THRESHOLD: "float"},
        "cvar_signals": {_THRESHOLD: _SIGNAL},
        "cvar_depends_on": {_THRESHOLD: _TVARS},
        "cvar_signal_inputs": {_THRESHOLD: _SIGNAL_INPUTS},
    }
    params.update(overrides)
    return CompositeProgram(**params)


def _code(exc: pytest.ExceptionInfo[ValueError]) -> str:
    return str(exc.value).split(":", 1)[0]


def _signal(state):
    return state[_SIGNAL]


def _verified_state(
    *,
    draft: str,
    score: float,
    revision: str,
    contradiction_score: float = 0.1,
) -> dict[str, object]:
    return {
        "draft": draft,
        "verification_questions": 3,
        "verification_answers": f"checked:{draft}",
        "verifier_pass_score": score,
        "contradiction_score": contradiction_score,
        "revision": revision,
        "independent_context": "offline_reference",
    }


# Golden expansion and P8 canary -------------------------------------------- #


GOLDEN_VERIFICATION_GATE = "19ca7c1aae7c56cb4c3c274619cd029f02066f21b8179dbf082d0b1b671936a9"  # pragma: allowlist secret


def test_golden_hash_pinned_and_deterministic():
    actual = _golden_hash(_vg().structure)
    assert actual == GOLDEN_VERIFICATION_GATE, actual
    assert _golden_hash(_vg().structure) == _golden_hash(_vg().structure)


def test_expands_to_signal_accept_loop_family():
    ck = _vg()
    body = ck.structure.body

    assert ck.telemetry_names == ("iterations_used", "stop_reason")
    assert body.body == StageArm(
        _STAGE,
        (
            "verification_style",
            "verification_question_count",
            "verifier_model",
            "independent_context",
            "revision_policy",
        ),
    )
    assert body.stop.kind is StopKind.SIGNAL_ACCEPT
    assert body.stop.threshold == _THRESHOLD
    assert body.stop.signal == SignalUse(signal=_SIGNAL, inputs=_SIGNAL_INPUTS)
    assert body.state_keys == _STATE_KEYS
    assert body.max_iters == 2


def test_p8_canary_raw_verifier_params_never_serialize_into_provenance():
    sentinel = "SECRET-CANARY-verifier-param"
    ck = _vg(
        verification_style=sentinel,
        verifier_model=f"{sentinel}:model",
        revision_policy=f"{sentinel}:revision",
    )

    prov_blob = json.dumps(_prov_to_dict(ck.structure.provenance), sort_keys=True)
    assert sentinel not in prov_blob
    assert ck.structure.provenance.param_hash is not None
    assert sentinel not in ck.structure.provenance.param_hash
    assert canonical_hash(_prov_to_dict(ck.structure.provenance))


# Factory and program validation -------------------------------------------- #


def test_contradiction_score_max_rejects_at_factory_boundary():
    with pytest.raises(ValueError, match=r"^unsupported_compound_stop"):
        _vg(contradiction_score_max="contradiction_score_max")


def test_max_iters_is_literal_int_not_tuned_structural_bound():
    with pytest.raises(ValueError, match="invalid_max_iters"):
        _vg(max_iters="max_repair_rounds")  # type: ignore[arg-type]


def test_validate_program_accepts_well_bound_gate():
    validate_program(_program())


def test_validate_program_rejects_missing_calibration_signal():
    with pytest.raises(ValueError) as exc:
        validate_program(_program(cvar_signals={_THRESHOLD: None}))
    assert _code(exc) == "missing_calibration_signal"


def test_validate_program_rejects_signal_mismatch():
    with pytest.raises(ValueError) as exc:
        validate_program(_program(cvar_signals={_THRESHOLD: "other_signal"}))
    assert _code(exc) == "signal_mismatch"


def test_validate_program_rejects_unbound_signal_inputs():
    with pytest.raises(ValueError) as exc:
        validate_program(_program(cvar_signal_inputs={_THRESHOLD: None}))
    assert _code(exc) == "unbound_signal_inputs"


def test_validate_program_rejects_missing_tvar_parent_coverage():
    with pytest.raises(ValueError) as exc:
        validate_program(
            _program(
                cvar_depends_on={
                    _THRESHOLD: frozenset(_TVARS - {"revision_policy"})
                }
            )
        )
    assert _code(exc) == "missing_composite_parent"


# Runtime behavior ----------------------------------------------------------- #


def test_runtime_accepts_on_first_iteration():
    seen_states: list[dict[str, object]] = []

    def body(_config, state):
        seen_states.append(dict(state))
        return LoopBodyResult(
            output="draft-1",
            state=_verified_state(draft="draft-1", score=0.94, revision="none"),
        )

    result = execute_composite(
        _vg().structure,
        {_STAGE: LoopBodyRunner(run=body)},
        config={},
        calibrated_values={_THRESHOLD: 0.9},
        signals={_SIGNAL: _signal},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "draft-1"
    assert result.measures["iterations_used"] == 1
    assert result.measures["stop_reason"] == StopReason.SIGNAL_ACCEPT.value
    assert seen_states == [{}]


def test_runtime_revises_then_accepts_on_second_iteration():
    calls = {"n": 0}
    seen_states: list[dict[str, object]] = []

    def body(_config, state):
        seen_states.append(dict(state))
        calls["n"] += 1
        if calls["n"] == 1:
            return LoopBodyResult(
                output="draft-0",
                state=_verified_state(
                    draft="draft-0", score=0.3, revision="needs_revision"
                ),
            )
        assert state["draft"] == "draft-0"
        return LoopBodyResult(
            output="draft-1",
            state=_verified_state(draft="draft-1", score=0.93, revision="revised"),
        )

    result = execute_composite(
        _vg().structure,
        {_STAGE: LoopBodyRunner(run=body)},
        config={},
        calibrated_values={_THRESHOLD: 0.9},
        signals={_SIGNAL: _signal},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "draft-1"
    assert result.measures["iterations_used"] == 2
    assert result.measures["stop_reason"] == StopReason.SIGNAL_ACCEPT.value
    assert seen_states[0] == {}
    assert seen_states[1]["revision"] == "needs_revision"


def test_runtime_exhausts_to_no_accept():
    scores = iter([0.2, 0.4])

    def body(_config, _state):
        score = next(scores)
        return LoopBodyResult(
            output=f"draft-{score}",
            state=_verified_state(
                draft=f"draft-{score}", score=score, revision="still_low"
            ),
        )

    result = execute_composite(
        _vg().structure,
        {_STAGE: LoopBodyRunner(run=body)},
        config={},
        calibrated_values={_THRESHOLD: 0.9},
        signals={_SIGNAL: _signal},
    )

    assert result.result_kind is ResultKind.NO_ACCEPT
    assert result.output is None
    assert result.measures["iterations_used"] == 2
    assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value


# bounded_refine_loop recipe boundary --------------------------------------- #


def test_bounded_refine_loop_recipe_threads_improvement_state_to_signal():
    stage_tuned_params = (
        "critic_model",
        "feedback_rubric",
        "repair_prompt",
        "max_repair_rounds",
        "stop_condition",
    )
    signal_inputs = ("draft", "score", "previous_score", "improvement_delta")
    ck = self_refine(
        name="bounded_refine_loop",
        stage="critique_repair",
        signal="refine_accept_score",
        threshold="acceptance_threshold",
        max_iters=3,
        state_keys=(
            "draft",
            "critique",
            "score",
            "previous_score",
            "improvement_delta",
            "round",
        ),
        signal_inputs=signal_inputs,
        stage_tuned_params=stage_tuned_params,
    )
    validate_program(
        CompositeProgram(
            composites={ck.name: ck.structure},
            tvars=frozenset(stage_tuned_params),
            cvars=frozenset({"acceptance_threshold"}),
            cvar_types={"acceptance_threshold": "float"},
            cvar_signals={"acceptance_threshold": "refine_accept_score"},
            cvar_depends_on={"acceptance_threshold": frozenset(stage_tuned_params)},
            cvar_signal_inputs={"acceptance_threshold": signal_inputs},
        )
    )

    scores = iter([0.4, 0.86])
    signal_views: list[dict[str, float]] = []

    def body(_config, state):
        previous = float(state.get("score", 0.0))
        score = next(scores)
        return LoopBodyResult(
            output=f"draft:{score}",
            state={
                "draft": f"draft:{score}",
                "critique": "stubbed critique",
                "score": score,
                "previous_score": previous,
                "improvement_delta": score - previous,
                "round": int(state.get("round", 0)) + 1,
            },
        )

    def refine_accept_score(state):
        signal_views.append(
            {
                "score": state["score"],
                "previous_score": state["previous_score"],
                "improvement_delta": state["improvement_delta"],
            }
        )
        return state["score"]

    result = execute_composite(
        ck.structure,
        {"critique_repair": LoopBodyRunner(run=body)},
        config={},
        calibrated_values={"acceptance_threshold": 0.8},
        signals={"refine_accept_score": refine_accept_score},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "draft:0.86"
    assert result.measures["stop_reason"] == StopReason.SIGNAL_ACCEPT.value
    assert signal_views[1]["score"] == pytest.approx(0.86)
    assert signal_views[1]["previous_score"] == pytest.approx(0.4)
    assert signal_views[1]["improvement_delta"] == pytest.approx(0.46)


def test_dual_threshold_acceptance_plus_improvement_stop_is_not_expressible():
    threshold_fields = [
        field.name for field in fields(StopDecl) if "threshold" in field.name
    ]
    assert threshold_fields == ["threshold"]
    assert "improvement_threshold" not in inspect.signature(StopDecl).parameters

    with pytest.raises(TypeError):
        StopDecl(
            kind=StopKind.SIGNAL_ACCEPT,
            threshold="acceptance_threshold",
            signal=SignalUse(signal="refine_accept_score"),
            improvement_threshold="improvement_min_delta",  # type: ignore[call-arg]
        )
