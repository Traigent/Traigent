"""react_tool_loop pattern tests (RFC 0002 loop-family catalog).

This file is intentionally self-contained. It copies the local golden
serializer convention from test_pattern_factories.py instead of editing that
shared test module.
"""

from __future__ import annotations

import json

import pytest

from traigent.knobs import Calibrated, Knob, Ref, SignalSpec, TargetProperty
from traigent.knobs.canonical import canonical_hash
from traigent.knobs.composites import (
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeProgram,
    EnsembleBody,
    LoopBody,
    StageArm,
    cal_fold,
    validate_program,
)
from traigent.knobs.patterns import react_tool_loop
from traigent.knobs.runtime import (
    LoopBodyResult,
    LoopBodyRunner,
    ResultKind,
    StopReason,
    execute_composite,
)

# --------------------------------------------------------------------------- #
# Byte-stable serializer (local copy of the golden-expansion oracle)          #
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


# --------------------------------------------------------------------------- #
# Shared fixtures                                                             #
# --------------------------------------------------------------------------- #


STAGE = "react_tool_step"
SIGNAL = "tool_confidence"
THRESHOLD = "tool_confidence_min"
STAGE_TVARS = (
    "planner_style",
    "tool_allowlist",
    "observation_format",
    "failure_handler",
)
STATE_KEYS = (
    "scratchpad",
    "tool_calls",
    "observations",
    "confidence",
    "last_error",
)
GOLDEN_REACT_TOOL_LOOP = "5f881392fed4ed0231445b40007cd859f6c72084f97fab4561982affe5ad4aad"  # pragma: allowlist secret


def _code(excinfo: pytest.ExceptionInfo[ValueError]) -> str:
    return str(excinfo.value).split(":", 1)[0]


def _react_loop(**overrides):
    args = {
        "name": "qa_react",
        "stage": STAGE,
        "signal": SIGNAL,
        "tool_confidence_min": THRESHOLD,
        "max_tool_calls": 4,
    }
    args.update(overrides)
    return react_tool_loop(**args)


def _program(ck, **overrides) -> CompositeProgram:
    base = {
        "composites": {ck.name: ck.structure},
        "tvars": frozenset(STAGE_TVARS),
        "cvars": frozenset({THRESHOLD}),
        "cvar_types": {THRESHOLD: "float"},
        "cvar_signals": {THRESHOLD: SIGNAL},
        "cvar_depends_on": {THRESHOLD: frozenset(STAGE_TVARS)},
        "cvar_signal_inputs": {THRESHOLD: STATE_KEYS},
    }
    base.update(overrides)
    return CompositeProgram(**base)


def _signal_spec(name: str) -> SignalSpec:
    return SignalSpec(
        name=name,
        version="1",
        score_function="offline_stub",
        score_function_version="1",
        comparator="gte",
        comparator_version="1",
    )


def _target() -> TargetProperty:
    return TargetProperty(name="loop_stop_adequacy", mode="require_calibration")


# --------------------------------------------------------------------------- #
# Golden expansion + P8 provenance canary                                     #
# --------------------------------------------------------------------------- #


class TestGoldenExpansion:
    def test_golden_hash_pinned(self):
        ck = _react_loop()

        assert _golden_hash(ck.structure) == GOLDEN_REACT_TOOL_LOOP

    def test_exact_loop_shape_and_telemetry(self):
        ck = _react_loop()
        body = ck.structure.body

        assert ck.kind is CompositeKind.LOOP
        assert isinstance(body, LoopBody)
        assert body.body == StageArm(STAGE, STAGE_TVARS)
        assert body.max_iters == 4
        assert body.state_keys == STATE_KEYS
        assert body.stop.kind.value == "signal_accept"
        assert body.stop.threshold == THRESHOLD
        assert body.stop.signal.signal == SIGNAL
        assert body.stop.signal.inputs == STATE_KEYS
        assert ck.structure.provenance.pattern == "react_tool_loop"
        assert ck.provenance == ck.structure.provenance
        assert ck.telemetry_names == ("iterations_used", "stop_reason")


class TestP8Canary:
    def test_raw_params_never_serialize_into_provenance(self):
        sentinel = "SECRET-CANARY-react-tool-loop-d4f1c0de"
        ck = react_tool_loop(
            "qa_react",
            stage=sentinel,
            signal=SIGNAL,
            tool_confidence_min=THRESHOLD,
            max_tool_calls=2,
            stage_tuned_params=(sentinel + "_tvar",),
        )

        prov_blob = json.dumps(_prov_to_dict(ck.structure.provenance), sort_keys=True)
        assert sentinel not in prov_blob
        assert ck.structure.provenance.param_hash is not None
        assert sentinel not in ck.structure.provenance.param_hash
        assert canonical_hash(_prov_to_dict(ck.structure.provenance))


# --------------------------------------------------------------------------- #
# Validation rejects delegated to the sealed IR/program checks                #
# --------------------------------------------------------------------------- #


class TestValidationRejects:
    @pytest.mark.parametrize("max_tool_calls", [0, True, "4"])
    def test_invalid_max_tool_calls_rejects(self, max_tool_calls):
        with pytest.raises(ValueError) as e:
            _react_loop(max_tool_calls=max_tool_calls)

        assert _code(e) == "invalid_max_iters"

    def test_empty_threshold_rejects_missing_stop_threshold(self):
        with pytest.raises(ValueError) as e:
            _react_loop(tool_confidence_min="")

        assert _code(e) == "missing_stop_threshold"

    def test_signal_inputs_must_be_within_state_keys(self):
        with pytest.raises(ValueError) as e:
            _react_loop(signal_inputs=("observations", "unknown"))

        assert _code(e) == "stop_signal_outside_state"

    def test_default_program_validates(self):
        validate_program(_program(_react_loop()))

    def test_missing_threshold_ref_rejects(self):
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    _react_loop(),
                    cvars=frozenset(),
                    cvar_types={},
                    cvar_signals={},
                    cvar_depends_on={},
                    cvar_signal_inputs={},
                )
            )

        assert _code(e) == "missing_ref"

    def test_threshold_must_be_numeric_cvar(self):
        with pytest.raises(ValueError) as e:
            validate_program(_program(_react_loop(), cvar_types={THRESHOLD: "string"}))

        assert _code(e) == "invalid_threshold_type"

    def test_missing_calibration_signal_rejects(self):
        with pytest.raises(ValueError) as e:
            validate_program(_program(_react_loop(), cvar_signals={THRESHOLD: None}))

        assert _code(e) == "missing_calibration_signal"

    def test_signal_mismatch_rejects(self):
        with pytest.raises(ValueError) as e:
            validate_program(_program(_react_loop(), cvar_signals={THRESHOLD: "other"}))

        assert _code(e) == "signal_mismatch"

    def test_unbound_signal_inputs_rejects(self):
        with pytest.raises(ValueError) as e:
            validate_program(_program(_react_loop(), cvar_signal_inputs={}))

        assert _code(e) == "unbound_signal_inputs"

    def test_invalid_stage_tuned_param_rejects(self):
        ck = _react_loop(stage_tuned_params=("planner_style", "ghost_tvar"))

        with pytest.raises(ValueError) as e:
            validate_program(_program(ck))

        assert _code(e) == "invalid_tuned_param"

    def test_body_tvar_must_be_covered_by_threshold_parentage(self):
        with pytest.raises(ValueError) as e:
            validate_program(
                _program(
                    _react_loop(),
                    cvar_depends_on={
                        THRESHOLD: frozenset(
                            {
                                "planner_style",
                                "tool_allowlist",
                                "observation_format",
                            }
                        )
                    },
                )
            )

        assert _code(e) == "missing_composite_parent"


# --------------------------------------------------------------------------- #
# Runtime behavior over execute_composite                                     #
# --------------------------------------------------------------------------- #


def _react_step(confidences: list[float]) -> LoopBodyRunner:
    counter = {"i": 0}

    def run(_item, state):
        i = counter["i"]
        counter["i"] += 1
        confidence = confidences[min(i, len(confidences) - 1)]
        tool_calls = [*state.get("tool_calls", ()), f"tool-{i + 1}"]
        observations = [*state.get("observations", ()), f"obs-{i + 1}"]
        return LoopBodyResult(
            output=f"answer-{i + 1}",
            state={
                "scratchpad": f"thought-{i + 1}",
                "tool_calls": tool_calls,
                "observations": observations,
                "confidence": confidence,
                "last_error": None,
                "undeclared": "dropped",
            },
        )

    return LoopBodyRunner(run=run)


def _tool_confidence(state):
    return state["confidence"]


class TestRuntime:
    def test_accepts_on_iteration_n_and_emits_loop_telemetry_only(self):
        ck = _react_loop(max_tool_calls=4)

        result = execute_composite(
            ck.structure,
            {STAGE: _react_step([0.2, 0.6, 0.91])},
            config={},
            calibrated_values={THRESHOLD: 0.9},
            signals={SIGNAL: _tool_confidence},
        )

        assert result.result_kind is ResultKind.OUTPUT
        assert result.output == "answer-3"
        assert result.measures == {
            "iterations_used": 3,
            "stop_reason": StopReason.SIGNAL_ACCEPT.value,
        }
        assert tuple(result.measures) == ck.telemetry_names

    def test_exhaustion_returns_no_accept(self):
        ck = _react_loop(max_tool_calls=2)

        result = execute_composite(
            ck.structure,
            {STAGE: _react_step([0.2, 0.6])},
            config={},
            calibrated_values={THRESHOLD: 0.9},
            signals={SIGNAL: _tool_confidence},
        )

        assert result.result_kind is ResultKind.NO_ACCEPT
        assert result.output is None
        assert result.measures == {
            "iterations_used": 2,
            "stop_reason": StopReason.EXHAUSTED.value,
        }


# --------------------------------------------------------------------------- #
# Cost-cap exclusion guard                                                    #
# --------------------------------------------------------------------------- #


def test_tool_cost_cap_is_documented_not_active_in_current_loop_algebra():
    tool_cost_cap = Knob(
        name="tool_cost_cap",
        binding=Calibrated(
            signal=_signal_spec("tool_cost"),
            target=_target(),
            depends_on=(Ref(knob="planner_style"),),
        ),
    )
    ck = _react_loop(members={"tool_cost_cap": tool_cost_cap})
    program = _program(
        ck,
        cvars=frozenset({THRESHOLD, "tool_cost_cap"}),
        cvar_types={THRESHOLD: "float", "tool_cost_cap": "float"},
        cvar_signals={THRESHOLD: SIGNAL, "tool_cost_cap": "tool_cost"},
        cvar_depends_on={
            THRESHOLD: frozenset(STAGE_TVARS),
            "tool_cost_cap": frozenset({"planner_style"}),
        },
        cvar_signal_inputs={THRESHOLD: STATE_KEYS},
    )

    validate_program(program)
    assert "tool_cost_cap" in ck.members
    assert cal_fold(program) == frozenset({THRESHOLD})
    assert "tool_cost_cap" not in json.dumps(_node_to_dict(ck.structure))

    result = execute_composite(
        ck.structure,
        {STAGE: _react_step([0.1, 0.2, 0.3, 0.4])},
        config={},
        calibrated_values={THRESHOLD: 0.9, "tool_cost_cap": 0.0},
        signals={SIGNAL: _tool_confidence},
    )

    assert result.result_kind is ResultKind.NO_ACCEPT
    assert result.measures["iterations_used"] == 4
    assert result.measures["stop_reason"] == StopReason.EXHAUSTED.value
