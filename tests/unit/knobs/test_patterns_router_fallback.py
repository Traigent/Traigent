"""router/fallback catalog tests (pre-dispatch + no_accept fallback).

This file is self-contained. It replicates the local golden-hash serializer
instead of editing the shared ``test_pattern_factories.py`` fixture.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from traigent.knobs import canonical_hash
from traigent.knobs.composites import (
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeProgram,
    EnsembleBody,
    LoopBody,
    Placement,
    StageArm,
    validate_program,
)
from traigent.knobs.patterns import (
    binary_cascade,
    fallback,
    n_cascade,
    router,
    self_consistency,
)
from traigent.knobs.runtime import ResultKind, StageRunner, execute_composite

POST_TELEMETRY = (
    "escalation_rate",
    "stage_selected",
    "gate_margin_pass_rate",
)
PRE_TELEMETRY = (
    "route_selected",
    "dispatch_signal_margin",
    "gate_signal_adequate",
)
ROUTER_TVARS = frozenset(
    {
        "retrieval_mode",
        "query_complexity_strategy",
        "retrieval_k",
        "reranker",
        "web_fallback",
        "decompose_query",
    }
)
ROUTER_GATE_PARENTS = frozenset(
    {"retrieval_mode", "query_complexity_strategy", "retrieval_k"}
)


# Byte-stable serializer (local copy of the golden-expansion oracle) -------- #


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


def _stage(outputs: list[str]) -> StageRunner:
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


class _SpyStage:
    def __init__(self, value: str) -> None:
        self.value = value
        self.called = False

    def runner(self) -> StageRunner:
        def run(_item):
            self.called = True
            return [self.value]

        return StageRunner(run=run, key_fn=lambda x: x, samples=1)


def _router():
    return router(
        "adaptive_rag_gate",
        arms=("rag_light", "rag_heavy"),
        signals=("query_light_adequacy",),
        thresholds=("complexity_threshold",),
        tuned_params=(
            ("retrieval_mode", "query_complexity_strategy", "retrieval_k"),
            (
                "retrieval_mode",
                "query_complexity_strategy",
                "retrieval_k",
                "reranker",
                "web_fallback",
                "decompose_query",
            ),
        ),
        signal_inputs=(("question",),),
    )


def _fallback():
    return fallback(
        "llm_provider_fallback",
        arms=("primary", "secondary", "last_resort"),
        thresholds=("fallback_margin_floor", "fallback_margin_floor"),
        tuned_params=(("primary_model",), ("secondary_model",), ()),
    )


GOLDEN = {
    "router": "33004b6c6d6ea398498ede7da76a5a6a15aaed0403b48661ddd1baf8b176e999",  # pragma: allowlist secret
    "fallback": "2ac8b027c0f1a91443f31b3dd9762d7e05b3cf50d877d749e1b81f6fb4643f6e",  # pragma: allowlist secret
}


def test_router_golden_hash_pinned_and_expands_to_pre_cascade() -> None:
    knob = _router()
    body = knob.structure.body

    assert _golden_hash(knob.structure) == GOLDEN["router"]
    assert knob.structure.kind is CompositeKind.CASCADE
    assert isinstance(body, CascadeBody)
    assert body.placement is Placement.PRE
    assert body.gates[0].kind.value == "signal_below"
    assert body.gates[0].signal is not None
    assert body.gates[0].signal.signal == "query_light_adequacy"
    assert body.gates[0].signal.inputs == ("question",)
    assert knob.structure.provenance.pattern == "router"


def test_fallback_golden_hash_pinned_and_expands_to_post_cascade() -> None:
    knob = _fallback()
    body = knob.structure.body

    assert _golden_hash(knob.structure) == GOLDEN["fallback"]
    assert knob.structure.kind is CompositeKind.CASCADE
    assert isinstance(body, CascadeBody)
    assert body.placement is Placement.POST
    assert [gate.kind.value for gate in body.gates] == [
        "margin_below",
        "margin_below",
    ]
    assert [gate.threshold for gate in body.gates] == [
        "fallback_margin_floor",
        "fallback_margin_floor",
    ]
    assert knob.structure.provenance.pattern == "fallback"


def test_p8_canary_router_and_fallback_provenance_does_not_leak_raw_params() -> None:
    sentinel = "SECRET-CANARY-router-fallback-d4f1c0de"
    r = router(
        "sentinel_router",
        arms=(sentinel, "terminal"),
        signals=(sentinel + "_signal",),
        thresholds=(sentinel + "_theta",),
        tuned_params=((sentinel + "_tvar",), ()),
    )
    f = fallback(
        "sentinel_fallback",
        arms=(sentinel, "terminal"),
        thresholds=(sentinel + "_theta",),
        tuned_params=((sentinel + "_tvar",), ()),
    )

    for knob in (r, f):
        prov_blob = json.dumps(_prov_to_dict(knob.structure.provenance))
        assert sentinel not in prov_blob
        assert knob.structure.provenance.param_hash is not None
        assert sentinel not in knob.structure.provenance.param_hash


def test_telemetry_names_are_truthful_for_pre_and_post_cascades() -> None:
    assert _router().telemetry_names == PRE_TELEMETRY
    assert "escalation_rate" not in _router().telemetry_names

    assert _fallback().telemetry_names == POST_TELEMETRY
    assert (
        binary_cascade(
            "bc",
            base_stage="cheap",
            expert_stage="strong",
            threshold="theta",
        ).telemetry_names
        == POST_TELEMETRY
    )
    assert (
        n_cascade("nc", stages=("a", "b", "c"), thresholds=("t0", "t1")).telemetry_names
        == POST_TELEMETRY
    )


def test_fallback_docstring_states_no_accept_not_error_contract() -> None:
    doc = fallback.__doc__ or ""

    assert "leaf StageRunner cannot signal failure" in doc
    assert "theta=0.0" in doc
    assert "no_accept-triggered, NOT error-triggered" in doc
    assert "errors stay absorbing" in doc


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "arms": (),
                "signals": (),
                "thresholds": (),
            },
            "empty_arms",
        ),
        (
            {
                "arms": ("a", "b", "c"),
                "signals": ("s0",),
                "thresholds": ("t0", "t1"),
            },
            "cascade_arity",
        ),
        (
            {
                "arms": ("a", "b", "c"),
                "signals": ("s0", "s1"),
                "thresholds": ("t0",),
            },
            "cascade_arity",
        ),
        (
            {
                "arms": ("a", "b"),
                "signals": ("s0",),
                "thresholds": ("t0",),
                "tuned_params": (("m",),),
            },
            "invalid_tuned_param",
        ),
        (
            {
                "arms": ("a", "b"),
                "signals": ("s0",),
                "thresholds": ("t0",),
                "signal_inputs": (("question",), ("unused",)),
            },
            "invalid_signal_use",
        ),
        (
            {
                "arms": ("a", "b"),
                "signals": ("s0",),
                "thresholds": ("t0",),
                "signal_inputs": (["question"],),
            },
            "invalid_signal_use",
        ),
        (
            {
                "arms": ("a", "b"),
                "signals": ("",),
                "thresholds": ("t0",),
            },
            "invalid_signal_use",
        ),
        (
            {
                "arms": ("a", "a"),
                "signals": ("s0",),
                "thresholds": ("t0",),
            },
            "duplicate_stage",
        ),
    ],
)
def test_router_rejects_bad_factory_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        router("bad_router", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"arms": (), "thresholds": ()}, "empty_arms"),
        ({"arms": ("a", "b", "c"), "thresholds": ("t0",)}, "cascade_arity"),
        (
            {
                "arms": ("a", "b"),
                "thresholds": ("t0",),
                "tuned_params": (("m",),),
            },
            "invalid_tuned_param",
        ),
        ({"arms": ("a", "a"), "thresholds": ("t0",)}, "duplicate_stage"),
    ],
)
def test_fallback_rejects_bad_factory_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        fallback("bad_fallback", **kwargs)


def test_router_program_validation_rejects_missing_calibration_signal() -> None:
    program = CompositeProgram(
        composites={"adaptive_rag_gate": _router().structure},
        tvars=ROUTER_TVARS,
        cvars=frozenset({"complexity_threshold"}),
        cvar_types={"complexity_threshold": "float"},
        cvar_depends_on={"complexity_threshold": ROUTER_GATE_PARENTS},
    )

    with pytest.raises(ValueError, match="missing_calibration_signal"):
        validate_program(program)


def test_router_program_validation_rejects_missing_threshold_ref() -> None:
    program = CompositeProgram(
        composites={"adaptive_rag_gate": _router().structure},
        tvars=ROUTER_TVARS,
        cvars=frozenset(),
        cvar_depends_on={"complexity_threshold": ROUTER_GATE_PARENTS},
    )

    with pytest.raises(ValueError, match="missing_ref"):
        validate_program(program)


def test_router_program_validation_rejects_mismatched_signal() -> None:
    program = CompositeProgram(
        composites={"adaptive_rag_gate": _router().structure},
        tvars=ROUTER_TVARS,
        cvars=frozenset({"complexity_threshold"}),
        cvar_types={"complexity_threshold": "float"},
        cvar_signals={"complexity_threshold": "other_signal"},
        cvar_depends_on={"complexity_threshold": ROUTER_GATE_PARENTS},
        cvar_signal_inputs={"complexity_threshold": ("question",)},
    )

    with pytest.raises(ValueError, match="signal_mismatch"):
        validate_program(program)


def test_router_program_validation_rejects_uncovered_signal_inputs() -> None:
    program = CompositeProgram(
        composites={"adaptive_rag_gate": _router().structure},
        tvars=ROUTER_TVARS,
        cvars=frozenset({"complexity_threshold"}),
        cvar_types={"complexity_threshold": "float"},
        cvar_signals={"complexity_threshold": "query_light_adequacy"},
        cvar_depends_on={"complexity_threshold": ROUTER_GATE_PARENTS},
    )

    with pytest.raises(ValueError, match="unbound_signal_inputs"):
        validate_program(program)


def test_router_program_validation_rejects_missing_leaf_parent_coverage() -> None:
    program = CompositeProgram(
        composites={"adaptive_rag_gate": _router().structure},
        tvars=ROUTER_TVARS,
        cvars=frozenset({"complexity_threshold"}),
        cvar_types={"complexity_threshold": "float"},
        cvar_signals={"complexity_threshold": "query_light_adequacy"},
        cvar_signal_inputs={"complexity_threshold": ("question",)},
        cvar_depends_on={"complexity_threshold": frozenset({"retrieval_mode"})},
    )

    with pytest.raises(ValueError, match="missing_composite_parent"):
        validate_program(program)


def test_fallback_program_validation_requires_vote_margin_signal() -> None:
    program = CompositeProgram(
        composites={"llm_provider_fallback": _fallback().structure},
        tvars=frozenset({"primary_model", "secondary_model"}),
        cvars=frozenset({"fallback_margin_floor"}),
        cvar_types={"fallback_margin_floor": "float"},
        cvar_signals={"fallback_margin_floor": "provider_health"},
        cvar_depends_on={
            "fallback_margin_floor": frozenset({"primary_model", "secondary_model"})
        },
    )

    with pytest.raises(ValueError, match="signal_mismatch"):
        validate_program(program)


def test_fallback_program_validation_rejects_non_numeric_threshold_type() -> None:
    program = CompositeProgram(
        composites={"llm_provider_fallback": _fallback().structure},
        tvars=frozenset({"primary_model", "secondary_model"}),
        cvars=frozenset({"fallback_margin_floor"}),
        cvar_types={"fallback_margin_floor": "str"},
        cvar_signals={"fallback_margin_floor": "vote_margin"},
        cvar_depends_on={
            "fallback_margin_floor": frozenset({"primary_model", "secondary_model"})
        },
    )

    with pytest.raises(ValueError, match="invalid_threshold_type"):
        validate_program(program)


def test_router_routes_through_shipped_pre_cascade_executor() -> None:
    light = _SpyStage("light-answer")
    heavy = _SpyStage("heavy-answer")
    result = execute_composite(
        _router().structure,
        {"rag_light": light.runner(), "rag_heavy": heavy.runner()},
        config={"question": "short factual question"},
        calibrated_values={"complexity_threshold": 0.5},
        signals={"query_light_adequacy": lambda payload: 0.9},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "light-answer"
    assert light.called is True
    assert heavy.called is False
    assert result.measures["route_selected"] == 0
    assert result.measures["dispatch_signal_margin"] == pytest.approx(0.4)
    assert result.measures["gate_signal_adequate"] == {0: 1}


def test_router_absent_signal_routes_to_terminal() -> None:
    light = _SpyStage("light-answer")
    heavy = _SpyStage("heavy-answer")
    result = execute_composite(
        _router().structure,
        {"rag_light": light.runner(), "rag_heavy": heavy.runner()},
        config={"question": "question with no complexity signal"},
        calibrated_values={"complexity_threshold": 0.5},
        signals={"query_light_adequacy": lambda payload: None},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "heavy-answer"
    assert light.called is False
    assert heavy.called is True
    assert result.measures["route_selected"] == 1
    assert result.measures["gate_signal_adequate"] == {0: 0}
    assert "dispatch_signal_margin" not in result.measures


def test_router_malformed_signal_absorbs_as_error() -> None:
    result = execute_composite(
        _router().structure,
        {"rag_light": _stage(["light"]), "rag_heavy": _stage(["heavy"])},
        config={"question": "bad score"},
        calibrated_values={"complexity_threshold": 0.5},
        signals={"query_light_adequacy": lambda payload: True},
    )

    assert result.result_kind is ResultKind.ERROR
    assert result.measures == {}
    assert "must return a finite number or None" in (result.error or "")


def test_fallback_escalates_when_nested_non_terminal_arm_no_accepts() -> None:
    inner = self_consistency(
        "primary_gate",
        stage="primary_leaf",
        cardinality="k",
        accept_threshold="accept_margin",
    ).structure
    base = fallback(
        "provider_fallback",
        arms=("primary_gate", "secondary"),
        thresholds=("fallback_margin_floor",),
    ).structure
    body = replace(
        base.body,
        arms=(CompositeArm("primary_gate"), StageArm("secondary")),
    )
    outer = replace(base, body=body)

    result = execute_composite(
        outer,
        {
            "primary_leaf": _stage(["a", "b"]),
            "secondary": _stage(["secondary-answer"]),
        },
        config={"k": 2},
        calibrated_values={
            "accept_margin": 0.9,
            "fallback_margin_floor": 0.0,
        },
        registry={"provider_fallback": outer, "primary_gate": inner},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "secondary-answer"
    assert result.measures["stage_selected"] == 1
    assert result.measures["escalation_rate"] == 1.0


def test_fallback_absorbs_errors_instead_of_trying_next_arm() -> None:
    secondary = _SpyStage("secondary-answer")

    def boom(_item):
        raise RuntimeError("primary provider failed")

    result = execute_composite(
        fallback(
            "provider_fallback",
            arms=("primary", "secondary"),
            thresholds=("fallback_margin_floor",),
        ).structure,
        {
            "primary": StageRunner(run=boom, key_fn=lambda x: x, samples=1),
            "secondary": secondary.runner(),
        },
        config={},
        calibrated_values={"fallback_margin_floor": 0.0},
    )

    assert result.result_kind is ResultKind.ERROR
    assert secondary.called is False
    assert "primary provider failed" in (result.error or "")
