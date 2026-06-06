"""Mixture-of-experts pattern factory tests (committee ensemble macro)."""

from __future__ import annotations

import json

import pytest

from traigent.knobs import canonical_hash
from traigent.knobs.composites import (
    AggregateKind,
    CascadeBody,
    CompositeArm,
    CompositeProgram,
    EnsembleBody,
    LoopBody,
    StageArm,
    validate_program,
)
from traigent.knobs.patterns import CompositeKnob, moe
from traigent.knobs.runtime import ResultKind, StageRunner, execute_composite


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


MOE_ACCEPT_THRESHOLD = "patch_moe.vote_margin_min"
MOE_EXPERT_TVARS = frozenset(
    {
        "repo_context_strategy",
        "edit_granularity",
        "test_selection_strategy",
        "patch_review_mode",
    }
)


def _moe_vote() -> CompositeKnob:
    return moe(
        "patch_moe",
        experts=("fast_patch", "semantic_patch", "test_driven_patch"),
        aggregate="vote",
        accept_threshold=MOE_ACCEPT_THRESHOLD,
        expert_tuned_params=(
            ("repo_context_strategy", "edit_granularity"),
            ("repo_context_strategy",),
            ("test_selection_strategy", "patch_review_mode"),
        ),
    )


def _moe_judge() -> CompositeKnob:
    return moe(
        "review_moe",
        experts=("draft_a", "draft_b"),
        aggregate="judge",
        judge_stage="rubric_judge",
        expert_tuned_params=(("style",), ("tests",)),
        judge_tuned_params=("judge_rubric", "judge_model"),
    )


def _moe_vote_program(
    *,
    cvar_signals: dict[str, str | None] | None = None,
    cvar_depends_on: dict[str, frozenset[str]] | None = None,
) -> CompositeProgram:
    return CompositeProgram(
        composites={"patch_moe": _moe_vote().structure},
        tvars=MOE_EXPERT_TVARS,
        cvars=frozenset({MOE_ACCEPT_THRESHOLD}),
        cvar_types={MOE_ACCEPT_THRESHOLD: "float"},
        cvar_signals=(
            {MOE_ACCEPT_THRESHOLD: "vote_margin"}
            if cvar_signals is None
            else cvar_signals
        ),
        cvar_depends_on=(
            {MOE_ACCEPT_THRESHOLD: MOE_EXPERT_TVARS}
            if cvar_depends_on is None
            else cvar_depends_on
        ),
    )


GOLDEN = {
    "moe_vote": "e81cb5ef92f166cb1fdbc0647ef2f544b4b2b1979681a91a456b01a1af73a816",  # pragma: allowlist secret
    "moe_judge": "0392e0c0916286fd1896a79f7c6690c76c0a30e8c42d8be4155044477f4f4e41",  # pragma: allowlist secret
}


def test_moe_vote_golden_hash_pinned() -> None:
    assert _golden_hash(_moe_vote().structure) == GOLDEN["moe_vote"]


def test_moe_judge_golden_hash_pinned() -> None:
    assert _golden_hash(_moe_judge().structure) == GOLDEN["moe_judge"]


def test_moe_returns_composite_knob_and_ensemble_telemetry() -> None:
    knob = _moe_vote()

    assert isinstance(knob, CompositeKnob)
    assert knob.name == "patch_moe"
    assert isinstance(knob.structure.body, EnsembleBody)
    assert knob.structure.provenance.pattern == "moe"
    assert knob.telemetry_names == (
        "vote_agreement",
        "vote_margin",
        "candidates_evaluated",
        "candidates_excluded",
    )


def test_moe_vote_expands_to_committee_majority_vote_with_accept_decl() -> None:
    body = _moe_vote().structure.body

    assert isinstance(body, EnsembleBody)
    assert body.cardinality is None
    assert [arm.name for arm in body.arms if isinstance(arm, StageArm)] == [
        "fast_patch",
        "semantic_patch",
        "test_driven_patch",
    ]
    assert body.aggregate.kind is AggregateKind.MAJORITY_VOTE
    assert body.aggregate.accept is not None
    assert body.aggregate.accept.stat.value == "vote_margin"
    assert body.aggregate.accept.threshold == "patch_moe.vote_margin_min"


def test_moe_judge_expands_to_judge_max_with_judge_tuned_params() -> None:
    body = _moe_judge().structure.body

    assert isinstance(body, EnsembleBody)
    assert body.aggregate.kind is AggregateKind.JUDGE_MAX
    assert isinstance(body.aggregate.judge, StageArm)
    assert body.aggregate.judge.name == "rubric_judge"
    assert body.aggregate.judge.tuned_params == ("judge_rubric", "judge_model")
    assert body.aggregate.accept is None


class TestMoeAcceptThresholdAdmission:
    def test_missing_calibration_signal_binding_rejects(self) -> None:
        program = _moe_vote_program(
            cvar_signals={MOE_ACCEPT_THRESHOLD: None},
        )

        with pytest.raises(ValueError, match="missing_calibration_signal"):
            validate_program(program)

    def test_signal_mismatch_rejects(self) -> None:
        program = _moe_vote_program(
            cvar_signals={MOE_ACCEPT_THRESHOLD: "other_signal"},
        )

        with pytest.raises(ValueError, match="signal_mismatch"):
            validate_program(program)

    def test_missing_parent_coverage_rejects(self) -> None:
        program = _moe_vote_program(
            cvar_depends_on={
                MOE_ACCEPT_THRESHOLD: MOE_EXPERT_TVARS - {"patch_review_mode"}
            },
        )

        with pytest.raises(ValueError, match="missing_composite_parent"):
            validate_program(program)


def test_p8_canary_raw_params_do_not_enter_provenance() -> None:
    sentinel = "SECRET-CANARY-moe-d4f1c0de"
    knob = moe(
        "sentinel_moe",
        experts=(sentinel, "expert_b"),
        accept_threshold=sentinel + "_threshold",
        expert_tuned_params=((sentinel + "_tvar",), ()),
    )

    prov_blob = json.dumps(_prov_to_dict(knob.structure.provenance), sort_keys=True)
    assert sentinel not in prov_blob
    assert knob.structure.provenance.param_hash is not None
    assert sentinel not in knob.structure.provenance.param_hash


def test_committee_runtime_majority_vote_populates_telemetry() -> None:
    node = moe(
        "runtime_moe",
        experts=("expert_a", "expert_b", "expert_c"),
        aggregate="vote",
    ).structure
    result = execute_composite(
        node,
        {
            "expert_a": _stage(["PATCH"]),
            "expert_b": _stage(["PATCH"]),
            "expert_c": _stage(["OTHER"]),
        },
        config={},
        calibrated_values={},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "PATCH"
    assert result.measures["candidates_evaluated"] == 3
    assert result.measures["candidates_excluded"] == 0
    assert result.measures["vote_margin"] == pytest.approx(2 / 3)
    assert result.measures["vote_agreement"] == pytest.approx(1.0)


def test_judge_runtime_uses_judge_tuned_params_declaration() -> None:
    knob = _moe_judge()
    result = execute_composite(
        knob.structure,
        {
            "draft_a": _stage(["x"]),
            "draft_b": _stage(["y"]),
            "rubric_judge": StageRunner(
                run=lambda candidate: [0.9 if candidate == "y" else 0.1],
                samples=1,
            ),
        },
        config={},
        calibrated_values={},
    )

    assert result.result_kind is ResultKind.OUTPUT
    assert result.output == "y"
    assert result.measures["candidates_evaluated"] == 2
    assert result.measures["candidates_excluded"] == 0
    assert result.measures["vote_margin"] == 0.0
    assert result.measures["vote_agreement"] == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"experts": ("only",)}, r"^committee_arity"),
        ({"experts": ("a", "")}, r"^committee_arity"),
        ({"experts": ("a", "a")}, r"^duplicate_stage"),
        ({"experts": ("a", "b"), "aggregate": "rank"}, r"^invalid_aggregate"),
        (
            {"experts": ("a", "b"), "expert_tuned_params": (("x",),)},
            r"^invalid_tuned_param",
        ),
        (
            {"experts": ("a", "b"), "aggregate": "judge"},
            r"^aggregate_argument_mismatch",
        ),
        (
            {"experts": ("a", "b"), "aggregate": "vote", "judge_stage": "j"},
            r"^aggregate_argument_mismatch",
        ),
        (
            {
                "experts": ("a", "b"),
                "aggregate": "vote",
                "judge_tuned_params": ("judge_rubric",),
            },
            r"^aggregate_argument_mismatch",
        ),
        (
            {"experts": ("a", "b"), "aggregate": "judge", "judge_stage": "a"},
            r"^duplicate_stage",
        ),
        (
            {
                "experts": ("a", "b"),
                "aggregate": "judge",
                "judge_stage": "j",
                "accept_threshold": "theta",
            },
            r"^aggregate_argument_mismatch",
        ),
    ],
)
def test_moe_rejects_invalid_factory_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        moe("bad_moe", **kwargs)
