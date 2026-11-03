"""Golden characterization for the ``moe`` pattern factory (validation order, output)."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.knobs.composites import AcceptDecl, AggregateKind, StageArm, StatKind
from traigent.knobs.patterns import moe


def _kw(**kwargs: Any) -> dict[str, Any]:
    return kwargs


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (_kw(experts=["a", "b"]), "committee_arity"),
        (_kw(experts=("a",)), "committee_arity"),
        (_kw(experts=("a", "")), "committee_arity"),
        (_kw(experts=("a", 3)), "committee_arity"),
        (_kw(experts=("a", "a")), "duplicate_stage: moe experts"),
        (_kw(experts=("a", "b"), aggregate="mean"), "invalid_aggregate"),
        (
            _kw(experts=("a", "b"), expert_tuned_params=(("x",),)),
            "invalid_tuned_param: .*got 1 for 2",
        ),
        (_kw(experts=("a", "b"), judge_stage="j"), "forbids judge_stage"),
        (_kw(experts=("a", "b"), judge_tuned_params=("p",)), "forbids judge_tuned"),
        (
            _kw(experts=("a", "b"), aggregate="judge"),
            "requires judge_stage",
        ),
        (
            _kw(experts=("a", "b"), aggregate="judge", judge_stage=""),
            "requires judge_stage",
        ),
        (
            _kw(experts=("a", "b"), aggregate="judge", judge_stage="a"),
            "judge_stage must be distinct",
        ),
        (
            _kw(
                experts=("a", "b"),
                aggregate="judge",
                judge_stage="j",
                accept_threshold="t",
            ),
            "does not support accept_threshold",
        ),
    ],
)
def test_validation_errors(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        moe("m", **kwargs)


def test_validation_precedence_with_multiple_faults() -> None:
    # arity beats duplicates, duplicates beat aggregate, aggregate beats tuned-params
    with pytest.raises(ValueError, match="committee_arity"):
        moe("m", experts=("a",), aggregate="bad", judge_stage="j")
    with pytest.raises(ValueError, match="duplicate_stage: moe experts"):
        moe("m", experts=("a", "a"), aggregate="bad")
    with pytest.raises(ValueError, match="invalid_aggregate"):
        moe("m", experts=("a", "b"), aggregate="bad", expert_tuned_params=())
    # tuned-param arity beats the aggregate-specific checks
    with pytest.raises(ValueError, match="invalid_tuned_param"):
        moe("m", experts=("a", "b"), expert_tuned_params=(), judge_stage="j")
    # vote: judge_stage checked before judge_tuned_params
    with pytest.raises(ValueError, match="forbids judge_stage"):
        moe("m", experts=("a", "b"), judge_stage="j", judge_tuned_params=("p",))
    # judge: missing stage beats accept_threshold; distinctness beats accept_threshold
    with pytest.raises(ValueError, match="requires judge_stage"):
        moe("m", experts=("a", "b"), aggregate="judge", accept_threshold="t")
    with pytest.raises(ValueError, match="must be distinct"):
        moe(
            "m",
            experts=("a", "b"),
            aggregate="judge",
            judge_stage="b",
            accept_threshold="t",
        )


def test_vote_default_output() -> None:
    k = moe("m", experts=("a", "b"))
    body = k.structure.body
    assert (
        k.provenance.param_hash
        == (
            "931059b14f44dcebe3afbd451986501db20388f406361be555b9d792e258fe79"  # pragma: allowlist secret
        )
    )
    assert body.aggregate.kind is AggregateKind.MAJORITY_VOTE
    assert body.aggregate.accept is None and body.aggregate.judge is None
    assert body.arms == (StageArm("a", ()), StageArm("b", ()))
    assert body.cardinality is None
    assert k.members == {}


def test_vote_with_accept_and_tuned_params() -> None:
    k = moe(
        "m",
        experts=("a", "b"),
        accept_threshold="th",
        expert_tuned_params=(("x",), ()),
    )
    assert (
        k.provenance.param_hash
        == (
            "b732b83f267fbaba1e0454edff4e2b78936db4a93124b79de2470394919eeace"  # pragma: allowlist secret
        )
    )
    assert k.structure.body.aggregate.accept == AcceptDecl(
        stat=StatKind.VOTE_MARGIN, threshold="th"
    )
    assert k.structure.body.arms[0] == StageArm("a", ("x",))


def test_judge_output_and_members_copied() -> None:
    members: dict[str, Any] = {}
    k = moe(
        "m",
        experts=("a", "b", "c"),
        aggregate="judge",
        judge_stage="j",
        judge_tuned_params=("q",),
        members=members,
    )
    assert (
        k.provenance.param_hash
        == (
            "20b6bcfa4ebef8d5b0722eb7995df8e199f57d520148d51a1193bc0a2c9b3f83"  # pragma: allowlist secret
        )
    )
    agg = k.structure.body.aggregate
    assert agg.kind is AggregateKind.JUDGE_MAX
    assert agg.judge == StageArm("j", ("q",))
    assert k.members == {} and k.members is not members
