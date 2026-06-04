"""CascadePolicy tests (SDK packet 6-D, RFC 0001 §3.8).

Includes the binary-router EQUIVALENCE BATTERY: a reference transcription of
the unmerged Router's decide() semantics (margin vote over key_fn, escalate
when margin < θ, terminal output vs stage-0 representative) is compared
against the m=2 cascade across vote shapes — so when the Router branch
merges, its adapter over CascadePolicy is a mechanical, pre-verified diff.
"""

from __future__ import annotations

import pytest

from traigent.knobs.cascade import (
    CascadePolicy,
    Gate,
    GateKind,
    StageSpec,
    VoteStats,
    vote_over,
)


def _stage(name: str, outputs, key_fn=lambda x: x, samples=None):
    return StageSpec(
        name=name,
        run=lambda _item: list(outputs),
        key_fn=key_fn,
        samples=samples or max(len(outputs), 1),
    )


def _binary(cheap_outputs, strong_output, threshold_holder):
    return CascadePolicy(
        stages=(
            _stage("cheap", cheap_outputs),
            _stage("strong", [strong_output]),
        ),
        gates=(Gate(threshold_ref=lambda: threshold_holder["theta"]),),
    )


class TestArityAndTotality:
    def test_m1_degenerate_no_gates(self):
        policy = CascadePolicy(stages=(_stage("only", ["A"]),))
        step = policy.decide("item")
        assert step.stage_index == 0
        assert step.output == "A"
        assert step.escalations == 0

    def test_arity_invariant_enforced(self):
        with pytest.raises(ValueError, match="arity"):
            CascadePolicy(stages=(_stage("a", ["x"]), _stage("b", ["y"])))
        with pytest.raises(ValueError, match="at least one stage"):
            CascadePolicy(stages=())

    def test_three_stage_cascade_walks_gates_in_order(self):
        holder = {"theta": 0.9}
        policy = CascadePolicy(
            stages=(
                _stage("s1", ["a", "b", "c"]),  # margin 1/3 < 0.9 -> escalate
                _stage("s2", ["x", "x", "y"]),  # margin 2/3 < 0.9 -> escalate
                _stage("s3", ["final"]),
            ),
            gates=(
                Gate(threshold_ref=lambda: holder["theta"]),
                Gate(threshold_ref=lambda: holder["theta"]),
            ),
        )
        step = policy.decide("item")
        assert step.stage_index == 2
        assert step.output == "final"
        assert step.escalations == 2

    def test_unset_threshold_raises(self):
        policy = _binary(["A", "A"], "S", {"theta": None})
        with pytest.raises(ValueError, match="calibrate the CVAR"):
            policy.decide("item")

    def test_stage_exception_propagates_never_degrades(self):
        def explode(_item):
            raise RuntimeError("provider 404")

        policy = CascadePolicy(
            stages=(
                StageSpec(name="cheap", run=explode, key_fn=lambda x: x),
                _stage("strong", ["S"]),
            ),
            gates=(Gate(threshold_ref=lambda: 0.5),),
        )
        with pytest.raises(RuntimeError, match="provider 404"):
            policy.decide("item")

    def test_voting_stage_requires_comparator(self):
        policy = CascadePolicy(
            stages=(
                StageSpec(name="cheap", run=lambda _i: ["a"], key_fn=None),
                _stage("strong", ["S"]),
            ),
            gates=(Gate(threshold_ref=lambda: 0.5),),
        )
        with pytest.raises(ValueError, match="key_fn"):
            policy.decide("item")


class TestGateSemantics:
    def test_empty_vote_escalates_for_positive_theta(self):
        policy = _binary([None, None], "S", {"theta": 0.6})
        step = policy.decide("item")
        assert step.stage_index == 1  # margin 0 < 0.6

    def test_theta_zero_never_escalates(self):
        policy = _binary([None, None], "S", {"theta": 0.0})
        step = policy.decide("item")
        assert step.stage_index == 0  # strict inequality: 0 < 0 is False

    def test_live_threshold_read(self):
        """Re-calibration is observed at decide time, never snapshotted."""
        holder = {"theta": 0.9}
        policy = _binary(["A", "A", "B"], "S", holder)  # margin 2/3
        assert policy.decide("item").stage_index == 1  # 2/3 < 0.9 -> escalate
        holder["theta"] = 0.5  # the CVAR was re-fit
        assert policy.decide("item").stage_index == 0  # 2/3 >= 0.5 -> stop

    def test_tie_does_not_change_selection(self):
        holder = {"theta": 0.4}
        tied = _binary(["A", "A", "B", "B"], "S", holder)  # margin 1/2, tie
        untied = _binary(["A", "A", "B", "C"], "S", holder)  # margin 1/2
        assert tied.decide("i").stage_index == untied.decide("i").stage_index

    def test_gate_kind_registry(self):
        assert Gate().kind is GateKind.MARGIN_BELOW


class TestVoteOver:
    def test_abstentions_lower_valid_rate_not_margin_base(self):
        vote = vote_over(["a", "a", None, None], 4)
        assert vote.margin == 0.5
        assert vote.valid_rate == 0.5
        assert vote.top_key == "a"

    def test_tie_representative_deterministic(self):
        vote = vote_over(["b", "a"], 2)
        assert vote.tie is True
        assert vote.top_key == "a"  # lexicographic over serialized keys


class TestBinaryRouterEquivalence:
    """The equivalence battery: the m=2 cascade against a reference
    transcription of the unmerged Router.decide() semantics."""

    @staticmethod
    def _reference_router_decide(cheap_samples, strong_output, key_fn, threshold):
        """Transcription of routing/router.py decide() (unmerged branch):
        vote over key_fn, escalate iff margin < threshold, output = strong on
        escalation else the representative."""
        keys = [key_fn(sample) for sample in cheap_samples]
        vote = vote_over(keys, len(cheap_samples))
        representative = next(
            (
                sample
                for sample, key in zip(cheap_samples, keys, strict=False)
                if key == vote.top_key
            ),
            cheap_samples[0] if cheap_samples else None,
        )
        escalated = vote.margin < threshold
        output = strong_output if escalated else representative
        return escalated, output, representative, vote.margin

    BATTERY = [
        # (cheap samples, threshold) — unanimous, majority, tie, all-distinct,
        # abstain-heavy, empty
        (["A", "A", "A"], 0.6),
        (["A", "A", "B"], 0.6),
        (["A", "B"], 0.6),
        (["A", "B", "C"], 0.6),
        (["A", None, None], 0.5),
        ([None, None], 0.5),
        (["A", "A", "B"], 0.0),
        (["A", "B", "C"], 1.0),
    ]

    @pytest.mark.parametrize("cheap,theta", BATTERY)
    def test_equivalence(self, cheap, theta):
        key_fn = lambda x: x  # noqa: E731
        expected_escalated, expected_output, expected_repr, expected_margin = (
            self._reference_router_decide(cheap, "STRONG", key_fn, theta)
        )
        policy = _binary(cheap, "STRONG", {"theta": theta})
        step = policy.decide("item")
        assert (step.stage_index == 1) == expected_escalated, (cheap, theta)
        assert step.output == expected_output
        assert (step.vote.margin if step.stage_index == 0 else expected_margin) == (
            expected_margin
        )
        if step.stage_index == 0:
            assert step.representative == expected_repr
