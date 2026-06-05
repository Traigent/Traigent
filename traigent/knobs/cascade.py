"""Cascade policies — staged escalate-or-stop execution (RFC 0001 §3.8).

A cascade of arity ``m`` is ``(stages, gates)`` with ``|gates| = m - 1``.
For an input, stage ``j`` is selected where::

    j = min{ i : i = m ∨ gate_i(vote_i) = stop }

The v1 gate predicate is ``escalate ⟺ margin < θ`` where each θ is a
**calibrated variable** (a CVAR — certificate-backed, never searched). Gate
thresholds are read LIVE via ``threshold_ref`` so a re-calibration (e.g.
``Router.fit``-style assignment) is observed at decide time, never
snapshotted at construction.

Totality (RFC §3.8): an empty/abstain-only vote has ``margin = 0`` and
escalates for any θ > 0 (θ = 0 means "never escalate" by the strict
inequality); ties affect representative choice only, never the margin or
the selected stage; a stage or gate exception PROPAGATES — the cascade
never silently degrades to some stage's output.

NOTE (consolidation obligation, FR-SDK-KNOBS-CVARS-V1 trail): the binary
``Router`` (fit/decide/route) lives on an unmerged branch — when it lands,
``Router`` becomes a thin adapter over ``CascadePolicy`` with
``threshold_ref=lambda: self.threshold``; its public surface and test suite
must pass unchanged. The equivalence battery in
``tests/unit/knobs/test_cascade.py`` transcribes the router's voting
semantics so that adapter is a documented, mechanical diff.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "CascadePolicy",
    "CascadeStep",
    "Gate",
    "GateKind",
    "StageSpec",
    "VoteStats",
    "vote_over",
]

VoteKey = Hashable | None


class GateKind(StrEnum):
    """v1 gate registry (RFC 0001 §3.8)."""

    MARGIN_BELOW = "margin_below"


@dataclass(frozen=True)
class VoteStats:
    """Deterministic, content-free vote summary over equivalence keys."""

    top_key: VoteKey
    margin: float
    valid_rate: float
    n_distinct: int
    tie: bool


def vote_over(keys: Sequence[VoteKey], n: int) -> VoteStats:
    """Majority vote over hashable keys; ``None`` is an abstention.

    The empty/abstain-only vote yields ``margin = 0`` (it escalates for any
    θ > 0). Tie-breaking for the representative key is deterministic
    (lexicographic over serialized keys) and never affects the margin.
    """
    if n < len(keys) or (keys and n < 1):
        raise ValueError(
            f"vote_over denominator n={n} is smaller than the number of "
            f"samples ({len(keys)}); margins would exceed 1"
        )
    counts: Counter[VoteKey] = Counter(key for key in keys if key is not None)
    if not counts:
        return VoteStats(
            top_key=None, margin=0.0, valid_rate=0.0, n_distinct=0, tie=False
        )
    top_count = max(counts.values())
    tied = sorted((key for key, count in counts.items() if count == top_count), key=str)
    denominator = n or 1
    return VoteStats(
        top_key=tied[0],
        margin=top_count / denominator,
        valid_rate=sum(counts.values()) / denominator,
        n_distinct=len(counts),
        tie=len(tied) > 1,
    )


@dataclass(frozen=True)
class Gate:
    """A cascade gate: escalate when the stage's vote margin is below θ.

    ``threshold_ref`` is a LIVE read — re-calibration of the underlying CVAR
    is observed at decide time. An unset threshold (``None``) is a hard
    error, mirroring the router's "calibrate before routing" contract.
    """

    kind: GateKind = GateKind.MARGIN_BELOW
    threshold_ref: Callable[[], float | None] = lambda: None

    def should_escalate(self, vote: VoteStats) -> bool:
        threshold = self.threshold_ref()
        if threshold is None:
            raise ValueError(
                "Gate threshold is unset; calibrate the CVAR before routing."
            )
        if not math.isfinite(threshold):
            # NaN comparisons are always False — a NaN threshold would
            # silently NEVER escalate (fail-open). Non-finite thresholds are
            # a calibration defect: fail closed.
            raise ValueError(
                f"Gate threshold is non-finite ({threshold!r}); "
                "re-calibrate the CVAR."
            )
        return vote.margin < threshold


@dataclass(frozen=True)
class StageSpec:
    """One cascade stage: a runtime execution unit.

    ``samples`` is the per-stage CARDINALITY knob (the router's ``k``);
    ``run`` returns the stage's samples for an item; ``key_fn`` maps outputs
    to equivalence keys (the comparator — required for voting stages).
    """

    name: str
    run: Callable[[Any], Sequence[Any]]
    key_fn: Callable[[Any], VoteKey] | None = None
    samples: int = 1

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("StageSpec.samples must be >= 1")


@dataclass(frozen=True)
class CascadeStep:
    """The outcome of one cascade evaluation.

    ``stage_votes`` carries the vote statistics of EVERY evaluated stage (in
    order), so escalated decisions remain fully observable — the terminal
    stage's vote is also exposed as ``vote`` for convenience.
    """

    stage_index: int
    stage_name: str
    output: Any
    representative: Any
    vote: VoteStats | None
    escalations: int
    stage_votes: tuple[VoteStats | None, ...] = ()


@dataclass(frozen=True)
class CascadePolicy:
    """Staged escalate-or-stop execution policy (arity = ``len(stages)``)."""

    stages: tuple[StageSpec, ...]
    gates: tuple[Gate, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.stages) < 1:
            raise ValueError("CascadePolicy requires at least one stage")
        if len(self.gates) != len(self.stages) - 1:
            raise ValueError(
                f"CascadePolicy arity violation: {len(self.gates)} gate(s) for "
                f"{len(self.stages)} stage(s); |gates| must equal |stages| - 1"
            )

    @property
    def arity(self) -> int:
        return len(self.stages)

    def decide(self, item: Any) -> CascadeStep:
        """Evaluate the cascade for one item (RFC §3.8 semantics).

        Stage/gate exceptions PROPAGATE — never a silent degradation to a
        stage's output. Given the per-stage sample multisets, the selected
        stage is a deterministic function of (votes, thresholds).
        """
        escalations = 0
        stage_votes: list[VoteStats | None] = []
        for index, stage in enumerate(self.stages):
            samples = list(stage.run(item))
            if len(samples) != stage.samples:
                # samples is the stage's declared CARDINALITY knob value — a
                # mismatch means the effectuation did not happen as declared.
                raise ValueError(
                    f"stage {stage.name!r} declared samples={stage.samples} "
                    f"but produced {len(samples)}"
                )
            is_last = index == len(self.stages) - 1
            if stage.key_fn is None:
                vote = None
                representative = samples[0] if samples else None
            else:
                keys = [stage.key_fn(sample) for sample in samples]
                vote = vote_over(keys, len(samples))
                representative = next(
                    (
                        sample
                        for sample, key in zip(samples, keys, strict=False)
                        if key == vote.top_key
                    ),
                    samples[0] if samples else None,
                )
            stage_votes.append(vote)
            if is_last:
                return CascadeStep(
                    stage_index=index,
                    stage_name=stage.name,
                    output=representative,
                    representative=representative,
                    vote=vote,
                    escalations=escalations,
                    stage_votes=tuple(stage_votes),
                )
            gate = self.gates[index]
            if vote is None:
                raise ValueError(
                    f"stage {stage.name!r} feeds a gate but has no key_fn "
                    "comparator; voting stages require one"
                )
            if gate.should_escalate(vote):
                escalations += 1
                continue
            return CascadeStep(
                stage_index=index,
                stage_name=stage.name,
                output=representative,
                representative=representative,
                vote=vote,
                escalations=escalations,
                stage_votes=tuple(stage_votes),
            )
        raise AssertionError("unreachable: the last stage always returns")
