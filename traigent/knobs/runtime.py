"""The composite EXECUTION runtime — the full §3.2 algebra (RFC 0002).

This module executes EVERY composite kind end-to-end: the post-cascade
(cross-referenced as RFC 0001 §3.8), the PRE-cascade dispatch/router (§3.2
``route(x) = arm_j where j = min({i < m : σ_i(x) ≥ θ_i} ∪ {m})`` — left-to-right
gates, first-match-wins, exactly one terminal-like arm runs), the ensemble
(sampling and committee forms, ``majority_vote`` / ``judge_max``), and the loop
(``signal_accept`` / ``external_accept`` / ``exhausted`` stops), closed under
nested-composite arms, plus the §3.8 K-chain unroll executor.

It does NOT reimplement vote/margin math: cascade and ensemble ``majority_vote``
both route through the shipped :func:`~traigent.knobs.cascade.vote_over`
(content-free majority vote over equivalence keys, RFC 0001 tie/abstain rules,
deterministic total order over serialized keys). The cascade kind ADAPTS the
§3.2 ``CascadeBody`` IR into the shipped
:class:`~traigent.knobs.cascade.CascadePolicy` and reads LIVE calibrated gate
thresholds through ``threshold_ref`` closures over ``calibrated_values``.

The result codomain is the §3.2.1 algebra (closed, total, disjoint), named once:

- ``output`` — a produced output (the selected arm/aggregate/accepted state);
- ``no_accept`` — ran, but nothing met the construct's acceptance condition (an
  HONEST no-output outcome, NOT an error). Reachable in THIS packet:
  ensemble accept-gate failure (``stat < θ``); a loop exhausting ``max_iters``
  without ``signal_accept`` firing (or an ``exhausted`` loop whose final body
  result is ``no_accept``); a committee whose candidates are ALL excluded by
  ``no_accept`` (§3.2.1). A ``no_accept`` reaching the root is honest no-output.
- ``error`` — evaluation failed (a stage/body/judge/stop exception, a missing
  stage/signal/predicate, a missing/non-finite gate threshold, a judge
  contract violation with NO survivor, a committee ALL-excluded-by-error, an
  out-of-range cardinality). Fail CLOSED: ``error`` is absorbing (§3.2.1) —
  it propagates outward to the root, never a silent degradation.

Telemetry (:attr:`CompositeRunResult.measures`) is the §3.10 content-free dict,
per kind:

- post-cascade: ``escalation_rate``, ``stage_selected``, per-gate (INDEX-keyed)
  ``gate_margin_pass_rate``;
- pre-cascade dispatch/router: ``route_selected``, ``gate_signal_adequate``,
  and, on gated selection, ``dispatch_signal_margin``;
- ensemble: ``vote_agreement``, ``vote_margin``, ``candidates_evaluated``,
  ``candidates_excluded``;
- loop: ``iterations_used``, ``stop_reason`` (enum over StopDecl kinds).

Counts/rates/enums/finite numbers only — never content.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .cascade import (
    CascadePolicy,
    CascadeStep,
    Gate,
    GateKind,
    StageSpec,
    VoteKey,
    VoteStats,
    vote_over,
)
from .composites import (
    AcceptDecl,
    AggregateKind,
    Arm,
    CascadeBody,
    CompositeArm,
    CompositeNode,
    EnsembleBody,
)
from .composites import GateKind as IRGateKind
from .composites import LoopBody, Placement, StageArm, StatKind, StopKind

__all__ = [
    "CompositeRunResult",
    "LoopBodyResult",
    "LoopBodyRunner",
    "ResultKind",
    "StageRunner",
    "StopReason",
    "execute_composite",
    "execute_kchain",
]


class ResultKind(StrEnum):
    """The §3.2.1 result-algebra codomain (closed, total, disjoint)."""

    OUTPUT = "output"
    NO_ACCEPT = "no_accept"
    ERROR = "error"


class StopReason(StrEnum):
    """The §3.10 loop ``stop_reason`` enum over the StopDecl kinds.

    A loop that runs until a ``signal_accept`` stop fires reports
    ``signal_accept``; an ``external_accept`` predicate firing reports
    ``external_accept``; a loop reaching ``max_iters`` without acceptance (the
    ``exhausted`` stop kind, OR a ``signal_accept`` loop that never accepts)
    reports ``exhausted``.
    """

    SIGNAL_ACCEPT = "signal_accept"
    EXTERNAL_ACCEPT = "external_accept"
    EXHAUSTED = "exhausted"


# --------------------------------------------------------------------------- #
# Runtime execution units (calling conventions)                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StageRunner:
    """A runtime execution unit for one opaque stage (§3.2 StageRef).

    ``run`` returns the stage's sample sequence for an item; ``key_fn`` maps an
    output to its content-free equivalence key (REQUIRED for a voting stage —
    one that feeds a margin gate, OR an ensemble ``majority_vote`` candidate
    stage); ``samples`` is the declared per-stage cardinality (the §3.4 ``k``
    knob — the engine rejects a run that produces a different count, mirroring
    :class:`~traigent.knobs.cascade.StageSpec`).

    A bare ``Callable`` may be supplied directly in the ``stages`` mapping; it
    is wrapped as a NON-VOTING single-sample runner (``key_fn=None`` — a leaf
    stage that feeds no gate; supplying one to a gated position fails closed). A
    stage that DOES feed a margin gate (or votes in an ensemble) MUST be
    supplied as a :class:`StageRunner` carrying a ``key_fn``.
    """

    run: Callable[[Any], Sequence[Any]]
    key_fn: Callable[[Any], VoteKey] | None = None
    samples: int = 1


@dataclass(frozen=True, slots=True)
class LoopBodyResult:
    """The frozen outcome of ONE loop-body invocation (§3.2 loop semantics).

    ``output`` is the body's produced output for this iteration; ``state`` is
    the threaded state the body produced — iteration ``i+1`` observes EXACTLY
    these keys (restricted to the loop's declared ``state_keys`` by the runtime;
    undeclared keys are dropped, invisible to ``stop``). ``result_kind`` lets a
    body honestly report a ``no_accept`` (the iteration completes unaccepted;
    the loop CONTINUES, §3.2.1) — defaulting to ``OUTPUT`` keeps the common
    "produced something" case terse. A body that raises is the §3.2.1
    absorbing ``error`` (caught by the runtime, never reaches here).
    """

    output: Any
    state: Mapping[str, Any] = field(default_factory=dict)
    result_kind: ResultKind = ResultKind.OUTPUT


@dataclass(frozen=True, slots=True)
class LoopBodyRunner:
    """A runtime execution unit for a loop body (§3.2 loop calling convention).

    ``run`` is called once per iteration as ``run(item, state)`` where ``state``
    is the threaded-state mapping RESTRICTED to the loop's declared
    ``state_keys`` (iteration 1 receives ``{}``; iteration ``i+1`` receives the
    keys iteration ``i`` produced, ∩ ``state_keys``). It returns a
    :class:`LoopBodyResult` (or a bare value, wrapped as
    ``LoopBodyResult(output=value)`` with empty produced state — a pure body).
    Undeclared produced keys are dropped by the runtime (documented: invisible
    to ``stop``).

    A bare ``Callable`` may be supplied in the ``stages`` mapping for a loop
    body; it is wrapped as ``run(item, _state) -> LoopBodyResult(output=call(item))``
    (a pure, state-free body — the ``state_keys = []`` case).
    """

    run: Callable[[Any, Mapping[str, Any]], LoopBodyResult | Any]


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompositeRunResult:
    """The frozen outcome of one composite execution (§3.2.1 + §3.10).

    ``output`` is the selected output when ``result_kind`` is ``OUTPUT``
    (``None`` otherwise — ``no_accept``/``error`` carry no output).
    ``result_kind`` is the §3.2.1 algebra tag. ``measures`` is the §3.10
    content-free telemetry dict; ``error`` carries a fail-closed diagnostic
    string when ``result_kind`` is ``ERROR`` (never a partial output).
    """

    output: Any
    result_kind: ResultKind
    measures: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _error(message: str) -> CompositeRunResult:
    """A fail-closed ``error`` result (no output, no measures, §3.2.1)."""
    return CompositeRunResult(
        output=None, result_kind=ResultKind.ERROR, measures={}, error=message
    )


def _no_accept(measures: dict[str, Any]) -> CompositeRunResult:
    """An honest ``no_accept`` result carrying its telemetry (§3.2.1)."""
    return CompositeRunResult(
        output=None, result_kind=ResultKind.NO_ACCEPT, measures=measures, error=None
    )


# --------------------------------------------------------------------------- #
# Shared resolution helpers                                                   #
# --------------------------------------------------------------------------- #

#: A ``stages`` mapping entry: a voting/sampling runner, a loop-body runner, or
#: a bare callable normalized to whichever convention its position demands.
StageEntry = StageRunner | LoopBodyRunner | Callable[..., Any]


def _coerce_stage_runner(stage: StageEntry) -> StageRunner:
    """Normalize a ``stages`` entry into a :class:`StageRunner` (vote position).

    A bare callable becomes a single-sample, identity-keyed runner. A
    :class:`LoopBodyRunner` is NOT a stage runner — handing one to a vote
    position is the caller's error (it has no ``run(item)`` arity), surfaced as
    a fail-closed ``error`` by the enclosing executor's exception guard.
    """
    if isinstance(stage, StageRunner):
        return stage
    if isinstance(stage, LoopBodyRunner):
        raise TypeError(
            "a LoopBodyRunner was supplied to a voting/sampling stage position "
            "(it has no run(item) arity); supply a StageRunner or a bare callable"
        )
    call: Callable[[Any], Any] = stage

    def _run(item: Any) -> Sequence[Any]:
        return [call(item)]

    return StageRunner(run=_run, key_fn=None, samples=1)


def _coerce_loop_runner(stage: StageEntry) -> LoopBodyRunner:
    """Normalize a ``stages`` entry into a :class:`LoopBodyRunner` (loop body).

    A bare callable becomes a pure, state-free body (``run(item, _state)``
    returning ``LoopBodyResult(output=call(item))``).
    """
    if isinstance(stage, LoopBodyRunner):
        return stage
    if isinstance(stage, StageRunner):
        raise TypeError(
            "a StageRunner was supplied to a loop-body position (it has no "
            "run(item, state) arity); supply a LoopBodyRunner or a bare callable"
        )
    call: Callable[[Any], Any] = stage

    def _run(item: Any, _state: Mapping[str, Any]) -> LoopBodyResult:
        return LoopBodyResult(output=call(item))

    return LoopBodyRunner(run=_run)


def _resolve_cardinality(
    name: str,
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
) -> int:
    """Resolve an ensemble cardinality ``k`` (§3.2; R9 ``k < 1``).

    ``k`` is a TVAR (tuned, in ``config``) or CVAR (calibrated, in
    ``calibrated_values``) ref; we look in ``config`` first, then
    ``calibrated_values``. A missing or non-integer or ``< 1`` value raises —
    caught as a fail-closed ``error`` by the caller (R9
    ``invalid_cardinality_value``).
    """
    raw: Any = config.get(name, calibrated_values.get(name))
    if raw is None:
        raise ValueError(f"ensemble cardinality {name!r} is unresolved (no value)")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"ensemble cardinality {name!r} must resolve to an int, got {raw!r}"
        )
    if raw < 1:
        raise ValueError(f"ensemble cardinality {name!r} resolved to {raw} < 1 (R9)")
    return int(raw)


def _build_threshold_ref(
    threshold: str, calibrated_values: Mapping[str, float | int]
) -> Callable[[], float | None]:
    """A LIVE ``threshold_ref`` closure (§ cascade.py contract).

    Reads ``calibrated_values[threshold]`` at decide time so a re-calibration
    is observed, never snapshotted. A missing threshold returns ``None`` — the
    :class:`~traigent.knobs.cascade.Gate` then fails CLOSED, which this runtime
    maps to an ``error`` result (§3.2.1).
    """

    def _ref() -> float | None:
        value = calibrated_values.get(threshold)
        return None if value is None else float(value)

    return _ref


def _resolve_threshold(
    threshold: str, calibrated_values: Mapping[str, float | int]
) -> float:
    """Resolve a finite acceptance/stop threshold or fail closed.

    A missing threshold ("calibrate the CVAR before routing") or a non-finite
    value (a NaN would make ``stat >= θ`` silently True/False) is a calibration
    defect — raise, mapping to an ``error`` (§3.2.1 non-finite threshold).
    """
    value = calibrated_values.get(threshold)
    if value is None:
        raise ValueError(
            f"threshold {threshold!r} is unset; calibrate the CVAR before routing"
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(
            f"threshold {threshold!r} is non-finite ({fvalue!r}); re-calibrate"
        )
    return fvalue


# --------------------------------------------------------------------------- #
# Cascade execution (§3.2 post-cascade; ADAPTS the shipped CascadePolicy)     #
# --------------------------------------------------------------------------- #


def _adapt_cascade(
    body: CascadeBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CascadePolicy:
    """Adapt a §3.2 post-cascade ``CascadeBody`` into a ``CascadePolicy``.

    A stage arm becomes a leaf :class:`StageSpec`; a nested-composite arm
    becomes a synthetic single-sample stage whose ``run`` recursively executes
    the nested composite and lifts its result into the cascade's escalation
    line (§3.2.1 propagation). Each gate is a ``margin_below`` gate whose
    ``threshold_ref`` reads ``calibrated_values`` live. This does NOT
    reimplement gate math — it constructs the engine that owns it.
    """
    stage_specs: list[StageSpec] = []
    for arm in body.arms:
        if isinstance(arm, StageArm):
            runner = _coerce_stage_runner(stages[arm.name])
            stage_specs.append(
                StageSpec(
                    name=arm.name,
                    run=runner.run,
                    key_fn=runner.key_fn,
                    samples=runner.samples,
                )
            )
        else:  # CompositeArm — nested execution lifted onto the escalation line
            stage_specs.append(
                _nested_cascade_stage(
                    arm,
                    stages,
                    registry,
                    config,
                    calibrated_values,
                    signals,
                    predicates,
                )
            )

    gates: list[Gate] = []
    for gate in body.gates:
        if gate.kind is not IRGateKind.MARGIN_BELOW:  # pragma: no cover - IR-guarded
            raise ValueError(
                f"post-cascade gate kind {gate.kind.value!r} is not executable "
                "(margin_below only)"
            )
        gates.append(
            Gate(
                kind=GateKind.MARGIN_BELOW,
                threshold_ref=_build_threshold_ref(gate.threshold, calibrated_values),
            )
        )

    return CascadePolicy(stages=tuple(stage_specs), gates=tuple(gates))


class _NestedNoAccept(Exception):
    """Signals a nested-composite arm returned ``no_accept`` (NOT an error).

    Carries the nested composite name for diagnostics. The cascade driver maps
    this back into §3.2.1 post-cascade ``no_accept`` propagation (escalate, or
    yield ``no_accept`` at the last arm); never an ``error``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"nested composite {name!r} yielded no_accept")
        self.name = name


def _nested_cascade_stage(
    arm: CompositeArm,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> StageSpec:
    """A synthetic non-voting stage that executes a nested composite arm.

    A nested ``error`` raises (absorbing — fails the parent item via the
    cascade engine's exception propagation). A nested ``no_accept`` raises
    :class:`_NestedNoAccept` (the parent cascade then escalates / yields
    ``no_accept`` per §3.2.1). Only a nested ``output`` produces a sample. Such
    a stage is NON-VOTING (``key_fn=None``); the IR forbids a margin gate over a
    non-margin-bearing nested arm (``gate_arm_incompatible``), so a nested arm
    only sits in a NON-gated cascade position.
    """
    node = registry[arm.ref]  # presence guaranteed by the caller's pre-flight

    def _run(item: Any) -> Sequence[Any]:
        inner = _execute_node(
            node, stages, registry, config, calibrated_values, signals, predicates
        )
        if inner.result_kind is ResultKind.ERROR:
            raise RuntimeError(f"nested composite {arm.ref!r} failed: {inner.error}")
        if inner.result_kind is ResultKind.NO_ACCEPT:
            raise _NestedNoAccept(arm.ref)
        return [inner.output]

    return StageSpec(name=arm.ref, run=_run, key_fn=None, samples=1)


@dataclass(frozen=True, slots=True)
class _ArmOutcome:
    """One cascade arm's evaluated outcome for the manual driver (F1).

    ``vote`` is the margin-bearing producer's :class:`VoteStats` (a stage's vote
    over its samples, OR a nested majority_vote ensemble's OWN vote surfaced to
    the gate); ``None`` for a non-voting terminal arm. ``kind`` is the §3.2.1
    outcome of THIS arm (``output`` / ``no_accept`` / ``error``).
    """

    output: Any
    vote: VoteStats | None
    kind: ResultKind
    error: str | None = None


def _has_gated_nested_arm(body: CascadeBody) -> bool:
    """A cascade has a GATED nested-composite arm iff a :class:`CompositeArm`
    sits at a NON-terminal index (i < m-1, i.e. a position followed by a gate).

    Only this case needs the manual driver (the gate must read the nested arm's
    OWN vote stats, and a gated arm's no_accept must ESCALATE — neither of which
    the shipped ``CascadePolicy.decide`` supports for a synthetic stage). The
    stage-only and terminal-nested cases keep the engine fast path verbatim.
    """
    last = len(body.arms) - 1
    return any(isinstance(arm, CompositeArm) for arm in body.arms[:last])


def _run_nested_majority_vote(
    node: CompositeNode,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> _ArmOutcome:
    """Run a nested majority_vote ensemble and SURFACE its VoteStats (F1a).

    The ONLY legal gated nested arm is a ``majority_vote`` ensemble (the §3.2
    margin-bearing-arm rule, ``gate_arm_incompatible``), whose own vote feeds the
    cascade gate. This collects the nested candidates, runs the SHIPPED
    ``vote_over`` (never a reimplementation), applies the nested ensemble's OWN
    accept gate, and returns the representative output + the surfaced vote:
    ``no_accept`` (accept-gate failure / all-excluded-by-no_accept) and ``error``
    (absorbing) are reported via :class:`_ArmOutcome.kind` so the cascade driver
    can ESCALATE (gated) or propagate (§3.2.1).
    """
    body = node.body
    if not (
        isinstance(body, EnsembleBody)
        and body.aggregate.kind is AggregateKind.MAJORITY_VOTE
    ):
        # Defensive: the IR forbids a non-margin-bearing gated nested arm; if one
        # reaches here it is a fail-closed error (never a silent gate read).
        return _ArmOutcome(
            output=None,
            vote=None,
            kind=ResultKind.ERROR,
            error=(
                f"composite-runtime: gated nested arm {node.name!r} is not a "
                "majority_vote ensemble (no margin to feed the gate)"
            ),
        )
    try:
        candidates, excluded_no_accept = _collect_candidates(
            body, stages, registry, config, calibrated_values, signals, predicates
        )
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return _ArmOutcome(
            output=None,
            vote=None,
            kind=ResultKind.ERROR,
            error=f"composite-runtime: {type(exc).__name__}: {exc}",
        )

    if not candidates:
        # All-excluded-by-no_accept -> no_accept; truly-no-candidates -> error.
        if excluded_no_accept > 0:
            return _ArmOutcome(output=None, vote=None, kind=ResultKind.NO_ACCEPT)
        return _ArmOutcome(
            output=None,
            vote=None,
            kind=ResultKind.ERROR,
            error="composite-runtime: nested ensemble produced no candidates",
        )

    output, vote = _majority_vote(candidates)
    accept = body.aggregate.accept
    if accept is not None:
        try:
            if not _accept_passes(accept, vote, calibrated_values):
                # The nested ensemble's accept gate failed -> nested no_accept;
                # the surfaced vote is still carried for the gate's margin read.
                return _ArmOutcome(output=None, vote=vote, kind=ResultKind.NO_ACCEPT)
        except Exception as exc:  # noqa: BLE001 - non-finite threshold -> error
            return _ArmOutcome(
                output=None,
                vote=None,
                kind=ResultKind.ERROR,
                error=f"composite-runtime: {type(exc).__name__}: {exc}",
            )
    return _ArmOutcome(output=output, vote=vote, kind=ResultKind.OUTPUT)


def _evaluate_stage_arm(
    arm: StageArm,
    stages: Mapping[str, StageEntry],
    *,
    gated: bool,
    config: Mapping[str, Any],
) -> _ArmOutcome:
    """Evaluate a leaf stage arm for the manual driver, mirroring the engine.

    Reproduces ``CascadePolicy.decide``'s per-stage semantics verbatim (sample
    cardinality enforcement, ``vote_over`` over the keys, deterministic
    representative). A gated stage with no ``key_fn`` is the SAME fail-closed
    error the engine raises ("feeds a gate but has no key_fn").
    """
    runner = _coerce_stage_runner(stages[arm.name])
    if runner.samples < 1:
        # Parity with the engine path: ``StageSpec.samples must be >= 1``
        # (cascade.py). Without this, a samples=0 stage self-consistently
        # produces zero outputs and the driver would emit a silent
        # ``OUTPUT None`` — fail closed instead.
        raise ValueError(
            f"stage {arm.name!r} declared samples={runner.samples}; "
            "StageSpec.samples must be >= 1"
        )
    samples = list(runner.run(config))
    if len(samples) != runner.samples:
        raise ValueError(
            f"stage {arm.name!r} declared samples={runner.samples} "
            f"but produced {len(samples)}"
        )
    if runner.key_fn is None:
        if gated:
            raise ValueError(
                f"stage {arm.name!r} feeds a gate but has no key_fn comparator; "
                "voting stages require one"
            )
        representative = samples[0] if samples else None
        return _ArmOutcome(output=representative, vote=None, kind=ResultKind.OUTPUT)
    keys = [runner.key_fn(sample) for sample in samples]
    vote = vote_over(keys, len(samples))
    representative = next(
        (
            sample
            for sample, key in zip(samples, keys, strict=False)
            if key == vote.top_key
        ),
        samples[0] if samples else None,
    )
    return _ArmOutcome(output=representative, vote=vote, kind=ResultKind.OUTPUT)


def _drive_cascade(
    body: CascadeBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """The manual post-cascade driver for the GATED-nested-arm case (F1).

    Reuses the SHIPPED gate math (:meth:`Gate.should_escalate`) and ``vote_over``
    — never a reimplementation. Semantics (§3.2.1):

    - a gated arm's ``no_accept`` ESCALATES (i < m); only the TERMINAL arm's
      ``no_accept`` becomes the cascade's ``no_accept`` (F1b);
    - a gated nested majority_vote ensemble's OWN vote feeds the gate (F1a);
    - an ``error`` is absorbing (propagates to the cascade error).
    """
    gates: list[Gate] = []
    for gate in body.gates:
        if gate.kind is not IRGateKind.MARGIN_BELOW:  # pragma: no cover - IR-guarded
            return _error(
                f"composite-runtime: post-cascade gate kind {gate.kind.value!r} "
                "is not executable (margin_below only)"
            )
        gates.append(
            Gate(
                kind=GateKind.MARGIN_BELOW,
                threshold_ref=_build_threshold_ref(gate.threshold, calibrated_values),
            )
        )

    escalations = 0
    last = len(body.arms) - 1
    for index, arm in enumerate(body.arms):
        is_last = index == last
        try:
            if isinstance(arm, StageArm):
                outcome = _evaluate_stage_arm(
                    arm, stages, gated=not is_last, config=config
                )
            elif is_last:
                # A terminal nested arm: any composite kind; no vote needed.
                inner = _execute_node(
                    registry[arm.ref],
                    stages,
                    registry,
                    config,
                    calibrated_values,
                    signals,
                    predicates,
                )
                outcome = _ArmOutcome(
                    output=inner.output,
                    vote=None,
                    kind=inner.result_kind,
                    error=inner.error,
                )
            else:
                # A GATED nested arm MUST be a majority_vote ensemble (F1a).
                outcome = _run_nested_majority_vote(
                    registry[arm.ref],
                    stages,
                    registry,
                    config,
                    calibrated_values,
                    signals,
                    predicates,
                )
        except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

        if outcome.kind is ResultKind.ERROR:
            return _error(
                outcome.error
                or f"composite-runtime: cascade arm {index} failed (no detail)"
            )

        if is_last:
            step = CascadeStep(
                stage_index=index,
                stage_name="",
                output=outcome.output,
                representative=outcome.output,
                vote=outcome.vote,
                escalations=escalations,
            )
            if outcome.kind is ResultKind.NO_ACCEPT:
                # Only the terminal arm's no_accept becomes the cascade's (F1b).
                return _no_accept(_cascade_measures(step, body))
            return CompositeRunResult(
                output=outcome.output,
                result_kind=ResultKind.OUTPUT,
                measures=_cascade_measures(step, body),
                error=None,
            )

        # A non-terminal (gated) arm.
        if outcome.kind is ResultKind.NO_ACCEPT:
            # F1b: a gated arm's no_accept ESCALATES to the next stage.
            escalations += 1
            continue
        runtime_gate = gates[index]
        if outcome.vote is None:  # pragma: no cover - guarded above for stages
            return _error(
                f"composite-runtime: gated cascade arm {index} has no vote to "
                "feed the gate (margin-bearing arm required)"
            )
        try:
            escalate = runtime_gate.should_escalate(outcome.vote)
        except Exception as exc:  # noqa: BLE001 - non-finite/unset threshold
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")
        if escalate:
            escalations += 1
            continue
        step = CascadeStep(
            stage_index=index,
            stage_name="",
            output=outcome.output,
            representative=outcome.output,
            vote=outcome.vote,
            escalations=escalations,
        )
        return CompositeRunResult(
            output=outcome.output,
            result_kind=ResultKind.OUTPUT,
            measures=_cascade_measures(step, body),
            error=None,
        )

    return _error(  # pragma: no cover - the loop always returns at the last arm
        "composite-runtime: cascade driver reached an unreachable terminal state"
    )


def _cascade_measures(step: CascadeStep, body: CascadeBody) -> dict[str, Any]:
    """The §3.10 cascade telemetry for one decided cascade — content-free.

    Keyed by GATE INDEX (duplicate threshold refs across gates are legal —
    ``n_cascade`` may reuse one CVAR — and name-keying would collapse per-gate
    values, the duplicate-ref lesson). ``escalation_rate`` is the per-run 0/1
    escalated indicator; ``stage_selected`` is the selected arm index.
    """
    selected = step.stage_index
    escalated = 1 if step.escalations > 0 else 0
    per_gate: dict[int, float] = {}
    for index, _gate in enumerate(body.gates):
        if index < selected:
            per_gate[index] = 0.0
        elif index == selected and index < len(body.arms) - 1:
            per_gate[index] = 1.0
    return {
        "escalation_rate": float(escalated),
        "stage_selected": selected,
        "gate_margin_pass_rate": per_gate,
    }


def _execute_cascade(
    node: CompositeNode,
    body: CascadeBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute one cascade body for one item (§3.2 / §3.2.1).

    Dispatches on placement: ``post`` runs the §3.8 escalation cascade (the
    fast :class:`CascadePolicy` path, or the manual driver when a GATED
    nested-composite arm needs the nested arm's own vote); ``pre`` runs the
    §3.2 dispatch (router) executor — gates score the input LEFT TO RIGHT,
    first match wins, exactly one routed arm runs (terminal-like).
    """
    if body.placement is Placement.PRE:
        return _execute_pre_cascade(
            node, body, stages, registry, config, calibrated_values, signals, predicates
        )

    # F1: a GATED nested-composite arm needs the manual driver (the gate must
    # read the nested arm's OWN vote, and a gated arm's no_accept ESCALATES).
    # Every other shape keeps the shipped CascadePolicy.decide fast path.
    if _has_gated_nested_arm(body):
        return _drive_cascade(
            body, stages, registry, config, calibrated_values, signals, predicates
        )

    policy = _adapt_cascade(
        body, stages, registry, config, calibrated_values, signals, predicates
    )
    try:
        step = policy.decide(config)
    except _NestedNoAccept:
        # §3.2.1: the cascade engine escalated past every gate (a nested arm's
        # no_accept never stops the cascade); reaching here means the LAST arm
        # was a nested composite that yielded no_accept -> cascade no_accept.
        return _no_accept({})
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

    return CompositeRunResult(
        output=step.output,
        result_kind=ResultKind.OUTPUT,
        measures=_cascade_measures(step, body),
        error=None,
    )


# --------------------------------------------------------------------------- #
# Pre-cascade DISPATCH execution (§3.2 routing; the router path)              #
# --------------------------------------------------------------------------- #


class _SignalRaised(Exception):
    """A pre-gate signal produced an ABSORBING ERROR — fails the item (§3.2.1).

    Carries the three malformed-signal cases that fail the item's evaluation
    (never silently routing), feeding the fail-closed law under strict modes:

    1. the signal *raised* an exception;
    2. the signal returned a ``bool`` or non-numeric value (a malformed score is
       a defect, not an abstention);
    3. the signal returned a non-finite ``float`` (NaN/±inf — it cannot enter the
       §3.2.1 finite ``σ ≥ θ`` comparison).

    All three are at PARITY with the loop ``signal_accept`` stop (runtime.py
    ~1587-1593), which likewise raises on bool/non-numeric/non-finite. The ONLY
    abstention §3.2 permits is a signal returning ``None`` (absent VALUE →
    inadequate → route onward), which is NOT carried here. This wrapper holds the
    detail so the dispatch driver can map it to a fail-closed ``error`` result.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _dispatch_adequate(
    gate: Any,
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
) -> tuple[bool, float | None, float | None]:
    """Evaluate ONE pre-gate against the dispatch input (§3.2 ``σ_i(x) ≥ θ_i``).

    Returns ``(adequate, sigma, theta)``:

    - ``adequate`` is ``σ(x) ≥ θ`` — selecting arm ``i``;
    - ``sigma`` / ``theta`` are the resolved finite values when the gate was
      evaluated numerically (for the §3.10 ``dispatch_signal_margin``);
      ``sigma`` is ``None`` only when the signal ABSTAINED — i.e. it returned
      ``None`` (the signal id maps to an absent VALUE). An abstention is
      INADEQUATE per §3.2 (route onward, "absent/abstaining signal value is
      INADEQUATE"); ``theta`` is still returned for symmetry but no margin is
      emitted.

    The signal is invoked with the dispatch input ``config`` (the item/config
    payload the stages receive) as a SINGLE positional argument — the SAME
    one-argument convention the ``signal_accept`` loop stop uses, with the
    dispatch payload instead of the threaded loop state. ``SignalUse.inputs``
    for a pre-gate is an opaque calibration/freshness declaration (§3.2 item 11
    ``signal_inputs`` / §3.5 ctx_ext binding), NOT a runtime input filter, so it
    never narrows what the signal observes (mirrors the loop's documented note).

    VALUE rules (parity with the loop ``signal_accept`` stop, §3.2.1 finite
    comparison law — "comparisons are over finite numbers ... never a silent
    comparison"):

    - ``None`` → ABSTENTION → inadequate, route onward (§3.2 explicit license);
    - ``bool`` or any non-numeric (str/dict/...) → RAISE (a malformed score is a
      defect, not an abstention) — absorbing ``error`` at the caller;
    - a non-finite ``float`` (NaN/±inf) → RAISE — a non-finite σ cannot enter the
      finite ``σ ≥ θ`` comparison; it is a malformed score, never a silent route.

    A missing/non-finite THRESHOLD raises (a calibration defect → ``error`` at
    the caller, exactly as POST does). A signal that RAISES is the §3.2
    absorbing-error case (re-raised as :class:`_SignalRaised`).
    """
    # θ is read LIVE and fails closed on missing/non-finite (calibration defect).
    theta = _resolve_threshold(gate.threshold, calibrated_values)

    signal_use = gate.signal  # presence guaranteed by IR (signal_below requires it)
    signal_id = signal_use.signal
    # The signal is preflighted into the registry (fail-closed); guard anyway.
    if signal_id not in signals:  # pragma: no cover - preflight short-circuits
        raise ValueError(
            f"signal {signal_id!r} is not provided (fail-closed: a missing "
            "dispatch signal fails the item's evaluation)"
        )
    try:
        raw = signals[signal_id](config)
    except Exception as exc:  # noqa: BLE001 - a RAISING signal is absorbing (§3.2)
        raise _SignalRaised(f"{type(exc).__name__}: {exc}") from exc

    # None ABSTAINS (§3.2: absent/abstaining value is INADEQUATE → route onward).
    if raw is None:
        return False, None, theta
    # A bool/non-numeric value is a MALFORMED SCORE, not an abstention: fail
    # closed (parity with the loop signal_accept rule, runtime.py ~1587). bool is
    # not a number (parity with the §3.10 measure rule).
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _SignalRaised(
            f"signal {signal_id!r} must return a finite number or None, got {raw!r}"
        )
    sigma = float(raw)
    if not math.isfinite(sigma):
        # A non-finite σ cannot enter the §3.2.1 finite ``σ ≥ θ`` comparison;
        # it is a malformed score → absorbing error (parity with the loop,
        # runtime.py ~1592), NOT a silent fail-toward-the-stronger-arm route.
        raise _SignalRaised(f"signal {signal_id!r} returned non-finite {sigma!r}")
    return sigma >= theta, sigma, theta


def _run_routed_arm(
    arm: Arm,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> _ArmOutcome:
    """Execute the routed arm — TERMINAL-like (§3.2.1: its result IS final).

    A stage arm runs exactly like a non-gated cascade arm (a routed stage is
    never feeding a gate — routing already decided — so a non-voting
    representative is fine; a key_fn'd stage still votes for its representative).
    A nested-composite arm runs its OWN §3.2 semantics (any kind), mirroring the
    POST terminal-nested branch — the dispatch restriction "gated arms must be
    margin-bearing" does NOT apply because a PRE-routed arm is terminal, never
    gated. The arm's ``no_accept`` becomes the cascade's ``no_accept`` (NO
    re-route) and its ``error`` is absorbing — both surfaced via
    :class:`_ArmOutcome.kind`.
    """
    if isinstance(arm, StageArm):
        # gated=False: the routed stage feeds no gate (routing already decided).
        return _evaluate_stage_arm(arm, stages, gated=False, config=config)
    inner = _execute_node(
        registry[arm.ref],
        stages,
        registry,
        config,
        calibrated_values,
        signals,
        predicates,
    )
    return _ArmOutcome(
        output=inner.output,
        vote=None,
        kind=inner.result_kind,
        error=inner.error,
    )


def _pre_cascade_measures(
    selected: int,
    adequacy: dict[int, int],
    selected_margin: float | None,
) -> dict[str, Any]:
    """The §3.10 pre-cascade (dispatch) telemetry — content-free.

    - ``route_selected``: the selected arm index (ALWAYS emitted);
    - ``gate_signal_adequate``: an INDEX-keyed ``{i: 0|1}`` map of per-gate
      adequacy for every EVALUATED gate (gates not reached — beyond the
      selected arm — are absent, never a false 0);
    - ``dispatch_signal_margin``: ``σ_sel − θ_sel``, emitted ONLY when a GATED
      arm was selected (``selected_margin is not None``); OMITTED for terminal
      fall-through (no gate selected the terminal arm).
    """
    measures: dict[str, Any] = {
        "route_selected": selected,
        "gate_signal_adequate": adequacy,
    }
    if selected_margin is not None:
        measures["dispatch_signal_margin"] = selected_margin
    return measures


def _execute_pre_cascade(
    node: CompositeNode,
    body: CascadeBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute one pre-cascade (dispatch) body for one item (§3.2 / §3.2.1).

    The §3.2 dispatch law::

        route(x) = arm_j  where  j = min({ i < m : σ_i(x) ≥ θ_i } ∪ {m})

    Gates are evaluated LEFT TO RIGHT on the dispatch input ``config``; the
    FIRST adequate gate selects its arm and STOPS (no other gate is even
    evaluated); all gates inadequate → the terminal fallback arm ``m``. Exactly
    ONE arm executes. Semantics:

    - a signal returning ``None`` ABSTAINS (absent VALUE) → INADEQUATE: routing
      continues to the next gate / terminal arm (§3.2 explicit license:
      "absent/abstaining signal value is INADEQUATE"), NEVER an error;
    - a signal returning ``bool``/non-numeric/non-finite is a MALFORMED SCORE →
      absorbing ``error`` (a malformed score is a defect, not an abstention) —
      parity with the loop ``signal_accept`` stop and the §3.2.1 finite-
      comparison law; no arm executes;
    - a signal that RAISES is the §3.2 absorbing ``error`` — no arm executes;
    - a reached gate's missing/non-finite THRESHOLD is a calibration defect →
      ``error`` (fail closed, exactly as POST);
    - the routed arm is TERMINAL: its ``no_accept`` is the cascade's
      ``no_accept`` (NO re-route, §3.2.1) and its ``error`` is absorbing.
    """
    gates = body.gates
    last = len(body.arms) - 1  # the terminal fallback arm index (m)
    adequacy: dict[int, int] = {}
    selected_index = last
    selected_margin: float | None = None

    for index, gate in enumerate(gates):
        try:
            adequate, sigma, theta = _dispatch_adequate(
                gate, config, calibrated_values, signals
            )
        except _SignalRaised as raised:
            # §3.2.1: a signal that RAISES, or returns a malformed score
            # (bool/non-numeric/non-finite), fails the item (absorbing error).
            return _error(f"composite-runtime: {raised.detail}")
        except Exception as exc:  # noqa: BLE001 - missing/non-finite θ -> error
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")
        adequacy[index] = 1 if adequate else 0
        if adequate:
            selected_index = index
            # σ and θ are both finite here (adequate implies a numeric σ).
            selected_margin = float(sigma) - float(theta)  # type: ignore[arg-type]
            break

    # Exactly the selected arm executes (the routed arm is TERMINAL-like).
    try:
        outcome = _run_routed_arm(
            body.arms[selected_index],
            stages,
            registry,
            config,
            calibrated_values,
            signals,
            predicates,
        )
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

    measures = _pre_cascade_measures(selected_index, adequacy, selected_margin)

    if outcome.kind is ResultKind.ERROR:
        return _error(
            outcome.error
            or f"composite-runtime: routed arm {selected_index} failed (no detail)"
        )
    if outcome.kind is ResultKind.NO_ACCEPT:
        # §3.2.1: the routed arm's no_accept IS the cascade's no_accept — routing
        # selects the arm; it does NOT re-route on non-acceptance.
        return _no_accept(measures)
    return CompositeRunResult(
        output=outcome.output,
        result_kind=ResultKind.OUTPUT,
        measures=measures,
        error=None,
    )


# --------------------------------------------------------------------------- #
# Ensemble execution (§3.2 sampling / committee; majority_vote / judge_max)   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One ensemble candidate output plus its content-free equivalence key."""

    output: Any
    key: VoteKey


def _collect_candidates(
    body: EnsembleBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> tuple[list[_Candidate], int]:
    """Run the ensemble arms and collect candidate outputs (§3.2).

    Sampling form (``|arms| = 1``): the single arm runs ``k`` times (``k`` the
    resolved cardinality). Committee form (``|arms| > 1``): each arm runs once;
    a committee arm yielding ``no_accept`` is EXCLUDED from the vote (§3.2.1),
    an arm error propagates (absorbing). Returns the surviving candidates and
    the count excluded BY ``no_accept`` (NOT by error — error already raised).
    """
    excluded_no_accept = 0
    if body.cardinality is not None:  # sampling form
        k = _resolve_cardinality(body.cardinality, config, calibrated_values)
        arm = body.arms[0]
        if isinstance(arm, StageArm):
            runner = _coerce_stage_runner(stages[arm.name])
            samples = list(runner.run(config))
            key_fn = runner.key_fn or (lambda x: x)
            candidates = [_Candidate(output=s, key=key_fn(s)) for s in samples]
            # Sampling over a stage produces k outputs; cap the declared vote
            # denominator at k so margins never exceed 1 (the vote_over contract).
            if len(candidates) != k:
                raise ValueError(
                    f"sampling ensemble arm {arm.name!r} declared k={k} "
                    f"but produced {len(candidates)} samples"
                )
            return candidates, excluded_no_accept
        # F2: a single-arm SAMPLING ensemble over a nested COMPOSITE runs that
        # composite k times — each run is ONE candidate (keyed by its output via
        # the nested run's own selection semantics). A nested error is absorbing;
        # a nested no_accept excludes that draw (§3.2.1).
        node = registry[arm.ref]
        sampled: list[_Candidate] = []
        for _draw in range(k):
            inner = _execute_node(
                node, stages, registry, config, calibrated_values, signals, predicates
            )
            if inner.result_kind is ResultKind.ERROR:
                raise RuntimeError(
                    f"sampling ensemble arm {arm.ref!r} failed: {inner.error}"
                )
            if inner.result_kind is ResultKind.NO_ACCEPT:
                excluded_no_accept += 1
                continue
            sampled.append(_Candidate(output=inner.output, key=inner.output))
        return sampled, excluded_no_accept

    candidates = []  # committee form
    for arm in body.arms:
        if isinstance(arm, StageArm):
            runner = _coerce_stage_runner(stages[arm.name])
            samples = list(runner.run(config))
            if not samples:
                continue  # an arm that produced nothing contributes no candidate
            # F3: a committee arm contributes EXACTLY ONE candidate — its
            # SELECTED output (the majority winner over its OWN samples per the
            # RFC 0001 vote semantics), NOT one candidate per raw sample (which
            # would give a multi-sample member inflated vote weight).
            key_fn = runner.key_fn or (lambda x: x)
            arm_keys = [key_fn(sample) for sample in samples]
            arm_vote = vote_over(arm_keys, len(arm_keys))
            representative = next(
                (
                    sample
                    for sample, key in zip(samples, arm_keys, strict=False)
                    if key == arm_vote.top_key
                ),
                samples[0],
            )
            candidates.append(
                _Candidate(output=representative, key=key_fn(representative))
            )
        else:  # nested-composite committee arm
            inner = _execute_node(
                registry[arm.ref],
                stages,
                registry,
                config,
                calibrated_values,
                signals,
                predicates,
            )
            if inner.result_kind is ResultKind.ERROR:
                raise RuntimeError(f"committee arm {arm.ref!r} failed: {inner.error}")
            if inner.result_kind is ResultKind.NO_ACCEPT:
                excluded_no_accept += 1
                continue
            candidates.append(_Candidate(output=inner.output, key=inner.output))
    return candidates, excluded_no_accept


def _majority_vote(candidates: Sequence[_Candidate]) -> tuple[Any, VoteStats]:
    """Aggregate candidates by ``majority_vote`` via the shipped ``vote_over``.

    Reuses :func:`~traigent.knobs.cascade.vote_over` verbatim (RFC 0001
    tie/abstain, deterministic total order over serialized keys) — never a
    reimplementation. Returns the winning representative output and the vote
    stats (the margin-bearing producer's ``vote_stats``, §3.2.1).
    """
    keys = [c.key for c in candidates]
    vote = vote_over(keys, len(keys))
    representative = next(
        (c.output for c in candidates if c.key == vote.top_key),
        candidates[0].output if candidates else None,
    )
    return representative, vote


def _judge_max(
    candidates: Sequence[_Candidate],
    judge: Arm,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> tuple[Any, int]:
    """Aggregate candidates by ``judge_max`` (§3.2 judge output contract).

    The judge arm is invoked once per candidate and MUST yield a FINITE numeric
    score. A candidate whose judging raises or yields NaN/±∞/non-numeric is
    EXCLUDED (judge-contract exclusion). If ALL candidates are excluded the
    ensemble FAILS (raise — §3.2.1 all-excluded-by-error). Selection is the max
    score; ties break by the deterministic total order over serialized keys
    (the §3.2 tie-break). Returns the winning output and the
    ``candidates_excluded`` count (judge-contract violations).
    """
    judge_run = _judge_callable(
        judge, stages, registry, config, calibrated_values, signals, predicates
    )
    scored: list[tuple[float, str, Any]] = []
    excluded = 0
    for candidate in candidates:
        try:
            raw = judge_run(candidate.output)
        except Exception:  # noqa: BLE001 - judge-contract exclusion (§3.2)
            excluded += 1
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            excluded += 1
            continue
        score = float(raw)
        if not math.isfinite(score):
            excluded += 1
            continue
        scored.append((score, str(candidate.key), candidate.output))
    if not scored:
        # ALL excluded BY ERROR (judge contract) -> the ensemble FAILS (§3.2.1).
        raise RuntimeError(
            "judge_max: all candidates excluded by judge-contract violation "
            "(no finite numeric score) — the ensemble evaluation fails (§3.2.1)"
        )
    # max score; deterministic tie-break by the serialized-key total order.
    best = max(scored, key=lambda t: (t[0], _neg_key(t[1])))
    return best[2], excluded


def _neg_key(serialized: str) -> Any:
    """A tie-break helper: among equal max scores, the SMALLEST serialized key
    wins (the §3.2 deterministic total order over serialized keys). ``max``
    selects the LARGEST tuple, so we invert the key ordering with a wrapper that
    reverses string comparison."""
    return _ReverseStr(serialized)


@dataclass(frozen=True, slots=True)
class _ReverseStr:
    """A string wrapper whose ordering is REVERSED, so that under ``max`` the
    lexicographically SMALLEST serialized key is chosen on a score tie (§3.2
    deterministic total order)."""

    value: str

    def __lt__(self, other: _ReverseStr) -> bool:
        return self.value > other.value


def _judge_callable(
    judge: Arm,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> Callable[[Any], Any]:
    """Build the per-candidate judge invocation.

    A stage judge runs once per candidate (the candidate output is the judge's
    item). A nested-composite judge executes the nested composite per candidate;
    its ``output`` is the score (an error/no_accept from the judge composite is
    a judge-contract exclusion at the ``_judge_max`` call site via the raise).
    """
    if isinstance(judge, StageArm):
        runner = _coerce_stage_runner(stages[judge.name])

        def _stage_judge(candidate_output: Any) -> Any:
            produced = list(runner.run(candidate_output))
            if not produced:
                raise ValueError("judge stage produced no output")
            return produced[0]

        return _stage_judge

    node = registry[judge.ref]

    def _composite_judge(candidate_output: Any) -> Any:
        inner = _execute_node(
            node,
            stages,
            registry,
            candidate_output,
            calibrated_values,
            signals,
            predicates,
        )
        if inner.result_kind is not ResultKind.OUTPUT:
            raise RuntimeError(
                f"nested judge composite {judge.ref!r} did not produce an output"
            )
        return inner.output

    return _composite_judge


def _stage_name(arm: Arm) -> str:
    """The lookup key for an arm in ``stages`` (stage name or composite ref)."""
    if isinstance(arm, StageArm):
        return str(arm.name)
    return str(arm.ref)


def _ensemble_measures(
    vote: VoteStats | None, evaluated: int, excluded: int
) -> dict[str, Any]:
    """The §3.10 ensemble telemetry — content-free.

    ``vote_agreement``/``vote_margin`` come from the vote stats (0.0 when there
    is no vote, e.g. a ``judge_max`` aggregate — no majority vote ran);
    ``candidates_evaluated`` is the count fed to aggregation;
    ``candidates_excluded`` is the judge-contract / no_accept exclusion count.
    """
    return {
        "vote_agreement": float(vote.valid_rate) if vote is not None else 0.0,
        "vote_margin": float(vote.margin) if vote is not None else 0.0,
        "candidates_evaluated": int(evaluated),
        "candidates_excluded": int(excluded),
    }


def _execute_ensemble(
    body: EnsembleBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute one ensemble body for one item (§3.2 / §3.2.1)."""
    try:
        candidates, excluded_no_accept = _collect_candidates(
            body, stages, registry, config, calibrated_values, signals, predicates
        )
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

    aggregate = body.aggregate
    evaluated = len(candidates)

    # A committee whose candidates are ALL excluded by no_accept -> the ensemble
    # yields no_accept (§3.2.1, DISTINCT from all-excluded-by-error).
    if not candidates:
        if excluded_no_accept > 0:
            return _no_accept(
                _ensemble_measures(None, evaluated=0, excluded=excluded_no_accept)
            )
        return _error(
            "composite-runtime: ensemble produced no candidates (no arms ran)"
        )

    vote: VoteStats | None = None
    judge_excluded = 0
    try:
        if aggregate.kind is AggregateKind.MAJORITY_VOTE:
            output, vote = _majority_vote(candidates)
        else:  # JUDGE_MAX
            if aggregate.judge is None:  # pragma: no cover - IR-guarded
                raise ValueError("judge_max aggregate requires a judge arm")
            output, judge_excluded = _judge_max(
                candidates,
                aggregate.judge,
                stages,
                registry,
                config,
                calibrated_values,
                signals,
                predicates,
            )
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

    total_excluded = excluded_no_accept + judge_excluded
    measures = _ensemble_measures(vote, evaluated=evaluated, excluded=total_excluded)

    # The §3.2.1 accept gate: stat >= θ, the OPPOSITE direction of cascade
    # escalation. Failing it is an HONEST no_accept (NOT an error).
    accept = aggregate.accept
    if accept is not None:
        try:
            decision = _accept_passes(accept, vote, calibrated_values)
        except Exception as exc:  # noqa: BLE001 - non-finite threshold -> error
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")
        if not decision:
            return _no_accept(measures)

    return CompositeRunResult(
        output=output, result_kind=ResultKind.OUTPUT, measures=measures, error=None
    )


def _accept_passes(
    accept: AcceptDecl,
    vote: VoteStats | None,
    calibrated_values: Mapping[str, float | int],
) -> bool:
    """Evaluate the §3.2 acceptance inequality ``stat >= θ``.

    The named content-free statistic (``vote_margin`` / ``vote_agreement``) is
    read from the vote stats. A ``judge_max`` aggregate has no vote, so an
    accept decl over it has no statistic to read — that is an IR-level
    misconfiguration (accept stats are vote stats), surfaced as an error here.
    """
    theta = _resolve_threshold(accept.threshold, calibrated_values)
    if vote is None:
        raise ValueError(
            f"accept stat {accept.stat.value!r} has no vote statistic to read "
            "(an accept decl over a non-voting aggregate is misconfigured)"
        )
    if accept.stat is StatKind.VOTE_MARGIN:
        stat = float(vote.margin)
    else:  # VOTE_AGREEMENT
        stat = float(vote.valid_rate)
    return bool(stat >= theta)


# --------------------------------------------------------------------------- #
# Loop execution (§3.2 signal_accept / external_accept / exhausted)           #
# --------------------------------------------------------------------------- #


def _restrict_state(
    state: Mapping[str, Any], state_keys: tuple[str, ...]
) -> dict[str, Any]:
    """Project ``state`` onto the declared ``state_keys`` (§3.2).

    Undeclared keys are DROPPED (documented: invisible to ``stop``). Declared
    keys the body did not produce are simply absent.
    """
    allowed = set(state_keys)
    return {k: v for k, v in state.items() if k in allowed}


def _execute_loop(
    node: CompositeNode,
    body: LoopBody,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute one loop body up to ``max_iters`` (§3.2 / §3.2.1 / §3.10).

    State threads over the declared ``state_keys`` (iteration ``i+1`` observes
    exactly the keys iteration ``i`` produced, ∩ ``state_keys``). Stop kinds:
    ``signal_accept`` (σ(state) ≥ θ accepts), ``external_accept`` (opaque
    predicate), ``exhausted`` (runs all ``max_iters``). A body/stop exception
    fails the item (absorbing error). A loop exhausting without acceptance, or
    whose final body result is ``no_accept``, yields ``no_accept``.
    """
    stop = body.stop
    try:
        runner = _loop_body_runner(body.body, stages, registry)
    except Exception as exc:  # noqa: BLE001 - missing body callable -> error
        return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

    state: dict[str, Any] = {}
    last_output: Any = None
    last_kind: ResultKind = ResultKind.NO_ACCEPT
    iterations_used = 0

    for _iteration in range(body.max_iters):
        iterations_used += 1
        try:
            result = _invoke_loop_body(runner, config, state, body.state_keys)
        except Exception as exc:  # noqa: BLE001 - body exception is absorbing
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

        if result.result_kind is ResultKind.ERROR:  # a nested body error
            return _error(
                f"composite-runtime: loop body of {node.name!r} failed: "
                f"{result.error if isinstance(result, CompositeRunResult) else 'error'}"
            )

        last_output = result.output
        last_kind = result.result_kind
        state = _restrict_state(result.state, body.state_keys)

        # §3.2.1: a no_accept body result completes the iteration unaccepted;
        # the loop CONTINUES (no stop is evaluated on a non-output iteration —
        # there is no accepted state to test).
        if result.result_kind is ResultKind.NO_ACCEPT:
            continue

        # An output iteration: evaluate the stop on the produced state.
        try:
            accepted = _stop_accepts(
                stop, state, calibrated_values, signals, predicates
            )
        except Exception as exc:  # noqa: BLE001 - stop exception is absorbing
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")

        if accepted:
            measures = _loop_measures(iterations_used, _accept_reason(stop.kind))
            return CompositeRunResult(
                output=last_output,
                result_kind=ResultKind.OUTPUT,
                measures=measures,
                error=None,
            )

    # max_iters reached without acceptance, OR an exhausted loop whose final
    # body result is no_accept -> the loop yields no_accept (§3.2.1).
    measures = _loop_measures(iterations_used, StopReason.EXHAUSTED)
    if stop.kind is StopKind.EXHAUSTED and last_kind is ResultKind.OUTPUT:
        # An exhausted loop with a produced final output: that output IS the
        # loop's output (the exhausted stop has no acceptance predicate, so the
        # final produced output is the result, NOT no_accept) — §3.2 exhausted
        # "always runs max_iters iterations" and yields its final body output.
        return CompositeRunResult(
            output=last_output,
            result_kind=ResultKind.OUTPUT,
            measures=measures,
            error=None,
        )
    return _no_accept(measures)


def _loop_body_runner(
    body_arm: Arm,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
) -> LoopBodyRunner | CompositeNode:
    """Resolve the loop body to a :class:`LoopBodyRunner` or a nested node.

    A stage body resolves through ``stages``; a nested-composite body resolves
    through the registry (executed per iteration with the threaded state passed
    as ``config`` overlay — see :func:`_invoke_loop_body`).
    """
    if isinstance(body_arm, StageArm):
        return _coerce_loop_runner(stages[body_arm.name])
    return registry[body_arm.ref]


def _invoke_loop_body(
    runner: LoopBodyRunner | CompositeNode,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    state_keys: tuple[str, ...],
) -> LoopBodyResult:
    """Invoke the loop body for one iteration with the restricted state.

    A :class:`LoopBodyRunner` is called ``run(item, restricted_state)``; a bare
    value return is wrapped as an output with empty produced state. A nested
    composite is NOT executed here (it has no state-producing convention in v1)
    — a nested-composite loop body is rejected upstream as a deferral.
    """
    if isinstance(runner, CompositeNode):  # pragma: no cover - rejected upstream
        raise NotImplementedError(
            "composite-runtime: nested-composite loop bodies are deferred "
            "(no v1 state-producing convention for a composite body)"
        )
    restricted = _restrict_state(state, state_keys)
    raw = runner.run(config, restricted)
    if isinstance(raw, LoopBodyResult):
        return raw
    return LoopBodyResult(output=raw)


def _stop_accepts(
    stop: Any,
    state: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> bool:
    """Evaluate the loop stop on the produced state (§3.2).

    - ``signal_accept``: σ(threaded state) ≥ θ accepts. ``state`` is ALREADY
      restricted to the loop's declared ``state_keys`` by the caller (undeclared
      keys dropped, §3.2). The signal receives that FULL threaded view — the
      SAME view the K-chain's σ receives (§3.8 trace-equality). The declared
      ``signal.inputs ⊆ state_keys`` is a calibration/freshness COVERAGE
      declaration (§3.2 item 11 / `signal_inputs`), NOT a runtime input filter,
      so it never narrows what σ observes. A missing signal callable or a
      non-finite/absent threshold fails closed (raise -> error); σ MUST return a
      finite number.
    - ``external_accept``: the opaque predicate over the state decides; a
      missing predicate fails closed.
    - ``exhausted``: never accepts mid-loop (runs all iterations).
    """
    if stop.kind is StopKind.SIGNAL_ACCEPT:
        if stop.signal is None or stop.threshold is None:  # pragma: no cover - IR
            raise ValueError("signal_accept stop missing signal/threshold")
        signal_id = stop.signal.signal
        if signal_id not in signals:
            raise ValueError(
                f"signal {signal_id!r} is not provided (fail-closed: a missing "
                "signal fails the item's evaluation)"
            )
        theta = _resolve_threshold(stop.threshold, calibrated_values)
        raw = signals[signal_id](dict(state))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"signal {signal_id!r} must return a finite number, got {raw!r}"
            )
        sigma = float(raw)
        if not math.isfinite(sigma):
            raise ValueError(f"signal {signal_id!r} returned non-finite {sigma!r}")
        return sigma >= theta

    if stop.kind is StopKind.EXTERNAL_ACCEPT:
        predicate_id = stop.predicate
        if predicate_id not in predicates:
            raise ValueError(
                f"predicate {predicate_id!r} is not provided (fail-closed: a "
                "missing predicate fails the item's evaluation)"
            )
        return bool(predicates[predicate_id](dict(state)))

    return False  # EXHAUSTED never accepts mid-loop


def _accept_reason(stop_kind: StopKind) -> StopReason:
    """Map an accepting stop kind to its §3.10 ``stop_reason`` enum value."""
    if stop_kind is StopKind.SIGNAL_ACCEPT:
        return StopReason.SIGNAL_ACCEPT
    return StopReason.EXTERNAL_ACCEPT


def _loop_measures(iterations_used: int, stop_reason: StopReason) -> dict[str, Any]:
    """The §3.10 loop telemetry — content-free (count + enum)."""
    return {
        "iterations_used": int(iterations_used),
        "stop_reason": stop_reason.value,
    }


# --------------------------------------------------------------------------- #
# K-chain unroll execution (§3.8 — signal_accept loops only)                  #
# --------------------------------------------------------------------------- #


def execute_kchain(
    chain: Any,
    body_runner: LoopBodyRunner | Callable[..., Any],
    *,
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute a §3.8 K-chain IR object (the ``self_refine.unroll(K)`` return).

    A K-chain is the SDK intermediate-representation compilation of a
    ``signal_accept`` loop: ``stages = [b₁ .. b_K]`` escalating to stage ``i+1``
    exactly when ``σ(state_i) < θ`` (the acceptance-direction complement). This
    executor walks the chain stages, threading the body's state over
    ``chain.state_keys`` and reading the live ``chain.threshold`` and
    ``chain.signal`` — so its execution TRACE (selected stage index, accept
    decision, output) matches the live :func:`_execute_loop` under the §3.8
    conditions 1–3 (the property test asserts trace-equality).

    Args:
        chain: a :class:`~traigent.knobs.patterns.KChain` (duck-typed:
            ``signal``, ``threshold``, ``state_keys``, ``stages``).
        body_runner: the loop body, SAME convention as the loop's body — a
            :class:`LoopBodyRunner` (or a bare callable wrapped pure).
        config: the trial assignment passed as the body item.
        calibrated_values: holds ``chain.threshold`` (read via the live ref).
        signals: holds ``chain.signal`` (fail-closed if absent).

    Returns:
        ``OUTPUT`` at the first chain stage whose ``σ(state_i) ≥ θ`` accepts;
        ``no_accept`` when the whole chain escalates without accepting (the
        exhausted-equivalent of the live loop); ``error`` (absorbing) for a
        body/signal exception, a missing signal, or a non-finite threshold.
    """
    runner = _coerce_loop_runner(body_runner)
    state_keys = tuple(chain.state_keys)
    state: dict[str, Any] = {}
    last_output: Any = None
    stages = list(chain.stages)

    for index, _stage in enumerate(stages):
        iterations_used = index + 1
        try:
            raw = runner.run(config, _restrict_state(state, state_keys))
            result = raw if isinstance(raw, LoopBodyResult) else LoopBodyResult(raw)
        except Exception as exc:  # noqa: BLE001 - body exception is absorbing
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")
        if result.result_kind is ResultKind.ERROR:
            # F4 parity: a body ERROR result-kind is the §3.2.1 absorbing error
            # — the SAME outcome the live loop produces. The chain must NOT
            # treat the error cell's .output as a produced output (fail closed).
            return _error(
                f"composite-runtime: K-chain body of {chain.signal!r} failed: "
                f"{result.error if isinstance(result, CompositeRunResult) else 'error'}"
            )
        if result.result_kind is ResultKind.NO_ACCEPT:
            # a no_accept body iteration: escalate (no accepted state to test).
            state = _restrict_state(result.state, state_keys)
            continue
        last_output = result.output
        state = _restrict_state(result.state, state_keys)
        try:
            sigma = _kchain_sigma(chain, state, signals)
            theta = _resolve_threshold(chain.threshold, calibrated_values)
        except Exception as exc:  # noqa: BLE001 - signal/threshold absorbing
            return _error(f"composite-runtime: {type(exc).__name__}: {exc}")
        if sigma >= theta:  # ¬escalate: the signal_accept stop fired
            return CompositeRunResult(
                output=last_output,
                result_kind=ResultKind.OUTPUT,
                measures=_loop_measures(iterations_used, StopReason.SIGNAL_ACCEPT),
                error=None,
            )
        # σ < θ: escalate to the next chain stage (acceptance-direction
        # complement, §3.8).

    return _no_accept(_loop_measures(len(stages), StopReason.EXHAUSTED))


def _kchain_sigma(
    chain: Any, state: Mapping[str, Any], signals: Mapping[str, Callable[..., Any]]
) -> float:
    """Evaluate the chain's ``signal_accept`` signal σ over the threaded state.

    Identical signal contract to the live loop's ``signal_accept`` (§3.8
    condition 1/2): a missing signal fails closed; σ must return a finite
    number. The signal reads the state restricted to the chain's declared
    state_keys (the chain carries no separate ``inputs`` — the loop's
    ``signal.inputs ⊆ state_keys`` is structurally enforced, §3.8 condition 2).
    """
    signal_id = chain.signal
    if signal_id not in signals:
        raise ValueError(
            f"signal {signal_id!r} is not provided (fail-closed: a missing "
            "signal fails the item's evaluation)"
        )
    raw = signals[signal_id](dict(state))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"signal {signal_id!r} must return a finite number")
    sigma = float(raw)
    if not math.isfinite(sigma):
        raise ValueError(f"signal {signal_id!r} returned non-finite {sigma!r}")
    return sigma


# --------------------------------------------------------------------------- #
# Dispatch + the public surface                                               #
# --------------------------------------------------------------------------- #


def _missing_stage_names(
    node: CompositeNode, stages: Mapping[str, StageEntry]
) -> list[str]:
    """The stage-arm names a node references but ``stages`` does not provide.

    Composite-arm refs are validated against the registry (a separate
    fail-closed pre-flight); this only checks the LEAF stage callables.
    """
    body = node.body
    needed: list[str] = []
    if isinstance(body, CascadeBody):
        needed = [arm.name for arm in body.arms if isinstance(arm, StageArm)]
    elif isinstance(body, EnsembleBody):
        needed = [arm.name for arm in body.arms if isinstance(arm, StageArm)]
        judge = body.aggregate.judge
        if isinstance(judge, StageArm):
            needed.append(judge.name)
    else:  # LoopBody
        if isinstance(body.body, StageArm):
            needed = [body.body.name]
    return [name for name in needed if name not in stages]


def _missing_composite_refs(
    node: CompositeNode, registry: Mapping[str, CompositeNode]
) -> list[str]:
    """The composite-arm refs a node references but the registry lacks."""
    body = node.body
    refs: list[str] = []
    if isinstance(body, (CascadeBody, EnsembleBody)):
        refs = [arm.ref for arm in body.arms if isinstance(arm, CompositeArm)]
        if isinstance(body, EnsembleBody) and isinstance(
            body.aggregate.judge, CompositeArm
        ):
            refs.append(body.aggregate.judge.ref)
    elif isinstance(body, LoopBody) and isinstance(body.body, CompositeArm):
        refs = [body.body.ref]
    return [ref for ref in refs if ref not in registry]


def _preflight_reachable(
    root: CompositeNode,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult | None:
    """F5: a deep fail-closed preflight over ALL composites reachable from the
    root through the registry (§3.2.1).

    The per-node check in :func:`_execute_node` only sees the CURRENT node, so a
    missing leaf stage/signal/predicate inside an UNSELECTED nested arm would
    slip through and the root would return a silent output. This walks the whole
    reachable sub-DAG up front: ANY reachable missing composite ref, leaf stage
    callable, ``signal_accept`` signal, or ``external_accept`` predicate is a
    fail-closed ``error`` BEFORE any arm runs. Returns the error result, or
    ``None`` when every reachable obligation is satisfied.
    """
    visited: set[str] = set()
    stack: list[CompositeNode] = [root]
    while stack:
        node = stack.pop()
        if node.name in visited:
            continue
        visited.add(node.name)

        missing_stages = _missing_stage_names(node, stages)
        if missing_stages:
            return _error(
                "composite-runtime: missing stage callable(s) "
                f"{sorted(missing_stages)!r} for composite {node.name!r} "
                "(fail-closed: a missing stage fails the item's evaluation)"
            )
        missing_refs = _missing_composite_refs(node, registry)
        if missing_refs:
            return _error(
                "composite-runtime: missing composite ref(s) "
                f"{sorted(missing_refs)!r} for composite {node.name!r} "
                "(fail-closed: an unresolved nested arm fails the item's evaluation)"
            )

        body = node.body
        if isinstance(body, LoopBody):
            stop = body.stop
            if stop.kind is StopKind.SIGNAL_ACCEPT and stop.signal is not None:
                if stop.signal.signal not in signals:
                    return _error(
                        f"composite-runtime: signal {stop.signal.signal!r} is not "
                        f"provided for composite {node.name!r} (fail-closed: a "
                        "missing signal fails the item's evaluation)"
                    )
            elif stop.kind is StopKind.EXTERNAL_ACCEPT and stop.predicate is not None:
                if stop.predicate not in predicates:
                    return _error(
                        f"composite-runtime: predicate {stop.predicate!r} is not "
                        f"provided for composite {node.name!r} (fail-closed: a "
                        "missing predicate fails the item's evaluation)"
                    )
        elif isinstance(body, CascadeBody) and body.placement is Placement.PRE:
            # A pre-cascade (dispatch) gate's routing signal must be in the
            # registry FAIL-CLOSED — a missing dispatch signal fails the item's
            # evaluation BEFORE any arm runs, mirroring the signal_accept stop
            # preflight (the IR guarantees a signal_below gate carries a signal).
            for gate in body.gates:
                if gate.signal is not None and gate.signal.signal not in signals:
                    return _error(
                        f"composite-runtime: signal {gate.signal.signal!r} is not "
                        f"provided for composite {node.name!r} (fail-closed: a "
                        "missing dispatch signal fails the item's evaluation)"
                    )

        # Recurse into every present nested composite ref (missing refs already
        # short-circuited above, so every ref here resolves).
        for arm in _reachable_refs(node):
            child = registry.get(arm)
            if child is not None and child.name not in visited:
                stack.append(child)
    return None


def _reachable_refs(node: CompositeNode) -> list[str]:
    """The nested-composite refs a node directly references (arms + judge +
    loop body) — the edges of the §3.2 reference sub-DAG the preflight walks."""
    body = node.body
    refs: list[str] = []
    if isinstance(body, (CascadeBody, EnsembleBody)):
        refs = [arm.ref for arm in body.arms if isinstance(arm, CompositeArm)]
        if isinstance(body, EnsembleBody) and isinstance(
            body.aggregate.judge, CompositeArm
        ):
            refs.append(body.aggregate.judge.ref)
    elif isinstance(body, LoopBody) and isinstance(body.body, CompositeArm):
        refs = [body.body.ref]
    return refs


def _execute_node(
    node: CompositeNode,
    stages: Mapping[str, StageEntry],
    registry: Mapping[str, CompositeNode],
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]],
    predicates: Mapping[str, Callable[..., Any]],
) -> CompositeRunResult:
    """Execute ONE composite node (the recursive dispatch over §3.2 kinds).

    Fail-closed pre-flight on EVERY node (incl. nested): a missing leaf stage
    callable or a missing composite ref is an ``error`` (§3.2.1) — never a
    silent pick. A nested-composite loop body is the one v1 deferral and raises
    NotImplementedError (no state-producing convention for a composite body).
    """
    missing_stages = _missing_stage_names(node, stages)
    if missing_stages:
        return _error(
            "composite-runtime: missing stage callable(s) "
            f"{sorted(missing_stages)!r} for composite {node.name!r} "
            "(fail-closed: a missing stage fails the item's evaluation)"
        )
    missing_refs = _missing_composite_refs(node, registry)
    if missing_refs:
        return _error(
            "composite-runtime: missing composite ref(s) "
            f"{sorted(missing_refs)!r} for composite {node.name!r} "
            "(fail-closed: an unresolved nested arm fails the item's evaluation)"
        )

    body = node.body
    if isinstance(body, CascadeBody):
        return _execute_cascade(
            node, body, stages, registry, config, calibrated_values, signals, predicates
        )
    if isinstance(body, EnsembleBody):
        return _execute_ensemble(
            body, stages, registry, config, calibrated_values, signals, predicates
        )
    if isinstance(body, LoopBody):
        if isinstance(body.body, CompositeArm):
            raise NotImplementedError(
                f"composite-runtime: nested-composite loop bodies are deferred "
                f"(composite {node.name!r}); a composite loop body has no v1 "
                "state-producing convention. Stage-bodied loops execute fully."
            )
        return _execute_loop(
            node, body, stages, registry, config, calibrated_values, signals, predicates
        )
    raise NotImplementedError(  # pragma: no cover - kind/body checked in IR
        f"composite-runtime: unsupported composite body {type(body).__name__} "
        f"(composite {node.name!r})"
    )


def execute_composite(
    knob: CompositeNode,
    stages: Mapping[str, StageEntry],
    *,
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
    signals: Mapping[str, Callable[..., Any]] | None = None,
    predicates: Mapping[str, Callable[..., Any]] | None = None,
    registry: Mapping[str, CompositeNode] | None = None,
) -> CompositeRunResult:
    """Execute a composite for one item — the FULL §3.2 algebra (RFC 0002).

    Args:
        knob: the composite IR root (:class:`CompositeNode`). Cascade
            (post), ensemble (sampling/committee, ``majority_vote`` /
            ``judge_max``), and loop (``signal_accept`` / ``external_accept`` /
            ``exhausted``) kinds all execute, closed under nested-composite arms.
        stages: ``StageRef`` name -> a :class:`StageRunner` (voting/sampling),
            a :class:`LoopBodyRunner` (a loop body), or a bare callable wrapped
            to whichever convention its position demands. A voting stage MUST
            carry a ``key_fn``.
        config: the current trial's tuned assignment (Mapping). It is the item
            passed to stage/body runners; tuned ensemble cardinalities resolve
            from it.
        calibrated_values: resolved CVAR values (gate/accept/stop thresholds,
            calibrated cardinalities), read LIVE. A missing/non-finite
            threshold fails CLOSED.
        signals: signal id -> callable over the (restricted) state, for
            ``signal_accept`` loop stops. A referenced-but-missing signal fails
            CLOSED (an ``error``), same as a missing stage.
        predicates: external-accept id -> callable over the state, for
            ``external_accept`` loop stops. A referenced-but-missing predicate
            fails CLOSED.
        registry: the ``N_X`` composites map (name -> :class:`CompositeNode`)
            for nested-composite arm resolution. Defaults to ``{knob.name: knob}``
            (a single-node program); a nested arm whose ref is absent fails
            CLOSED (``missing composite ref``).

    Returns:
        A frozen :class:`CompositeRunResult` whose ``result_kind`` is the
        §3.2.1 algebra tag and whose ``measures`` is the §3.10 telemetry.

    Raises:
        NotImplementedError: for a nested-composite LOOP body — the one
            remaining v1 deferral, named explicitly (never faked). The
            ``placement: pre`` cascade (dispatch/router) now executes
            end-to-end; every other deferral has been replaced by tested
            behavior.
    """
    sigs: Mapping[str, Callable[..., Any]] = signals or {}
    preds: Mapping[str, Callable[..., Any]] = predicates or {}
    reg: Mapping[str, CompositeNode] = (
        dict(registry) if registry is not None else {knob.name: knob}
    )
    # The root must be resolvable in its own registry (so nested self-reference
    # and the single-node default both work).
    if knob.name not in reg:
        reg = {**reg, knob.name: knob}
    # F5: deep fail-closed preflight over EVERY reachable composite — a missing
    # leaf stage/signal/predicate inside an unselected nested arm fails closed
    # up front, before any arm runs (never a silent root output).
    preflight = _preflight_reachable(knob, stages, reg, sigs, preds)
    if preflight is not None:
        return preflight
    return _execute_node(knob, stages, reg, config, calibrated_values, sigs, preds)
