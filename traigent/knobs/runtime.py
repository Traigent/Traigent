"""The composite EXECUTION runtime — post-cascade only (RFC 0002 §3.2).

This packet executes the ``placement: post`` cascade kind end-to-end and
**only** that kind. Per the §3.2 cross-reference, a post-cascade is EXACTLY
RFC 0001 §3.8 cascade execution, so this module does NOT reimplement gate math:
it ADAPTS the §3.2 ``CascadeBody`` IR into the shipped
:class:`~traigent.knobs.cascade.CascadePolicy` (the cascade execution engine —
vote/margin/escalation, the fail-closed exception rule, totality/determinism)
and reads the LIVE calibrated gate thresholds through ``threshold_ref``
closures over ``calibrated_values``.

Ensemble and loop execution are **deferred to a follow-up packet** (captain
decision). They are NOT silently skipped: :func:`execute_composite` raises
:class:`NotImplementedError` naming the deferral when handed an ``ensemble`` or
``loop`` root. Nested-composite arms are likewise deferred and raise loudly —
post-cascade-over-stages is the bounded surface this packet ships.

The result codomain is the §3.2.1 algebra, named once:

- ``output`` — a produced output (the selected arm's output);
- ``no_accept`` — ran, but nothing met the construct's acceptance condition (an
  HONEST no-output outcome, NOT an error). In a v1 post-cascade over opaque
  stage runners there is no arm-level acceptance predicate, so a post-cascade
  always lands in ``output`` or ``error``; ``no_accept`` is part of the named
  codomain (the result mirrors the full algebra) and becomes reachable when the
  deferred ensemble ``accept`` decls / loop exhaustion land. It is declared
  here, never faked into firing.
- ``error`` — evaluation failed (a stage exception, a missing stage callable,
  a missing/non-finite gate threshold). Fail CLOSED: a stage exception fails
  the item's evaluation, never a silent degradation (RFC 0001 §3.8 / §3.2.1
  "error is absorbing").

Telemetry (:attr:`CompositeRunResult.measures`) is the §3.10 content-free dict:
``escalation_rate`` (the per-run 0/1 escalated indicator), ``stage_selected``
(the selected arm index), and per-gate ``gate_margin_pass_rate`` (1.0 when the
gate did NOT escalate — the margin passed — 0.0 when it did, for every gate
actually evaluated). Counts/rates/enums/finite numbers only — never content.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .cascade import CascadePolicy, CascadeStep, Gate, GateKind, StageSpec, VoteKey
from .composites import (
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    EnsembleBody,
)
from .composites import GateKind as IRGateKind
from .composites import LoopBody, Placement, StageArm

__all__ = [
    "CompositeRunResult",
    "ResultKind",
    "StageRunner",
    "execute_composite",
]


class ResultKind(StrEnum):
    """The §3.2.1 result-algebra codomain (closed, total, disjoint)."""

    OUTPUT = "output"
    NO_ACCEPT = "no_accept"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StageRunner:
    """A runtime execution unit for one opaque stage (§3.2 StageRef).

    ``run`` returns the stage's sample sequence for an item; ``key_fn`` maps an
    output to its content-free equivalence key (REQUIRED for a voting stage —
    one that feeds a margin gate); ``samples`` is the declared per-stage
    cardinality (the §3.4 ``k`` knob — the engine rejects a run that produces a
    different count, mirroring :class:`~traigent.knobs.cascade.StageSpec`).

    A bare ``Callable`` may be supplied directly in the ``stages`` mapping; it
    is wrapped as a NON-VOTING single-sample runner (``key_fn=None`` — a leaf
    stage that feeds no gate; supplying one to a gated position fails closed). A stage that DOES feed a margin gate must be supplied as a
    :class:`StageRunner` carrying a ``key_fn``.
    """

    run: Callable[[Any], Sequence[Any]]
    key_fn: Callable[[Any], VoteKey] | None = None
    samples: int = 1


@dataclass(frozen=True, slots=True)
class CompositeRunResult:
    """The frozen outcome of one composite execution (§3.2.1 + §3.10).

    ``output`` is the selected arm's output when ``result_kind`` is
    ``OUTPUT`` (``None`` otherwise — ``no_accept``/``error`` carry no output).
    ``result_kind`` is the §3.2.1 algebra tag. ``measures`` is the §3.10
    content-free telemetry dict; ``error`` carries a fail-closed diagnostic
    string when ``result_kind`` is ``ERROR`` (never a partial output).
    """

    output: Any
    result_kind: ResultKind
    measures: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _coerce_runner(stage: StageRunner | Callable[[Any], Any]) -> StageRunner:
    """Normalize a ``stages`` entry into a :class:`StageRunner`.

    A bare callable becomes a single-sample, identity-keyed runner. A missing
    callable is the caller's problem (caught up-front by
    :func:`execute_composite` as a fail-closed ``error`` — never reached here).
    """
    if isinstance(stage, StageRunner):
        return stage

    call: Callable[[Any], Any] = stage

    def _run(item: Any) -> Sequence[Any]:
        return [call(item)]

    return StageRunner(run=_run, key_fn=None, samples=1)


def _build_threshold_ref(
    threshold: str, calibrated_values: Mapping[str, float | int]
) -> Callable[[], float | None]:
    """A LIVE ``threshold_ref`` closure (§ cascade.py contract).

    Reads ``calibrated_values[threshold]`` at decide time so a re-calibration
    is observed, never snapshotted. A missing threshold returns ``None`` — the
    :class:`~traigent.knobs.cascade.Gate` then fails CLOSED ("calibrate the
    CVAR before routing"), which this runtime maps to an ``error`` result
    (§3.2.1: a non-finite/absent threshold fails the item's evaluation).
    """

    def _ref() -> float | None:
        value = calibrated_values.get(threshold)
        return None if value is None else float(value)

    return _ref


def _adapt_cascade(
    body: CascadeBody,
    stages: Mapping[str, StageRunner | Callable[[Any], Any]],
    calibrated_values: Mapping[str, float | int],
) -> CascadePolicy:
    """Adapt a §3.2 post-cascade ``CascadeBody`` into a ``CascadePolicy``.

    Every arm MUST be a stage arm (nested-composite arms are deferred, raised
    by the caller). Each gate is a ``margin_below`` gate (the only v1 post
    gate) whose ``threshold_ref`` reads ``calibrated_values`` live. This does
    NOT reimplement gate math — it constructs the engine that owns it.
    """
    stage_specs: list[StageSpec] = []
    for arm in body.arms:
        # Guaranteed StageArm by the caller's deferral guard; assert for -O.
        if not isinstance(arm, StageArm):  # pragma: no cover - guarded upstream
            raise NotImplementedError(
                "composite-runtime: nested-composite cascade arms are deferred "
                f"(arm {arm!r}); this packet executes post-cascade over stages only"
            )
        runner = _coerce_runner(stages[arm.name])
        stage_specs.append(
            StageSpec(
                name=arm.name,
                run=runner.run,
                key_fn=runner.key_fn,
                samples=runner.samples,
            )
        )

    gates: list[Gate] = []
    for gate in body.gates:
        # margin_below is the only v1 post gate (IR-enforced); map it onto the
        # cascade engine's single gate kind with a LIVE threshold_ref.
        if gate.kind is not IRGateKind.MARGIN_BELOW:  # pragma: no cover - IR-guarded
            raise NotImplementedError(
                f"composite-runtime: post-cascade gate kind {gate.kind.value!r} "
                "is not executable in this packet (margin_below only)"
            )
        gates.append(
            Gate(
                kind=GateKind.MARGIN_BELOW,
                threshold_ref=_build_threshold_ref(gate.threshold, calibrated_values),
            )
        )

    return CascadePolicy(stages=tuple(stage_specs), gates=tuple(gates))


def _cascade_measures(step: CascadeStep, body: CascadeBody) -> dict[str, Any]:
    """The §3.10 cascade telemetry for one decided cascade — content-free.

    ``escalation_rate`` is the per-run 0/1 escalated indicator (1 iff any
    escalation happened); ``stage_selected`` is the selected arm index;
    ``gate_margin_pass_rate`` is a per-gate map (keyed by gate INDEX —
    threshold names may repeat across gates) over the gates actually
    EVALUATED: 1.0 where the gate did not escalate (the margin passed) and 0.0
    where it did. A gate is evaluated for arm ``i`` iff stage ``i`` ran and was
    not the selected terminal stage (i.e. ``i < stage_selected``: those
    escalated; the gate at ``stage_selected`` itself only fires if a non-last
    stage was kept, which never escalates).
    """
    selected = step.stage_index
    escalated = 1 if step.escalations > 0 else 0
    # Keyed by GATE INDEX, not threshold name: duplicate threshold refs across
    # gates are legal (n_cascade may reuse one CVAR) and name-keying collapsed
    # per-gate values (codex runtime-round blocker). §3.10: per-gate.
    per_gate: dict[int, float] = {}
    for index, _gate in enumerate(body.gates):
        if index < selected:
            # stage index < selected ran a gate that escalated.
            per_gate[index] = 0.0
        elif index == selected and index < len(body.arms) - 1:
            # the selected non-terminal stage's gate evaluated and STOPPED.
            per_gate[index] = 1.0
        # gates beyond the selected stage never evaluated — omitted (honest).
    return {
        "escalation_rate": float(escalated),
        "stage_selected": selected,
        "gate_margin_pass_rate": per_gate,
    }


def _has_nested_arm(body: CascadeBody) -> bool:
    return any(isinstance(arm, CompositeArm) for arm in body.arms)


def execute_composite(
    knob: CompositeNode,
    stages: Mapping[str, StageRunner | Callable[[Any], Any]],
    *,
    config: Mapping[str, Any],
    calibrated_values: Mapping[str, float | int],
) -> CompositeRunResult:
    """Execute a composite for one item — POST-CASCADE ONLY (RFC 0002 §3.2).

    Args:
        knob: the composite IR root (:class:`CompositeNode`). Only a
            ``placement: post`` cascade over stage arms executes in this packet.
        stages: ``StageRef`` name -> a :class:`StageRunner` (or a bare callable
            wrapped as a single-sample, identity-keyed runner). A voting stage
            (one that feeds a margin gate) MUST be a ``StageRunner`` carrying a
            ``key_fn``.
        config: the current trial's tuned assignment (Mapping) — carried for
            symmetry with the resolver chokepoint and future per-stage
            parameterization; this packet does not read individual entries.
        calibrated_values: resolved CVAR values (gate thresholds), read LIVE
            via ``threshold_ref``. A missing threshold fails CLOSED.

    Returns:
        A frozen :class:`CompositeRunResult` whose ``result_kind`` is the
        §3.2.1 algebra tag and whose ``measures`` is the §3.10 telemetry.

    Raises:
        NotImplementedError: for an ``ensemble`` or ``loop`` root, a
            ``placement: pre`` cascade, or a nested-composite cascade arm —
            each named explicitly (deferred, never faked).
    """
    body = knob.body

    # --- deferrals: loud, never silent (NotImplementedError naming the gap) --
    if isinstance(body, EnsembleBody) or knob.kind is CompositeKind.ENSEMBLE:
        raise NotImplementedError(
            f"composite-runtime: ensemble execution is deferred to a follow-up "
            f"packet (composite {knob.name!r}); this packet executes the "
            "post-cascade kind only"
        )
    if isinstance(body, LoopBody) or knob.kind is CompositeKind.LOOP:
        raise NotImplementedError(
            f"composite-runtime: loop execution is deferred to a follow-up "
            f"packet (composite {knob.name!r}); this packet executes the "
            "post-cascade kind only"
        )
    if not isinstance(body, CascadeBody):  # pragma: no cover - kind/body checked in IR
        raise NotImplementedError(
            f"composite-runtime: unsupported composite body {type(body).__name__} "
            f"(composite {knob.name!r})"
        )
    if body.placement is not Placement.POST:
        raise NotImplementedError(
            f"composite-runtime: pre-cascade (dispatch) execution is deferred to "
            f"a follow-up packet (composite {knob.name!r}); this packet executes "
            "the post-cascade kind only"
        )
    if _has_nested_arm(body):
        raise NotImplementedError(
            f"composite-runtime: nested-composite arms are deferred to a follow-up "
            f"packet (composite {knob.name!r}); this packet executes post-cascade "
            "over stage arms only"
        )

    # --- fail-closed pre-flight: every stage callable must be present --------
    missing = [
        arm.name
        for arm in body.arms
        if isinstance(arm, StageArm) and arm.name not in stages
    ]
    if missing:
        return CompositeRunResult(
            output=None,
            result_kind=ResultKind.ERROR,
            measures={},
            error=(
                "composite-runtime: missing stage callable(s) "
                f"{sorted(missing)!r} for composite {knob.name!r} "
                "(fail-closed: a missing stage fails the item's evaluation)"
            ),
        )

    policy = _adapt_cascade(body, stages, calibrated_values)

    # --- decide; a stage/gate exception (incl. missing/non-finite threshold)
    #     is the §3.2.1 absorbing `error` — fail CLOSED, never degrade. --------
    item = config
    try:
        step = policy.decide(item)
    except Exception as exc:  # noqa: BLE001 - error is absorbing (§3.2.1)
        return CompositeRunResult(
            output=None,
            result_kind=ResultKind.ERROR,
            measures={},
            error=f"composite-runtime: {type(exc).__name__}: {exc}",
        )

    return CompositeRunResult(
        output=step.output,
        result_kind=ResultKind.OUTPUT,
        measures=_cascade_measures(step, body),
        error=None,
    )
