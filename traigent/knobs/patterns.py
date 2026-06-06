"""The pattern catalog v1: named macros over the composite algebra (RFC 0002).

Patterns are the human-facing layer. Each catalog factory is a pure function
of validated params that expands DETERMINISTICALLY into the sealed algebra of
:mod:`traigent.knobs.composites`, stamping every emitted node with a closed
:class:`~traigent.knobs.composites.Provenance` value (``pattern = <name>``,
``param_hash = H_c(canonical validated params)``). Raw pattern params are
NEVER stored — only their canonical hash rides provenance (§3.7, the P8
admission-contract canary).

A factory returns a :class:`CompositeKnob`: a declaration bundle the caller
spreads into a ``ConfigSpace``. It exposes ``.structure`` (the IR root node),
``.members`` (the member ``Knob`` declarations — bindings stay
``Tuned | Calibrated | Fixed``; a composite never binds a value), ``.provenance``,
and ``.telemetry_names`` (the §3.10 per-kind standard measure names). Adding a
pattern is an SDK release, not a language or schema change; the algebra never
grows because a pattern was added.

INTEGRATION BOUNDARY (deliberate, captain decision): factories return
DECLARATIONS only. Deep orchestrator/resolver wiring is a follow-up packet —
nothing here touches ``traigent.core`` or ``traigent.cloud``.

Unroll (§3.8): ``self_refine`` (a ``signal_accept`` loop) is unroll-eligible
and exposes :meth:`CompositeKnob.unroll` returning the internal K-chain IR.
``self_debug`` (an ``external_accept`` loop) offers no unroll and raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bindings import Knob
from .canonical import canonical_hash
from .composites import (
    AcceptDecl,
    AggregateDecl,
    AggregateKind,
    CascadeBody,
    CompositeKind,
    CompositeNode,
    EnsembleBody,
    GateDecl,
    GateKind,
    LoopBody,
    Placement,
    Provenance,
    SignalUse,
    StageArm,
    StatKind,
    StopDecl,
    StopKind,
)

__all__ = [
    "CompositeKnob",
    "KChain",
    "KChainStage",
    "best_of_n",
    "binary_cascade",
    "n_cascade",
    "self_consistency",
    "self_debug",
    "self_refine",
    "react_tool_loop",
    "verification_gate",
    "moe",
]

#: §3.10 standard telemetry measure names, per constructor kind. Content-free:
#: counts, rates, enums, finite numbers only (P8 discipline).
_TELEMETRY: dict[CompositeKind, tuple[str, ...]] = {
    CompositeKind.CASCADE: (
        "escalation_rate",
        "stage_selected",
        "gate_margin_pass_rate",
    ),
    CompositeKind.ENSEMBLE: (
        "vote_agreement",
        "vote_margin",
        "candidates_evaluated",
        "candidates_excluded",
    ),
    CompositeKind.LOOP: ("iterations_used", "stop_reason"),
}


# --------------------------------------------------------------------------- #
# K-chain IR (§3.8 Loop -> K-chain semantic compilation)                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class KChainStage:
    """One stage of an unrolled ``signal_accept`` loop (§3.8).

    ``body`` is the loop body specialized to iteration ``index``'s threaded
    state; ``escalate_when`` carries the acceptance-direction complement
    ``σ(state_i) < θ`` (the chain escalates to stage ``i+1`` exactly when the
    ``signal_accept`` stop has NOT fired).
    """

    index: int
    body: StageArm
    escalate_when: str  # human-readable record of the escalation predicate


@dataclass(frozen=True, slots=True)
class KChain:
    """The internal K-chain execution form of a ``signal_accept`` loop (§3.8).

    This is an SDK intermediate representation, NOT a surface ``Cascade_post``
    (the v1 gate registry has no state-predicate gate). ``signal``/``threshold``
    are the loop stop's ``SignalUse.signal`` and threshold CVAR; ``state_keys``
    are the declared threaded-state keys.
    """

    signal: str
    threshold: str
    state_keys: tuple[str, ...]
    stages: tuple[KChainStage, ...]
    provenance: Provenance | None = None


# --------------------------------------------------------------------------- #
# CompositeKnob: the factory return type                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompositeKnob:
    """A named composite declaration bundle (the pattern-factory return type).

    ``structure`` is the IR root :class:`CompositeNode`; ``members`` are the
    member ``Knob`` declarations the caller spreads into their ``ConfigSpace``
    (bindings stay ``Tuned | Calibrated | Fixed`` — a composite NEVER binds a
    value); ``provenance`` is stamped by the expander (``pattern = <name>``,
    ``param_hash`` only — raw params never stored); ``telemetry_names`` is the
    §3.10 per-kind standard measure tuple.
    """

    name: str
    structure: CompositeNode
    members: dict[str, Knob[Any]] = field(default_factory=dict)
    provenance: Provenance | None = None
    telemetry_names: tuple[str, ...] = ()

    @property
    def kind(self) -> CompositeKind:
        return self.structure.kind

    def unroll(self, k: int) -> KChain:
        """Return the §3.8 K-chain IR (``signal_accept`` loops only).

        Raises ``ValueError`` for any non-``signal_accept`` loop (an
        ``external_accept``/``exhausted`` loop offers no unroll) or any
        non-loop composite.
        """
        body = self.structure.body
        if not isinstance(body, LoopBody):
            raise ValueError(
                f"unroll is only defined for loop composites; "
                f"{self.name!r} is a {self.structure.kind.value}"
            )
        if body.stop.kind is not StopKind.SIGNAL_ACCEPT:
            raise ValueError(
                f"unroll is only defined for signal_accept loops; "
                f"{self.name!r} stop kind is {body.stop.kind.value}"
            )
        if k < 1:
            raise ValueError(f"unroll K must be >= 1, got {k}")
        if not isinstance(body.body, StageArm):
            raise ValueError("unroll v1 supports a stage-bodied loop only")
        # A signal_accept stop always carries signal + threshold (guaranteed by
        # StopDecl.__post_init__); guard explicitly so this holds under -O too.
        if body.stop.signal is None or body.stop.threshold is None:
            raise ValueError("signal_accept stop is missing its signal/threshold")
        signal = body.stop.signal.signal
        threshold = body.stop.threshold
        stages = tuple(
            KChainStage(
                index=i,
                body=body.body,
                escalate_when=f"{signal}(state_{i}) < {threshold}",
            )
            for i in range(1, k + 1)
        )
        return KChain(
            signal=signal,
            threshold=threshold,
            state_keys=body.state_keys,
            stages=stages,
            provenance=self.provenance,
        )


def _provenance(name: str, params: dict[str, Any]) -> Provenance:
    """Stamp closed-shape provenance: ``pattern`` + ``param_hash`` only.

    ``param_hash`` is ``H_c`` of the CANONICAL validated params; the raw
    ``params`` dict is hashed and discarded — it never enters the returned
    object (the P8 canary asserts a sentinel param value never serializes).
    """
    return Provenance(pattern=name, param_hash=canonical_hash(params))


# --------------------------------------------------------------------------- #
# Catalog v1 (§3.7)                                                           #
# --------------------------------------------------------------------------- #


def binary_cascade(
    name: str,
    *,
    base_stage: str,
    expert_stage: str,
    threshold: str,
    base_tuned_params: tuple[str, ...] = (),
    expert_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Cascade(arms=[stage(base), stage(expert)], gates=[margin_below θ], post)``.

    The RFC 0001 cascade policy's exact shape — the migration target (§4).
    """
    params = {
        "name": name,
        "base_stage": base_stage,
        "expert_stage": expert_stage,
        "threshold": threshold,
        "base_tuned_params": list(base_tuned_params),
        "expert_tuned_params": list(expert_tuned_params),
    }
    prov = _provenance("binary_cascade", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(
            arms=(
                StageArm(base_stage, base_tuned_params),
                StageArm(expert_stage, expert_tuned_params),
            ),
            gates=(GateDecl(kind=GateKind.MARGIN_BELOW, threshold=threshold),),
            placement=Placement.POST,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.CASCADE],
    )


def n_cascade(
    name: str,
    *,
    stages: tuple[str, ...],
    thresholds: tuple[str, ...],
    tuned_params: tuple[tuple[str, ...], ...] | None = None,
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Cascade(arms=[stage(a_1)..stage(a_m)], gates=[θ_1..θ_{m-1}], post)``.

    Ordered escalation. ``|thresholds|`` must equal ``|stages| - 1`` (the IR
    enforces ``cascade_arity``).

    ``tuned_params``, WHEN PROVIDED, declares the per-stage parentage and MUST
    have EXACTLY one tuple per stage (``len(tuned_params) == len(stages)``) — a
    short list is NOT silently backfilled with ``()``, because that would
    under-declare an arm's parent coverage (§3.5). Omitting the argument
    entirely keeps every stage's ``tuned_params`` empty (the documented
    policy-migration ratchet, §4); a PARTIAL declaration rejects with
    ``invalid_tuned_param``.
    """
    if tuned_params is None:
        tuned_params = tuple(() for _ in stages)
    elif len(tuned_params) != len(stages):
        raise ValueError(
            "invalid_tuned_param: tuned_params must declare exactly one tuple "
            f"per stage (got {len(tuned_params)} for {len(stages)} stage(s)); "
            "omit tuned_params entirely to leave all stages undeclared, but "
            "partial per-stage declarations are not silently backfilled"
        )
    params = {
        "name": name,
        "stages": list(stages),
        "thresholds": list(thresholds),
        "tuned_params": [list(tp) for tp in tuned_params],
    }
    prov = _provenance("n_cascade", params)
    arms = tuple(StageArm(stage, tuned_params[i]) for i, stage in enumerate(stages))
    gates = tuple(
        GateDecl(kind=GateKind.MARGIN_BELOW, threshold=theta) for theta in thresholds
    )
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(arms=arms, gates=gates, placement=Placement.POST),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.CASCADE],
    )


def self_consistency(
    name: str,
    *,
    stage: str,
    cardinality: str,
    accept_threshold: str | None = None,
    stage_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Ensemble(arms=[stage(a)], cardinality=k, majority_vote, accept?)``.

    ``k`` may be tuned or calibrated. An optional ``accept_threshold`` adds a
    ``stat_at_least(vote_margin, θ)`` acceptance decl (§3.7 note).
    """
    params = {
        "name": name,
        "stage": stage,
        "cardinality": cardinality,
        "accept_threshold": accept_threshold,
        "stage_tuned_params": list(stage_tuned_params),
    }
    prov = _provenance("self_consistency", params)
    accept = (
        AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold=accept_threshold)
        if accept_threshold is not None
        else None
    )
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=(StageArm(stage, stage_tuned_params),),
            aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE, accept=accept),
            cardinality=cardinality,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.ENSEMBLE],
    )


def best_of_n(
    name: str,
    *,
    stage: str,
    judge_stage: str,
    cardinality: str,
    stage_tuned_params: tuple[str, ...] = (),
    judge_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Ensemble(arms=[stage(a)], cardinality=k, judge_max(stage(judge)))``.

    The judge output contract (finite numeric score) is §3.2 execution
    semantics — runtime-enforced, not part of this declaration.
    """
    params = {
        "name": name,
        "stage": stage,
        "judge_stage": judge_stage,
        "cardinality": cardinality,
        "stage_tuned_params": list(stage_tuned_params),
        "judge_tuned_params": list(judge_tuned_params),
    }
    prov = _provenance("best_of_n", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=(StageArm(stage, stage_tuned_params),),
            aggregate=AggregateDecl(
                kind=AggregateKind.JUDGE_MAX,
                judge=StageArm(judge_stage, judge_tuned_params),
            ),
            cardinality=cardinality,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.ENSEMBLE],
    )


def self_debug(
    name: str,
    *,
    stage: str,
    predicate: str,
    max_iters: int,
    state_keys: tuple[str, ...] = ("attempt", "critique"),
    stage_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Loop(body=stage(a), state_keys, stop=external_accept(tests), max_iters=K)``.

    The "2-step knob" at ``max_iters = 1``. The ``external_accept`` stop is an
    opaque runtime predicate — NO unroll (§3.8); :meth:`CompositeKnob.unroll`
    raises.
    """
    params = {
        "name": name,
        "stage": stage,
        "predicate": predicate,
        "max_iters": max_iters,
        "state_keys": list(state_keys),
        "stage_tuned_params": list(stage_tuned_params),
    }
    prov = _provenance("self_debug", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm(stage, stage_tuned_params),
            stop=StopDecl(kind=StopKind.EXTERNAL_ACCEPT, predicate=predicate),
            state_keys=state_keys,
            max_iters=max_iters,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.LOOP],
    )


def self_refine(
    name: str,
    *,
    stage: str,
    signal: str,
    threshold: str,
    max_iters: int,
    state_keys: tuple[str, ...] = ("draft",),
    signal_inputs: tuple[str, ...] = (),
    stage_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Loop(body=stage(a), state_keys, stop=signal_accept(σ, θ), max_iters=K)``.

    The calibrated ``signal_accept`` stop is unroll-eligible (§3.8):
    :meth:`CompositeKnob.unroll` returns the K-chain IR. ``signal_inputs``, when
    non-empty, must be a subset of ``state_keys`` (the IR enforces
    ``stop_signal_outside_state``).
    """
    params = {
        "name": name,
        "stage": stage,
        "signal": signal,
        "threshold": threshold,
        "max_iters": max_iters,
        "state_keys": list(state_keys),
        "signal_inputs": list(signal_inputs),
        "stage_tuned_params": list(stage_tuned_params),
    }
    prov = _provenance("self_refine", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm(stage, stage_tuned_params),
            stop=StopDecl(
                kind=StopKind.SIGNAL_ACCEPT,
                threshold=threshold,
                signal=SignalUse(signal=signal, inputs=signal_inputs),
            ),
            state_keys=state_keys,
            max_iters=max_iters,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.LOOP],
    )


def react_tool_loop(
    name: str,
    *,
    stage: str,
    signal: str,
    tool_confidence_min: str,
    max_tool_calls: int,
    state_keys: tuple[str, ...] = (
        "scratchpad",
        "tool_calls",
        "observations",
        "confidence",
        "last_error",
    ),
    signal_inputs: tuple[str, ...] = (
        "scratchpad",
        "tool_calls",
        "observations",
        "confidence",
        "last_error",
    ),
    stage_tuned_params: tuple[str, ...] = (
        "planner_style",
        "tool_allowlist",
        "observation_format",
        "failure_handler",
    ),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Loop(body=stage(a), state_keys, stop=signal_accept(tool_confidence, θ), max_iters=K)``.

    A catalog ReAct-style tool loop over the existing loop algebra only. Each
    body invocation is assumed to perform at most one ReAct tool step, so
    ``max_tool_calls`` maps to the sealed IR's literal ``max_iters`` bound. If
    the opaque stage performs multiple tool calls per invocation, this is a
    max-ReAct-iterations bound, not a precise tool-call cap.

    ``tool_cost_cap`` is deliberately not part of this expansion: the current
    algebra has no cost stop/gate or mid-loop budget hook, so this factory does
    not enforce cost stopping. Add a future cost-stop proposal before making
    that claim.
    """
    params = {
        "name": name,
        "stage": stage,
        "signal": signal,
        "tool_confidence_min": tool_confidence_min,
        "max_tool_calls": max_tool_calls,
        "state_keys": list(state_keys),
        "signal_inputs": list(signal_inputs),
        "stage_tuned_params": list(stage_tuned_params),
    }
    prov = _provenance("react_tool_loop", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm(stage, stage_tuned_params),
            stop=StopDecl(
                kind=StopKind.SIGNAL_ACCEPT,
                threshold=tool_confidence_min,
                signal=SignalUse(signal=signal, inputs=signal_inputs),
            ),
            state_keys=state_keys,
            max_iters=max_tool_calls,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.LOOP],
    )


def verification_gate(
    name: str,
    *,
    stage: str,
    verifier_signal: str,
    verifier_pass_threshold: str,
    verification_style: str,
    verification_question_count: str,
    verifier_model: str,
    independent_context: str,
    revision_policy: str,
    max_iters: int = 2,
    state_keys: tuple[str, ...] = (
        "draft",
        "verification_questions",
        "verification_answers",
        "verifier_pass_score",
        "contradiction_score",
        "revision",
        "independent_context",
    ),
    signal_inputs: tuple[str, ...] = (
        "draft",
        "verification_answers",
        "independent_context",
    ),
    contradiction_score_max: str | None = None,
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """CoVe-style verifier loop: ``signal_accept(verifier_signal, θ)``.

    This is a loop-family catalog macro over the sealed IR: the stage produces
    threaded verification state, then the verifier signal accepts when
    ``verifier_signal(state) >= verifier_pass_threshold``. Exhaustion yields
    the runtime's ordinary ``no_accept`` result.

    Not expressible in v1: a simultaneous ``contradiction_score_max`` stop, dual
    verifier-pass plus improvement/contradiction thresholds on one loop
    decision, or a tuned structural repair bound such as ``max_repair_rounds``;
    ``max_iters`` is the literal IR bound.
    """
    if contradiction_score_max is not None:
        raise ValueError(
            "unsupported_compound_stop: verification_gate cannot express "
            "contradiction_score_max until the loop IR supports compound stop "
            "decisions"
        )

    stage_tuned_params = (
        verification_style,
        verification_question_count,
        verifier_model,
        independent_context,
        revision_policy,
    )
    params = {
        "name": name,
        "stage": stage,
        "verifier_signal": verifier_signal,
        "verifier_pass_threshold": verifier_pass_threshold,
        "verification_style": verification_style,
        "verification_question_count": verification_question_count,
        "verifier_model": verifier_model,
        "independent_context": independent_context,
        "revision_policy": revision_policy,
        "max_iters": max_iters,
        "state_keys": list(state_keys),
        "signal_inputs": list(signal_inputs),
        "contradiction_score_max": contradiction_score_max,
    }
    prov = _provenance("verification_gate", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm(stage, stage_tuned_params),
            stop=StopDecl(
                kind=StopKind.SIGNAL_ACCEPT,
                threshold=verifier_pass_threshold,
                signal=SignalUse(signal=verifier_signal, inputs=signal_inputs),
            ),
            state_keys=state_keys,
            max_iters=max_iters,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.LOOP],
    )


def moe(
    name: str,
    *,
    experts: tuple[str, ...],
    aggregate: str = "vote",
    judge_stage: str | None = None,
    accept_threshold: str | None = None,
    expert_tuned_params: tuple[tuple[str, ...], ...] | None = None,
    judge_tuned_params: tuple[str, ...] = (),
    members: dict[str, Knob[Any]] | None = None,
) -> CompositeKnob:
    """``Ensemble(arms=[stage(expert_1)..stage(expert_m)], aggregate, committee)``.

    Mixture-of-experts is the committee form of an ensemble: each distinct
    expert stage contributes one representative candidate, then aggregation is
    either majority vote or ``judge_max``. Sampling-form cardinality is
    deliberately absent; use ``self_consistency``/``best_of_n`` for repeated
    draws from one generator.
    """
    if (
        not isinstance(experts, tuple)
        or len(experts) < 2
        or any(not isinstance(expert, str) or not expert for expert in experts)
    ):
        raise ValueError("moe experts must be at least two non-empty stage names")
    if len(set(experts)) != len(experts):
        raise ValueError("moe experts must be distinct")

    if aggregate not in {"vote", "judge"}:
        raise ValueError("moe aggregate must be 'vote' or 'judge'")
    if expert_tuned_params is None:
        expert_tuned_params = tuple(() for _ in experts)
    elif len(expert_tuned_params) != len(experts):
        raise ValueError(
            "invalid_tuned_param: expert_tuned_params must declare exactly one "
            f"tuple per expert (got {len(expert_tuned_params)} for "
            f"{len(experts)} expert(s)); omit expert_tuned_params entirely to "
            "leave all experts undeclared"
        )

    if aggregate == "vote":
        if judge_stage is not None:
            raise ValueError("aggregate='vote' forbids judge_stage")
        if judge_tuned_params:
            raise ValueError("aggregate='vote' forbids judge_tuned_params")
        aggregate_decl = AggregateDecl(
            kind=AggregateKind.MAJORITY_VOTE,
            accept=(
                AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold=accept_threshold)
                if accept_threshold is not None
                else None
            ),
        )
    else:
        if not isinstance(judge_stage, str) or not judge_stage:
            raise ValueError("aggregate='judge' requires judge_stage")
        if judge_stage in experts:
            raise ValueError("aggregate='judge' judge_stage must be distinct")
        if accept_threshold is not None:
            raise ValueError("aggregate='judge' does not support accept_threshold")
        aggregate_decl = AggregateDecl(
            kind=AggregateKind.JUDGE_MAX,
            judge=StageArm(judge_stage, judge_tuned_params),
        )

    params = {
        "name": name,
        "experts": list(experts),
        "aggregate": aggregate,
        "judge_stage": judge_stage,
        "accept_threshold": accept_threshold,
        "expert_tuned_params": [list(tp) for tp in expert_tuned_params],
        "judge_tuned_params": list(judge_tuned_params),
    }
    prov = _provenance("moe", params)
    structure = CompositeNode(
        name=name,
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=tuple(
                StageArm(expert, expert_tuned_params[i])
                for i, expert in enumerate(experts)
            ),
            aggregate=aggregate_decl,
            cardinality=None,
        ),
        provenance=prov,
    )
    return CompositeKnob(
        name=name,
        structure=structure,
        members=dict(members or {}),
        provenance=prov,
        telemetry_names=_TELEMETRY[CompositeKind.ENSEMBLE],
    )
