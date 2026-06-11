"""The composite-knob internal IR: a sealed control-flow algebra (RFC 0002).

A ``CompositeKnob`` is NOT a binding kind (RFC 0001's one-Knob model is
untouched): it is a namespace/expansion/provenance unit that *groups* member
knobs and gate declarations under a declared control-flow shape. Nothing in
this module binds, carries, or defaults a value — value binding stays on
``Tuned | Fixed | Calibrated`` knobs exclusively.

This module is the machine-facing layer: three disjoint structural
constructors (:class:`CompositeKind` ``∈ {cascade, ensemble, loop}``), closed
under nesting, that the optimizer projection, the certificate coverage fold,
the freshness dependency compilation, the cost models, and the model checker
consume. The frozen dataclasses below mirror the §3.2 algebra exactly:

- tagged arms — :class:`StageArm` (opaque, with declared ``tuned_params``) and
  :class:`CompositeArm` (a nesting reference into ``N_X``);
- bodies — :class:`CascadeBody`, :class:`EnsembleBody`, :class:`LoopBody`,
  discriminated by :class:`CompositeKind`;
- declarations — :class:`GateDecl`, :class:`AcceptDecl`, :class:`StopDecl`,
  :class:`AggregateDecl`, :class:`SignalUse`;
- :class:`Provenance`, a CLOSED shape (identifiers + one canonical hash; no
  raw-params field exists, §3.7).

Structural well-formedness (§3.2 items 1–11, §3.5, §3.11) is checked here and
raises :class:`ValueError` whose message is PREFIXED with the exact §3.11
error code. Node-local rules check on construction; cross-composite rules
(cycle detection, ``ambiguous_arm``, ``missing_composite_ref``) need the whole
``N_X`` and a TVAR/CVAR namespace view, so they run in
:func:`validate_program` over a :class:`CompositeProgram`.

The derived functions :func:`leaf_tvars`, :func:`required_parents`,
:func:`cal_fold`, and :func:`roots` are the §3.5/§3.6 compilations onto leaf
TVARs and calibratable members. No value binding appears anywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "AcceptDecl",
    "AggregateDecl",
    "AggregateKind",
    "Arm",
    "CascadeBody",
    "CompositeArm",
    "CompositeKind",
    "CompositeNode",
    "CompositeProgram",
    "EnsembleBody",
    "GateDecl",
    "GateKind",
    "LoopBody",
    "Placement",
    "Provenance",
    "Scope",
    "SignalUse",
    "StageArm",
    "StatKind",
    "StopDecl",
    "StopKind",
    "cal_fold",
    "leaf_tvars",
    "required_parents",
    "roots",
    "validate_program",
]


def _reject(code: str, detail: str) -> ValueError:
    """A §3.11 rejection: a ``ValueError`` whose message is PREFIXED with the
    exact error code so callers can assert on the code, never on prose."""
    return ValueError(f"{code}: {detail}")


# --------------------------------------------------------------------------- #
# Sealed registries (§3.2)                                                    #
# --------------------------------------------------------------------------- #


class CompositeKind(StrEnum):
    """The SEALED structural-constructor registry (§3.2).

    A fourth kind is a new RFC, not a registry entry — DAG-shaped
    configurations are the closure of nesting, not a fourth "DAG kind".
    """

    CASCADE = "cascade"
    ENSEMBLE = "ensemble"
    LOOP = "loop"


class Placement(StrEnum):
    """Cascade placement (§3.2); ``post`` is the default."""

    PRE = "pre"
    POST = "post"


class GateKind(StrEnum):
    """v1 gate registry (§3.2 ``GateDecl``).

    ``margin_below`` is POST-only and consumes the just-executed arm's vote
    statistics; ``signal_below`` is PRE-only and scores the input before any
    arm runs (strict placement symmetry, item 5).
    """

    MARGIN_BELOW = "margin_below"
    SIGNAL_BELOW = "signal_below"


class StatKind(StrEnum):
    """Content-free vote statistics for an :class:`AcceptDecl` (§3.2)."""

    VOTE_MARGIN = "vote_margin"
    VOTE_AGREEMENT = "vote_agreement"


class AggregateKind(StrEnum):
    """v1 ensemble aggregate registry (§3.2 ``AggregateDecl``)."""

    MAJORITY_VOTE = "majority_vote"
    JUDGE_MAX = "judge_max"


class StopKind(StrEnum):
    """v1 loop-stop registry (§3.2 ``StopDecl``)."""

    SIGNAL_ACCEPT = "signal_accept"
    EXTERNAL_ACCEPT = "external_accept"
    EXHAUSTED = "exhausted"


# --------------------------------------------------------------------------- #
# Scope (RFC 0001 §3.7 scope_spec — carried verbatim onto a composite, §3.2)  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Scope:
    """RFC 0001 §3.7 ``scope_spec`` — ``⟨ node?, agent?, workflow? ⟩``.

    Carried VERBATIM onto a composite with the SAME semantics as
    ``PolicyDecl.scope`` (§3.2, §4 subsumption): each field is an optional
    ``Ident`` recording node/agent/workflow ownership. v1 semantics are
    informational (RFC 0001 metadata) — a composite still binds no value. The
    closed shape (three optional identifiers, no content-typed field) keeps it
    OUTSIDE the P8 surface, exactly as the policy form's scope.
    """

    node: str | None = None
    agent: str | None = None
    workflow: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("node", self.node),
            ("agent", self.agent),
            ("workflow", self.workflow),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise _reject(
                    "invalid_arm_shape",
                    f"Scope.{label} must be a non-empty identifier or None",
                )


# --------------------------------------------------------------------------- #
# Provenance (closed shape, §3.7)                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Provenance:
    """Closed-shape expansion-output metadata stamped on every emitted node.

    Identifiers and one canonical hash only — there is NO raw-params field and
    NO content-typed field, so provenance can ride operational metadata
    without widening the P8 surface (§3.7, §7). ``param_hash`` is
    ``H_c(canonical validated params)``; raw pattern params NEVER serialize.
    """

    pattern: str
    pattern_version: str | None = None
    param_hash: str | None = None
    node_path: str | None = None


# --------------------------------------------------------------------------- #
# Signal use (§3.2 SignalUse / §3.9 SignalSurface)                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SignalUse:
    """A named signal reference plus its declared input keys (§3.2).

    ``signal`` names a registry signal exactly as RFC 0001's module-facing
    ``calibration.signal`` does (operationally resolved; one signal model
    across the language). ``inputs`` are the declared input keys the signal
    reads; for loop stops they MUST be a subset of the loop's ``state_keys``
    (checked by the enclosing :class:`LoopBody`). A malformed object — missing
    ``signal`` or a non-tuple ``inputs`` — rejects ``invalid_signal_use``.
    """

    signal: str
    inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.signal, str) or not self.signal:
            raise _reject(
                "invalid_signal_use", "SignalUse.signal must be a non-empty identifier"
            )
        if not isinstance(self.inputs, tuple):
            raise _reject(
                "invalid_signal_use", "SignalUse.inputs must be a tuple of identifiers"
            )
        for key in self.inputs:
            if not isinstance(key, str) or not key:
                raise _reject(
                    "invalid_signal_use",
                    f"SignalUse.inputs entry not an identifier: {key!r}",
                )


# --------------------------------------------------------------------------- #
# Arms (tagged; §3.2 Arm / §3.9 ArmSurface)                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StageArm:
    """An opaque stage arm with DECLARED tuned parentage (§3.2, §3.5).

    ``tuned_params`` declares the TVARs parameterizing this opaque stage
    (possibly empty — the author's declaration). Stages never consult the
    namespace; TVAR-only resolution of these entries is checked at program
    level (item 9, ``invalid_tuned_param``).
    """

    name: str
    tuned_params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _reject(
                "invalid_arm_shape", "StageArm.name must be a non-empty identifier"
            )
        if not isinstance(self.tuned_params, tuple):
            raise _reject(
                "invalid_arm_shape",
                "StageArm.tuned_params must be a tuple of identifiers",
            )
        for tvar in self.tuned_params:
            if not isinstance(tvar, str) or not tvar:
                raise _reject(
                    "invalid_arm_shape",
                    f"tuned_params entry not an identifier: {tvar!r}",
                )


@dataclass(frozen=True, slots=True)
class CompositeArm:
    """A tagged nesting reference; exact-match into ``N_X`` (§3.2 item 3).

    Resolution is checked at program level (``missing_composite_ref``); the
    bare/ambiguous-identifier case (``ambiguous_arm``) is a SURFACE concern and
    cannot arise in this tagged IR (it is parsed there), so it is N/A here.
    """

    ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref:
            raise _reject(
                "invalid_arm_shape", "CompositeArm.ref must be a non-empty identifier"
            )


#: The tagged arm union — isinstance-discriminated, mirroring the bindings
#: module's discriminator-free style.
Arm = StageArm | CompositeArm


# --------------------------------------------------------------------------- #
# Declarations (§3.2)                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GateDecl:
    """A cascade gate (§3.2 ``GateDecl``).

    ``threshold`` MUST resolve to a numeric CVAR (program-level kind/type
    checks). ``signal`` is REQUIRED iff ``kind = signal_below``
    (``missing_gate_signal``) and FORBIDDEN otherwise.
    """

    kind: GateKind
    threshold: str
    signal: SignalUse | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GateKind):
            raise _reject(
                "unknown_gate_kind", f"gate kind outside the v1 registry: {self.kind!r}"
            )
        if not isinstance(self.threshold, str) or not self.threshold:
            raise _reject(
                "invalid_arm_shape", "GateDecl.threshold must be a non-empty identifier"
            )
        if self.kind is GateKind.SIGNAL_BELOW and self.signal is None:
            raise _reject(
                "missing_gate_signal", "signal_below gate requires a declared signal"
            )
        if self.kind is GateKind.MARGIN_BELOW and self.signal is not None:
            raise _reject(
                "unknown_composite_field", "margin_below gate carries no signal field"
            )


@dataclass(frozen=True, slots=True)
class AcceptDecl:
    """An ensemble acceptance decl (§3.2 ``AcceptDecl``).

    The acceptance inequality is ``stat ≥ θ`` — deliberately the OPPOSITE
    direction of cascade escalation (``margin < θ``), a distinct registry so
    the two cannot drift. ``threshold`` MUST resolve to a numeric CVAR.
    """

    stat: StatKind
    threshold: str
    kind: str = "stat_at_least"

    def __post_init__(self) -> None:
        if self.kind != "stat_at_least":
            raise _reject(
                "unknown_aggregate_kind", f"unknown accept kind: {self.kind!r}"
            )
        if not isinstance(self.stat, StatKind):
            raise _reject(
                "unknown_aggregate_kind", f"unknown accept stat: {self.stat!r}"
            )
        if not isinstance(self.threshold, str) or not self.threshold:
            raise _reject(
                "invalid_arm_shape",
                "AcceptDecl.threshold must be a non-empty identifier",
            )


@dataclass(frozen=True, slots=True)
class AggregateDecl:
    """An ensemble aggregate (§3.2 ``AggregateDecl``).

    ``judge`` is REQUIRED iff ``kind = judge_max`` (``missing_judge``) and is
    stage- or composite-tagged. ``accept`` is optional for either kind.
    """

    kind: AggregateKind
    judge: Arm | None = None
    accept: AcceptDecl | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AggregateKind):
            raise _reject(
                "unknown_aggregate_kind",
                f"aggregate kind outside the v1 registry: {self.kind!r}",
            )
        if self.kind is AggregateKind.JUDGE_MAX and self.judge is None:
            raise _reject("missing_judge", "judge_max aggregate requires a judge arm")
        if self.kind is AggregateKind.MAJORITY_VOTE and self.judge is not None:
            raise _reject(
                "unknown_composite_field",
                "majority_vote aggregate carries no judge field",
            )
        if self.judge is not None and not isinstance(
            self.judge, (StageArm, CompositeArm)
        ):
            raise _reject(
                "invalid_arm_shape", "AggregateDecl.judge must be a tagged Arm"
            )


@dataclass(frozen=True, slots=True)
class StopDecl:
    """A loop stop rule (§3.2 ``StopDecl``).

    - ``signal_accept`` REQUIRES ``threshold`` (``missing_stop_threshold``)
      AND ``signal`` (``missing_stop_signal``); ``signal.inputs`` MUST ⊆ the
      loop's ``state_keys`` (checked by :class:`LoopBody`).
    - ``external_accept`` REQUIRES ``predicate`` (``missing_stop_predicate``);
      the predicate is an OPAQUE runtime id, outside the P8 surface.
    - ``exhausted`` carries none of the above.
    """

    kind: StopKind
    threshold: str | None = None
    signal: SignalUse | None = None
    predicate: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StopKind):
            raise _reject(
                "unknown_stop_kind", f"stop kind outside the v1 registry: {self.kind!r}"
            )
        if self.kind is StopKind.SIGNAL_ACCEPT:
            if not isinstance(self.threshold, str) or not self.threshold:
                raise _reject(
                    "missing_stop_threshold", "signal_accept stop requires a threshold"
                )
            if self.signal is None:
                raise _reject(
                    "missing_stop_signal", "signal_accept stop requires a signal"
                )
            if self.predicate is not None:
                raise _reject(
                    "unknown_composite_field",
                    "signal_accept stop carries no predicate field",
                )
        elif self.kind is StopKind.EXTERNAL_ACCEPT:
            if not isinstance(self.predicate, str) or not self.predicate:
                raise _reject(
                    "missing_stop_predicate",
                    "external_accept stop requires a non-empty predicate identifier",
                )
            if self.threshold is not None or self.signal is not None:
                raise _reject(
                    "unknown_composite_field",
                    "external_accept stop carries no threshold/signal field",
                )
        else:  # EXHAUSTED
            if (
                self.threshold is not None
                or self.signal is not None
                or self.predicate is not None
            ):
                raise _reject(
                    "unknown_composite_field",
                    "exhausted stop carries no threshold/signal/predicate field",
                )


# --------------------------------------------------------------------------- #
# Bodies (§3.2)                                                               #
# --------------------------------------------------------------------------- #


def _check_arms_nonempty(arms: tuple[Arm, ...]) -> None:
    if not isinstance(arms, tuple):
        raise _reject("invalid_arm_shape", "arms must be a tuple of tagged Arms")
    if not arms:
        raise _reject("empty_arms", "arms must be non-empty for every constructor")
    for arm in arms:
        if not isinstance(arm, (StageArm, CompositeArm)):
            raise _reject("invalid_arm_shape", f"arm is not a tagged Arm: {arm!r}")


def _check_duplicate_stages(arms: Iterable[Arm]) -> None:
    seen: set[str] = set()
    for arm in arms:
        if isinstance(arm, StageArm):
            if arm.name in seen:
                raise _reject(
                    "duplicate_stage",
                    f"duplicate stage {arm.name!r} within one composite",
                )
            seen.add(arm.name)


def _arm_is_margin_bearing(
    arm: Arm, registry: Mapping[str, CompositeNode] | None
) -> bool:
    """Item 5: a margin-bearing arm is a stage (RFC 0001 vote semantics) or an
    ensemble whose ``aggregate.kind = majority_vote`` (committee votes).

    With no registry the nested-composite case is conservatively rejected as
    non-margin-bearing — node-local construction only sees stages as
    definitively margin-bearing. Program-level validation passes the registry.
    """
    if isinstance(arm, StageArm):
        return True
    if registry is None:
        return False
    node = registry.get(arm.ref)
    if node is None:
        return False
    body = node.body
    return (
        isinstance(body, EnsembleBody)
        and body.aggregate.kind is AggregateKind.MAJORITY_VOTE
    )


@dataclass(frozen=True, slots=True)
class CascadeBody:
    """A cascade body (§3.2 ``Cascade``).

    ``|gates| = |arms| − 1`` (``cascade_arity``); the degenerate ``m = 1``
    cascade has no gates and always returns its arm. ``placement`` defaults
    POST. Gate kind/placement symmetry (item 5) is enforced here for the
    placement axis and for the margin-bearing-arm rule in the stage case; the
    nested-composite margin-bearing case is finalized at program level.
    """

    arms: tuple[Arm, ...]
    gates: tuple[GateDecl, ...] = ()
    placement: Placement = Placement.POST

    def __post_init__(self) -> None:
        _check_arms_nonempty(self.arms)
        if not isinstance(self.placement, Placement):
            raise _reject(
                "unknown_composite_field", f"unknown placement: {self.placement!r}"
            )
        if not isinstance(self.gates, tuple):
            raise _reject("invalid_arm_shape", "gates must be a tuple of GateDecl")
        if len(self.gates) != len(self.arms) - 1:
            raise _reject(
                "cascade_arity",
                f"|gates|={len(self.gates)} must equal |arms|-1={len(self.arms) - 1}",
            )
        _check_duplicate_stages(self.arms)
        self._check_gate_placement(registry=None)

    def _check_gate_placement(
        self, registry: Mapping[str, CompositeNode] | None
    ) -> None:
        for index, gate in enumerate(self.gates):
            if self.placement is Placement.PRE and gate.kind is GateKind.MARGIN_BELOW:
                raise _reject(
                    "gate_kind_placement_mismatch", "margin_below gate is POST-only"
                )
            if self.placement is Placement.POST and gate.kind is GateKind.SIGNAL_BELOW:
                raise _reject(
                    "gate_kind_placement_mismatch", "signal_below gate is PRE-only"
                )
            if (
                self.placement is Placement.POST
                and gate.kind is GateKind.MARGIN_BELOW
                and not _arm_is_margin_bearing(self.arms[index], registry)
            ):
                # Defer the nested-composite verdict to program level: only
                # raise here when a registry is present (final verdict) or when
                # the gated arm is a CompositeArm we cannot yet resolve.
                if registry is not None or isinstance(self.arms[index], StageArm):
                    raise _reject(
                        "gate_arm_incompatible",
                        f"margin_below gate {index} gates a non-margin-bearing arm",
                    )


@dataclass(frozen=True, slots=True)
class EnsembleBody:
    """An ensemble body (§3.2 ``Ensemble``).

    ``cardinality`` is REQUIRED iff ``|arms| = 1`` (sampling form) and
    FORBIDDEN if ``|arms| > 1`` (committee form) — ``cardinality_arity_mismatch``.
    It is a namespace ref to a TVAR or CVAR of TVL type ``int`` (kind/type
    checked at program level; resolution-time ``k < 1`` is rejection R9).
    """

    arms: tuple[Arm, ...]
    aggregate: AggregateDecl
    cardinality: str | None = None

    def __post_init__(self) -> None:
        _check_arms_nonempty(self.arms)
        if not isinstance(self.aggregate, AggregateDecl):
            raise _reject(
                "unknown_aggregate_kind", "ensemble requires an AggregateDecl"
            )
        is_sampling = len(self.arms) == 1
        if is_sampling and self.cardinality is None:
            raise _reject(
                "cardinality_arity_mismatch",
                "single-arm (sampling) ensemble requires cardinality",
            )
        if not is_sampling and self.cardinality is not None:
            raise _reject(
                "cardinality_arity_mismatch",
                "multi-arm (committee) ensemble forbids cardinality",
            )
        if self.cardinality is not None and (
            not isinstance(self.cardinality, str) or not self.cardinality
        ):
            raise _reject(
                "invalid_arm_shape", "cardinality must be a non-empty identifier"
            )
        # §3.2 item 6: duplicates within ONE composite — the judge arm is part
        # of this composite's stage scope, so it is checked alongside arms.
        judge = self.aggregate.judge
        scope = self.arms if judge is None else (*self.arms, judge)
        _check_duplicate_stages(scope)


@dataclass(frozen=True, slots=True)
class LoopBody:
    """A loop body (§3.2 ``Loop``).

    ``state_keys`` are content-free identifiers (``[]`` ⟺ body treated pure).
    ``stop.signal.inputs`` (when present) MUST ⊆ ``state_keys``
    (``stop_signal_outside_state``). ``max_iters`` is the REQUIRED totality
    bound (``invalid_max_iters`` when ``< 1``).
    """

    body: Arm
    stop: StopDecl
    state_keys: tuple[str, ...] = ()
    max_iters: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.body, (StageArm, CompositeArm)):
            raise _reject("invalid_arm_shape", "Loop.body must be a tagged Arm")
        if not isinstance(self.stop, StopDecl):
            raise _reject("unknown_stop_kind", "Loop requires a StopDecl")
        if not isinstance(self.state_keys, tuple):
            raise _reject(
                "invalid_arm_shape", "state_keys must be a tuple of identifiers"
            )
        for key in self.state_keys:
            if not isinstance(key, str) or not key:
                raise _reject(
                    "invalid_arm_shape", f"state_key not an identifier: {key!r}"
                )
        if not isinstance(self.max_iters, int) or isinstance(self.max_iters, bool):
            raise _reject("invalid_max_iters", "max_iters must be an integer")
        if self.max_iters < 1:
            raise _reject(
                "invalid_max_iters", f"max_iters must be >= 1, got {self.max_iters}"
            )
        if (
            self.stop.kind is StopKind.SIGNAL_ACCEPT
            and self.stop.signal is not None
            and not set(self.stop.signal.inputs) <= set(self.state_keys)
        ):
            raise _reject(
                "stop_signal_outside_state",
                "signal_accept stop inputs must be a subset of state_keys",
            )


Body = CascadeBody | EnsembleBody | LoopBody


@dataclass(frozen=True, slots=True)
class CompositeNode:
    """One composite declaration: a member of ``N_X`` (§3.2 ``Composite``).

    ``kind`` discriminates ``body``; a kind/body mismatch rejects
    (``unknown_composite_field``). ``scope`` is the RFC 0001 §3.7 ``scope_spec``
    carried VERBATIM with PolicyDecl semantics (§3.2, §4 subsumption — optional,
    frozen, OUTSIDE the P8 surface). ``provenance`` rides as expansion-output
    metadata. ``parameters`` is an opaque operational map OUTSIDE the P8
    surface, exactly as ``PolicyDecl.parameters`` (kept as an opaque mapping;
    never serialized through the typed wire).
    """

    name: str
    kind: CompositeKind
    body: Body
    scope: Scope | None = None
    provenance: Provenance | None = None
    parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _reject(
                "invalid_arm_shape", "CompositeNode.name must be a non-empty identifier"
            )
        if not isinstance(self.kind, CompositeKind):
            raise _reject(
                "unknown_composite_kind",
                f"kind outside the closed registry: {self.kind!r}",
            )
        if self.scope is not None and not isinstance(self.scope, Scope):
            raise _reject(
                "unknown_composite_field",
                f"scope must be a Scope, got {type(self.scope).__name__}",
            )
        expected = {
            CompositeKind.CASCADE: CascadeBody,
            CompositeKind.ENSEMBLE: EnsembleBody,
            CompositeKind.LOOP: LoopBody,
        }[self.kind]
        if not isinstance(self.body, expected):
            raise _reject(
                "unknown_composite_field",
                f"kind {self.kind.value!r} requires {expected.__name__}, "
                f"got {type(self.body).__name__}",
            )


# --------------------------------------------------------------------------- #
# Program (the N_X registry + namespace view) and cross-composite validation  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompositeProgram:
    """A module's composite registry plus the TVAR/CVAR namespace view.

    ``composites`` is ``N_X`` (keyed by name). ``tvars`` and ``cvars`` are the
    declared identifier names of the surrounding module's tuned/calibrated
    knobs (a namespace VIEW only — no bindings, no values). ``cvar_types``
    maps each CVAR name to its declared TVL type (for the int/float threshold
    and int-cardinality checks); ``tvar_types`` is the PARALLEL mapping for
    TVARs (a TVAR-bound cardinality must also be int-typed — §3.2 item 7). A
    name absent from its type map is treated as untyped/unknown and rejected by
    the type checks. ``cvar_signals`` maps each CVAR to its declared
    ``calibration.signal`` id (``None`` when absent) for the §3.2 item-11
    binding lint, and ``cvar_depends_on`` to its declared TVAR parents for the
    §3.5 ``missing_composite_parent`` lint. ``cvar_signal_inputs`` maps each
    CVAR to the input list covered under the ``signal_inputs``
    freshness-context extension (``None`` when uncovered).
    """

    composites: Mapping[str, CompositeNode]
    tvars: frozenset[str] = frozenset()
    cvars: frozenset[str] = frozenset()
    cvar_types: Mapping[str, str] = field(default_factory=dict)
    tvar_types: Mapping[str, str] = field(default_factory=dict)
    cvar_signals: Mapping[str, str | None] = field(default_factory=dict)
    cvar_depends_on: Mapping[str, frozenset[str]] = field(default_factory=dict)
    cvar_signal_inputs: Mapping[str, tuple[str, ...] | None] = field(
        default_factory=dict
    )


def _all_arms(body: Body) -> tuple[Arm, ...]:
    """Every arm/body/judge an expansion node directly contains."""
    if isinstance(body, CascadeBody):
        return body.arms
    if isinstance(body, EnsembleBody):
        if body.aggregate.judge is not None:
            return (*body.arms, body.aggregate.judge)
        return body.arms
    return (body.body,)  # LoopBody


def _detect_cycle(program: CompositeProgram) -> None:
    """Item 4: the ``composite(·)`` reference graph over ``N_X`` is acyclic.

    DFS with a recursion stack; a back-edge is ``composite_cycle``. A
    reference to an undeclared composite is ``missing_composite_ref`` (item 3).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(program.composites, WHITE)

    def visit(name: str) -> None:
        color[name] = GRAY
        node = program.composites[name]
        for arm in _all_arms(node.body):
            if isinstance(arm, CompositeArm):
                if arm.ref not in program.composites:
                    raise _reject(
                        "missing_composite_ref",
                        f"composite({arm.ref!r}) resolves to no N_X member",
                    )
                if color[arm.ref] == GRAY:
                    raise _reject(
                        "composite_cycle", f"cycle through composite({arm.ref!r})"
                    )
                if color[arm.ref] == WHITE:
                    visit(arm.ref)
        color[name] = BLACK

    for name in program.composites:
        if color[name] == WHITE:
            visit(name)


def _check_tuned_params(program: CompositeProgram) -> None:
    """Item 9: every ``tuned_params`` entry resolves to a declared TVAR."""
    for node in program.composites.values():
        for arm in _all_arms(node.body):
            if isinstance(arm, StageArm):
                for tvar in arm.tuned_params:
                    if tvar not in program.tvars:
                        raise _reject(
                            "invalid_tuned_param",
                            f"tuned_param {tvar!r} does not resolve to a TVAR",
                        )


def _check_threshold(program: CompositeProgram, threshold: str) -> None:
    """Item 6 + item 10: a gate/accept/stop threshold resolves to a numeric
    (int/float) CVAR."""
    if threshold not in program.cvars:
        raise _reject(
            "missing_ref", f"threshold {threshold!r} does not resolve to a CVAR"
        )
    declared = program.cvar_types.get(threshold)
    if declared not in ("int", "float"):
        raise _reject(
            "invalid_threshold_type",
            f"threshold CVAR {threshold!r} has non-numeric type {declared!r}",
        )


def _check_cardinality(program: CompositeProgram, cardinality: str) -> None:
    """Item 7: cardinality references a TVAR or CVAR of declared TVL type
    ``int``. The int-type obligation applies to BOTH a tuned and a calibrated
    cardinality — a non-int cardinality (e.g. a float TVAR) rejects
    ``invalid_cardinality_type`` regardless of binding kind. Resolution-time
    ``k < 1`` is the separate runtime rejection R9 (``invalid_cardinality_value``).
    """
    if cardinality in program.tvars:
        if program.tvar_types.get(cardinality) != "int":
            raise _reject(
                "invalid_cardinality_type",
                f"cardinality TVAR {cardinality!r} is not int-typed",
            )
        return  # int-typed tuned cardinality; k<1 is R9 (runtime)
    if cardinality not in program.cvars:
        raise _reject(
            "missing_ref",
            f"cardinality {cardinality!r} does not resolve to a TVAR/CVAR",
        )
    if program.cvar_types.get(cardinality) != "int":
        raise _reject(
            "invalid_cardinality_type",
            f"cardinality CVAR {cardinality!r} is not int-typed",
        )


def _sig_of_construct(
    *,
    gate: GateDecl | None = None,
    accept: AcceptDecl | None = None,
    stop: StopDecl | None = None,
) -> str:
    """§3.2 item 11: the canonical signal id a thresholded construct
    determines."""
    if gate is not None:
        if gate.kind is GateKind.SIGNAL_BELOW:
            if gate.signal is None:  # guaranteed by GateDecl.__post_init__
                raise _reject("missing_gate_signal", "signal_below gate has no signal")
            return gate.signal.signal
        return StatKind.VOTE_MARGIN.value  # margin_below -> vote_margin
    if accept is not None:
        return accept.stat.value
    if stop is None or stop.signal is None:  # guaranteed by StopDecl.__post_init__
        raise _reject("missing_stop_signal", "signal_accept stop has no signal")
    return stop.signal.signal


def _check_signal_binding(
    program: CompositeProgram,
    threshold: str,
    expected_signal: str,
    inputs: tuple[str, ...],
) -> None:
    """§3.2 item 11 (composite USE-SITE): the referenced threshold CVAR must
    declare ``calibration.signal = sig(construct)``, and when the governing
    ``SignalUse.inputs ≠ []`` must cover ``signal_inputs`` with that exact
    ordered list."""
    declared = program.cvar_signals.get(threshold)
    if declared is None:
        raise _reject(
            "missing_calibration_signal",
            f"composite-referenced threshold {threshold!r} declares no calibration.signal",
        )
    if declared != expected_signal:
        raise _reject(
            "signal_mismatch",
            f"threshold {threshold!r} calibration.signal {declared!r} != sig {expected_signal!r}",
        )
    if inputs:
        covered = program.cvar_signal_inputs.get(threshold)
        if covered is None or tuple(covered) != inputs:
            raise _reject(
                "unbound_signal_inputs",
                f"threshold {threshold!r} does not cover signal_inputs {inputs!r}",
            )


def _check_node_semantics(program: CompositeProgram, node: CompositeNode) -> None:
    """Per-node threshold/signal/placement checks with the registry present."""
    body = node.body
    if isinstance(body, CascadeBody):
        body._check_gate_placement(
            registry=program.composites
        )  # final margin-bearing verdict
        for gate in body.gates:
            _check_threshold(program, gate.threshold)
            inputs = gate.signal.inputs if gate.signal is not None else ()
            _check_signal_binding(
                program, gate.threshold, _sig_of_construct(gate=gate), inputs
            )
    elif isinstance(body, EnsembleBody):
        if body.cardinality is not None:
            _check_cardinality(program, body.cardinality)
        accept = body.aggregate.accept
        if accept is not None:
            _check_threshold(program, accept.threshold)
            _check_signal_binding(
                program, accept.threshold, _sig_of_construct(accept=accept), ()
            )
    else:  # LoopBody
        stop = body.stop
        if stop.kind is StopKind.SIGNAL_ACCEPT and stop.threshold is not None:
            _check_threshold(program, stop.threshold)
            inputs = stop.signal.inputs if stop.signal is not None else ()
            _check_signal_binding(
                program, stop.threshold, _sig_of_construct(stop=stop), inputs
            )


def validate_program(program: CompositeProgram) -> None:
    """Run every cross-composite well-formedness check (§3.2, §3.5, §3.11).

    Node-local rules already ran on construction. This adds the rules that
    need the whole ``N_X`` and the namespace view: acyclicity
    (``composite_cycle``) + reference resolution (``missing_composite_ref``),
    TVAR-only ``tuned_params`` (``invalid_tuned_param``), numeric/typed
    thresholds and cardinality, the §3.2 item-11 signal binding, the final
    margin-bearing-arm verdict (``gate_arm_incompatible``), and the §3.5
    parent-coverage obligation (``missing_composite_parent``).
    """
    _detect_cycle(program)
    _check_tuned_params(program)
    for node in program.composites.values():
        _check_node_semantics(program, node)
    _check_parent_coverage(program)


# --------------------------------------------------------------------------- #
# §3.5 leafT / required_parents                                               #
# --------------------------------------------------------------------------- #


def _leaf_names_raw(
    arm: Arm, registry: Mapping[str, CompositeNode], _seen: frozenset[str] | None = None
) -> frozenset[str]:
    """NAMESPACE-BLIND leaf walk over the §3.5 ``leafT`` structure.

    Collects every ``tuned_params`` entry plus every sampling-ensemble
    ``cardinality`` name, recursively through nesting — WITHOUT consulting the
    ``N_T`` partition. The cardinality term it folds in may therefore be a CVAR
    or a TVAR; this raw set is INTERNAL only and must be intersected with
    ``N_T`` before it is exposed (the public :func:`leaf_tvars` does so). Using
    this set directly on the public surface would violate the TVAR-only / P5
    boundary (§3.5, C6) by returning calibrated names as leaves.
    """
    seen = _seen or frozenset()
    if isinstance(arm, StageArm):
        return frozenset(arm.tuned_params)
    # CompositeArm
    if arm.ref in seen:  # acyclic by validation; guard anyway
        return frozenset()
    node = registry.get(arm.ref)
    if node is None:  # unresolved ref (caught elsewhere as missing_composite_ref)
        return frozenset()
    return _leaf_names_of_node(node, registry, seen | {arm.ref})


def _leaf_names_of_node(
    node: CompositeNode, registry: Mapping[str, CompositeNode], seen: frozenset[str]
) -> frozenset[str]:
    body = node.body
    leaves: frozenset[str] = frozenset()
    for member in _all_arms(body):
        leaves |= _leaf_names_raw(member, registry, seen)
    if isinstance(body, EnsembleBody) and body.cardinality is not None:
        # Raw fold: include the cardinality name WHATEVER its binding. A
        # Calibrated/absent cardinality contributes nothing to the final
        # ``leafT`` (it is a Cal member instead, §3.6), but that ∩ N_T cut is
        # the PUBLIC layer's job — never this namespace-blind walk's.
        leaves |= {body.cardinality}
    return leaves


def leaf_tvars(program: CompositeProgram, arm: Arm) -> frozenset[str]:
    """``leafT`` (§3.5): the tuned-TVAR leaf set an arm parameterizes.

    PUBLIC surface, TVAR-only (C6 / the P5 boundary). ``leafT(stage(s, ps)) =
    ps``; ``leafT(composite(x))`` unions ``leafT`` over all arms/body/judge of
    ``x`` PLUS its sampling-ensemble's tuned cardinality (``{cardinality} ∩
    N_T``), folding in recursively through nesting.

    The cardinality fold is intersected with ``N_T`` at EVERY layer (via the
    final ``& program.tvars`` cut over the namespace-blind walk), so a composite
    whose ensemble cardinality is bound Calibrated NEVER leaks a CVAR onto this
    helper — that CVAR is a :func:`cal_fold` member instead (§3.6). Requiring
    the ``program`` namespace is what makes this guarantee total.
    """
    return _leaf_names_raw(arm, program.composites) & program.tvars


def _leaf_tvars_program(arm: Arm, program: CompositeProgram) -> frozenset[str]:
    """Internal alias kept for the §3.5 ``required_parents`` call sites."""
    return leaf_tvars(program, arm)


def required_parents(
    program: CompositeProgram, node: CompositeNode
) -> dict[str, frozenset[str]]:
    """``required_parents`` (§3.5): per-threshold coverage obligations.

    Returns a map from each threshold CVAR in ``node`` to the TVAR set its
    certificate must depend on. TVAR-only (C6) — the cardinality term is
    intersected with ``N_T``.
    """
    out: dict[str, frozenset[str]] = {}
    body = node.body
    if isinstance(body, CascadeBody):
        if body.placement is Placement.POST:
            # θ_i covers leafT(a_1) ∪ … ∪ leafT(a_i)
            prefix: frozenset[str] = frozenset()
            for index, gate in enumerate(body.gates):
                prefix |= _leaf_tvars_program(body.arms[index], program)
                out[gate.threshold] = out.get(gate.threshold, frozenset()) | prefix
        else:  # PRE: θ_i covers leafT(a_i)
            for index, gate in enumerate(body.gates):
                out[gate.threshold] = out.get(
                    gate.threshold, frozenset()
                ) | _leaf_tvars_program(body.arms[index], program)
    elif isinstance(body, EnsembleBody):
        accept = body.aggregate.accept
        if accept is not None:
            leaves: frozenset[str] = frozenset()
            for member in _all_arms(body):
                leaves |= _leaf_tvars_program(member, program)
            if body.cardinality is not None and body.cardinality in program.tvars:
                leaves |= {body.cardinality}
            out[accept.threshold] = out.get(accept.threshold, frozenset()) | leaves
    else:  # LoopBody
        stop = body.stop
        if stop.kind is StopKind.SIGNAL_ACCEPT and stop.threshold is not None:
            out[stop.threshold] = out.get(
                stop.threshold, frozenset()
            ) | _leaf_tvars_program(body.body, program)
    return out


def _check_parent_coverage(program: CompositeProgram) -> None:
    """§3.5 ``missing_composite_parent``: ``required_parents(θ) ⊆ depends_on(θ)``."""
    for node in program.composites.values():
        for threshold, needed in required_parents(program, node).items():
            declared = program.cvar_depends_on.get(threshold, frozenset())
            if not needed <= declared:
                missing = needed - declared
                raise _reject(
                    "missing_composite_parent",
                    f"threshold {threshold!r} depends_on missing TVAR parents {sorted(missing)!r}",
                )


# --------------------------------------------------------------------------- #
# §3.6 Cal fold + roots                                                       #
# --------------------------------------------------------------------------- #


def _cal_of_arm(arm: Arm, registry: Mapping[str, CompositeNode]) -> frozenset[str]:
    """``Cal(stage) = ∅``; ``Cal(composite(x)) = Cal(body(x))``."""
    if isinstance(arm, StageArm):
        return frozenset()
    node = registry.get(arm.ref)
    if node is None:
        return frozenset()
    return _cal_of_node(node, registry)


def _cal_of_node(
    node: CompositeNode, registry: Mapping[str, CompositeNode]
) -> frozenset[str]:
    body = node.body
    members: frozenset[str] = frozenset()
    if isinstance(body, CascadeBody):
        for arm in body.arms:
            members |= _cal_of_arm(arm, registry)
        members |= {gate.threshold for gate in body.gates}
    elif isinstance(body, EnsembleBody):
        for arm in body.arms:
            members |= _cal_of_arm(arm, registry)
        if body.aggregate.judge is not None:
            members |= _cal_of_arm(body.aggregate.judge, registry)
        if body.aggregate.accept is not None:
            members |= {body.aggregate.accept.threshold}
        # ({k} ∩ N_C) — a Calibrated cardinality is a Cal member. The N_C test
        # needs the program; cal_fold(program, ...) intersects below.
        if body.cardinality is not None:
            members |= {body.cardinality}
    else:  # LoopBody
        members |= _cal_of_arm(body.body, registry)
        if body.stop.kind is StopKind.SIGNAL_ACCEPT and body.stop.threshold is not None:
            members |= {body.stop.threshold}
    return members


def roots(program: CompositeProgram) -> frozenset[str]:
    """§3.6 root consumption (v1, conservative): the roots of the ``N_X``
    reference DAG — every declared composite NOT referenced as a nested arm of
    another. v1 consumes all roots by every candidate (fail-closed)."""
    referenced: set[str] = set()
    for node in program.composites.values():
        for arm in _all_arms(node.body):
            if isinstance(arm, CompositeArm):
                referenced.add(arm.ref)
    return frozenset(name for name in program.composites if name not in referenced)


def cal_fold(program: CompositeProgram) -> frozenset[str]:
    """§3.6 certificate coverage fold: ``⋃_{r ∈ roots} Cal(r)`` — the set of
    calibratable member CVAR names whose certificates strict selection
    requires. Intersected with ``N_C`` so a tuned cardinality is excluded
    (it is a ``leafT`` member instead, §3.5). Fail-closed: the caller checks
    each member has a valid fresh certificate; any gap ⇒ no certified
    selection."""
    members: frozenset[str] = frozenset()
    for root in roots(program):
        members |= _cal_of_node(program.composites[root], program.composites)
    # ALWAYS intersect — an empty cvar namespace means an empty Cal set, never
    # a raw fall-through (codex delta round: the falsy-empty check leaked a
    # TUNED cardinality into the Cal set — the §3.5/§3.6 boundary dual of the
    # leafT cut).
    return members & program.cvars
