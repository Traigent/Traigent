"""The deterministic knob resolver (RFC 0001 §3.4) — SDK packet 6-C2.

Resolution turns an optimizer suggestion (Tuned values only, P2) into the
full executed configuration:

    config = suggestion ∪ fixed-binding values ∪ runtime fixed ∪ calibrated values

Acceptance is EXACTLY the complement of the rejection conditions plus
calibrator success (the Accept biconditional — P3/P4):

    Accept ⟺ ¬(R1 ∨ … ∨ R8) ∧ every calibrated value present

The resolver CHECKS certificate freshness; it never fits. In-loop
re-calibration and the CVAR optimizer are explicitly deferred (RFC §2).
Rejections are collected exhaustively (no short-circuit) and raised as one
typed, fail-closed :class:`ResolutionError`. Fallbacks are consumed only for
non-governed CVARs and are always recorded in the result — never silent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .bindings import Calibrated, Fixed, Tuned
from .certificates import Certificate, FreshnessContext, conformal_evidence_floor
from .resolution import ResolutionError, ResolutionRejection
from .signals import SignalObservation

if TYPE_CHECKING:
    from traigent.api.config_space import ConfigSpace

__all__ = ["CalibratedInput", "KnobResolver", "ResolvedConfiguration"]


@dataclass(frozen=True, slots=True)
class CalibratedInput:
    """The runtime inputs for one CVAR: its calibrated value + validity data.

    ``value is None`` means the calibrator abstained (⊥ / no_decision).
    """

    value: Any | None
    certificate: Certificate | None = None
    context: FreshnessContext | None = None
    observations: tuple[SignalObservation, ...] = field(default=())
    evidence_item_ids: frozenset[str] = frozenset()  # calibration pool item ids


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    """A successful resolution: the executed config + observability."""

    config: Mapping[str, Any]
    used_fallbacks: tuple[str, ...] = ()


def _type_conforms(value: Any, value_type: str) -> bool:
    """R5 type conformance (the model checker's transcription, productized)."""
    if value_type == "bool":
        return isinstance(value, bool)
    if value_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)  # enum[str] / str-like


class KnobResolver:
    """Deterministic, topological, fail-closed resolution over a ConfigSpace.

    Args:
        space: The configuration space whose ``knobs`` drive resolution.
        calibrated_inputs: Per-CVAR runtime inputs (value, certificate,
            context, observations). EVERY Calibrated knob needs an entry —
            a declared CVAR that cannot be produced is R4 (phase mismatch),
            consumed or not.
        runtime_fixed: The runtime-supplied fixed assignment ``f`` (N_F).
            Its domain must not intersect declared knob names (R3).
        eval_split: The evaluation split label; calibration evidence tagged
            with it (or item-overlapping it) is R7 evidence leakage.
        eval_item_ids: Optional eval-split item ids for the true-intersection
            form of R7 when callers track item identity.
    """

    def __init__(
        self,
        space: ConfigSpace,
        *,
        calibrated_inputs: Mapping[str, CalibratedInput] | None = None,
        runtime_fixed: Mapping[str, Any] | None = None,
        eval_split: str = "eval",
        eval_item_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._space = space
        self._calibrated_inputs = dict(calibrated_inputs or {})
        self._runtime_fixed = dict(runtime_fixed or {})
        self._eval_split = eval_split
        self._eval_item_ids = eval_item_ids

    # ------------------------------------------------------------------
    def resolve(self, suggestion: Mapping[str, Any]) -> ResolvedConfiguration:
        """Resolve one optimizer suggestion into the executed configuration.

        Raises:
            ResolutionError: carrying EVERY applicable rejection.
        """
        rejections: list[ResolutionRejection] = []
        details: list[str] = []
        knobs = dict(self._space.knobs or {})

        tuned_names = {n for n, k in knobs.items() if isinstance(k.binding, Tuned)}
        fixed_knobs = {
            n: k.binding for n, k in knobs.items() if isinstance(k.binding, Fixed)
        }
        calibrated_knobs = {
            n: k.binding for n, k in knobs.items() if isinstance(k.binding, Calibrated)
        }

        def reject(code: ResolutionRejection, detail: str) -> None:
            rejections.append(code)
            details.append(f"{code.value}: {detail}")

        # --- R3 duplicate providers -----------------------------------
        fixed_collisions = set(self._runtime_fixed) & set(knobs)
        for name in sorted(fixed_collisions):
            reject(
                ResolutionRejection.DUPLICATE_PROVIDER,
                f"runtime fixed assignment collides with declared knob {name!r}",
            )
        for name in sorted(set(self._calibrated_inputs) - set(calibrated_knobs)):
            reject(
                ResolutionRejection.DUPLICATE_PROVIDER,
                f"calibrated input provided for non-calibrated knob {name!r}",
            )
        for name in sorted(set(suggestion) - tuned_names):
            reject(
                ResolutionRejection.DUPLICATE_PROVIDER,
                f"suggestion provides non-tuned name {name!r} (dom(t) must be N_T)",
            )

        # --- R4 phase: every tuned value present, every CVAR producible -
        for name in sorted(tuned_names - set(suggestion)):
            reject(
                ResolutionRejection.PHASE_MISMATCH,
                f"suggestion is missing tuned knob {name!r}",
            )
        for name in sorted(set(calibrated_knobs) - set(self._calibrated_inputs)):
            reject(
                ResolutionRejection.PHASE_MISMATCH,
                f"declared CVAR {name!r} has no calibrated input (cannot be "
                "produced before use; resolution includes every n in N_C)",
            )

        # --- R2 refs + R1 cycles --------------------------------------
        for name in sorted(calibrated_knobs):
            for ref in calibrated_knobs[name].depends_on:
                if ref.knob in calibrated_knobs:
                    # v1: CVAR->CVAR parents are deferred; also feeds R1 below
                    reject(
                        ResolutionRejection.MISSING_REF,
                        f"CVAR {name!r} depends on CVAR {ref.knob!r} "
                        "(v1 parents must be Tuned)",
                    )
                elif ref.knob not in tuned_names:
                    reject(
                        ResolutionRejection.MISSING_REF,
                        f"CVAR {name!r} depends_on {ref.knob!r} which does not "
                        "resolve to a Tuned knob",
                    )
        self._detect_cycles(calibrated_knobs, reject)

        # --- per-CVAR resolution in deterministic (sorted) topo order --
        values: dict[str, Any] = {}
        used_fallbacks: list[str] = []
        for name in sorted(set(calibrated_knobs) & set(self._calibrated_inputs)):
            binding = calibrated_knobs[name]
            self._resolve_calibrated(
                name,
                binding,
                self._calibrated_inputs[name],
                values,
                used_fallbacks,
                reject,
            )

        if rejections:
            raise ResolutionError(tuple(dict.fromkeys(rejections)), "; ".join(details))

        config: dict[str, Any] = dict(suggestion)
        config.update({name: binding.value for name, binding in fixed_knobs.items()})
        config.update(self._runtime_fixed)
        config.update(values)
        return ResolvedConfiguration(
            config=config, used_fallbacks=tuple(used_fallbacks)
        )

    # ------------------------------------------------------------------
    def _resolve_calibrated(
        self,
        name: str,
        binding: Calibrated,
        provided: CalibratedInput,
        values: dict[str, Any],
        used_fallbacks: list[str],
        reject: Any,
    ) -> None:
        # Governance is DECLARED (on the binding/target), never inferred
        # from runtime-presented evidence — presenting a certificate is
        # evidence, not an opt-in to strictness.
        governed = (
            binding.require_calibration
            or binding.certificate is not None
            or binding.target.mode in ("certificate_backed", "guaranteed_selection")
        )

        # R7 evidence leakage — split tag or true item intersection
        for observation in provided.observations:
            if observation.split == self._eval_split:
                reject(
                    ResolutionRejection.EVIDENCE_LEAKAGE,
                    f"CVAR {name!r} consumed an observation from the eval split",
                )
                break
        if provided.context is not None and (
            provided.context.calibration_split == self._eval_split
        ):
            reject(
                ResolutionRejection.EVIDENCE_LEAKAGE,
                f"CVAR {name!r} calibration split equals the eval split",
            )
        if self._eval_item_ids and (provided.evidence_item_ids & self._eval_item_ids):
            # the TRUE intersection form: 𝓔_cal ∩ 𝓔_eval ≠ ∅ over item ids
            reject(
                ResolutionRejection.EVIDENCE_LEAKAGE,
                f"CVAR {name!r} calibration pool intersects the eval split items",
            )

        # R8 insufficient evidence — closed-form conformal floor. A declared
        # epsilon with NO context means the floor is unverifiable: fail
        # closed (unverifiable evidence IS insufficient evidence).
        if binding.target_epsilon is not None:
            floor = conformal_evidence_floor(binding.target_epsilon)
            if provided.context is None:
                reject(
                    ResolutionRejection.INSUFFICIENT_EVIDENCE,
                    f"CVAR {name!r} declares epsilon={binding.target_epsilon} "
                    "but no freshness context carries the evidence count",
                )
            elif provided.context.evidence_n < floor:
                reject(
                    ResolutionRejection.INSUFFICIENT_EVIDENCE,
                    f"CVAR {name!r} has n={provided.context.evidence_n} < "
                    f"conformal floor {floor} for epsilon={binding.target_epsilon}",
                )

        # calibrator ⊥ — the Accept biconditional requires 𝒦 ≠ ⊥ with NO
        # fallback carve-out (RFC §3.4); §3.5's non-strict fallback applies
        # to STALE CERTIFICATES only, never to abstention.
        if provided.value is None:
            reject(
                ResolutionRejection.NO_DECISION,
                f"calibrator for CVAR {name!r} abstained (⊥)",
            )
            return

        # R5 type + validity
        if not _type_conforms(provided.value, binding.value_type):
            reject(
                ResolutionRejection.INFEASIBLE_VALUE,
                f"CVAR {name!r} value {provided.value!r} does not conform to "
                f"declared type {binding.value_type!r}",
            )
            return

        # R6 certificate validity for governed CVARs
        if governed:
            certificate = provided.certificate or binding.certificate
            if certificate is None or provided.context is None:
                reject(
                    ResolutionRejection.STALE_CERTIFICATE,
                    f"governed CVAR {name!r} lacks a certificate/context",
                )
                return
            if not certificate.valid_for(
                name, binding.value_type, provided.value, provided.context
            ):
                reject(
                    ResolutionRejection.STALE_CERTIFICATE,
                    f"certificate for CVAR {name!r} is stale or invalid for the "
                    "current context",
                )
                return
        elif (
            binding.fallback is not None
            and provided.certificate is not None
            and provided.context is not None
            and not provided.certificate.valid_for(
                name, binding.value_type, provided.value, provided.context
            )
        ):
            # non-governed CVAR whose optional certificate is PROVEN stale
            # against a live context: the §3.5 carve-out — declared fallback
            # MAY be used, always recorded. With NO context, staleness is
            # not established: the unverifiable certificate is ignored and
            # the calibrated value stands (non-governed acceptance).
            values[name] = binding.fallback.value
            used_fallbacks.append(name)
            return

        values[name] = provided.value

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_cycles(calibrated_knobs: Mapping[str, Calibrated], reject: Any) -> None:
        """R1 — unreachable in v1 (CVAR->TVAR only) but kept normative for
        the deferred CVAR->CVAR extension."""
        adjacency = {
            name: {
                ref.knob for ref in binding.depends_on if ref.knob in calibrated_knobs
            }
            for name, binding in calibrated_knobs.items()
        }
        state: dict[str, int] = {}

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return True
            if state.get(node) == 2:
                return False
            state[node] = 1
            for neighbour in adjacency.get(node, ()):
                if visit(neighbour):
                    return True
            state[node] = 2
            return False

        for name in sorted(adjacency):
            if visit(name):
                reject(
                    ResolutionRejection.CYCLE,
                    f"dependency cycle involving CVAR {name!r}",
                )
                return
