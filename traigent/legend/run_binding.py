"""Pillar Legend run binding (SDK side): declared refs, drift handling and admission.

Contract: TraigentSchema ``docs/identity/pillar-legend-v0.md`` sections 6 and 9. The
Backend side is dark behind its own flag; this module is **driven only by the
``legend`` block of the session-create response** (and of each next-trial
suggestion). Without that block nothing here runs and no extra network call is made.

Flow (one :class:`LegendRunBinder` per optimization run):

1. :func:`parse_block` reads the Backend's closed ``legend`` block (opaque refs and
   closed enums only). Absent, ``enabled`` not true, or malformed => no binder.
2. :meth:`LegendRunBinder.admit_run` runs after run-identity preparation and BEFORE any
   evaluator or model call. Each pillar the SDK can observe from its existing envelopes
   is declared (stable idempotency keys), compared with the version the caller declared
   for that pillar (match => reuse, strict mismatch => refuse, permitted drift => a
   same-entity child version), and a COMPLETE run observation is sent to the admission
   route.
3. :meth:`LegendRunBinder.admit_trial` re-checks the configuration the trial will
   actually evaluate, right before evaluation, and sends the complete observation of
   that trial context.

Privacy: only opaque refs, closed enums and digests ever cross this module. Refusals
carry a closed code and a fixed message; they never echo a name, path or value. The
in-memory default transport of :mod:`traigent.legend.client` is NOT a server
declaration and is never used here: a transport must be supplied explicitly.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from traigent.legend import _contract as c
from traigent.legend import payloads
from traigent.legend.errors import LegendError
from traigent.legend.pillars import ConfigurationSource, Pillar, compute_pillar
from traigent.legend.transport import (  # noqa: F401 - re-exported
    BackendLegendTransport,
    DeclareResult,
    transport_for_client,
    with_retries,
)

MODE_STRICT = "strict"
MODE_PERMITTED = "permitted"
MODES = frozenset({MODE_STRICT, MODE_PERMITTED})

SCOPE_RUN = "run"
SCOPE_TRIAL = "trial"

#: Backend binding roles (contract section 3.3) per pillar kind.
RUN_ROLE_BY_KIND: dict[str, str] = {
    c.KIND_AGENT: "base_agent",
    c.KIND_CONFIGURATION: "run_base_configuration",
    c.KIND_DATASET: "dataset",
    c.KIND_EVALUATOR: "evaluator",
    c.KIND_SEARCH_SPACE: "search_space",
}
TRIAL_ROLE_BY_KIND: dict[str, str] = {
    c.KIND_AGENT: "trial_candidate",
    c.KIND_CONFIGURATION: "trial_configuration",
}
_KIND_BY_ROLE: dict[str, str] = {
    **{role: kind for kind, role in RUN_ROLE_BY_KIND.items()},
    **{role: kind for kind, role in TRIAL_ROLE_BY_KIND.items()},
    "search_set": c.KIND_DATASET,
}
_MAX_BINDINGS = 64
_RUN_REF_RE = re.compile(r"runref_[0-9a-f]{32}")
_TRIAL_REF_RE = re.compile(r"trialref_[0-9a-f]{32}")
_OPAQUE_ID_RE = re.compile(r"[A-Za-z0-9_.\-]{1,128}")

#: Closed local refusal codes (besides the contract's own).
CODE_DRIFT = "manifest_mismatch"
CODE_INCOMPLETE = "local_observation_incomplete"
CODE_NOTHING = "local_nothing_to_admit"
#: Roles only the Backend can attest (an immutable effective search space identity).
_ATTESTED_ROLES = frozenset({"search_space"})
CODE_NO_TRANSPORT = "local_transport_unavailable"
CODE_PRIVACY = "local_privacy_withholds_identity"

SlotKey = tuple[str, str, int]  # (binding_role, slot_scope, binding_slot)


class RunBindingTransport(Protocol):
    """Declares versions and posts the admission request. Injected; never built here."""

    def declare(self, payload: Mapping[str, Any]) -> DeclareResult: ...

    def admit(self, session_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DeclaredVersion:
    """A version the caller declared for a pillar kind (opaque refs only).

    ``entity_ref`` is needed to record permitted drift as a child in the same entity.
    """

    version_ref: str
    entity_ref: str | None = None

    def __post_init__(self) -> None:
        if not c.VERSION_REF_RE.fullmatch(self.version_ref or ""):
            raise LegendError(
                "invalid_request", "declared version must be an opaque ref"
            )
        if self.entity_ref is not None and not c.ENTITY_REF_RE.fullmatch(
            self.entity_ref
        ):
            raise LegendError(
                "invalid_request", "declared entity must be an opaque ref"
            )


@dataclass(frozen=True)
class LegendOptions:
    """Typed Legend options of ``@traigent.optimize`` and ``optimize()``.

    Giving options does not enable anything: the Backend's ``legend`` block does.
    ``declared`` maps a pillar kind to the version the caller expects to run.
    ``mode`` defaults to ``strict`` when options are given (a mismatch is refused before
    any evaluator call); a run with NO options is admitted as ``permitted`` (drift is
    recorded, never refuses a run that declared nothing).
    """

    declared: Mapping[str, DeclaredVersion] = field(default_factory=dict)
    mode: str = MODE_STRICT
    transport: RunBindingTransport | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise LegendError("invalid_request", "mode must be strict or permitted")
        clean: dict[str, DeclaredVersion] = {}
        for kind, value in dict(self.declared).items():
            if kind not in c.ENTITY_KINDS:
                raise LegendError("invalid_request", "unknown pillar kind")
            clean[kind] = (
                value
                if isinstance(value, DeclaredVersion)
                else DeclaredVersion(value)
                if isinstance(value, str)
                else _declared_from_mapping(value)
            )
        object.__setattr__(self, "declared", clean)

    @classmethod
    def coerce(
        cls, value: LegendOptions | Mapping[str, Any] | None
    ) -> LegendOptions | None:
        if value is None or isinstance(value, LegendOptions):
            return value
        if not isinstance(value, Mapping):
            raise LegendError("invalid_request", "legend options must be a mapping")
        unknown = set(value) - {"declared", "mode", "transport"}
        if unknown:
            raise LegendError("invalid_request", "unknown legend option")
        return cls(
            declared=value.get("declared") or {},
            mode=value.get("mode", MODE_STRICT),
            transport=value.get("transport"),
        )


def _declared_from_mapping(value: Any) -> DeclaredVersion:
    if not isinstance(value, Mapping) or set(value) - {"version_ref", "entity_ref"}:
        raise LegendError("invalid_request", "declared version must be a ref")
    return DeclaredVersion(value.get("version_ref"), value.get("entity_ref"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class LegendBlock:
    """The Backend's closed ``legend`` block: the run (and optionally trial) context."""

    run_ref: str
    trial_ref: str | None
    bindings: tuple[tuple[SlotKey, str], ...]  # ((role, scope, slot), version_ref)
    # Optional entity of each advertised version (opaque ref), when the Backend names it.
    # Declaring the observed identity UNDER that entity makes a matching fingerprint
    # resolve to the advertised version itself.
    entities: tuple[tuple[SlotKey, str], ...] = ()

    def entity_of(self, key: SlotKey) -> str | None:
        return dict(self.entities).get(key)

    def refs(self) -> dict[SlotKey, str]:
        return dict(self.bindings)


def parse_block(raw: Any) -> LegendBlock | None:
    """The block, or ``None`` when absent, not enabled, or malformed (never raises)."""
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return None
    run_ref = raw.get("run_ref")
    trial_ref = raw.get("trial_ref")
    items = raw.get("bindings")
    if not isinstance(run_ref, str) or not _RUN_REF_RE.fullmatch(run_ref):
        return None
    if trial_ref is not None and (
        not isinstance(trial_ref, str) or not _TRIAL_REF_RE.fullmatch(trial_ref)
    ):
        return None
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        return None
    if len(items) > _MAX_BINDINGS:
        return None
    parsed: list[tuple[SlotKey, str]] = []
    entities: list[tuple[SlotKey, str]] = []
    seen: set[SlotKey] = set()
    for item in items:
        if not isinstance(item, Mapping):
            return None
        role, scope = item.get("binding_role"), item.get("slot_scope")
        slot, ref = item.get("binding_slot"), item.get("version_ref")
        if (
            role not in _KIND_BY_ROLE
            or scope not in (SCOPE_RUN, SCOPE_TRIAL)
            or type(slot) is not int
            or not 0 <= slot <= 1023
            or not isinstance(ref, str)
            or not c.VERSION_REF_RE.fullmatch(ref)
        ):
            return None
        key = (role, scope, slot)
        if key in seen:
            return None
        seen.add(key)
        parsed.append((key, ref))
        entity = item.get("entity_ref")
        if entity is not None:
            if not isinstance(entity, str) or not c.ENTITY_REF_RE.fullmatch(entity):
                return None
            entities.append((key, entity))
    return LegendBlock(run_ref, trial_ref, tuple(parsed), tuple(entities))


def pillars_from_content_identity(
    wire: Mapping[str, Any] | None, config: Mapping[str, Any] | None
) -> dict[str, Pillar]:
    """The pillars already present in the session-create ``content_identity`` envelope.

    Nothing is recomputed from sources and no slot is invented: an unavailable slot is
    simply absent. ``config`` (the applied run-base configuration) is fingerprinted with
    the same digest the envelope's agent build used.
    """
    found: dict[str, Pillar] = {}
    wire = wire or {}
    agent = wire.get("agent")
    if isinstance(agent, Mapping) and isinstance(agent.get("build_digest"), str):
        found[c.KIND_AGENT] = Pillar(
            c.KIND_AGENT,
            c.SCHEME_BY_KIND[c.KIND_AGENT],
            None,
            agent["build_digest"],
            c.IDENTITY_SDK_FINGERPRINTED,
        )
    dataset = wire.get("dataset")
    if (
        isinstance(dataset, Mapping)
        and isinstance(dataset.get("dataset_root"), str)
        and isinstance(dataset.get("key_id"), str)
    ):
        found[c.KIND_DATASET] = Pillar(
            c.KIND_DATASET,
            c.SCHEME_BY_KIND[c.KIND_DATASET],
            dataset["key_id"],
            dataset["dataset_root"],
            c.IDENTITY_SDK_FINGERPRINTED,
            ordering=c.ORDERING_PRESERVED,
        )
    evaluator = wire.get("evaluator")
    if isinstance(evaluator, Mapping) and isinstance(
        evaluator.get("version_digest"), str
    ):
        found[c.KIND_EVALUATOR] = Pillar(
            c.KIND_EVALUATOR,
            c.SCHEME_BY_KIND[c.KIND_EVALUATOR],
            None,
            evaluator["version_digest"],
            c.IDENTITY_SDK_FINGERPRINTED,
        )
    if config is not None:
        try:
            found[c.KIND_CONFIGURATION] = compute_pillar(
                c.KIND_CONFIGURATION, ConfigurationSource(dict(config))
            )
        except LegendError:
            pass  # a configuration that cannot be fingerprinted is simply not observed
    return found


def _observation_items(observed: Mapping[SlotKey, str]) -> list[dict[str, Any]]:
    return [
        {
            "binding_role": role,
            "slot_scope": scope,
            "binding_slot": slot,
            "version_ref": ref,
        }
        for (role, scope, slot), ref in sorted(observed.items())
    ]


class LegendRunBinder:
    """Declares observed identities and admits them before execution (one per run)."""

    def __init__(
        self,
        *,
        session_id: str,
        block: LegendBlock,
        options: LegendOptions | None,
        transport: RunBindingTransport,
    ) -> None:
        self._session_id = session_id
        self._block = block
        self._options = options
        self._mode = options.mode if options is not None else MODE_PERMITTED
        self._transport = transport
        self._lock = threading.Lock()
        self._keys: dict[tuple[Any, ...], str] = {}
        self._declared: dict[tuple[Any, ...], DeclareResult] = {}
        self._run_observed: dict[SlotKey, str] = {}
        self._run_entities: dict[str, str] = {}
        self.run_tuple_digest: str | None = None
        self.run_admitted = False
        self.unadmitted_reason: str | None = None
        self.calls = 0

    @property
    def mode(self) -> str:
        return self._mode

    # -- declaration (stable idempotency keys) ------------------------------------------

    def _declare(
        self,
        pillar: Pillar,
        *,
        entity: str | None = None,
        previous: str | None = None,
        initial_parent: str | None = None,
        change_reason: str | None = None,
    ) -> DeclareResult:
        memo = (
            pillar.kind,
            pillar.fingerprint_scheme,
            pillar.key_id,
            pillar.fingerprint,
            entity,
            previous,
            initial_parent,
            change_reason,
        )
        with self._lock:
            done = self._declared.get(memo)
            if done is not None:
                return done
            key = self._keys.setdefault(memo, payloads.new_idempotency_key())
        payload = payloads.build_declaration_payload(
            pillar,
            entity_ref=entity,
            previous_version_ref=previous,
            change_reason=change_reason,
            initial_parent_version_ref=initial_parent,
            idempotency_key=key,
        )
        self.calls += 1
        result = with_retries(lambda: self._transport.declare(payload))
        with self._lock:
            self._declared[memo] = result
        return result

    def _observe(
        self, pillar: Pillar, declared: DeclaredVersion | None
    ) -> tuple[DeclareResult, bool]:
        """Declare an observed pillar; resolve drift against the declared version.

        Returns ``(result, drifted)``. Strict drift raises before anything runs;
        permitted drift records the observed identity as a child of the declared
        version in the declared entity.
        """
        entity = declared.entity_ref if declared is not None else None
        result = self._declare(pillar, entity=entity)
        if declared is None or result.version_ref == declared.version_ref:
            if declared is None and entity is None:
                self._note_new_entity(pillar.kind, result)
            return result, False
        if self._mode == MODE_STRICT:
            raise LegendError(CODE_DRIFT, "observed identity differs from declared")
        if declared.entity_ref is None:
            raise LegendError(
                "invalid_request", "permitted drift needs the declared entity"
            )
        child = self._declare(
            pillar,
            entity=declared.entity_ref,
            previous=declared.version_ref,
            initial_parent=None
            if result.entity_ref == declared.entity_ref
            else declared.version_ref,
            change_reason="unspecified",
        )
        return child, True

    def _note_new_entity(self, kind: str, result: DeclareResult) -> None:
        """An observation declared without any entity minted (or adopted) one.

        Documented path to reuse an entity across runs: resolve it from the local alias map
        (``AliasMap.resolve``) and pass it as ``DeclaredVersion.entity_ref``, or have the
        Backend name it in the block; otherwise each fresh fingerprint starts an entity."""
        self._run_entities.setdefault(kind, result.entity_ref)

    # -- admission ------------------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        data = with_retries(lambda: self._transport.admit(self._session_id, body))
        if not isinstance(data, Mapping) or not isinstance(
            data.get("run_tuple_digest"), str
        ):
            raise LegendError("internal_error", "admission returned no digest")
        return data

    def _incomplete(self, reason: str) -> bool:
        """An observation cannot be completed: strict refuses; otherwise stay unadmitted.

        Never fills a slot from the Backend's proposal. Returns True (unadmitted)."""
        if self._mode == MODE_STRICT:
            raise LegendError(CODE_INCOMPLETE, "observation is incomplete")
        self.unadmitted_reason = reason
        return True

    def admit_run(self, observed: Mapping[str, Pillar]) -> None:
        """Admit the run-base context. Call BEFORE any evaluator or model invocation.

        Every advertised run slot must be OBSERVED here (a pillar the SDK holds), except
        ``search_space``, which the Backend alone can attest (it identifies the immutable
        effective search space). Anything else missing is never copied from the Backend's
        proposal: strict refuses, permitted leaves the run unadmitted.
        """
        declared = self._options.declared if self._options is not None else {}
        run_keys = [(k, ref) for k, ref in self._block.bindings if k[1] == SCOPE_RUN]
        if not run_keys:
            self._incomplete(CODE_NOTHING)
            return
        for (role, _scope, _slot), _ref in run_keys:
            kind = _KIND_BY_ROLE[role]
            observable = role in _ATTESTED_ROLES or (
                RUN_ROLE_BY_KIND.get(kind) == role and kind in observed
            )
            if not observable:
                self._incomplete(CODE_INCOMPLETE)
                return
        # A declaration for a kind the run does not carry cannot be honoured.
        covered = {_KIND_BY_ROLE[role] for (role, _s, _n), _r in run_keys}
        if self._mode == MODE_STRICT and any(k not in covered for k in declared):
            raise LegendError(CODE_DRIFT, "declared pillar is not part of the run")
        run_refs: dict[SlotKey, str] = {}
        for key, advertised in run_keys:
            role = key[0]
            kind = _KIND_BY_ROLE[role]
            want = declared.get(kind)
            if role in _ATTESTED_ROLES:
                if (
                    want is not None
                    and want.version_ref != advertised
                    and self._mode == MODE_STRICT
                ):
                    raise LegendError(
                        CODE_DRIFT, "observed identity differs from declared"
                    )
                run_refs[key] = advertised  # Backend-attested, never SDK-observed
                continue
            entity = self._block.entity_of(key)
            baseline = want or (
                DeclaredVersion(advertised, entity) if entity is not None else None
            )
            result, _drifted = self._observe(observed[kind], baseline)
            run_refs[key] = result.version_ref
            self._run_entities[kind] = result.entity_ref
        self._run_observed = run_refs
        data = self._post(
            {"run_observed": _observation_items(run_refs), "mode": self._mode}
        )
        self.run_tuple_digest = str(data["run_tuple_digest"])
        self.run_admitted = True

    def admit_trial(
        self,
        *,
        trial_id: str,
        trial_block: LegendBlock,
        observed: Mapping[str, Pillar],
        suggested: Mapping[str, str] | None = None,
    ) -> None:
        """Re-check what this trial will actually execute, then admit it.

        ``observed`` maps a pillar kind to what is about to run: the configuration and
        the executed candidate (agent) identity. Inherited run slots are the ones already
        admitted. A trial slot with no observation is never filled from the Backend's
        suggestion: strict refuses, permitted leaves the trial unadmitted.
        """
        if not isinstance(trial_id, str) or not _OPAQUE_ID_RE.fullmatch(trial_id):
            raise LegendError("invalid_request", "trial id must be an opaque id")
        if trial_block.trial_ref is None:
            raise LegendError("invalid_request", "trial block carries no trial ref")
        suggested = suggested or {}
        if not self.run_admitted:
            return  # the run base never froze; a trial cannot be admitted (reason kept)
        for key, _ref in trial_block.bindings:
            role, scope, _slot = key
            kind = _KIND_BY_ROLE[role]
            if scope == SCOPE_RUN:
                if key not in self._run_observed:
                    self._incomplete(CODE_INCOMPLETE)
                    return
            elif not (TRIAL_ROLE_BY_KIND.get(kind) == role and kind in observed):
                self._incomplete(CODE_INCOMPLETE)
                return
        trial_observed: dict[SlotKey, str] = {}
        for key, advertised in trial_block.bindings:
            role, scope, _slot = key
            if scope == SCOPE_RUN:
                trial_observed[key] = self._run_observed[
                    key
                ]  # exactly what was admitted
                continue
            kind = _KIND_BY_ROLE[role]
            pillar = observed[kind]
            entity = trial_block.entity_of(key)
            if entity is not None:
                # Declared UNDER the advertised entity: the same fingerprint resolves to
                # the advertised version; a different one is drift (strict refuses,
                # permitted records a same-entity child).
                result, _drift = self._observe(
                    pillar, DeclaredVersion(advertised, entity)
                )
                trial_observed[key] = result.version_ref
            elif suggested.get(kind) is not None:
                if pillar.fingerprint == suggested[kind]:
                    # Entity not named: the identity is CONFIRMED by digest equality with
                    # the very suggestion the Backend digested, so its version stands.
                    trial_observed[key] = advertised
                elif self._mode == MODE_STRICT:
                    raise LegendError(CODE_DRIFT, "trial differs from its suggestion")
                else:
                    result = self._declare(pillar, entity=self._run_entities.get(kind))
                    trial_observed[key] = result.version_ref
            else:
                # no entity and no suggestion fingerprint: cannot be compared
                self._incomplete(CODE_INCOMPLETE)
                return
        self._post(
            {
                "run_observed": _observation_items(self._run_observed),
                "trial_id": trial_id,
                "trial_observed": _observation_items(trial_observed),
                "mode": self._mode,
            }
        )


def start_binder(
    *,
    session_id: str | None,
    raw_block: Any,
    options: LegendOptions | None,
    privacy_enabled: bool,
    fallback_transport: RunBindingTransport | None = None,
) -> LegendRunBinder | None:
    """A binder when (and only when) the Backend advertised Legend for this session.

    ``None`` => the run proceeds exactly as without Legend (zero additional calls).
    Fails closed, with a closed code, when the caller declared versions that cannot be
    honoured (privacy mode withholds identity; no transport to admit through).
    """
    block = parse_block(raw_block)
    wants_check = options is not None and bool(options.declared)
    if block is None or not session_id:
        advertised = isinstance(raw_block, Mapping) and raw_block.get("enabled") is True
        if advertised and wants_check:
            raise LegendError("invalid_request", "backend legend block is unusable")
        return None
    if privacy_enabled:
        if wants_check:
            raise LegendError(CODE_PRIVACY, "privacy mode withholds identity")
        return None
    transport = (
        options.transport if options is not None else None
    ) or fallback_transport
    if transport is None:
        if wants_check:
            raise LegendError(CODE_NO_TRANSPORT, "no Legend transport is configured")
        return None
    return LegendRunBinder(
        session_id=session_id, block=block, options=options, transport=transport
    )


def candidate_pillar(
    identity_run: Any, evaluation_config: Mapping[str, Any]
) -> Pillar | None:
    """The agent build the trial will EXECUTE (applied config included), or ``None``.

    Derived from the run's content-identity snapshot at the execution seam; ``None``
    when there is no snapshot or the build cannot be stated honestly."""
    if identity_run is None:
        return None
    try:
        version, _reason = identity_run._version_for(dict(evaluation_config))
    except Exception:  # noqa: BLE001 - identity never blocks a run
        return None
    digest = version.get("build_digest") if isinstance(version, Mapping) else None
    if not isinstance(digest, str):
        return None
    return Pillar(
        c.KIND_AGENT,
        c.SCHEME_BY_KIND[c.KIND_AGENT],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
    )


def admit_trial_config(
    binder: LegendRunBinder,
    trial_id: str,
    raw_trial_block: Any,
    evaluation_config: Mapping[str, Any],
    candidate: Pillar | None = None,
    suggested_fingerprint: str | None = None,
) -> None:
    """Re-check and admit one trial's ACTUAL configuration and executed candidate.

    A trial block that is not the Backend's enabled trial block (or carries no trial
    ref) means the Backend issued no Legend context for this trial: nothing to admit.
    An identity that cannot be derived stays missing (never taken from the suggestion).
    """
    block = parse_block(raw_trial_block)
    if block is None or block.trial_ref is None:
        return
    observed: dict[str, Pillar] = {}
    try:
        observed[c.KIND_CONFIGURATION] = compute_pillar(
            c.KIND_CONFIGURATION, ConfigurationSource(dict(evaluation_config))
        )
    except LegendError:
        pass
    if candidate is not None:
        observed[c.KIND_AGENT] = candidate
    suggested = (
        {c.KIND_CONFIGURATION: suggested_fingerprint}
        if isinstance(suggested_fingerprint, str)
        else None
    )
    binder.admit_trial(
        trial_id=trial_id, trial_block=block, observed=observed, suggested=suggested
    )


class LegendTrialRefused(LegendError):
    """A Legend refusal before a trial's evaluation; carries the ISSUED backend trial id."""

    def __init__(self, code: str, backend_trial_id: str) -> None:
        super().__init__(code, "trial refused by Legend admission")
        self.backend_trial_id = backend_trial_id


#: Closed contract error codes the Backend may answer with.
CLOSED_CODES = frozenset(
    {
        "invalid_request",
        "unknown_ref",
        "foreign_ref",
        "flag_disabled",
        "conflicting_report",
        "manifest_mismatch",
        "conflicting_idempotency_payload",
        "store_unavailable",
        "store_not_configured",
        "internal_error",
    }
)


def closed_error_code(status: int, body: Any) -> str:
    """A closed code from an error response: the body's own if closed, else the status's."""
    if isinstance(body, Mapping):
        for field_name in ("error_code", "code", "error"):
            value = body.get(field_name)
            if isinstance(value, str) and value in CLOSED_CODES:
                return value
    if status == 400:
        return "invalid_request"
    if status in (401, 403):
        return "foreign_ref"
    if status == 404:
        return "unknown_ref"
    if status == 409:
        return "conflicting_report"
    return "internal_error"
