"""Round-trip manifest for :class:`~traigent.api.types.OptimizationResult` (issue #2031).

Both on-disk result formats — ``ConfigStateManager.save/load_optimization_results``
and ``PersistenceManager.save_result``/``load_result`` — reconstructed the
dataclass with a hand-written constructor call that named a subset of its
fields. Every field added to ``OptimizationResult`` since those call sites were
written was therefore silently dropped on load: a reloaded run reported
``total_cost=None``, no ``stop_reason``, no ``warnings``, and ``source
== "backend"`` even for a local run (the #1265 regression, re-created by the
loader).

This module is the single place where "does this field survive a save→load
round trip?" is answered, as an explicit partition of **all** the dataclass's
fields into :data:`RESULT_RESTORE` and :data:`RESULT_RESET`. The partition is
asserted against ``dataclasses.fields(OptimizationResult)`` in
``tests/unit/utils/test_optimization_result_persistence_2031.py``, so a 28th
field fails that test on its first run until its author records a decision here.

Adding a restorable field is a five-line change, and **every one of the five is
test-enforced** — the checklist is not advisory, because forgetting any of it
makes every artifact already on disk unreadable while a fresh save→load test
stays green:

1. put it in :data:`RESULT_RESTORE`
   (``test_manifest_covers_every_optimization_result_field``),
2. give it a sentinel in :data:`_SENTINELS`
   (``test_sentinel_table_covers_every_restored_field``),
3. bump :data:`RESULT_SCHEMA_VERSION`,
4. record the *bumped* version in :data:`FIELD_INTRODUCED_IN` — which is
   **exact-coverage**, so an omitted entry is a test failure rather than an
   implicit "present since version 1"
   (``test_field_introduction_table_covers_every_restored_field_exactly``),
5. freeze a golden artifact for the version you just left behind, next to the
   existing ones in ``tests/unit/utils/test_optimization_result_persistence_2031.py``
   (``test_a_golden_artifact_is_frozen_for_every_readable_schema_version``).

Steps 3-5 are what keep artifacts already on disk loading: the reader accepts
any version up to its own and decodes a key that the artifact's version predates
as the field's declared dataclass default, which is what such a record genuinely
held. Only a version *newer* than this build is refused. The golden artifacts are
committed literals, so they cannot drift with the code that reads them: they fail
if the version is not bumped, and they keep failing until the introduction table
agrees with the bump.

Import note: this module is deliberately **not** re-exported from
``traigent/utils/__init__.py``. That barrel is eager, and ``traigent.api.types``
imports from ``traigent.utils.*``; adding this module to the barrel would create
``types -> utils/__init__ -> optimization_result_persistence -> types``. Both
loaders import it by its full dotted path instead.

Traceability: CONC-Layer-Data CONC-Quality-Reliability FUNC-STORAGE REQ-STOR-007
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from traigent.api.types import (
    ADVISORY_SELECTION_GRADE,
    OptimizationResult,
    OptimizationStatus,
    PresetSelection,
    TrialError,
    TrialResult,
    TrialStatus,
)

logger = logging.getLogger(__name__)

#: Version stamped into every artifact written by a loader-aware writer. An
#: artifact carrying version ``v`` is trusted to hold every
#: :data:`RESULT_RESTORE` field that already existed at version ``v`` — so a
#: missing key is corruption — but says nothing about fields introduced later.
#: Bump this whenever a field is added to :data:`RESULT_RESTORE`, and record the
#: new version in :data:`FIELD_INTRODUCED_IN`. See :func:`decode_result`.
RESULT_SCHEMA_VERSION = 1

#: Key under which :data:`RESULT_SCHEMA_VERSION` is stored.
SCHEMA_VERSION_KEY = "_schema_version"

#: ``optimization_id`` given to a result restored from a pre-#2031 artifact that
#: never persisted the real id. A sentinel rather than a fabricated-looking
#: ``loaded_<name>``: the id is genuinely unavailable, and code that correlates
#: results with backend records must be able to tell the difference.
UNRESTORED_OPTIMIZATION_ID_PREFIX = "unrestored-legacy:"

#: ``source`` for a result whose provenance was never recorded — restored from a
#: pre-#2031 artifact, or encoded from an object that carries no ``source`` at
#: all (see :data:`_UNRESTORED_DEFAULTS`). Deliberately not ``"backend"`` (the
#: dataclass default): claiming backend provenance for a run that may well have
#: been local is the #1265 regression. Consumers branching on ``source`` must
#: treat this third value as "not known", not as the negation of ``"local"``.
UNRESTORED_SOURCE = "unknown"

#: Fields whose declared dataclass default must never stand in for a value the
#: writer did not hold. Every other defaultable field defaults to an *absence*
#: (``None`` / ``{}`` / ``[]``), which is a faithful encoding of "this object
#: carried nothing here"; ``source`` defaults to ``"backend"``, which is a
#: positive provenance claim about a run that may well have been local — the
#: #1265 regression. :func:`_decode_legacy` refuses that substitution when
#: *reading* a pre-#2031 artifact; this table is the same refusal when
#: *writing* one from an object that has no ``source`` at all.
_UNRESTORED_DEFAULTS: dict[str, Any] = {"source": UNRESTORED_SOURCE}

_LEGACY_CONFIG_STATE = "config_state"
_LEGACY_PERSISTENCE = "persistence"
_LEGACY_FORMATS = frozenset({_LEGACY_CONFIG_STATE, _LEGACY_PERSISTENCE})

# Fields that MUST survive a save -> load round trip. Each is durable data about
# what the run did, not a handle into live machine-local state.
RESULT_RESTORE: frozenset[str] = frozenset(
    {
        "trials",  # the run itself; each format keeps its own trial decoder
        "best_config",  # the answer the run produced
        "best_score",  # the answer's score
        "optimization_id",  # durable run identity; correlates with backend records
        "duration",  # measured wall clock
        "convergence_info",  # measured convergence statistics
        "status",  # terminal outcome; UNKNOWN is a real answer, COMPLETED is a lie
        "objectives",  # what was optimized
        "algorithm",  # how it was optimized
        "timestamp",  # when the run completed (NOT when it was saved)
        "metadata",  # verbatim; consumers read function_name/session_summary/... from it
        "preset_selection",  # advisory selection record (#1667)
        "total_cost",  # money spent; silently None-ing it under-reports spend
        "total_tokens",  # tokens spent
        "metrics",  # aggregated run metrics
        "stop_reason",  # why execution stopped
        "reason_code",  # why selection produced no winner
        "experiment_id",  # backend provenance
        "cloud_url",  # backend provenance (a link; never re-validated on load)
        "run_label",  # human-readable run identity
        "experiment_run_id",  # backend provenance for analytics reads
        "warnings",  # money-correctness warnings the user must still see (#1407)
        "warning_codes",  # structured form of the above; clamps success_rate
        "source",  # local vs backend provenance (#1265)
        "best_config_margin",  # winner-vs-runner-up significance (#1866)
    }
)

# Fields that are deliberately NOT restored. Both are live/derived state, not
# facts about the run.
RESULT_RESET: frozenset[str] = frozenset(
    {
        # Store-relative live handle into THIS machine's local session store
        # (#2020). A reloaded result may come from another machine or a store
        # since cleaned, so a restored id would name a record `traigent sync`
        # rejects. Contract: traigent/api/types.py, OptimizationResult docstring
        # ("sync_session_id:"). Pinned by
        # tests/unit/core/test_config_state_manager_sync_session_id_2020.py and
        # tests/unit/utils/test_persistence.py::
        # test_load_result_does_not_restore_sync_session_id.
        "sync_session_id",
        # init=False memoization cache for the `experiment_stats` property;
        # recomputed on demand from `trials`.
        "_experiment_stats",
    }
)

_FIELDS: dict[str, dataclasses.Field[Any]] = {
    field.name: field for field in dataclasses.fields(OptimizationResult)
}

# Restored by the caller's own format-specific trial decoder, so the generic
# field loop never reads or writes it.
_CALLER_SUPPLIED = "trials"


# ---------------------------------------------------------------------------
# TrialResult manifest (#2047) — the nested twin of the result manifest above.
# ---------------------------------------------------------------------------

#: Every :class:`TrialResult` field that MUST survive a save -> load round trip.
#:
#: Both persisted formats reconstructed trials from a hand-written 8-field
#: constructor call, so ``error`` and ``score`` were dropped on load: a crashed
#: trial came back indistinguishable from one that merely scored badly, and any
#: failure-rate or error-clustering analysis over reloaded results silently
#: under-reported. The ``config_state`` writer had been putting both on disk all
#: along (it dumps via ``asdict``); only the decoder threw them away.
TRIAL_RESTORE: frozenset[str] = frozenset(
    {
        "trial_id",  # durable trial identity; winning_trial_ids (#1854) matches on it
        "config",  # the configuration this trial actually ran
        "metrics",  # measured per-trial metrics
        "status",  # terminal outcome; FAILED must not decay to COMPLETED
        "duration",  # measured wall clock
        "timestamp",  # when the trial ran
        "error_message",  # human-readable failure text
        "metadata",  # verbatim; consumers read successful_examples/... from it
        "error",  # structured diagnosis: the difference between "bad" and "crashed"
        "score",  # the optimization signal best_config argmaxes (#1845)
    }
)

#: Fields deliberately NOT restored. Empty — and the emptiness is the claim, not
#: an oversight: unlike :class:`OptimizationResult` (which carries
#: ``sync_session_id``, a handle into *this* machine's store, and a memoization
#: cache), every :class:`TrialResult` field is a durable fact about a trial that
#: already ran. Pinned by ``test_trial_manifest_covers_every_field``: a field
#: added to the dataclass must be classified into one set or the other.
TRIAL_RESET: frozenset[str] = frozenset()

_TRIAL_FIELDS: dict[str, dataclasses.Field[Any]] = {
    field.name: field for field in dataclasses.fields(TrialResult)
}

#: Reference value for every :data:`TRIAL_RESTORE` field: a non-default,
#: round-trippable example. This is the manifest's proof-of-coverage table, and
#: it is what makes the completeness guard real rather than clerical -- without
#: it a new field could be added to :data:`TRIAL_RESTORE` and still be silently
#: dropped by both formats while every test stayed green, because a hand-written
#: fixture compares a new field's default against itself. Mirrors the role
#: :data:`_SENTINELS` plays for the result manifest.
_TRIAL_SENTINELS: dict[str, Any] = {
    "trial_id": "trial-manifest-sentinel",
    "config": {"model": "smart", "temperature": 0.7},
    "metrics": {"accuracy": 0.61, "score": 0.61},
    "status": TrialStatus.FAILED,
    "duration": 3.25,
    "timestamp": datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC),
    "error_message": "provider refused the request",
    "metadata": {"replicate": 2},
    "error": TrialError(
        message="provider refused the request",
        error_type="RuntimeError",
        traceback="Traceback (most recent call last):\n  RuntimeError: refused",
        timestamp=datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC),
        config={"model": "smart"},
    ),
    "score": 0.61,
    # A field added to TRIAL_RESTORE without an entry here fails
    # test_trial_sentinel_table_covers_every_restored_field.
}


def decode_trial_error(raw: Any) -> TrialError | None:
    """Rebuild a :class:`TrialError` from whatever a persisted artifact holds.

    Delegates to :meth:`TrialError.from_dict`, which is the codebase's already
    decided answer for this object rather than a second, stricter parallel one.
    An earlier revision of this function hand-rolled its own strict decoder that
    raised on any payload it did not recognise; that was wrong twice over. It
    rejected artifacts already on disk -- ``asdict`` does not recurse into a
    duck-typed ``trial.error``, so the legacy ``json.dump(default=str)`` writer
    stringified it, and a whole otherwise-good run then failed to load. And it
    applied a write-time contract retroactively to files written before that
    contract existed.

    ``None`` (absent key, or a trial that did not fail) stays ``None``. A
    non-mapping payload is a legacy artifact rather than a decodable error, and
    is reported so the loss is visible in the log rather than silent.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        logger.warning(
            "Trial 'error' payload is %s, not a mapping -- this artifact predates "
            "structured error persistence. Restoring the trial without its "
            "structured diagnosis; status and error_message are unaffected.",
            type(raw).__name__,
        )
        return None
    return TrialError.from_dict(raw)


def decode_trial_score(raw: Any) -> float | None:
    """Rebuild a trial's optimization signal.

    Absent or ``None`` is a real answer -- a crashed trial has no objective
    value -- so it is preserved rather than coerced to ``0.0``, which would
    enter selection as a legitimate losing score.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        # Our own writer produces this: `_safe_json_value` has no branch for
        # Decimal (or any other non-primitive number), so it falls through to
        # `str(value)`. Refusing the string here would discard a score this
        # code path itself wrote.
        try:
            return float(raw)
        except ValueError:
            pass
    logger.warning(
        "Trial 'score' payload is %s, not a number; restoring it as None "
        "rather than guessing a value that would enter selection.",
        type(raw).__name__,
    )
    return None


#: The schema version in which each persisted field first appeared — **every**
#: one of them, explicitly, including the version-1 fields.
#:
#: This is what makes the format evolvable. Without it the strict "a versioned
#: artifact promises every restorable field" rule would reject every artifact
#: already on disk the moment a field was added: :func:`_decode_versioned` uses
#: this table to decode a key the artifact's own version predates as the field's
#: **declared dataclass default**, which is not a fabricated value but exactly
#: what a record written before the field existed genuinely held.
#:
#: The version-1 entries are spelled out rather than left implicit, and the table
#: is asserted to cover the restorable fields **exactly** (by
#: ``test_field_introduction_table_covers_every_restored_field_exactly``). A
#: subset assertion with an implicit "absent means 1" default is not a guard at
#: all: the one mistake it has to catch — adding a field and forgetting its entry
#: — is precisely the case that silently reads as "present since version 1", and
#: then every v1 artifact on disk is reported as *corrupt* for missing a key it
#: could not possibly have carried. Requiring the entry means that mistake fails
#: a test instead of a user's reload. See :func:`_introduced_in`.
FIELD_INTRODUCED_IN: dict[str, int] = {
    "algorithm": 1,
    "best_config": 1,
    "best_config_margin": 1,
    "best_score": 1,
    "cloud_url": 1,
    "convergence_info": 1,
    "duration": 1,
    "experiment_id": 1,
    "experiment_run_id": 1,
    "metadata": 1,
    "metrics": 1,
    "objectives": 1,
    "optimization_id": 1,
    "preset_selection": 1,
    "reason_code": 1,
    "run_label": 1,
    "source": 1,
    "status": 1,
    "stop_reason": 1,
    "timestamp": 1,
    "total_cost": 1,
    "total_tokens": 1,
    "warning_codes": 1,
    "warnings": 1,
    # The next field added to RESULT_RESTORE goes here at the *bumped*
    # RESULT_SCHEMA_VERSION, e.g. "billing_currency": 2 — never at 1, which
    # would claim every artifact ever written already carried it.
}

#: Reference value for every :data:`RESULT_RESTORE` field: a non-default,
#: round-trippable example. This is the manifest's proof-of-coverage table — the
#: #2031 suites build a fully-populated result from it and assert every field
#: comes back — and it doubles as documentation of each field's on-the-wire
#: shape. A field added to ``RESULT_RESTORE`` without an entry here fails
#: ``test_sentinel_table_covers_every_restored_field``.
_SENTINELS: dict[str, Any] = {
    "trials": [
        TrialResult(
            trial_id="trial-sentinel-0",
            config={"model": "cheap"},
            metrics={"accuracy": 0.75},
            status=TrialStatus.COMPLETED,
            duration=1.5,
            timestamp=datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC),
            metadata={"replicate": 1},
        )
    ],
    "best_config": {"model": "cheap", "temperature": 0.0},
    "best_score": 0.91,
    "optimization_id": "opt-2031-sentinel",
    "duration": 45.25,
    "convergence_info": {"converged": True, "iterations": 7},
    "status": OptimizationStatus.CANCELLED,
    "objectives": ["accuracy", "cost"],
    "algorithm": "bayesian",
    "timestamp": datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC),
    "metadata": {
        "function_name": "answer_question",
        "configuration_space": {"model": ["cheap", "smart"]},
        "session_summary": {"winning_trial_ids": ["trial-sentinel-0"]},
        "source": "local",
    },
    "preset_selection": PresetSelection(
        preset_name="balanced",
        params={"temperature": 0.0},
        selection_grade=ADVISORY_SELECTION_GRADE,
        # Deliberately longer than the 280-char clamp in
        # PresetSelection.to_metadata(): the encoder must use `asdict`, not
        # `to_dict`, so a long rationale survives verbatim.
        selection_rationale="balanced beat the alternatives on accuracy. " * 10,
        status="selected",
        selected_config={"model": "cheap"},
        selected_configs=[{"model": "cheap"}, {"model": "smart"}],
        selected_trial_indices=[0],
    ),
    "total_cost": 0.0234,
    "total_tokens": 15420,
    "metrics": {"accuracy": 0.91, "cost": 0.0234},
    "stop_reason": "max_trials_reached",
    "reason_code": "OBJECTIVE_UNMATCHED",
    "experiment_id": "exp-2031",
    "cloud_url": "https://portal.example.invalid/experiments/exp-2031",
    "run_label": "answer_question_20260315_143022_a3f1b2",
    "experiment_run_id": "run-2031",
    "warnings": ["Model 'mystery-1' priced at $0 — spend is under-reported."],
    "warning_codes": ["OBJECTIVE_UNMATCHED"],
    "source": "local",
    "best_config_margin": {"verdict": "statistical_tie", "delta": 0.01},
}


def _default_for(name: str) -> Any:
    """Return the dataclass default for ``name``, or ``MISSING`` when required.

    ``default_factory`` is invoked per call, so a defaulted collection is always
    a fresh instance and can never be shared between two loaded results.
    """
    field = _FIELDS[name]
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    if field.default is not dataclasses.MISSING:
        return field.default
    return dataclasses.MISSING


#: The restorable fields whose persisted form is not their in-memory form, so
#: both halves of the round trip handle them explicitly (:func:`_encode_value` /
#: :func:`_decode_value`); every other field is stored as it stands. Named
#: rather than left implicit in the ``if`` chains because these are exactly the
#: fields a *decoder* can refuse, and therefore exactly the fields any writer
#: that stamps :data:`SCHEMA_VERSION_KEY` must run through the encoder first —
#: see :func:`encode_whole_result_dump`. Pinned by
#: ``test_explicitly_encoded_names_the_fields_whose_form_actually_changes``.
_EXPLICITLY_ENCODED: tuple[str, ...] = ("preset_selection", "status", "timestamp")


def _encode_timestamp(value: Any) -> str:
    """Encode a run timestamp, refusing any value the reader could not decode.

    Deliberately defined in terms of :func:`_decode_timestamp` (below) rather
    than repeating its accepted forms: the encoder's job is to write only what
    the reader can read back, and a second hand-written notion of "timestamp-ish"
    here is exactly how the two halves drift apart again.

    The earlier version passed a non-``datetime`` through unchanged, so
    ``timestamp=None`` or ``"not-a-date"`` produced an artifact ``save_result``
    wrote happily and ``load_result`` then refused — the failure surfaced at read
    time, on data already on disk, arbitrarily far from the caller that caused
    it. Failing the *save* is what the invalid-``status`` branch above already
    does, for the same reason.

    A string that the reader accepts is still accepted here, so duck-typed
    callers holding an already-serialized timestamp keep working; it is
    normalized to ``isoformat()`` on the way out, which loses nothing (both
    forms decode to the same instant) and makes the on-disk format one shape.
    """
    try:
        return _decode_timestamp(value).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"Cannot persist an optimization result whose 'timestamp' is "
            f"{value!r}: a persisted timestamp must be a datetime or an "
            f"ISO-8601 string, and writing this value would produce an "
            f"artifact that load_result cannot read back ({exc})."
        ) from exc


def _encode_value(name: str, value: Any) -> Any:
    """Convert one field value to a JSON-safe form.

    Every branch here must be at least as strict as its :func:`_decode_value`
    counterpart. A value the encoder accepts and the decoder rejects is not
    caught at all until the artifact is read back, by which point the bad value
    is on disk and the caller that supplied it is long gone.
    """
    if name == "status":
        return OptimizationStatus(value).value
    if name == "timestamp":
        return _encode_timestamp(value)
    if name == "preset_selection":
        # `asdict`, not `to_dict`: to_dict() routes through to_metadata(), which
        # truncates selection_rationale to 280 chars and drops it entirely when
        # falsy. A persistence encoder must not lose data the reader can see.
        return dataclasses.asdict(value) if value is not None else None
    return value


def _decode_status(value: Any) -> OptimizationStatus:
    """Decode a persisted status, falling back to UNKNOWN (#1302 AC3)."""
    if isinstance(value, OptimizationStatus):
        return value
    try:
        return OptimizationStatus(value)
    except ValueError:
        logger.warning(
            "Unrecognized OptimizationStatus %r in a persisted result; "
            "restoring it as UNKNOWN rather than asserting success or failure.",
            value,
        )
        return OptimizationStatus.UNKNOWN


def _decode_timestamp(value: Any) -> datetime:
    """Decode a persisted timestamp written by ``isoformat`` or ``str``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Covers both writers: PersistenceManager stores `isoformat()`;
        # ConfigStateManager dumps with `default=str` (space separator).
        return datetime.fromisoformat(value)
    raise ValueError(f"Unreadable persisted timestamp: {value!r}")


def _decode_value(name: str, value: Any) -> Any:
    """Decode one persisted field value back to its dataclass type."""
    if name == "status":
        return _decode_status(value)
    if name == "timestamp":
        return _decode_timestamp(value)
    if name == "preset_selection":
        return PresetSelection.from_dict(value)
    return value


def _read_field(result: Any, name: str) -> Any:
    """Read one restorable field off ``result``, defaulting an absent attribute.

    ``save_result`` historically read ~10 attributes, so callers have long been
    able to hand it any object shaped like a result. Reading all 24 encoded
    fields by plain attribute access would silently narrow that contract to
    ``OptimizationResult`` alone, breaking duck-typed callers with a bare
    ``AttributeError`` (the regression pinned by
    ``test_encodes_a_duck_typed_partial_result_using_declared_defaults``).

    The fallback invents nothing, and cannot mask a genuinely missing field:

    * ``OptimizationResult`` is a dataclass, so a real instance always carries
      all 27 attributes. This branch is therefore only ever reachable for an
      object that is *not* an ``OptimizationResult``.
    * The value used is the field's **own** declared default, read from
      ``dataclasses.fields(OptimizationResult)`` via :func:`_default_for` — i.e.
      exactly what constructing the result without that argument would have
      produced. There is deliberately no second hand-written default table here
      that could drift away from the dataclass.

    The one exception is :data:`_UNRESTORED_DEFAULTS`: a declared default that is
    a positive *claim* rather than an absence must not be written into a
    versioned artifact on behalf of an object that never made it. Unlike the
    legacy path, where a defaulted field is announced by
    :func:`_decode_legacy`'s warning, an encoded value is indistinguishable from
    a recorded one once the artifact is on disk — so the honest sentinel has to
    be chosen at write time or not at all.

    A field with neither ``default`` nor ``default_factory`` is required, so
    there is no declared value to fall back to and none is fabricated: that is a
    real contract violation and still raises, but as a ``TypeError`` naming both
    the field and the offending type rather than a bare ``AttributeError``.
    """
    value = getattr(result, name, dataclasses.MISSING)
    if value is not dataclasses.MISSING:
        return value
    if name in _UNRESTORED_DEFAULTS:
        unrestored = _UNRESTORED_DEFAULTS[name]
        logger.warning(
            "Persisting a %s as an optimization result: it has no '%s' "
            "attribute, and OptimizationResult's declared default (%r) is a "
            "claim rather than an absence, so %r is recorded instead.",
            type(result).__name__,
            name,
            _default_for(name),
            unrestored,
        )
        return unrestored
    default = _default_for(name)
    if default is dataclasses.MISSING:
        raise TypeError(
            f"Cannot persist a {type(result).__name__} as an optimization "
            f"result: it has no '{name}' attribute and OptimizationResult "
            f"declares no default for that field, so there is no value to "
            f"record."
        )
    return default


def encode_result_fields(result: OptimizationResult) -> dict[str, Any]:
    """Return every restorable field of ``result`` except ``trials``.

    Used by ``PersistenceManager.save_result``, whose metadata.json is a curated
    dict rather than a whole-dataclass dump; trials are stored in their own
    file. Values are JSON-ready apart from free-form containers (``metadata``,
    ``metrics``, ``convergence_info``, ...), which the caller passes through its
    own JSON sanitizer.

    The returned mapping carries its own :data:`SCHEMA_VERSION_KEY` so it can be
    handed straight back to :func:`decode_result` without the caller having to
    re-declare the version of the envelope it was nested in.

    Fields in :data:`RESULT_RESET` are never written. Fields absent from a
    duck-typed ``result`` fall back to their declared dataclass default — see
    :func:`_read_field` for why that is a faithful encoding rather than a
    fabricated one — except for :data:`_UNRESTORED_DEFAULTS`, where the declared
    default is a claim and the honest sentinel is written instead: an absent
    ``source`` is recorded as :data:`UNRESTORED_SOURCE`, never as ``"backend"``.

    Fidelity is per *field*, and the free-form containers round-trip as their
    JSON-native equivalents rather than as their exact Python types: the on-disk
    format is JSON, so a tuple nested in ``metadata`` comes back as a list, a
    nested enum as its value, and a nested dataclass as a dict. The three fields
    with a declared non-JSON type (``status``, ``timestamp``,
    ``preset_selection``) are encoded and decoded explicitly and do keep their
    types.

    Raises:
        ValueError: If ``status`` or ``timestamp`` holds a value
            :func:`decode_result` could not read back. Refusing the *write* is
            the point: the alternative is a well-formed-looking artifact that
            fails on load, long after the caller that produced it.
        TypeError: If a field required by ``OptimizationResult`` is absent from
            a duck-typed ``result`` (see :func:`_read_field`), or if
            ``preset_selection`` is neither ``None`` nor a dataclass instance.
    """
    encoded: dict[str, Any] = {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION}
    encoded.update(
        {
            name: _encode_value(name, _read_field(result, name))
            for name in sorted(RESULT_RESTORE - {_CALLER_SUPPLIED})
        }
    )
    return encoded


def encode_whole_result_dump(result: OptimizationResult) -> dict[str, Any]:
    """Return a stamped, validated whole-dataclass dump of ``result``.

    ``ConfigStateManager.save_optimization_results`` persists the entire
    dataclass rather than the curated subset :func:`encode_result_fields` builds
    for ``PersistenceManager``, so it cannot use that function — but it stamps
    the same :data:`SCHEMA_VERSION_KEY`, which is a promise to *its own* reader
    that every restorable field is present and readable. A writer that stamps
    the version without running the encoder can persist an artifact its own
    loader then refuses: ``asdict`` + ``json.dump(default=str)`` happily writes
    ``timestamp: None`` or ``status: "bogus"``, and the failure surfaces at read
    time, on data already on disk, arbitrarily far from the caller that caused
    it. That is the asymmetry :func:`_encode_timestamp` exists to close, and it
    has to close on both writers or it is not closed.

    Only the fields whose persisted form differs from their in-memory one are
    re-encoded (:data:`_EXPLICITLY_ENCODED`) — those are exactly the fields
    :func:`_decode_value` treats specially, so they are exactly the ones a
    decoder can refuse. Everything else keeps its ``asdict`` form, which
    recurses into nested dataclasses and is what this format has always written.

    Fields in :data:`RESULT_RESET` are left in the dump (the loader drops them
    on read — see ``decode_result``) rather than stripped here, so the on-disk
    shape of this format does not change.

    Raises:
        ValueError: If ``status`` or ``timestamp`` holds a value
            :func:`decode_result` could not read back.
        TypeError: If ``preset_selection`` is neither ``None`` nor a dataclass.
    """
    dump = dataclasses.asdict(result)
    dump[SCHEMA_VERSION_KEY] = RESULT_SCHEMA_VERSION
    for name in _EXPLICITLY_ENCODED:
        # Read from `result`, not from `dump`: `asdict` has already flattened
        # nested dataclasses, so `preset_selection` would arrive here as a dict
        # that `_encode_value` (rightly) refuses.
        dump[name] = _encode_value(name, _read_field(result, name))
    return dump


def _legacy_view(
    data: dict[str, Any], legacy_format: str, artifact_name: str | None
) -> tuple[dict[str, Any], list[str]]:
    """Map a pre-#2031 artifact onto dataclass field names.

    Returns the mapped view plus human-readable notes about values that were
    degraded rather than restored, for the single warning emitted by
    :func:`_decode_legacy`.
    """
    if legacy_format == _LEGACY_CONFIG_STATE:
        # ConfigStateManager dumps `asdict(result)`, so its keys are already
        # field names; only fields added after a given artifact was written are
        # absent, and those take dataclass defaults below.
        return dict(data), []

    label = artifact_name or "<unnamed>"
    view: dict[str, Any] = {
        key: data[key]
        for key in (
            "best_config",
            "best_score",
            "duration",
            "convergence_info",
            "objectives",
            "algorithm",
            "preset_selection",
        )
        if key in data
    }
    notes: list[str] = []

    if "created_at" in data:
        # The legacy format never stored the result's own `timestamp`; the only
        # time it recorded is save time. Documented as such rather than silently
        # presented as the run-completion time.
        view["timestamp"] = data["created_at"]
        notes.append("timestamp <- created_at (artifact save time, not run completion)")

    # The curated legacy metadata subset. CLI apply/auto-load paths read
    # function_name / configuration_space, and #1854 needs session_summary for
    # id-matched best_metrics.
    legacy_metadata = {
        key: data[key]
        for key in (
            "function_name",
            "configuration_space",
            "session_summary",
        )
        if data.get(key) is not None
    }
    if legacy_metadata:
        view["metadata"] = legacy_metadata

    view["optimization_id"] = f"{UNRESTORED_OPTIMIZATION_ID_PREFIX}{label}"
    notes.append(f"optimization_id -> {view['optimization_id']} (never persisted)")
    view["status"] = OptimizationStatus.UNKNOWN
    notes.append("status -> unknown (never persisted; not assumed completed)")
    return view, notes


def _decode_legacy(
    data: dict[str, Any],
    trials: list[TrialResult],
    legacy_format: str,
    artifact_name: str | None,
    label: str,
) -> OptimizationResult:
    """Restore an artifact written before the schema version existed."""
    view, notes = _legacy_view(data, legacy_format, artifact_name)

    if "source" not in view:
        # The one field whose dataclass default is NOT a safe fallback: an
        # artifact that never recorded provenance may well be a local run, and
        # silently calling it "backend" is the #1265 regression. Applies to
        # both legacy formats.
        view["source"] = UNRESTORED_SOURCE
        notes.append("source -> unknown (never persisted; not assumed backend)")

    kwargs: dict[str, Any] = {}
    defaulted: list[str] = []
    for name in sorted(RESULT_RESTORE - {_CALLER_SUPPLIED}):
        if name in view:
            kwargs[name] = _decode_value(name, view[name])
            continue
        default = _default_for(name)
        if default is dataclasses.MISSING:
            raise ValueError(
                f"Corrupted optimization result artifact {label}: "
                f"required field '{name}' is missing and has no default"
            )
        kwargs[name] = default
        defaulted.append(name)

    if defaulted or notes:
        # Signal via the log only. Injecting entries into `warnings` /
        # `warning_codes` / `metadata` would corrupt the user-facing payload
        # with a loader-internal concern and break the verbatim round trip
        # those fields are pinned to (#2026).
        logger.warning(
            "Restored a pre-#2031 %s artifact (%s) with reduced fidelity: "
            "fields not present in the artifact and left at their defaults: %s. %s",
            legacy_format,
            label,
            ", ".join(defaulted) or "(none)",
            " ".join(notes) or "",
        )

    return OptimizationResult(trials=trials, **kwargs)


def _introduced_in(name: str) -> int:
    """Return the schema version that first declared ``name``.

    Deliberately not ``FIELD_INTRODUCED_IN.get(name, 1)``. An implicit default
    turns "somebody added a field and forgot its entry" — the one authoring
    mistake this table exists to catch — into the claim that every artifact ever
    written already carried the key, so every one of them is then reported as
    corrupt. A missing entry is a defect in *this build*, not in the file being
    read, and says so.
    """
    try:
        return FIELD_INTRODUCED_IN[name]
    except KeyError:
        raise RuntimeError(
            f"traigent/utils/optimization_result_persistence.py is inconsistent: "
            f"'{name}' is in RESULT_RESTORE but has no FIELD_INTRODUCED_IN entry, "
            f"so this build cannot tell whether an artifact that lacks the key is "
            f"damaged or merely predates the field. Add "
            f"FIELD_INTRODUCED_IN['{name}'] = <the RESULT_SCHEMA_VERSION that "
            f"introduced it>."
        ) from None


def _missing_declared_field_error(
    label: str, name: str, version: int, introduced: int
) -> ValueError:
    """Build the error for a versioned artifact missing a field it declared.

    Two things can put us here, and the message must not pick one of them:

    * the file really is damaged or truncated, or
    * this build added ``name`` to :data:`RESULT_RESTORE` and recorded it in
      :data:`FIELD_INTRODUCED_IN` at a version the artifact already declares —
      typically by forgetting to bump :data:`RESULT_SCHEMA_VERSION` and writing
      ``1``. Then *no* artifact written before this build carries the key, and
      calling each of them damaged sends their owner hunting for file corruption
      that does not exist.

    The second cause is the more likely one whenever the field was introduced at
    a version this build can also write, so it is named first there.
    """
    reader_added_it = introduced >= RESULT_SCHEMA_VERSION
    causes = [
        f"this build of the SDK declares that field '{name}' already existed at "
        f"schema version {introduced}. If '{name}' was in fact added to "
        f"RESULT_RESTORE recently, that declaration is wrong — bump "
        f"RESULT_SCHEMA_VERSION and set FIELD_INTRODUCED_IN['{name}'] to the "
        f"bumped version — and no artifact written before this build carries the "
        f"key, so they are all readable again once it is corrected",
        "the artifact is damaged or was truncated",
    ]
    if not reader_added_it:
        causes.reverse()
    return ValueError(
        f"Cannot read optimization result artifact {label}: it declares schema "
        f"version {version}, which this build says already included field "
        f"'{name}', but that key is missing. Either {causes[0]}; or {causes[1]}. "
        f"(This reader writes schema version {RESULT_SCHEMA_VERSION}.)"
    )


def _decode_versioned(
    data: dict[str, Any], trials: list[TrialResult], label: str, version: int
) -> OptimizationResult:
    """Restore an artifact that declares schema ``version``.

    ``version`` is at most :data:`RESULT_SCHEMA_VERSION` (a newer one is
    rejected by :func:`decode_result`), so every key this reader knows about is
    in one of two states:

    * declared by the artifact's own version — it promised the key, so an absent
      one is corruption. No blanket ``.get(name, default)``: that is exactly the
      silent-drop defect #2031 fixes, re-armed for the next field somebody adds.
    * introduced *after* the artifact was written — the writer could not have
      recorded a field that did not exist, so the field's declared dataclass
      default is what the record genuinely held. Restoring it is faithful, not
      fabricated, and it is what keeps already-persisted results readable across
      a schema bump.
    """
    kwargs: dict[str, Any] = {}
    predating: list[str] = []
    for name in sorted(RESULT_RESTORE - {_CALLER_SUPPLIED}):
        if name in data:
            kwargs[name] = _decode_value(name, data[name])
            continue

        introduced = _introduced_in(name)
        if introduced <= version:
            raise _missing_declared_field_error(label, name, version, introduced)

        default = _default_for(name)
        if default is dataclasses.MISSING:
            # Unreachable for a well-formed manifest: a field added to a
            # dataclass after defaulted ones must itself carry a default. Kept
            # so the failure names the real cause instead of surfacing as a
            # TypeError from the constructor.
            raise ValueError(
                f"Cannot read optimization result artifact {label} (schema "
                f"version {version}): field '{name}' was introduced in schema "
                f"version {introduced} and the artifact predates it, but the "
                f"field declares no default, so there is no value it can have "
                f"held."
            )
        kwargs[name] = default
        predating.append(name)

    if predating:
        # Deliberately not a warning: nothing was lost. The artifact is
        # complete for the version it declares, and each field below is
        # restored to the value the dataclass itself would have produced when
        # the record was written.
        logger.info(
            "Optimization result artifact %s declares schema version %d; this "
            "build reads version %d. Fields introduced after version %d are "
            "restored at their declared defaults: %s.",
            label,
            version,
            RESULT_SCHEMA_VERSION,
            version,
            ", ".join(predating),
        )

    return OptimizationResult(trials=trials, **kwargs)


def verify_envelope_version(
    envelope: dict[str, Any], payload: Any, *, artifact_name: str | None = None
) -> None:
    """Check that a nested artifact declares its schema version exactly once.

    ``PersistenceManager``'s ``metadata.json`` stamps the schema version twice:
    on the envelope (so a reader can classify the file without descending into
    it) and on the ``result_fields`` payload (so the payload can be handed
    straight to :func:`decode_result`). Only the payload's copy is decoded, so
    without this check the envelope's copy is decorative — and a file carrying
    an envelope version this build cannot read could still be decoded, at
    whatever version its payload happened to claim, with everything the higher
    version encodes silently dropped. That is exactly the outcome the
    future-version refusal exists to prevent, reachable with a structurally
    valid file.

    The two stamps are written together, so anything other than "both present
    and equal" or "neither present" (a pre-#2031 artifact) means the file was
    hand-edited or truncated and is not trustworthy.

    Args:
        envelope: The outer mapping.
        payload: The nested field mapping, or whatever was found in its place.
        artifact_name: Name used in the error message.

    Raises:
        ValueError: If exactly one of the two stamps is present, or if both are
            present and they disagree.
    """
    label = f"'{artifact_name}'" if artifact_name else "<unnamed>"
    outer_declared = SCHEMA_VERSION_KEY in envelope
    inner_declared = isinstance(payload, dict) and SCHEMA_VERSION_KEY in payload

    if not outer_declared and not inner_declared:
        return  # a pre-#2031 artifact: neither half is versioned

    if outer_declared != inner_declared:
        stamped, missing = (
            ("envelope", "'result_fields' payload")
            if outer_declared
            else ("'result_fields' payload", "envelope")
        )
        raise ValueError(
            f"Corrupted optimization result artifact {label}: its {stamped} "
            f"declares '{SCHEMA_VERSION_KEY}' but its {missing} does not. A "
            f"loader-written artifact stamps both, so this file was hand-edited "
            f"or truncated and its declared version cannot be trusted."
        )

    outer = envelope[SCHEMA_VERSION_KEY]
    inner = payload[SCHEMA_VERSION_KEY]
    if outer != inner:
        raise ValueError(
            f"Corrupted optimization result artifact {label}: it declares "
            f"'{SCHEMA_VERSION_KEY}' {outer!r} on the envelope but {inner!r} on "
            f"its 'result_fields' payload. Both are written together and must "
            f"agree; decoding the payload at its own version would ignore the "
            f"envelope's claim and silently drop whatever the other version "
            f"encodes. (This build reads schema version "
            f"{RESULT_SCHEMA_VERSION}.)"
        )


def decode_result(
    data: dict[str, Any],
    *,
    trials: list[TrialResult],
    legacy_format: str | None = None,
    artifact_name: str | None = None,
) -> OptimizationResult:
    """Reconstruct an ``OptimizationResult`` from a persisted field mapping.

    Args:
        data: The persisted mapping. Either a versioned artifact (carrying
            :data:`SCHEMA_VERSION_KEY`) or a pre-#2031 one.
        trials: Decoded trials. Always supplied by the caller — each format
            keeps its own trial decoder.
        legacy_format: ``"config_state"`` or ``"persistence"``; required to read
            an artifact with no schema version, ignored when one is present.
        artifact_name: Name used in error messages, log lines, and the
            :data:`UNRESTORED_OPTIMIZATION_ID_PREFIX` sentinel.

    Returns:
        The restored result. Every :data:`RESULT_RESTORE` field is taken from
        ``data``; every :data:`RESULT_RESET` field is left at its default even
        when ``data`` supplies a value.

    Schema versions are read forward-compatibly in one direction only. Any
    version *at or below* :data:`RESULT_SCHEMA_VERSION` is readable: keys the
    artifact's version predates take their declared dataclass default (see
    :data:`FIELD_INTRODUCED_IN`), so records already on disk keep loading after
    a field is added. A version *above* it is refused, because such an artifact
    may encode semantics this reader cannot honour and would silently drop.

    Raises:
        ValueError: If ``data`` is not a mapping, declares a schema version this
            build cannot read (including a present-but-``null`` one, which is an
            invalid declaration rather than an absent one), is versioned but
            missing a field its own version declared, or is unversioned with no
            (or an unrecognized) ``legacy_format``.
        RuntimeError: If this build's :data:`FIELD_INTRODUCED_IN` does not cover
            every restorable field — an SDK defect, not an artifact one.
    """
    label = f"'{artifact_name}'" if artifact_name else "<unnamed>"
    if not isinstance(data, dict):
        raise ValueError(
            f"Corrupted optimization result artifact {label}: "
            f"expected a JSON object, got {type(data).__name__}"
        )

    # Presence, not truthiness or non-None-ness: an explicit `"_schema_version":
    # null` is a *declared* version that is invalid, not an absent one. Treating
    # it as unversioned would route a damaged versioned artifact down the legacy
    # path, where every missing field is expected and silently defaulted — the
    # blanket `.get(name, default)` this module exists to remove, reachable by
    # hand-editing one value.
    if SCHEMA_VERSION_KEY in data:
        version = data[SCHEMA_VERSION_KEY]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(
                f"Corrupted optimization result artifact {label}: "
                f"'{SCHEMA_VERSION_KEY}' is {version!r}, which is not a schema "
                f"version this format ever used (versions are integers from 1; "
                f"this build writes {RESULT_SCHEMA_VERSION})."
            )
        if version > RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"Optimization result artifact {label} was written by schema "
                f"version {version}; this build of the SDK understands version "
                f"{RESULT_SCHEMA_VERSION}. The artifact is not corrupt, it is "
                f"newer than this reader and may record fields and semantics "
                f"this build would silently drop. Upgrade the SDK to read it."
            )
        return _decode_versioned(data, trials, label, version)

    if legacy_format not in _LEGACY_FORMATS:
        raise ValueError(
            f"Optimization result artifact {label} carries no "
            f"'{SCHEMA_VERSION_KEY}' and no legacy format was named "
            f"(expected one of {sorted(_LEGACY_FORMATS)})"
        )
    return _decode_legacy(data, trials, legacy_format, artifact_name, label)
