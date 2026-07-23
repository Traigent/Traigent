"""Fail-closed validation against the exact economics Schema (WI-B).

The closed-pipe design only holds if the client can prove, before transmission,
that a batch conforms to the authoritative contract — including that no key
outside the schema (``additionalProperties: false`` everywhere) rides along — and
that the schema it validated against is the exact accepted contract, not merely a
package that happens to expose the right schema NAMES.

So this emitter:

* REQUIRES the ``traigent-schema`` economics contract and fails closed when it is
  absent, too old to carry the economics schemas, or raises while validating;
* verifies a deterministic CONTENT FINGERPRINT of the economics contract material
  actually used (request + response + referenced definitions + endpoint bindings)
  against the pinned commit, so a name-only impostor or any mutation fails closed
  before transport — the version string is never trusted on its own;
* validates the request AND every ingest response body against the exact schema.

There is no fail-open path and no dependence on a generic strict-validation
toggle. Errors are payload-free (schema digests are public contract hashes, never
request/response content).

Bumping the fingerprint (only when the exact Git pin in ``pyproject.toml`` /
``uv.lock`` changes): install the new pinned ``traigent-schema`` and recompute::

    python -c "from traigent.economics.schema import compute_economics_schema_fingerprint as f; print(f())"

then set ``EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT`` to the printed digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from traigent.economics.errors import (
    EconomicsSchemaUnavailable,
    EconomicsTelemetryContractError,
)

#: Request schema name (file stem).
REQUEST_SCHEMA_NAME = "economics_telemetry_ingest_request_schema"

#: Response schema name per HTTP status. 200 replays (replayed=true); 201 and the
#: all-rejected 422 are initial (replayed=false).
RESPONSE_SCHEMA_BY_STATUS = {
    200: "economics_telemetry_ingest_response_replay_schema",
    201: "economics_telemetry_ingest_response_initial_schema",
    422: "economics_telemetry_ingest_response_initial_schema",
}

# Ordered economics contract material the emitter depends on. The fingerprint is
# computed over exactly this set, so it is meaningful: it covers the request and
# response envelopes, every referenced event/definition schema, and the endpoint
# bindings. Order is fixed so the digest is deterministic.
_FINGERPRINT_FILES = (
    "economics_common_schema",
    "economics_characterization_vocabulary_schema",
    "economics_funnel_event_schema",
    "economics_run_event_schema",
    "economics_receipt_event_schema",
    "economics_telemetry_event_schema",
    "economics_telemetry_ingest_request_schema",
    "economics_telemetry_ingest_response_schema",
    "economics_telemetry_ingest_response_initial_schema",
    "economics_telemetry_ingest_response_replay_schema",
    "economics_endpoints",
)

#: Expected content fingerprint of the accepted economics contract
#: (TraigentSchema 01f3e2a2bbc1ca7d1b1cc8dde94f82d73dbe822a). See module docstring
#: for how to recompute when the exact Git pin changes.
EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT = (
    "fc51000a51e2c29f2742fda1c8ee3e47a3a3467b8cf4498710f7dd4e2fe1cd5e"
)

_UNSET = object()
_BUNDLE_CACHE: Any = _UNSET

# Sentinel: the validator raised while checking a body. Used so the validator
# exception (which can carry the request/response payload) is confined to its
# handler and never chained into the public error.
_VALIDATOR_RAISED = object()


def _economics_schema_dir() -> str:
    try:
        import traigent_schema
    except Exception as exc:  # noqa: BLE001 - any import failure is fail-closed
        raise EconomicsSchemaUnavailable(
            "the economics telemetry contract requires the exact traigent-schema "
            "economics schemas, which are not importable"
        ) from exc
    return os.path.join(
        os.path.dirname(traigent_schema.__file__), "schemas", "economics"
    )


def compute_economics_schema_fingerprint(schema_dir: str | None = None) -> str:
    """Compute the deterministic content fingerprint of the economics contract.

    Reads each pinned schema file, canonicalizes its JSON (sorted keys, compact),
    and folds ``name\\0canonical\\n`` into a single SHA-256. Raises
    :class:`EconomicsSchemaUnavailable` when the material is missing or unreadable.
    """
    directory = schema_dir if schema_dir is not None else _economics_schema_dir()
    digest = hashlib.sha256()
    for stem in _FINGERPRINT_FILES:
        path = os.path.join(directory, stem + ".json")
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            canonical = json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except (OSError, ValueError) as exc:
            raise EconomicsSchemaUnavailable(
                f"the economics contract material is missing or unreadable ({stem})"
            ) from exc
        digest.update(stem.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_bundle() -> Any:
    """Load and verify the exact-contract validator, fail-closed.

    Verifies the content fingerprint BEFORE trusting the validator, so a package
    that merely exposes the right schema names but differs in content is rejected.
    """
    actual = compute_economics_schema_fingerprint()
    if actual != EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT:
        # Digests are public contract hashes, not payload — safe to name.
        raise EconomicsSchemaUnavailable(
            "installed economics contract fingerprint does not match the pinned "
            f"commit (expected {EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT}, "
            f"got {actual})"
        )

    try:
        from traigent_schema.validator import SchemaValidator
    except Exception as exc:  # noqa: BLE001 - any import failure is fail-closed
        raise EconomicsSchemaUnavailable(
            "the economics telemetry contract requires the exact traigent-schema "
            "economics schemas, which are not importable"
        ) from exc

    try:
        validator = SchemaValidator()
    except Exception as exc:  # noqa: BLE001 - construction failure is fail-closed
        raise EconomicsSchemaUnavailable(
            "the traigent-schema validator could not be constructed"
        ) from exc

    available = getattr(validator, "available_schemas", [])
    required = {REQUEST_SCHEMA_NAME, *RESPONSE_SCHEMA_BY_STATUS.values()}
    if not required.issubset(set(available)):
        raise EconomicsSchemaUnavailable(
            "the installed traigent-schema predates the economics telemetry "
            "contract (a required economics schema is absent)"
        )
    return validator


def _get_bundle() -> Any:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not _UNSET:
        return _BUNDLE_CACHE
    validator = _load_bundle()
    _BUNDLE_CACHE = validator
    return validator


def get_request_validator() -> Any:
    """Return the cached, fingerprint-verified validator (fail-closed on load)."""
    return _get_bundle()


def reset_request_validator_cache() -> None:
    """Test hook: drop the cached validator/bundle so discovery is re-evaluated."""
    global _BUNDLE_CACHE
    _BUNDLE_CACHE = _UNSET


def validate_request_or_fail(body: dict[str, Any]) -> None:
    """Validate a request body against the exact economics request schema.

    Raises:
        EconomicsSchemaUnavailable: If the exact schema is absent/old/mismatched,
            or the validator raises (fail closed — no transmission).
        EconomicsTelemetryContractError: If the body is present-and-invalid
            against the schema (e.g. an arbitrary extra key).
    """
    validator = _get_bundle()
    # Confine any validator exception to its handler and RAISE outside it: a
    # jsonschema failure can carry the offending instance value, so it must never
    # ride along as __cause__/__context__.
    try:
        errors = validator.validate_json(body, REQUEST_SCHEMA_NAME)
    except Exception:  # noqa: BLE001 - a validator that throws fails closed
        errors = _VALIDATOR_RAISED
    if errors is _VALIDATOR_RAISED:
        raise EconomicsSchemaUnavailable(
            "the economics schema validator raised while validating the request"
        )
    if errors:
        # Never echo the offending payload: the count and a fixed message only.
        raise EconomicsTelemetryContractError(
            f"economics telemetry request failed schema validation "
            f"({len(errors)} error(s))"
        )


def response_schema_name(http_status: int) -> str:
    """Return the response schema name bound to an ingest-result status."""
    try:
        return RESPONSE_SCHEMA_BY_STATUS[http_status]
    except KeyError as exc:
        raise EconomicsSchemaUnavailable(
            f"no economics response schema is bound to status {http_status}"
        ) from exc


def validate_response_or_fail(body: Any, *, http_status: int) -> None:
    """Validate an ingest response body against the exact per-status schema.

    Fails closed on schema unavailability/mismatch (``EconomicsSchemaUnavailable``)
    and on a validator exception or a schema-invalid body
    (``EconomicsSchemaUnavailable`` / ``EconomicsTelemetryContractError``
    respectively is avoided here: a malformed RESPONSE is the backend's contract
    breach, so it surfaces as the response error the caller expects). The body is
    never logged.
    """
    from traigent.economics.errors import EconomicsResponseError

    validator = _get_bundle()
    schema_name = response_schema_name(http_status)
    # Confine any validator exception to its handler and RAISE outside it: the
    # validator failure can carry the response payload, so it must never ride
    # along as __cause__/__context__.
    try:
        errors = validator.validate_json(body, schema_name)
    except Exception:  # noqa: BLE001 - a validator that throws fails closed
        errors = _VALIDATOR_RAISED
    if errors is _VALIDATOR_RAISED:
        raise EconomicsResponseError(
            "the economics schema validator raised while validating the response"
        )
    if errors:
        raise EconomicsResponseError(
            f"economics telemetry response failed schema validation "
            f"({len(errors)} error(s))"
        )


__all__ = [
    "EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT",
    "REQUEST_SCHEMA_NAME",
    "RESPONSE_SCHEMA_BY_STATUS",
    "compute_economics_schema_fingerprint",
    "get_request_validator",
    "reset_request_validator_cache",
    "response_schema_name",
    "validate_request_or_fail",
    "validate_response_or_fail",
]
