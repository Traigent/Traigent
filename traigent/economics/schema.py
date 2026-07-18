"""Fail-closed request validation against the exact economics Schema (WI-B).

The closed-pipe design only holds if the client can prove, before transmission,
that a batch conforms to the authoritative contract — including that no key
outside the schema (``additionalProperties: false`` everywhere) rides along. So
this emitter REQUIRES the exact ``traigent-schema`` economics contract and fails
closed when it is absent, too old to carry the economics schemas, raises while
validating, or reports the payload invalid. There is no fail-open path and no
dependence on a generic strict-validation toggle: an unvalidated economics batch
is never transmitted.
"""

from __future__ import annotations

from typing import Any

from traigent.economics.errors import (
    EconomicsSchemaUnavailable,
    EconomicsTelemetryContractError,
)

#: Schema name (file stem) the request contract is published under.
REQUEST_SCHEMA_NAME = "economics_telemetry_ingest_request_schema"

_UNSET = object()
_VALIDATOR_CACHE: Any = _UNSET


def _load_validator() -> Any:
    """Load a SchemaValidator that carries the economics request schema.

    Raises:
        EconomicsSchemaUnavailable: If ``traigent-schema`` is not importable or
            predates the economics contract (schema absent).
    """
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

    if REQUEST_SCHEMA_NAME not in getattr(validator, "available_schemas", []):
        raise EconomicsSchemaUnavailable(
            "the installed traigent-schema predates the economics telemetry "
            "contract (economics request schema is absent)"
        )
    return validator


def get_request_validator() -> Any:
    """Return the cached request validator, loading it fail-closed on first use.

    Only a successful load is cached; a failure re-raises on every call so a
    fixed environment recovers without a stale negative cache.
    """
    global _VALIDATOR_CACHE
    if _VALIDATOR_CACHE is not _UNSET:
        return _VALIDATOR_CACHE
    validator = _load_validator()
    _VALIDATOR_CACHE = validator
    return validator


def reset_request_validator_cache() -> None:
    """Test hook: drop the cached validator so discovery is re-evaluated."""
    global _VALIDATOR_CACHE
    _VALIDATOR_CACHE = _UNSET


def validate_request_or_fail(body: dict[str, Any]) -> None:
    """Validate a request body against the exact economics request schema.

    Raises:
        EconomicsSchemaUnavailable: If the exact schema is absent/old, or the
            validator raises while checking (fail closed — no transmission).
        EconomicsTelemetryContractError: If the body is present-and-invalid
            against the schema (e.g. an arbitrary extra key).
    """
    validator = get_request_validator()
    try:
        errors = validator.validate_json(body, REQUEST_SCHEMA_NAME)
    except Exception as exc:  # noqa: BLE001 - a validator that throws fails closed
        raise EconomicsSchemaUnavailable(
            "the economics schema validator raised while validating the request"
        ) from exc
    if errors:
        # Never echo the offending payload: the count and a fixed message only.
        raise EconomicsTelemetryContractError(
            f"economics telemetry request failed schema validation "
            f"({len(errors)} error(s))"
        )


__all__ = [
    "REQUEST_SCHEMA_NAME",
    "get_request_validator",
    "reset_request_validator_cache",
    "validate_request_or_fail",
]
