"""Runtime loading and validation for promoted best configs."""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003 REQ-STOR-007

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import math
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from traigent.core import best_config_diagnostics as diag
from traigent.utils.exceptions import ConfigurationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

BEST_CONFIG_SCHEMA_VERSION = "traigent.best_config.v1"
BEST_CONFIG_MANIFEST_SCHEMA_VERSION = "traigent.best_config_manifest.v1"
DEFAULT_BEST_CONFIG_CACHE_TTL_SECONDS = 24 * 60 * 60

_SPEC_REQUIRED_KEYS = {"schema_version", "config_id", "config"}
_SPEC_OPTIONAL_KEYS = {
    "function_ref",
    "environment",
    "provenance",
    "validation",
    "forward_compat",
}
_SPEC_ALLOWED_KEYS = _SPEC_REQUIRED_KEYS | _SPEC_OPTIONAL_KEYS
# The frozen spec_hash rule of the legacy ``traigent.best_config.v1`` envelope
# (TraigentSchema ``best_config_schema.json`` ``x-traigent-spec-hash``). The Backend has
# applied it since #819; provenance is NOT covered (v1 provenance carries event
# timestamps, so a republish of an identical config would otherwise look new).
_SPEC_HASH_KEYS = (
    "schema_version",
    "config_id",
    "function_ref",
    "environment",
    "config",
    "validation",
)
_SPEC_HASH_DEFAULT_ENVIRONMENT = "default"
# The Backend refuses a spec nested deeper than this (spec itself = depth 0); the
# frozen rule defines no digest for one (vector ``depth_boundary_refused``).
_SPEC_HASH_MAX_JSON_DEPTH = 16
# Removed from ``validation`` at any depth before hashing (by key name, not meaning):
# the guaranteed-selection claim and its result members are bound by the certificate,
# which itself carries the spec_hash, so they cannot be part of the hashed bytes.
_VALIDATION_HASH_EXCLUDED_KEYS = frozenset(
    {
        "guaranteed_selection",
        "certified_selection",
        "status",
        "deployable",
        "selected_config",
        "fallback_config",
        "certificate",
    }
)
# The rule this SDK used before it was aligned with the Backend (it also covered
# provenance and the unfiltered validation). Kept only to explain a stale stored hash.
_PRE_ALIGNMENT_SPEC_HASH_KEYS = (
    "schema_version",
    "config_id",
    "function_ref",
    "environment",
    "config",
    "provenance",
    "validation",
)

_SAFETY_SENSITIVE_KEYS = {
    "model",
    "provider",
    "system_prompt",
    "messages",
    "system_messages",
    "prompt_template",
    "prompt_templates",
    "tools",
    "tool_choice",
    "function_definitions",
    "tool_definitions",
    "response_format",
    "output_schema",
    "json_schema",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "safety_filter",
    "safety_filters",
    "guardrails",
    "moderation",
    "retrieval_sources",
    "grounding_sources",
    "data_sources",
    "auth",
    "credential",
    "credentials",
    "tenant",
    "account",
    "endpoint",
    "base_url",
}
_SAFETY_PREFIXES = (
    "safety_",
    "guard_",
    "policy_",
    "auth_",
    "credential_",
    "tenant_",
    "tools_",
    "tool_def_",
    "retriev",
    "ground",
)


class BestConfigSourceMode(StrEnum):
    """Durable best-config source selection mode."""

    OFF = "off"
    REPO = "repo"
    CLOUD = "cloud"
    REPO_THEN_CLOUD = "repo_then_cloud"
    CLOUD_THEN_REPO = "cloud_then_repo"
    #: The exact version the agent head pins (TRAIG-101 plan 8.5); fail closed.
    AGENT_HEAD = "agent_head"


class BestConfigSource(StrEnum):
    """Concrete source chosen during runtime resolution."""

    DEFAULT = "default"
    OVERRIDE = "override"
    APPLY_BEST_CONFIG = "apply_best_config"
    LOAD_FROM = "load_from"
    ENV = "env"
    REPO = "repo"
    CLOUD_CACHE = "cloud_cache"
    CLOUD_FETCH = "cloud_fetch"
    AGENT_HEAD = "agent_head"
    DEV_LOG = "dev_log"


class CloudPublishUnavailableReason(StrEnum):
    """Reason cloud best-config publish is unavailable."""

    DISABLED_BY_CONFIG = "disabled_by_config"
    BACKEND_NOT_IMPLEMENTED = "backend_not_implemented"
    REQUEST_FAILED = "request_failed"
    INTEGRITY_FAILED = "integrity_failed"
    STALE_HEAD_GENERATION = "stale_head_generation"
    HEAD_REFUSED = "head_refused"
    HEAD_CAPTURE_MISSING = "head_capture_missing"


class HeadRefusalCode(StrEnum):
    """Closed refusal codes of the content-less agent-head read and protected publish.

    Mirrors TraigentSchema ``optimization/best_config_head_refusal_schema.json``
    (``error_code``). ``UNRECOGNIZED`` is SDK-local: the backend answered a protected
    request with a body outside the closed envelope. ``CAPTURE_MISSING`` is SDK-local:
    the run could not capture the head at its start, so nothing was sent.
    """

    REFERENCE_NOT_FOUND = "reference_not_found"
    HEAD_CAPTURE_INVALID = "head_capture_invalid"
    HEAD_ENVIRONMENT_MISMATCH = "head_environment_mismatch"
    HEAD_AGENT_MISMATCH = "head_agent_mismatch"
    HEAD_GENERATION_STALE = "head_generation_stale"
    HEAD_WRITE_UNAUTHORIZED = "head_write_unauthorized"
    HEAD_BODY_INVALID = "head_body_invalid"
    HEAD_PUBLISH_CONFLICT = "head_publish_conflict"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    INTERNAL_ERROR = "internal_error"
    UNRECOGNIZED = "unrecognized"
    CAPTURE_MISSING = "capture_missing"


class CloudPublishUnavailable(ConfigurationError):
    """Raised when durable cloud best-config publishing cannot proceed."""

    def __init__(self, reason: CloudPublishUnavailableReason, message: str) -> None:
        super().__init__(message, details={"reason": reason.value})
        self.reason = reason


class CloudBestConfigStaleHeadError(CloudPublishUnavailable):
    """Raised when the agent head moved after the run that produced the winner began.

    The backend refused the promotion and wrote nothing: the published head is
    exactly as the other writer left it. ``best_config`` is the configuration
    this call tried to publish, so it can be re-published deliberately (for
    example after reviewing the newer head) instead of being lost.
    """

    def __init__(
        self,
        *,
        current_generation: int,
        expected_generation: int,
        message: str | None = None,
        agent_id: str | None = None,
        environment: str | None = None,
        best_config: dict[str, Any] | None = None,
    ) -> None:
        # trust only validated operands: exactly ``int`` (bool and subclasses, whose
        # ``__str__`` / ``__format__`` can be overridden, are excluded)
        operands_valid = (
            type(current_generation) is int and type(expected_generation) is int
        )
        if message is not None:
            default_message = message
        elif operands_valid:
            default_message = (
                f"Agent head is at generation {current_generation}, not the "
                f"generation {expected_generation} captured when the run started; "
                "the published head was left unchanged."
            )
        else:
            default_message = (
                "Agent head moved after the run captured its generation; "
                "the published head was left unchanged."
            )
        super().__init__(
            CloudPublishUnavailableReason.STALE_HEAD_GENERATION,
            default_message,
        )
        self.current_generation = current_generation
        self.expected_generation = expected_generation
        self.agent_id = agent_id
        self.environment = environment
        self.best_config = best_config
        if (
            message is None
            and operands_valid
            and type(self) is CloudBestConfigStaleHeadError
        ):
            # the default message holds only two validated integers
            diag._mark_safe(self)


class CloudBestConfigHeadRefusedError(CloudPublishUnavailable):
    """A content-less (protected) publish was refused with a closed code, or refused locally.

    Carries only the closed ``code`` (:class:`HeadRefusalCode`) and the HTTP status the
    backend answered with (None for a local refusal). The message is fixed SDK text:
    nothing of the response body, the ref, the ids or the headers is echoed. The backend
    wrote nothing; ``best_config`` is the refused configuration so the caller can decide
    deliberately what to do next.
    """

    def __init__(
        self,
        code: HeadRefusalCode,
        *,
        status_code: int | None = None,
        reason: CloudPublishUnavailableReason = CloudPublishUnavailableReason.HEAD_REFUSED,
        best_config: dict[str, Any] | None = None,
    ) -> None:
        # the closed code is validated by identity against the enum's own members, never
        # through the operand's ``__eq__`` / ``__hash__`` / ``__str__`` / ``value``
        valid_code = next((m for m in HeadRefusalCode if m is code), None)
        super().__init__(
            reason,
            (
                f"Protected best-config publish refused: {valid_code.value}."
                if valid_code is not None
                else "Protected best-config publish refused."
            ),
        )
        self.code = code
        self.status_code = status_code
        self.best_config = best_config
        if valid_code is not None and type(self) in (
            CloudBestConfigHeadRefusedError,
            CloudBestConfigHeadStaleError,
        ):
            # the message is built here from a validated closed code: trusted
            # (an unknown subclass or a forged code stays untrusted)
            diag._mark_safe(self)


class CloudBestConfigHeadStaleError(CloudBestConfigHeadRefusedError):
    """``head_generation_stale``: the agent head moved after this run captured its generation.

    The run's RESULT is stale. It must never be auto-retried, and the generation must
    never be refreshed and the same result resubmitted: start a new run (or re-evaluate
    against the new head) and publish with that run's own captured generation.
    """

    def __init__(
        self,
        *,
        status_code: int | None = 409,
        best_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            HeadRefusalCode.HEAD_GENERATION_STALE,
            status_code=status_code,
            reason=CloudPublishUnavailableReason.STALE_HEAD_GENERATION,
            best_config=best_config,
        )


class SafetySensitiveBestConfigError(ConfigurationError):
    """Raised when a best-config attempts an unsafe drift bypass."""


@dataclass(frozen=True)
class BestConfigSnapshot:
    """Immutable runtime snapshot used by invocation wrappers."""

    config_id: str | None
    config: MappingProxyType[str, Any]
    source: str
    schema_version: str = BEST_CONFIG_SCHEMA_VERSION
    config_hash: str | None = None
    spec_hash: str | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    provenance: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    @diag.boundary(
        diag.Category.RESULT_INVALID,
        "Best config snapshot could not be built",
    )
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        config_id: str | None,
        source: str,
        schema_version: str = BEST_CONFIG_SCHEMA_VERSION,
        spec_hash: str | None = None,
        loaded_at: datetime | None = None,
        expires_at: datetime | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> BestConfigSnapshot:
        """Build a deeply immutable snapshot from a mutable config dict."""

        normalized_config = _normalize_json_value(copy.deepcopy(config))
        if not isinstance(normalized_config, dict):
            raise diag._safe_error("Best config snapshot requires a dict config")
        frozen_config = _deep_freeze(normalized_config)
        normalized_provenance = _normalize_json_value(copy.deepcopy(provenance or {}))
        if not isinstance(normalized_provenance, dict):
            raise diag._safe_error("Best config provenance must be a dict")

        return cls(
            config_id=config_id,
            config=frozen_config,
            source=source,
            schema_version=schema_version,
            config_hash=sha256_digest(canonical_json(normalized_config)),
            spec_hash=spec_hash,
            loaded_at=loaded_at or datetime.now(UTC),
            expires_at=expires_at,
            provenance=_deep_freeze(normalized_provenance),
        )


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config value could not be serialized",
)
def canonical_json(value: Any) -> str:
    """Return canonical JSON for hash-stable, JSON-native values."""

    normalized = _normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config value could not be hashed",
)
def sha256_digest(value: str | bytes) -> str:
    """Return a prefixed sha256 digest."""

    if isinstance(value, str):
        try:
            data = value.encode("utf-8")
        except UnicodeEncodeError:
            # the error's repr/args carry the whole string (an unpaired surrogate in an
            # ordinary config value): fixed refusal, type name only
            raise diag.refusal(
                "Best config value is not valid UTF-8 text",
                category=diag.Category.RESULT_INVALID,
                error_type="UnicodeEncodeError",
            ) from None
    else:
        data = value
    return "sha256:" + hashlib.sha256(data).hexdigest()


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config could not be converted",
)
def thaw_config(config: Any) -> Any:
    """Convert a frozen config tree back to mutable JSON-native containers."""

    return _thaw(config)


def _thaw(config: Any) -> Any:
    if isinstance(config, MappingProxyType):
        return {key: _thaw(value) for key, value in config.items()}
    if isinstance(config, tuple):
        return [_thaw(value) for value in config]
    return config


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Function reference could not be derived",
)
def function_ref_for(func: Any) -> str:
    """Return canonical ``module:qualname`` for a callable after unwrapping."""

    target = inspect.unwrap(func)
    module = getattr(target, "__module__", None) or "__main__"
    qualname = getattr(target, "__qualname__", None) or getattr(
        target, "__name__", "unknown"
    )
    return f"{module}:{qualname}"


def _lookup_source_mode(mode: Any) -> BestConfigSourceMode | None:
    """Resolve ``mode`` to a member by value, or ``None``; never formats ``mode``.

    ``BestConfigSourceMode(mode)`` formats a rejected value with ``%r``, which runs a
    ``str`` subclass's own ``__repr__`` (and any exception it raises escapes). A member
    passes as itself; any other ``str`` is reduced to an exact ``str`` first (``join``
    does not call the subclass's ``__str__``, ``__eq__`` or ``__hash__``).
    """

    if isinstance(mode, BestConfigSourceMode):
        return mode
    if not isinstance(mode, str):
        return None
    return _SOURCE_MODES.get(str.join("", (mode,)))


_SOURCE_MODES: dict[str, BestConfigSourceMode] = {
    member.value: member for member in BestConfigSourceMode
}


@diag.boundary(
    diag.Category.SOURCE_MODE,
    "Best config source mode could not be resolved",
)
def source_order_for_mode(
    mode: str | BestConfigSourceMode,
) -> tuple[BestConfigSource, ...]:
    """Return durable source order for a best-config source mode."""

    normalized = _lookup_source_mode(mode)
    if normalized is None:
        raise diag.refusal(
            "Unknown best_config_source. "
            "Use one of: off, repo, cloud, repo_then_cloud, cloud_then_repo, agent_head.",
            category=diag.Category.SOURCE_MODE,
            best_config_source_type=type(mode).__name__,
        ) from None

    if normalized is BestConfigSourceMode.OFF:
        return ()
    if normalized is BestConfigSourceMode.REPO:
        return (BestConfigSource.REPO,)
    if normalized is BestConfigSourceMode.CLOUD:
        return (BestConfigSource.CLOUD_CACHE, BestConfigSource.CLOUD_FETCH)
    if normalized is BestConfigSourceMode.AGENT_HEAD:
        return (BestConfigSource.AGENT_HEAD,)
    if normalized is BestConfigSourceMode.REPO_THEN_CLOUD:
        return (
            BestConfigSource.REPO,
            BestConfigSource.CLOUD_CACHE,
            BestConfigSource.CLOUD_FETCH,
        )
    return (
        BestConfigSource.CLOUD_CACHE,
        BestConfigSource.CLOUD_FETCH,
        BestConfigSource.REPO,
    )


@diag.boundary(
    diag.Category.LOAD_FAILED,
    "Failed to load best config",
)
def load_best_config_spec(
    path: str | Path,
    *,
    configuration_space: dict[str, Any] | None = None,
    expected_config_id: str | None = None,
    expected_function_ref: str | None = None,
    expected_environment: str | None = None,
    manifest_entry: dict[str, Any] | None = None,
    source: str = BestConfigSource.REPO.value,
    strict: bool = False,
) -> BestConfigSnapshot | None:
    """Load, validate, and freeze one canonical best-config JSON spec."""

    spec_path = Path(path)
    try:
        with spec_path.open(encoding="utf-8") as handle:
            spec = json.load(handle)
        return snapshot_from_spec(
            spec,
            configuration_space=configuration_space,
            expected_config_id=expected_config_id,
            expected_function_ref=expected_function_ref,
            expected_environment=expected_environment,
            manifest_entry=manifest_entry,
            source=source,
        )
    except Exception as exc:
        if strict or isinstance(exc, ConfigurationError):
            if isinstance(exc, ConfigurationError):
                raise
            raise diag.refusal(
                "Failed to load best config",
                category=diag.Category.LOAD_FAILED,
                cause=exc,
                path=str(path),
            ) from None
        # fail-open(intentional): non-strict callers accept degraded fallback; strict callers re-raise
        diag.log(
            logger,
            logging.WARNING,
            "Ignoring invalid best config",
            category=diag.Category.LOAD_FAILED,
            exc=exc,
        )
        return None


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config spec could not be validated",
)
def snapshot_from_spec(
    spec: dict[str, Any],
    *,
    configuration_space: dict[str, Any] | None = None,
    expected_config_id: str | None = None,
    expected_function_ref: str | None = None,
    expected_environment: str | None = None,
    manifest_entry: dict[str, Any] | None = None,
    source: str = BestConfigSource.REPO.value,
) -> BestConfigSnapshot:
    """Validate a decoded best-config spec and return a snapshot.

    ``expected_environment`` binds the spec to the environment the caller asked for: the
    spec's effective environment (``environment``, else ``"default"``; trimmed, lower-cased)
    must equal it, otherwise another environment's config would be loaded.
    """

    if not isinstance(spec, dict):
        raise diag._safe_error("Best config spec must be a JSON object")

    unknown_top_level = set(spec) - _SPEC_ALLOWED_KEYS
    if unknown_top_level:
        raise diag._safe_error(
            "Unknown top-level best config fields",
            details={"fields": sorted(unknown_top_level)},
        )

    missing = _SPEC_REQUIRED_KEYS - set(spec)
    if missing:
        raise diag._safe_error(
            "Best config spec missing fields", details={"fields": sorted(missing)}
        )

    if spec["schema_version"] != BEST_CONFIG_SCHEMA_VERSION:
        raise diag._safe_error(
            "Unsupported best config schema_version",
            details={"schema_version": spec["schema_version"]},
        )

    config_id = _validate_config_id(spec["config_id"])
    if expected_config_id and config_id != expected_config_id:
        raise diag._safe_error(
            "Best config id does not match the requested config_id",
            details={"expected": expected_config_id, "actual": config_id},
        )

    _validate_spec_binding(spec, expected_function_ref, expected_environment)

    raw_config = spec["config"]
    if not isinstance(raw_config, dict):
        raise diag._safe_error("Best config field 'config' must be an object")

    raw_config_hash = sha256_digest(canonical_json(raw_config))
    raw_spec_hash = compute_spec_hash(spec)
    if manifest_entry:
        _verify_manifest_hashes(
            manifest_entry,
            config_hash=raw_config_hash,
            spec_hash=raw_spec_hash,
            spec=spec,
        )

    config = _validate_and_filter_config_keys(
        raw_config,
        configuration_space=configuration_space,
        forward_compat=bool(spec.get("forward_compat", False)),
    )
    spec_hash = raw_spec_hash

    provenance = spec.get("provenance", {})
    if not isinstance(provenance, dict):
        raise diag._safe_error("Best config provenance must be an object")

    snapshot = BestConfigSnapshot.from_config(
        config,
        config_id=config_id,
        source=source,
        spec_hash=spec_hash,
        provenance=provenance,
    )
    return snapshot


def _validate_spec_binding(
    spec: dict[str, Any],
    expected_function_ref: str | None,
    expected_environment: str | None,
) -> None:
    function_ref = spec.get("function_ref")
    if expected_function_ref:
        # A caller that names its function is only served a config bound to that
        # function: a missing / null ``function_ref`` cannot establish it. Only an
        # explicitly unbound read (no expected function) accepts a legacy unbound config.
        if function_ref is None:
            raise diag._safe_error(
                "Best config function_ref mismatch: the config is not bound to a "
                "function",
                details={"expected": expected_function_ref},
            )
        if function_ref != expected_function_ref:
            raise diag._safe_error(
                "Best config function_ref mismatch: the config is bound to "
                "another function",
                details={"expected": expected_function_ref, "actual": function_ref},
            )

    if expected_environment is not None:
        spec_environment = spec.get("environment")
        effective: str | None
        if isinstance(spec_environment, str):
            effective = spec_environment.strip().lower()
        elif spec_environment is None:
            effective = "default"
        else:
            effective = None
        if effective != expected_environment.strip().lower():
            raise diag._safe_error(
                "Best config environment mismatch: the spec is not for the "
                "requested environment"
            )


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config spec could not be hashed",
)
def compute_spec_hash(spec: dict[str, Any], *, environment: str | None = None) -> str:
    """Return the frozen ``spec_hash`` of a ``traigent.best_config.v1`` spec.

    Mirrors the Backend rule: NFC, trimmed ``config_id``, always-present effective
    environment (``spec.environment``, else ``environment``, else ``"default"``),
    provenance and ``forward_compat`` not covered, claim/result keys removed from
    ``validation`` recursively (``validation`` omitted when empty afterwards), then
    canonical JSON. ``environment`` is the publish request's environment, used only
    when the spec carries none. Any other envelope version has no rule here.
    """

    schema_version = spec.get("schema_version")
    if schema_version != BEST_CONFIG_SCHEMA_VERSION:
        raise diag._safe_error(
            "No spec_hash rule for this best config schema_version",
            details={"schema_version": schema_version},
        )
    return sha256_digest(
        _serialize_v1_hash_payload(_v1_hash_payload(spec, environment))
    )


def _check_spec_hash_finite(value: Any) -> None:
    """Refuse NaN/Infinity anywhere in the ORIGINAL spec (before NFC and exclusions)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise diag._safe_error("Best config JSON values cannot be NaN/Infinity")
    elif isinstance(value, dict):
        for item in value.values():
            _check_spec_hash_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_spec_hash_finite(item)


@diag.boundary(
    diag.Category.RESULT_INVALID,
    "Best config spec could not be checked",
)
def check_publishable_spec(spec: Any) -> None:
    """Refuse to SEND a spec holding object keys that are equal after Unicode NFC.

    The frozen v1 hash rule keeps the later key (so stored specs still verify), but a NEW
    publish with such keys is refused by the Backend; failing here avoids the round trip.
    The message never echoes a key.
    """
    _check_publishable(spec)


def _check_publishable(spec: Any) -> None:
    if isinstance(spec, dict):
        seen: set[str] = set()
        for key, item in spec.items():
            if not isinstance(key, str):
                raise diag._safe_error("Best config JSON object keys must be strings")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in seen:
                raise diag._safe_error(
                    "Best config spec has object keys that are equal after Unicode "
                    "normalization (NFC); rename one before publishing"
                )
            seen.add(normalized)
            _check_publishable(item)
    elif isinstance(spec, list):
        for item in spec:
            _check_publishable(item)


def _check_spec_hash_depth(value: Any, depth: int = 0) -> None:
    if depth > _SPEC_HASH_MAX_JSON_DEPTH:
        raise diag._safe_error("Best config spec exceeds the maximum JSON depth")
    if isinstance(value, dict):
        for item in value.values():
            _check_spec_hash_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_spec_hash_depth(item, depth + 1)


def _v1_hash_payload(spec: dict[str, Any], environment: str | None) -> dict[str, Any]:
    _check_spec_hash_finite(spec)
    _check_spec_hash_depth(spec)
    normalized = _normalize_json_value(copy.deepcopy(spec))
    if not isinstance(normalized, dict):
        raise diag._safe_error("Best config spec must be a JSON object")
    config_id = spec.get("config_id")
    if not isinstance(config_id, str):
        raise diag._safe_error("Best config config_id must be a string")
    normalized["config_id"] = config_id.strip()  # trimmed, not NFC (the Backend rule)
    function_ref = spec.get("function_ref")
    if isinstance(function_ref, str):
        normalized["function_ref"] = function_ref.strip()
    effective_environment = normalized.get("environment")
    if effective_environment is None:
        effective_environment = environment or _SPEC_HASH_DEFAULT_ENVIRONMENT
    if not isinstance(effective_environment, str):
        raise diag._safe_error("Best config environment must be a string")
    normalized["environment"] = effective_environment.strip().lower()

    payload = {key: normalized[key] for key in _SPEC_HASH_KEYS if key in normalized}
    validation = payload.get("validation")
    if isinstance(validation, dict):
        claim_free = _strip_validation_claims(validation)
        if claim_free:
            payload["validation"] = claim_free
        else:
            payload.pop("validation")
    return payload


def _strip_validation_claims(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_validation_claims(item)
            for key, item in value.items()
            if key not in _VALIDATION_HASH_EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_strip_validation_claims(item) for item in value]
    return value


def _serialize_v1_hash_payload(payload: dict[str, Any]) -> str:
    # Not canonical_json(): that would NFC-normalize config_id again, which the rule forbids.
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pre_alignment_spec_hash(spec: dict[str, Any]) -> str:
    payload = {key: spec[key] for key in _PRE_ALIGNMENT_SPEC_HASH_KEYS if key in spec}
    return sha256_digest(canonical_json(payload))


@diag.boundary(
    diag.Category.FILESYSTEM_FAILED,
    "Best config manifest could not be read",
)
def load_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Load a best-config manifest from a directory."""

    manifest_path = Path(directory) / "manifest.json"
    with diag.fs_guard("Best config manifest could not be read"):
        if not manifest_path.exists():
            return None
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise diag._safe_error("Best config manifest must be an object")
    if manifest.get("schema_version") != BEST_CONFIG_MANIFEST_SCHEMA_VERSION:
        raise diag._safe_error(
            "Unsupported best config manifest schema_version",
            details={"schema_version": manifest.get("schema_version")},
        )
    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        raise diag._safe_error("Best config manifest requires a configs object")
    return manifest


@diag.boundary(
    diag.Category.FILESYSTEM_FAILED,
    "Best config could not be written",
)
def write_repo_best_config(
    directory: str | Path,
    *,
    config_id: str,
    config: dict[str, Any],
    function_ref: str | None = None,
    provenance: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    environment: str | None = None,
) -> Path:
    """Write a canonical repo best-config spec and update manifest."""

    safe_config_id = _validate_config_id(config_id)
    output_dir = Path(directory)
    with diag.fs_guard("Best config directory could not be created"):
        output_dir.mkdir(parents=True, exist_ok=True)

    spec: dict[str, Any] = {
        "schema_version": BEST_CONFIG_SCHEMA_VERSION,
        "config_id": safe_config_id,
        "config": _normalize_json_value(copy.deepcopy(config)),
    }
    if function_ref:
        spec["function_ref"] = function_ref
    if environment:
        spec["environment"] = environment
    if provenance:
        spec["provenance"] = _normalize_json_value(copy.deepcopy(provenance))
    if validation:
        spec["validation"] = _normalize_json_value(copy.deepcopy(validation))

    config_hash = sha256_digest(canonical_json(spec["config"]))
    spec_hash = compute_spec_hash(spec)
    spec_path = output_dir / f"{safe_config_id}.json"
    _write_json_atomic(spec_path, spec)

    manifest_path = output_dir / "manifest.json"
    with diag.fs_guard("Best config manifest could not be inspected"):
        manifest_present = manifest_path.exists()
    loaded_manifest = load_manifest(output_dir) if manifest_present else None
    manifest: dict[str, Any] = loaded_manifest or {
        "schema_version": BEST_CONFIG_MANIFEST_SCHEMA_VERSION,
        "configs": {},
    }
    configs = manifest["configs"]
    if not isinstance(configs, dict):
        raise diag._safe_error("Best config manifest requires a configs object")
    configs[safe_config_id] = {
        "path": spec_path.name,
        "spec_hash": spec_hash,
        "config_hash": config_hash,
    }
    _write_json_atomic(manifest_path, manifest)
    return spec_path


@diag.boundary(
    diag.Category.CACHE_WRITE_FAILED,
    "Cloud best config cache could not be written",
)
def write_cloud_cache_best_config(
    directory: str | Path,
    spec: dict[str, Any],
    *,
    etag: str | None = None,
    version: int | str | None = None,
    loaded_at: datetime | None = None,
) -> Path:
    """Write a fetched cloud best-config spec into the local cloud cache."""

    if not etag and version is None:
        raise diag._safe_error("Cloud cache metadata requires etag or version")

    snapshot = snapshot_from_spec(spec, source=BestConfigSource.CLOUD_CACHE.value)
    safe_config_id = _validate_config_id(snapshot.config_id)
    output_dir = Path(directory)
    with diag.fs_guard("Best config directory could not be created"):
        output_dir.mkdir(parents=True, exist_ok=True)

    normalized_spec = _normalize_json_value(copy.deepcopy(spec))
    if not isinstance(normalized_spec, dict):
        raise diag._safe_error("Best config spec must be a JSON object")
    config_hash = sha256_digest(canonical_json(normalized_spec["config"]))
    spec_hash = compute_spec_hash(normalized_spec)
    spec_path = output_dir / f"{safe_config_id}.json"
    _write_json_atomic(spec_path, normalized_spec)

    manifest_path = output_dir / "manifest.json"
    with diag.fs_guard("Best config manifest could not be inspected"):
        manifest_present = manifest_path.exists()
    loaded_manifest = load_manifest(output_dir) if manifest_present else None
    manifest: dict[str, Any] = loaded_manifest or {
        "schema_version": BEST_CONFIG_MANIFEST_SCHEMA_VERSION,
        "configs": {},
    }
    configs = manifest["configs"]
    if not isinstance(configs, dict):
        raise diag._safe_error("Best config manifest requires a configs object")

    entry: dict[str, Any] = {
        "path": spec_path.name,
        "spec_hash": spec_hash,
        "config_hash": config_hash,
        "loaded_at": (loaded_at or datetime.now(UTC)).isoformat(),
    }
    if etag:
        entry["etag"] = etag
    if version is not None:
        entry["version"] = version
    configs[safe_config_id] = entry
    _write_json_atomic(manifest_path, manifest)
    return spec_path


@diag.boundary(
    diag.Category.REPO_INVALID,
    "Failed to resolve repo best config",
)
def resolve_repo_best_config(
    *,
    config_id: str | None,
    repo_root: str | Path | None = None,
    configuration_space: dict[str, Any] | None = None,
    expected_function_ref: str | None = None,
    expected_environment: str | None = None,
    strict: bool = False,
) -> BestConfigSnapshot | None:
    """Resolve a repo-local best config by config id.

    ``expected_function_ref`` and ``expected_environment`` (when given) bind the manifest's
    spec to what the caller asked for: a spec for another environment or function is
    refused (``strict``) or skipped, never loaded.
    """

    if not config_id:
        return None
    safe_config_id = _validate_config_id(config_id)
    directory = Path(repo_root or Path.cwd()) / ".traigent" / "best-configs"
    manifest_path = directory / "manifest.json"

    try:
        # Inside the guarded boundary: ``exists()`` raises ``OSError`` (text names the
        # path) for an overlong component or a permission failure.
        if not manifest_path.exists():
            return None
        manifest = load_manifest(directory) or {}
        entry = manifest.get("configs", {}).get(safe_config_id)
        if not isinstance(entry, dict):
            return None
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise diag._safe_error(
                "Manifest entry lacks path", details={"config_id": safe_config_id}
            )
        return load_best_config_spec(
            _resolve_entry_path(directory, path),
            configuration_space=configuration_space,
            expected_config_id=safe_config_id,
            expected_function_ref=expected_function_ref,
            expected_environment=expected_environment,
            manifest_entry=entry,
            source=BestConfigSource.REPO.value,
            strict=True,
        )
    except Exception as exc:
        if isinstance(exc, SafetySensitiveBestConfigError):
            raise
        if strict:
            if isinstance(exc, ConfigurationError):
                raise
            raise diag.refusal(
                "Failed to resolve repo best config",
                category=diag.Category.REPO_INVALID,
                cause=exc,
            ) from None
        # fail-open(intentional): non-strict callers accept degraded fallback; strict callers re-raise
        diag.log(
            logger,
            logging.WARNING,
            "Ignoring repo best config after validation failure",
            category=diag.Category.REPO_INVALID,
            exc=exc,
        )
        return None


@diag.boundary(
    diag.Category.CACHE_INVALID,
    "Failed to resolve cloud cache",
)
def resolve_cloud_cache_best_config(
    *,
    config_id: str | None,
    cache_dir: str | Path | None,
    configuration_space: dict[str, Any] | None = None,
    ttl_seconds: int = DEFAULT_BEST_CONFIG_CACHE_TTL_SECONDS,
    stale_ok_ttl_seconds: int | None = None,
    now: datetime | None = None,
    strict: bool = False,
    expected_function_ref: str | None = None,
    expected_environment: str | None = None,
) -> BestConfigSnapshot | None:
    """Resolve a locally cached cloud best config by config id.

    The cache is keyed by config id only, so ``expected_function_ref`` and
    ``expected_environment`` (when given) bind the cached spec to what the caller asked for.
    """

    if not config_id or not cache_dir:
        return None
    safe_config_id = _validate_config_id(config_id)

    directory = Path(cache_dir)

    current_time = now or datetime.now(UTC)
    try:
        entry = _existing_cache_entry(directory, safe_config_id)
        if entry is None:
            return None
        if not (entry.get("etag") or entry.get("version")):
            raise diag._safe_error("Cloud cache metadata requires etag or version")

        loaded_at = _parse_datetime(entry.get("loaded_at"))
        if _cache_is_stale(loaded_at, current_time, ttl_seconds, stale_ok_ttl_seconds):
            return None

        snapshot = load_best_config_spec(
            _resolve_entry_path(directory, entry["path"]),
            configuration_space=configuration_space,
            expected_config_id=safe_config_id,
            expected_function_ref=expected_function_ref,
            expected_environment=expected_environment,
            manifest_entry=entry,
            source=BestConfigSource.CLOUD_CACHE.value,
            strict=True,
        )
        expires_at = loaded_at + timedelta(seconds=ttl_seconds)
        object.__setattr__(snapshot, "loaded_at", loaded_at)
        object.__setattr__(snapshot, "expires_at", expires_at)
        return snapshot
    except Exception as exc:
        if isinstance(exc, SafetySensitiveBestConfigError):
            raise
        if strict:
            if isinstance(exc, ConfigurationError):
                raise
            raise diag.refusal(
                "Failed to resolve cloud cache",
                category=diag.Category.CACHE_INVALID,
                cause=exc,
            ) from None
        # fail-open(intentional): non-strict callers accept degraded fallback; strict callers re-raise
        diag.log(
            logger,
            logging.WARNING,
            "Ignoring cloud best-config cache",
            category=diag.Category.CACHE_INVALID,
            exc=exc,
        )
        return None


def _existing_cache_entry(directory: Path, config_id: str) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    return _load_cache_entry(directory, config_id)


def _cache_is_stale(
    loaded_at: datetime,
    current_time: datetime,
    ttl_seconds: int,
    stale_ok_ttl_seconds: int | None,
) -> bool:
    max_age = stale_ok_ttl_seconds if stale_ok_ttl_seconds is not None else ttl_seconds
    if current_time - loaded_at > timedelta(seconds=max_age):
        diag.log(
            logger,
            logging.WARNING,
            "Ignoring stale best-config cache",
            category=diag.Category.CACHE_STALE,
        )
        return True
    return False


def _validate_config_id(config_id: Any) -> str:
    if not isinstance(config_id, str) or not config_id:
        raise diag._safe_error("Best config config_id must be a non-empty string")
    if "\x00" in config_id or "/" in config_id or "\\" in config_id:
        raise diag._safe_error("Best config config_id must not contain path separators")
    if config_id in {".", ".."}:
        raise diag._safe_error("Best config config_id must not be a relative path")
    return config_id


def _resolve_entry_path(directory: Path, path: Any) -> Path:
    if not isinstance(path, str) or not path:
        raise diag._safe_error("Best config manifest entry lacks path")
    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in path:
        raise diag._safe_error(
            "Best config manifest path must stay inside its directory"
        )
    return directory / relative_path


def _load_cache_entry(directory: Path, config_id: str) -> dict[str, Any] | None:
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(directory) or {}
        entry = manifest.get("configs", {}).get(config_id)
        return entry if isinstance(entry, dict) else None

    metadata_path = directory / f"{config_id}.metadata.json"
    spec_path = directory / f"{config_id}.json"
    if not metadata_path.exists() or not spec_path.exists():
        return None
    with diag.fs_guard("Cloud cache metadata could not be read"):
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise diag._safe_error("Cloud cache metadata must be an object")
    metadata.setdefault("path", spec_path.name)
    return metadata


def _verify_manifest_hashes(
    manifest_entry: dict[str, Any],
    *,
    config_hash: str,
    spec_hash: str,
    spec: dict[str, Any] | None = None,
) -> None:
    expected_config_hash = manifest_entry.get("config_hash")
    expected_spec_hash = manifest_entry.get("spec_hash")
    if expected_config_hash and expected_config_hash != config_hash:
        raise diag._safe_error("Best config manifest config_hash mismatch")
    if expected_spec_hash and expected_spec_hash != spec_hash:
        if spec is not None and expected_spec_hash == _pre_alignment_spec_hash(spec):
            raise diag._safe_error(
                "Best config manifest spec_hash was written with an older spec_hash "
                "rule (it covered provenance) and is no longer valid; regenerate the "
                "manifest entry or refetch the cloud best config"
            )
        raise diag._safe_error("Best config manifest spec_hash mismatch")


def _validate_and_filter_config_keys(
    config: dict[str, Any],
    *,
    configuration_space: dict[str, Any] | None,
    forward_compat: bool,
) -> dict[str, Any]:
    normalized = _normalize_json_value(copy.deepcopy(config))
    if not isinstance(normalized, dict):
        raise diag._safe_error("Best config must be an object")

    if not configuration_space:
        return normalized

    allowed = set(configuration_space)
    unknown = set(normalized) - allowed
    if not unknown:
        return normalized

    denylisted = sorted(key for key in unknown if _is_safety_sensitive_key(key))
    if denylisted:
        raise diag._mark_safe(
            SafetySensitiveBestConfigError(
                "Unknown safety-sensitive best config keys",
                details={"keys": denylisted},
            )
        )

    if not forward_compat:
        raise diag._safe_error(
            "Unknown best config keys", details={"keys": sorted(unknown)}
        )

    logger.warning(
        "Ignoring %d forward-compatible best config key(s) outside configuration_space",
        len(unknown),
    )
    return {key: value for key, value in normalized.items() if key in allowed}


def _is_safety_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SAFETY_SENSITIVE_KEYS or lowered.startswith(_SAFETY_PREFIXES)


def _normalize_json_value(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _normalize_json_float(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)

    container_id = id(value)
    if isinstance(value, dict):
        return _normalize_json_dict(value, seen, container_id)

    if isinstance(value, list):
        return _normalize_json_list(value, seen, container_id)

    # numpy scalar types (e.g. np.int64 from Optuna integer dimensions) are not
    # JSON-native and do not subclass Python int/bool. Coerce a genuine 0-d numpy
    # scalar losslessly via .item(). We require ndim == 0 so size-one arrays
    # (np.array([1]) / np.array([[1]]), ndim >= 1) keep their shape and fall through
    # to reject instead of silently collapsing, and so arbitrary objects that merely
    # expose .item() (no ndim) are not coerced. We then accept ONLY when .item()
    # yields a plain JSON-native primitive; anything else (np.longdouble.item() ->
    # np.longdouble, datetime64/complex.item(), a 0-d object array's dict) is
    # rejected here rather than recursing without progress or passing silently.
    if _has_scalar_item(value):
        try:
            coerced = value.item()
        except (ValueError, TypeError):
            coerced = value  # falls through to the reject below
        if type(coerced) in (int, float, str, bool):
            return _normalize_json_value(coerced, seen)

    raise diag._safe_error(
        f"Best config contains non-JSON-native value {type(value).__name__}"
    )


def _has_scalar_item(value: Any) -> bool:
    return getattr(value, "ndim", None) == 0 and hasattr(value, "item")


def _normalize_json_float(value: float) -> float:
    if not math.isfinite(value):
        raise diag._safe_error("Best config JSON values cannot be NaN/Infinity")
    return value


def _normalize_json_list(
    value: list[Any], seen: set[int], container_id: int
) -> list[Any]:
    if container_id in seen:
        raise diag._safe_error("Best config contains a circular reference")
    seen.add(container_id)
    normalized_list = [_normalize_json_value(item, seen) for item in value]
    seen.remove(container_id)
    return normalized_list


def _normalize_json_dict(
    value: dict[Any, Any], seen: set[int], container_id: int
) -> dict[str, Any]:
    if container_id in seen:
        raise diag._safe_error("Best config contains a circular reference")
    seen.add(container_id)
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise diag._safe_error("Best config JSON object keys must be strings")
        normalized[unicodedata.normalize("NFC", key)] = _normalize_json_value(
            item, seen
        )
    seen.remove(container_id)
    return normalized


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise diag._safe_error("Cloud cache metadata requires loaded_at timestamp")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with diag.fs_guard("Best config file could not be written"):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
        except UnicodeEncodeError:
            # repr/args of the encode error carry the unencodable customer string
            tmp_path.unlink(missing_ok=True)
            raise diag.refusal(
                "Best config file could not be written",
                category=diag.Category.FILESYSTEM_FAILED,
                error_type="UnicodeEncodeError",
            ) from None
        tmp_path.replace(path)


__all__ = [
    "BEST_CONFIG_MANIFEST_SCHEMA_VERSION",
    "BEST_CONFIG_SCHEMA_VERSION",
    "BestConfigSnapshot",
    "BestConfigSource",
    "BestConfigSourceMode",
    "CloudBestConfigHeadRefusedError",
    "CloudBestConfigHeadStaleError",
    "CloudBestConfigStaleHeadError",
    "CloudPublishUnavailable",
    "CloudPublishUnavailableReason",
    "HeadRefusalCode",
    "SafetySensitiveBestConfigError",
    "canonical_json",
    "compute_spec_hash",
    "function_ref_for",
    "load_best_config_spec",
    "load_manifest",
    "resolve_cloud_cache_best_config",
    "resolve_repo_best_config",
    "sha256_digest",
    "snapshot_from_spec",
    "source_order_for_mode",
    "thaw_config",
    "write_cloud_cache_best_config",
    "write_repo_best_config",
]
