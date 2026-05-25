"""Runtime loading and validation for promoted best configs."""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003 REQ-STOR-007

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
_SPEC_HASH_KEYS = (
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
    DEV_LOG = "dev_log"


class CloudPublishUnavailableReason(StrEnum):
    """Reason cloud best-config publish is unavailable."""

    DISABLED_BY_CONFIG = "disabled_by_config"
    BACKEND_NOT_IMPLEMENTED = "backend_not_implemented"


class CloudPublishUnavailable(ConfigurationError):
    """Raised when durable cloud best-config publishing cannot proceed."""

    def __init__(self, reason: CloudPublishUnavailableReason, message: str) -> None:
        super().__init__(message, details={"reason": reason.value})
        self.reason = reason


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
            raise ConfigurationError("Best config snapshot requires a dict config")
        frozen_config = _deep_freeze(normalized_config)
        normalized_provenance = _normalize_json_value(copy.deepcopy(provenance or {}))
        if not isinstance(normalized_provenance, dict):
            raise ConfigurationError("Best config provenance must be a dict")

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


def sha256_digest(value: str | bytes) -> str:
    """Return a prefixed sha256 digest."""

    data = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(data).hexdigest()


def thaw_config(config: Any) -> Any:
    """Convert a frozen config tree back to mutable JSON-native containers."""

    if isinstance(config, MappingProxyType):
        return {key: thaw_config(value) for key, value in config.items()}
    if isinstance(config, tuple):
        return [thaw_config(value) for value in config]
    return config


def function_ref_for(func: Any) -> str:
    """Return canonical ``module:qualname`` for a callable after unwrapping."""

    target = inspect.unwrap(func)
    module = getattr(target, "__module__", None) or "__main__"
    qualname = getattr(target, "__qualname__", None) or getattr(
        target, "__name__", "unknown"
    )
    return f"{module}:{qualname}"


def source_order_for_mode(
    mode: str | BestConfigSourceMode,
) -> tuple[BestConfigSource, ...]:
    """Return durable source order for a best-config source mode."""

    try:
        normalized = BestConfigSourceMode(mode)
    except ValueError as exc:
        raise ConfigurationError(
            f"Unknown best_config_source {mode!r}. "
            "Use one of: off, repo, cloud, repo_then_cloud, cloud_then_repo."
        ) from exc

    if normalized is BestConfigSourceMode.OFF:
        return ()
    if normalized is BestConfigSourceMode.REPO:
        return (BestConfigSource.REPO,)
    if normalized is BestConfigSourceMode.CLOUD:
        return (BestConfigSource.CLOUD_CACHE, BestConfigSource.CLOUD_FETCH)
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


def load_best_config_spec(
    path: str | Path,
    *,
    configuration_space: dict[str, Any] | None = None,
    expected_config_id: str | None = None,
    expected_function_ref: str | None = None,
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
            manifest_entry=manifest_entry,
            source=source,
        )
    except Exception as exc:
        if strict or isinstance(exc, ConfigurationError):
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load best config {path}: {exc}"
            ) from exc
        logger.warning("Ignoring invalid best config %s: %s", path, exc)
        return None


def snapshot_from_spec(
    spec: dict[str, Any],
    *,
    configuration_space: dict[str, Any] | None = None,
    expected_config_id: str | None = None,
    expected_function_ref: str | None = None,
    manifest_entry: dict[str, Any] | None = None,
    source: str = BestConfigSource.REPO.value,
) -> BestConfigSnapshot:
    """Validate a decoded best-config spec and return a snapshot."""

    if not isinstance(spec, dict):
        raise ConfigurationError("Best config spec must be a JSON object")

    unknown_top_level = set(spec) - _SPEC_ALLOWED_KEYS
    if unknown_top_level:
        raise ConfigurationError(
            f"Unknown top-level best config fields: {sorted(unknown_top_level)}"
        )

    missing = _SPEC_REQUIRED_KEYS - set(spec)
    if missing:
        raise ConfigurationError(f"Best config spec missing fields: {sorted(missing)}")

    if spec["schema_version"] != BEST_CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported best config schema_version {spec['schema_version']!r}"
        )

    config_id = _validate_config_id(spec["config_id"])
    if expected_config_id and config_id != expected_config_id:
        raise ConfigurationError(
            f"Best config id mismatch: expected {expected_config_id!r}, got {config_id!r}"
        )

    function_ref = spec.get("function_ref")
    if (
        expected_function_ref
        and function_ref is not None
        and function_ref != expected_function_ref
    ):
        raise ConfigurationError(
            "Best config function_ref mismatch: "
            f"expected {expected_function_ref!r}, got {function_ref!r}"
        )

    raw_config = spec["config"]
    if not isinstance(raw_config, dict):
        raise ConfigurationError("Best config field 'config' must be an object")

    raw_config_hash = sha256_digest(canonical_json(raw_config))
    raw_spec_hash = compute_spec_hash(spec)
    if manifest_entry:
        _verify_manifest_hashes(
            manifest_entry,
            config_hash=raw_config_hash,
            spec_hash=raw_spec_hash,
        )

    config = _validate_and_filter_config_keys(
        raw_config,
        configuration_space=configuration_space,
        forward_compat=bool(spec.get("forward_compat", False)),
    )
    spec_hash = raw_spec_hash

    provenance = spec.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ConfigurationError("Best config provenance must be an object")

    snapshot = BestConfigSnapshot.from_config(
        config,
        config_id=config_id,
        source=source,
        spec_hash=spec_hash,
        provenance=provenance,
    )
    return snapshot


def compute_spec_hash(spec: dict[str, Any]) -> str:
    """Compute hash over the canonical self-hash-free spec payload."""

    payload = {key: spec[key] for key in _SPEC_HASH_KEYS if key in spec}
    return sha256_digest(canonical_json(payload))


def load_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Load a best-config manifest from a directory."""

    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ConfigurationError("Best config manifest must be an object")
    if manifest.get("schema_version") != BEST_CONFIG_MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported best config manifest schema_version "
            f"{manifest.get('schema_version')!r}"
        )
    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        raise ConfigurationError("Best config manifest requires a configs object")
    return manifest


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
    loaded_manifest = load_manifest(output_dir) if manifest_path.exists() else None
    manifest: dict[str, Any] = loaded_manifest or {
        "schema_version": BEST_CONFIG_MANIFEST_SCHEMA_VERSION,
        "configs": {},
    }
    configs = manifest["configs"]
    if not isinstance(configs, dict):
        raise ConfigurationError("Best config manifest requires a configs object")
    configs[safe_config_id] = {
        "path": spec_path.name,
        "spec_hash": spec_hash,
        "config_hash": config_hash,
    }
    _write_json_atomic(manifest_path, manifest)
    return spec_path


def resolve_repo_best_config(
    *,
    config_id: str | None,
    repo_root: str | Path | None = None,
    configuration_space: dict[str, Any] | None = None,
    expected_function_ref: str | None = None,
    strict: bool = False,
) -> BestConfigSnapshot | None:
    """Resolve a repo-local best config by config id."""

    if not config_id:
        return None
    safe_config_id = _validate_config_id(config_id)
    directory = Path(repo_root or Path.cwd()) / ".traigent" / "best-configs"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = load_manifest(directory) or {}
        entry = manifest.get("configs", {}).get(safe_config_id)
        if not isinstance(entry, dict):
            return None
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ConfigurationError(
                f"Manifest entry for {safe_config_id!r} lacks path"
            )
        return load_best_config_spec(
            _resolve_entry_path(directory, path),
            configuration_space=configuration_space,
            expected_config_id=safe_config_id,
            expected_function_ref=expected_function_ref,
            manifest_entry=entry,
            source=BestConfigSource.REPO.value,
            strict=True,
        )
    except Exception as exc:
        if strict:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to resolve repo best config: {exc}"
            ) from exc
        logger.warning(
            "Ignoring repo best config %s after validation failure: %s",
            safe_config_id,
            exc,
        )
        return None


def resolve_cloud_cache_best_config(
    *,
    config_id: str | None,
    cache_dir: str | Path | None,
    configuration_space: dict[str, Any] | None = None,
    ttl_seconds: int = DEFAULT_BEST_CONFIG_CACHE_TTL_SECONDS,
    stale_ok_ttl_seconds: int | None = None,
    now: datetime | None = None,
    strict: bool = False,
) -> BestConfigSnapshot | None:
    """Resolve a locally cached cloud best config by config id."""

    if not config_id or not cache_dir:
        return None
    safe_config_id = _validate_config_id(config_id)

    directory = Path(cache_dir)
    if not directory.exists():
        return None

    current_time = now or datetime.now(UTC)
    try:
        entry = _load_cache_entry(directory, safe_config_id)
        if entry is None:
            return None
        if not (entry.get("etag") or entry.get("version")):
            raise ConfigurationError("Cloud cache metadata requires etag or version")

        loaded_at = _parse_datetime(entry.get("loaded_at"))
        max_age = (
            stale_ok_ttl_seconds if stale_ok_ttl_seconds is not None else ttl_seconds
        )
        if current_time - loaded_at > timedelta(seconds=max_age):
            logger.warning("Ignoring stale best-config cache for %s", safe_config_id)
            return None

        snapshot = load_best_config_spec(
            _resolve_entry_path(directory, entry["path"]),
            configuration_space=configuration_space,
            expected_config_id=safe_config_id,
            manifest_entry=entry,
            source=BestConfigSource.CLOUD_CACHE.value,
            strict=True,
        )
        expires_at = loaded_at + timedelta(seconds=ttl_seconds)
        object.__setattr__(snapshot, "loaded_at", loaded_at)
        object.__setattr__(snapshot, "expires_at", expires_at)
        return snapshot
    except Exception as exc:
        if strict:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to resolve cloud cache: {exc}") from exc
        logger.warning(
            "Ignoring cloud best-config cache for %s: %s", safe_config_id, exc
        )
        return None


def _validate_config_id(config_id: Any) -> str:
    if not isinstance(config_id, str) or not config_id:
        raise ConfigurationError("Best config config_id must be a non-empty string")
    if "\x00" in config_id or "/" in config_id or "\\" in config_id:
        raise ConfigurationError(
            "Best config config_id must not contain path separators"
        )
    if config_id in {".", ".."}:
        raise ConfigurationError("Best config config_id must not be a relative path")
    return config_id


def _resolve_entry_path(directory: Path, path: Any) -> Path:
    if not isinstance(path, str) or not path:
        raise ConfigurationError("Best config manifest entry lacks path")
    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in path:
        raise ConfigurationError(
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
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ConfigurationError("Cloud cache metadata must be an object")
    metadata.setdefault("path", spec_path.name)
    return metadata


def _verify_manifest_hashes(
    manifest_entry: dict[str, Any],
    *,
    config_hash: str,
    spec_hash: str,
) -> None:
    expected_config_hash = manifest_entry.get("config_hash")
    expected_spec_hash = manifest_entry.get("spec_hash")
    if expected_config_hash and expected_config_hash != config_hash:
        raise ConfigurationError("Best config manifest config_hash mismatch")
    if expected_spec_hash and expected_spec_hash != spec_hash:
        raise ConfigurationError("Best config manifest spec_hash mismatch")


def _validate_and_filter_config_keys(
    config: dict[str, Any],
    *,
    configuration_space: dict[str, Any] | None,
    forward_compat: bool,
) -> dict[str, Any]:
    normalized = _normalize_json_value(copy.deepcopy(config))
    if not isinstance(normalized, dict):
        raise ConfigurationError("Best config must be an object")

    if not configuration_space:
        return normalized

    allowed = set(configuration_space)
    unknown = set(normalized) - allowed
    if not unknown:
        return normalized

    denylisted = sorted(key for key in unknown if _is_safety_sensitive_key(key))
    if denylisted:
        raise ConfigurationError(
            f"Unknown safety-sensitive best config keys: {denylisted}"
        )

    if not forward_compat:
        raise ConfigurationError(f"Unknown best config keys: {sorted(unknown)}")

    logger.warning(
        "Ignoring forward-compatible best config keys outside configuration_space: %s",
        sorted(unknown),
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
        if not math.isfinite(value):
            raise ConfigurationError("Best config JSON values cannot be NaN/Infinity")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)

    container_id = id(value)
    if isinstance(value, dict):
        if container_id in seen:
            raise ConfigurationError("Best config contains a circular reference")
        seen.add(container_id)
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError("Best config JSON object keys must be strings")
            normalized[unicodedata.normalize("NFC", key)] = _normalize_json_value(
                item, seen
            )
        seen.remove(container_id)
        return normalized

    if isinstance(value, list):
        if container_id in seen:
            raise ConfigurationError("Best config contains a circular reference")
        seen.add(container_id)
        normalized_list = [_normalize_json_value(item, seen) for item in value]
        seen.remove(container_id)
        return normalized_list

    raise ConfigurationError(
        f"Best config contains non-JSON-native value {type(value).__name__}"
    )


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
        raise ConfigurationError("Cloud cache metadata requires loaded_at timestamp")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
    tmp_path.replace(path)


__all__ = [
    "BEST_CONFIG_MANIFEST_SCHEMA_VERSION",
    "BEST_CONFIG_SCHEMA_VERSION",
    "BestConfigSnapshot",
    "BestConfigSource",
    "BestConfigSourceMode",
    "CloudPublishUnavailable",
    "CloudPublishUnavailableReason",
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
    "write_repo_best_config",
]
