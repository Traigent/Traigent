"""Privacy-safe TVAR observation helpers.

TVAR observations may be sent to the Traigent backend to power optimization.
They include knob names, enum/scalar values, and numeric metrics only; set
TRAIGENT_TVAR_OBSERVATION=off to disable them, or use the default hashed mode
to hash free-form string values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

_OBSERVATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "config_generator"
    / "catalog"
    / "schemas"
    / "tvar_observation_schema.json"
)
_VALID_KINDS = frozenset({"value", "cardinality", "topology", "policy"})
_VALID_SCOPES = frozenset({"opt_mini", "heldout", "isolation", "trial"})
_OBSERVATION_MODES = frozenset({"off", "hashed"})
_ENUM_SAFE_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_VARIABLE_VALUE_HASH_SALT = "traigent:tvar_observation:v1"
_SENSITIVE_CONFIG_NAMES = frozenset(
    {
        "prompt",
        "system_prompt",
        "user_prompt",
        "raw_prompt",
        "example",
        "examples",
        "response",
        "responses",
        "content",
        "input",
        "output",
        "text",
        "query",
        "answer",
    }
)
_SENSITIVE_CONFIG_SUFFIXES = (
    "_prompt",
    "_example",
    "_response",
    "_content",
    "_input",
    "_output",
    "_text",
    "_query",
    "_answer",
)


def build_tvar_observation(
    *,
    session_id: str,
    trial_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    primary_metric: str,
    comparability: dict[str, Any],
    catalog_entry_ids: tuple[str, ...] | list[str] = (),
    agent_type: str | None = None,
    config_space_id: str | None = None,
    sdk_version: str | None = None,
    effectuation_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build a schema-valid, content-free TVAR observation payload."""
    mode = _observation_mode()
    if mode == "off":
        return None

    observation: dict[str, Any] = {
        "schema_version": "1.0.0",
        "session_id": str(session_id),
        "trial_id": str(trial_id),
        "catalog_entry_ids": [str(entry_id) for entry_id in catalog_entry_ids],
        "variables": _build_variables(config, catalog_entry_ids, mode),
        "metrics": _numeric_metrics(metrics),
        "primary_metric": str(primary_metric),
        "comparability": _coerce_comparability(comparability),
        "privacy": {"raw_content_included": False},
        "sdk_version": str(sdk_version or _sdk_version()),
    }
    if agent_type:
        observation["agent_type"] = str(agent_type)
    if config_space_id:
        observation["config_space_id"] = str(config_space_id)
    safe_effectuation_events = _safe_effectuation_events(effectuation_events)
    if safe_effectuation_events:
        observation["effectuation_events"] = safe_effectuation_events

    _validate_observation(observation)
    return observation


def merge_tvar_observation_metadata(
    metadata: dict[str, Any] | None,
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach a TVAR observation under metadata['tvar_observation_v1']."""
    merged = dict(metadata or {})
    if observation is None:
        return merged
    merged["tvar_observation_v1"] = copy.deepcopy(observation)
    return merged


def _observation_mode() -> str:
    mode = os.getenv("TRAIGENT_TVAR_OBSERVATION", "hashed").strip().lower()
    if mode in _OBSERVATION_MODES:
        return mode
    return "hashed"


def _build_variables(
    config: dict[str, Any],
    catalog_entry_ids: tuple[str, ...] | list[str],
    mode: str,
) -> list[dict[str, Any]]:
    kind_by_name = _catalog_kind_by_name(catalog_entry_ids)
    variables: list[dict[str, Any]] = []
    for name, raw_value in config.items():
        variable_name = str(name)
        if _is_sensitive_config_name(variable_name):
            continue
        kind = kind_by_name.get(variable_name)
        safe_value = _safe_variable_value(
            raw_value,
            mode=mode,
            catalog_known=kind in _VALID_KINDS,
        )
        if safe_value is None:
            continue
        variable: dict[str, Any] = {"name": variable_name, "value": safe_value}
        if kind in _VALID_KINDS:
            variable["kind"] = kind
        variables.append(variable)
    return variables


def _is_sensitive_config_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized in _SENSITIVE_CONFIG_NAMES or normalized.endswith(
        _SENSITIVE_CONFIG_SUFFIXES
    )


def _catalog_kind_by_name(
    catalog_entry_ids: tuple[str, ...] | list[str],
) -> dict[str, str]:
    if not catalog_entry_ids:
        return {}
    requested_ids = {str(entry_id) for entry_id in catalog_entry_ids}
    try:
        from traigent.config_generator.catalog import catalog_entries

        return {
            str(entry["name"]): str(entry["kind"])
            for entry in catalog_entries()
            if str(entry.get("entry_id")) in requested_ids
        }
    except Exception:
        return {}


def _safe_variable_value(
    value: Any,
    *,
    mode: str,
    catalog_known: bool = False,
) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str) and len(value) <= 512:
        if _is_enum_safe_string(value, catalog_known):
            return value
        return _hash_variable_value(value)
    return None


def _safe_effectuation_events(
    events: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str | int | float | bool]]:
    safe_events: list[dict[str, str | int | float | bool]] = []
    if not events:
        return safe_events

    for event in events:
        if not isinstance(event, Mapping):
            continue
        safe_event: dict[str, str | int | float | bool] = {}
        for raw_key, raw_value in event.items():
            key = str(raw_key)
            if not _ENUM_SAFE_VALUE_PATTERN.fullmatch(key):
                continue
            value = _safe_effectuation_value(raw_value)
            if value is not None:
                safe_event[key] = value
        if safe_event:
            safe_events.append(safe_event)
    return safe_events


def _safe_effectuation_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str) and _is_enum_safe_string(value, catalog_known=False):
        return value
    return None


def _is_enum_safe_string(value: str, catalog_known: bool) -> bool:
    if len(value) > 64:
        return False
    if catalog_known:
        return True
    return bool(_ENUM_SAFE_VALUE_PATTERN.fullmatch(value))


def _hash_variable_value(value: str) -> str:
    digest = hashlib.sha256()
    digest.update(_VARIABLE_VALUE_HASH_SALT.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    numeric: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        numeric[str(key)] = value
    return numeric


def _coerce_comparability(comparability: dict[str, Any]) -> dict[str, Any]:
    scope = str(comparability.get("scope", "trial"))
    if scope not in _VALID_SCOPES:
        scope = "trial"
    try:
        n = int(comparability.get("n", 0))
    except (TypeError, ValueError):
        n = 0
    return {"scope": scope, "n": max(0, n)}


def _sdk_version() -> str:
    try:
        from traigent._version import get_version

        return get_version()
    except Exception:
        return "unknown"


def _validate_observation(observation: dict[str, Any]) -> None:
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
    except Exception:
        _minimal_validate_observation(observation)
        return

    try:
        validate(instance=observation, schema=_observation_schema())
    except ValidationError as exc:
        raise ValueError(f"Invalid TVAR observation: {exc.message}") from exc


@lru_cache(maxsize=1)
def _observation_schema() -> dict[str, Any]:
    with _OBSERVATION_SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("TVAR observation schema must be an object")
    return schema


def _minimal_validate_observation(observation: dict[str, Any]) -> None:
    for field in ("schema_version", "session_id", "trial_id", "privacy"):
        if field not in observation:
            raise ValueError(f"TVAR observation missing required field: {field}")
    privacy = observation.get("privacy")
    if (
        not isinstance(privacy, dict)
        or privacy.get("raw_content_included") is not False
    ):
        raise ValueError("TVAR observation privacy.raw_content_included must be false")
