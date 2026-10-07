"""Closed manifest parser and conservative runtime guarantee intersection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import yaml


class GuaranteeLevel(StrEnum):
    NATIVE = "native"
    EMULATED = "emulated"
    NONE = "none"


class ConditionalUpdate(StrEnum):
    ATOMIC = "atomic"
    READ_CHECK = "read_check"
    NONE = "none"


class GuaranteeState(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class GuaranteeReason(StrEnum):
    MANIFEST_NONE = "manifest_none"
    PERMISSION_LIMITED = "permission_limited"
    VERSION_MISMATCH = "version_mismatch"
    PROBE_UNAVAILABLE = "probe_unavailable"
    GUARANTEE_REDUCED = "guarantee_reduced"
    NOT_DECLARED = "not_declared"


@dataclass(frozen=True, slots=True)
class OperationManifest:
    idempotency: GuaranteeLevel = GuaranteeLevel.NONE
    conditional_update: ConditionalUpdate = ConditionalUpdate.NONE
    limits: Mapping[str, int] | None = None
    guarantees: Mapping[str, GuaranteeLevel] | None = None


@dataclass(frozen=True, slots=True)
class ConnectorManifest:
    schema_version: str
    connector: str
    version: str
    operations: Mapping[str, OperationManifest]


@dataclass(frozen=True, slots=True)
class GuaranteeResult:
    state: GuaranteeState
    reason: GuaranteeReason | None = None
    guarantees: Mapping[str, str] | None = None


_TOP_KEYS = {"schema_version", "connector", "version", "operations"}
_OP_KEYS = {"idempotency", "conditional_update", "limits", "guarantees"}
_LEVELS = {v.value: v for v in GuaranteeLevel}
_CONDITIONAL = {v.value: v for v in ConditionalUpdate}


def _closed_dict(value: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) - allowed:
        raise ValueError(f"invalid closed manifest object: {where}")
    return value


def load_manifest(source: str | Path | Mapping[str, Any]) -> ConnectorManifest:
    if isinstance(source, Mapping):
        raw: Any = dict(source)
    elif isinstance(source, Path):
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        candidate = Path(source)
        if "\n" not in source and candidate.is_file():
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        else:
            raw = (
                json.loads(source)
                if source.lstrip().startswith(("{", "["))
                else yaml.safe_load(source)
            )
    else:
        raise TypeError("manifest source must be a mapping, path, or YAML/JSON string")
    obj = _closed_dict(raw, _TOP_KEYS, "root")
    if set(obj) != _TOP_KEYS or obj["schema_version"] != "1":
        raise ValueError("manifest requires schema_version 1 and all root fields")
    if any(type(obj[k]) is not str or not obj[k] for k in ("connector", "version")):
        raise ValueError("connector and version must be nonempty strings")
    if type(obj["operations"]) is not dict or not obj["operations"]:
        raise ValueError("manifest operations must be a nonempty mapping")
    operations: dict[str, OperationManifest] = {}
    for name, values in obj["operations"].items():
        if type(name) is not str or not name:
            raise ValueError("operation names must be nonempty strings")
        op = _closed_dict(values, _OP_KEYS, f"operations.{name}")
        idem = op.get("idempotency", "none")
        conditional = op.get("conditional_update", "none")
        if idem not in _LEVELS or conditional not in _CONDITIONAL:
            raise ValueError(f"unknown guarantee value for operation {name}")
        limits = op.get("limits", {})
        if type(limits) is not dict or any(
            type(k) is not str or type(v) is not int or v < 0 for k, v in limits.items()
        ):
            raise ValueError(f"invalid limits for operation {name}")
        guarantees = op.get("guarantees", {})
        if type(guarantees) is not dict:
            raise ValueError(f"invalid guarantees for operation {name}")
        typed: dict[str, GuaranteeLevel] = {}
        for key, value in guarantees.items():
            if type(key) is not str or key in _OP_KEYS:
                raise ValueError(f"reserved guarantee key for operation {name}")
            if type(value) is not str or value not in _LEVELS:
                raise ValueError(f"unknown guarantee value for operation {name}")
            typed[key] = _LEVELS[value]
        operations[name] = OperationManifest(
            _LEVELS[idem], _CONDITIONAL[conditional], dict(limits), typed
        )
    return ConnectorManifest("1", obj["connector"], obj["version"], operations)


def _ordinal(value: str) -> int:
    if value == "none":
        return 0
    if value in {"emulated", "read_check"}:
        return 1
    return 2


def _allowed_values(key: str) -> Mapping[str, StrEnum]:
    return _CONDITIONAL if key == "conditional_update" else _LEVELS


def resolve_guarantee(
    manifest: ConnectorManifest,
    operation: str,
    permissions: Mapping[str, str],
    probed_version: str | None,
    required: Mapping[str, str] | None = None,
) -> GuaranteeResult:
    """Intersect declaration and probe; a probe can only lower guarantees."""
    if probed_version is None:
        return GuaranteeResult(
            GuaranteeState.UNAVAILABLE, GuaranteeReason.PROBE_UNAVAILABLE
        )
    if probed_version != manifest.version:
        return GuaranteeResult(
            GuaranteeState.UNAVAILABLE, GuaranteeReason.VERSION_MISMATCH
        )
    op = manifest.operations.get(operation)
    if op is None:
        return GuaranteeResult(
            GuaranteeState.UNAVAILABLE, GuaranteeReason.MANIFEST_NONE
        )
    declared = {
        "idempotency": op.idempotency.value,
        "conditional_update": op.conditional_update.value,
    }
    declared.update({k: v.value for k, v in (op.guarantees or {}).items()})
    for key, requirement in (required or {}).items():
        allowed = _allowed_values(key)
        if (
            type(key) is not str
            or type(requirement) is not str
            or key not in declared
            or requirement not in allowed
        ):
            return GuaranteeResult(
                GuaranteeState.UNAVAILABLE, GuaranteeReason.NOT_DECLARED, declared
            )
    final: dict[str, str] = {}
    reduced = False
    for key, value in declared.items():
        perm = permissions.get(key, "none")
        req = (required or {}).get(key, "none")
        allowed = _allowed_values(key)
        if perm not in allowed:
            perm = "none"
        actual = min(_ordinal(value), _ordinal(perm))
        final[key] = (
            ("none", "emulated", "native")[actual]
            if key == "idempotency" or key not in {"conditional_update"}
            else ("none", "read_check", "atomic")[actual]
        )
        if _ordinal(req) > actual:
            reduced = True
    if all(v == "none" for v in final.values()):
        reason = (
            GuaranteeReason.MANIFEST_NONE
            if all(value == "none" for value in declared.values())
            else GuaranteeReason.PERMISSION_LIMITED
        )
        return GuaranteeResult(GuaranteeState.UNAVAILABLE, reason, final)
    if reduced:
        return GuaranteeResult(
            GuaranteeState.DEGRADED, GuaranteeReason.PERMISSION_LIMITED, final
        )
    if any(_ordinal(v) < _ordinal(declared[k]) for k, v in final.items()):
        return GuaranteeResult(
            GuaranteeState.DEGRADED, GuaranteeReason.GUARANTEE_REDUCED, final
        )
    return GuaranteeResult(GuaranteeState.AVAILABLE, guarantees=final)
