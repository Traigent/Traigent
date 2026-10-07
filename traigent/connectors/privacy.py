"""Opaque customer-side identities and closed summary validation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
import re
import secrets
from collections.abc import Mapping
from typing import Any, cast

from jsonschema import Draft7Validator

from .models import ConnectionRef, VerifiedCodeFact, serialize_locator

_SCHEMA_DIR = Path(__file__).with_name("schemas")
_SCHEMA_FILES = {
    "connector_run_summary.json": frozenset(
        {
            "run_token",
            "connector_kind",
            "connection_token",
            "command",
            "status",
            "counts",
            "guarantees",
        }
    ),
    "dataset_revision_summary.json": frozenset(
        {
            "dataset_token",
            "source_connector_kind",
            "revision",
            "sampling_policy",
            "score_semantics",
        }
    ),
    "correlation_summary.json": frozenset(
        {"run_token", "tier_counts", "trials_total", "trials_linked", "trials_unknown"}
    ),
}
_SCHEMAS: dict[str, dict[str, Any]] = {}
_TOKEN_RE = re.compile(r"^tk_[0-9a-hjkmnp-tv-z]{26}$")
_MINT_CAPABILITY = object()
_LOCATOR_FIELDS = ("agent_function_ref", "agent_file_path")
_TIMESTAMP_FIELDS = frozenset({"started_at", "finished_at", "approval_at"})
_SAFE_POINTER_SEGMENTS = frozenset(
    {
        "agent_file_path",
        "agent_function_ref",
        "ambiguous",
        "approval_at",
        "approved",
        "command",
        "commit_name",
        "connection_token",
        "connector_kind",
        "counts",
        "dataset_token",
        "deployment",
        "direction",
        "error_code",
        "exact",
        "finished_at",
        "fraction",
        "guarantees",
        "holdout_count",
        "item_count",
        "items_written",
        "kind",
        "name_only",
        "observations_read",
        "operation",
        "pages",
        "reason",
        "revision",
        "rows_dropped_invalid",
        "run_token",
        "sample_size",
        "sampling_policy",
        "schema_version",
        "score_semantics",
        "score_token",
        "scores_read",
        "seed",
        "source_connector_kind",
        "started_at",
        "status",
        "support",
        "tier_counts",
        "trials_linked",
        "trials_total",
        "trials_unknown",
        "type",
    }
)


class SummaryValidationError(ValueError):
    """A schema violation with JSON Pointers only, never invalid values."""


class OpaqueToken:
    __slots__ = ("_value", "_connection_id")

    def __init__(
        self, value: str, capability: object = None, connection_id: str = ""
    ) -> None:
        if capability is not _MINT_CAPABILITY or not _TOKEN_RE.fullmatch(value):
            raise ValueError("opaque tokens can only be minted by CustomerSideMinter")
        self._value = value
        self._connection_id = connection_id

    @property
    def value(self) -> str:
        return self._value


def _schema_for(payload: Mapping[str, Any]) -> str:
    for name, discriminators in _SCHEMA_FILES.items():
        if discriminators.issubset(payload):
            return name
    raise SummaryValidationError("/")


def _pointer(path: Any) -> str:
    parts = []
    for part in path:
        if type(part) is int:
            parts.append(str(part))
        elif type(part) is str and part in _SAFE_POINTER_SEGMENTS:
            parts.append(part)
        else:
            parts.append("<unknown>")
    return "/" + "/".join(parts) if parts else "/"


def _error_pointer(error: Any) -> str:
    path = tuple(error.absolute_path)
    if error.validator == "additionalProperties":
        allowed = error.schema.get("properties", {})
        if any(key not in allowed for key in error.instance):
            path += ("<unknown>",)
    return _pointer(path)


def _schema(name: str) -> dict[str, Any]:
    if name not in _SCHEMAS:
        with (_SCHEMA_DIR / name).open(encoding="utf-8") as handle:
            _SCHEMAS[name] = json.load(handle)
        Draft7Validator.check_schema(_SCHEMAS[name])
    return _SCHEMAS[name]


def _normalize_json_primitive(value: Any, path: tuple[Any, ...] = ()) -> Any:
    """Return exact JSON primitives, rejecting subclasses and non-finite numbers."""
    value_type = type(value)
    if value_type in {str, int, bool, type(None)}:
        return value
    if value_type is float:
        if math.isfinite(value):
            return value
        raise SummaryValidationError(_pointer(path)) from None
    if value_type is list:
        return [
            _normalize_json_primitive(item, path + (index,))
            for index, item in enumerate(value)
        ]
    if value_type is dict:
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SummaryValidationError(_pointer(path + ("<unknown>",))) from None
            normalized[key] = _normalize_json_primitive(item, path + (key,))
        return normalized
    raise SummaryValidationError(_pointer(path)) from None


def _validate_timestamps(payload: dict[str, Any]) -> None:
    def visit(value: Any, path: tuple[Any, ...] = ()) -> None:
        if type(value) is dict:
            for key, item in value.items():
                item_path = path + (key,)
                if key in _TIMESTAMP_FIELDS and type(item) is str:
                    try:
                        parsed = datetime.fromisoformat(item)
                    except ValueError:
                        raise SummaryValidationError(_pointer(item_path)) from None
                    if parsed.tzinfo is not UTC or parsed.utcoffset() is None:
                        raise SummaryValidationError(_pointer(item_path)) from None
                visit(item, item_path)
        elif type(value) is list:
            for index, item in enumerate(value):
                visit(item, path + (index,))

    visit(payload)


def validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one of the three closed summary shapes using exact JSON primitives."""
    if type(payload) is not dict:
        raise SummaryValidationError("/")
    normalized = _normalize_json_primitive(payload)
    name = _schema_for(normalized)
    validator = Draft7Validator(
        _schema(name), format_checker=Draft7Validator.FORMAT_CHECKER
    )
    errors = sorted(
        validator.iter_errors(normalized),
        key=lambda err: (list(map(str, err.absolute_path)), err.validator or ""),
    )
    if errors:
        pointers = sorted({_error_pointer(err) for err in errors})
        raise SummaryValidationError("invalid summary at " + ", ".join(pointers))
    _validate_timestamps(normalized)
    return cast(dict[str, Any], normalized)


class CustomerSideMinter:
    """Mints random opaque tokens and HMAC digests under a per-connection key."""

    __slots__ = ("connection", "_key", "_connection_id", "_tokens", "connection_token")

    def __init__(self, connection: ConnectionRef, key: bytes | None = None) -> None:
        if type(connection) is not ConnectionRef:
            raise TypeError("connection must be a ConnectionRef")
        actual_key = secrets.token_bytes(32) if key is None else key
        if not isinstance(actual_key, bytes) or len(actual_key) < 32:
            raise ValueError("connection key must contain at least 32 bytes")
        self.connection = connection
        self._connection_id = secrets.token_hex(16)
        self._tokens: dict[str, OpaqueToken] = {}
        self.connection_token = self._mint_token()
        self._key = hmac.new(
            actual_key, self.connection_token.value.encode("utf-8"), sha256
        ).digest()

    def _mint_token(self) -> OpaqueToken:
        alphabet = "0123456789abcdefghjkmnpqrstvwxyz"
        bits = int.from_bytes(secrets.token_bytes(16), "big")
        raw = "".join(alphabet[(bits >> (5 * index)) & 31] for index in range(26))
        token = OpaqueToken("tk_" + raw, _MINT_CAPABILITY, self._connection_id)
        self._tokens[token.value] = token
        return token

    def mint(self, kind: str, source_id: str) -> OpaqueToken:
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError("token kind and source identity must be nonempty strings")
        return self._mint_token()

    def digest(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("digest input must be a string")
        return hmac.new(self._key, value.encode("utf-8"), sha256).hexdigest()

    def owns(self, token: Any) -> bool:
        return (
            type(token) is OpaqueToken
            and token._connection_id == self._connection_id
            and self._tokens.get(token.value) is token
        )


def serialize_summary(
    payload: Mapping[str, Any],
    *,
    minter: CustomerSideMinter,
    code_fact: VerifiedCodeFact | None = None,
) -> dict[str, Any]:
    """Validate a summary and require every opaque token to be owned by minter.

    Token positions must be supplied as minted :class:`OpaqueToken` instances;
    raw strings, including syntactically valid forged tokens, are rejected.
    """
    if type(minter) is not CustomerSideMinter:
        raise TypeError("a CustomerSideMinter is required")
    token_fields = {"run_token", "dataset_token", "connection_token", "score_token"}
    if type(payload) is not dict:
        raise SummaryValidationError("/")
    result = dict(payload)
    for field in _LOCATOR_FIELDS:
        if field in result:
            raise SummaryValidationError(_pointer((field,)))
    if code_fact is not None:
        if type(code_fact) is not VerifiedCodeFact:
            raise ValueError("code location requires verified code facts")
        result.update(serialize_locator(code_fact))

    def check_shape(node: Any, path: tuple[Any, ...] = ()) -> None:
        node_type = type(node)
        if node_type is dict:
            for key, value in node.items():
                if type(key) is not str:
                    raise SummaryValidationError(
                        _pointer(path + ("<unknown>",))
                    ) from None
                check_shape(value, path + (key,))
        elif node_type is list:
            for index, value in enumerate(node):
                check_shape(value, path + (index,))
        elif node_type not in {str, int, float, bool, type(None), OpaqueToken}:
            raise SummaryValidationError(_pointer(path)) from None

    check_shape(result)

    def visit(node: Any, path: tuple[Any, ...] = ()) -> None:
        if type(node) is dict:
            for key, value in node.items():
                if key in token_fields:
                    if not minter.owns(value):
                        raise SummaryValidationError(_pointer(path + (key,)))
                else:
                    visit(value, path + (key,))
        elif type(node) is list:
            for index, value in enumerate(node):
                visit(value, path + (index,))

    visit(result)

    def wire(node: Any) -> Any:
        if type(node) is OpaqueToken:
            return node.value
        if type(node) is dict:
            return {key: wire(value) for key, value in node.items()}
        if type(node) is list:
            return [wire(value) for value in node]
        return node

    return validate_summary(_normalize_json_primitive(wire(result)))
