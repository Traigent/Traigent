"""Opaque customer-side identities and closed summary validation."""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
import secrets
from collections.abc import Mapping
from typing import Any

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
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in path]
    return "/" + "/".join(parts) if parts else "/"


def _schema(name: str) -> dict[str, Any]:
    if name not in _SCHEMAS:
        with (_SCHEMA_DIR / name).open(encoding="utf-8") as handle:
            _SCHEMAS[name] = json.load(handle)
        Draft7Validator.check_schema(_SCHEMAS[name])
    return _SCHEMAS[name]


def validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and shallow-copy one of the three closed summary shapes."""
    if not isinstance(payload, Mapping):
        raise SummaryValidationError("/")
    name = _schema_for(payload)
    validator = Draft7Validator(
        _schema(name), format_checker=Draft7Validator.FORMAT_CHECKER
    )
    errors = sorted(
        validator.iter_errors(dict(payload)),
        key=lambda err: (list(map(str, err.absolute_path)), err.validator or ""),
    )
    if errors:
        pointers = sorted({_pointer(err.absolute_path) for err in errors})
        raise SummaryValidationError("invalid summary at " + ", ".join(pointers))
    return dict(payload)


class CustomerSideMinter:
    """Mints random opaque tokens and HMAC digests under a per-connection key."""

    __slots__ = ("connection", "_key", "_connection_id", "_tokens", "connection_token")

    def __init__(self, connection: ConnectionRef, key: bytes | None = None) -> None:
        if not isinstance(connection, ConnectionRef):
            raise TypeError("connection must be a ConnectionRef")
        actual_key = secrets.token_bytes(32) if key is None else key
        if not isinstance(actual_key, bytes) or len(actual_key) < 32:
            raise ValueError("connection key must contain at least 32 bytes")
        self.connection = connection
        self._key = actual_key
        self._connection_id = secrets.token_hex(16)
        self._tokens: dict[str, OpaqueToken] = {}
        self.connection_token = self._mint_token()

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
            isinstance(token, OpaqueToken)
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
    if not isinstance(minter, CustomerSideMinter):
        raise TypeError("a CustomerSideMinter is required")
    token_fields = {"run_token", "dataset_token", "connection_token", "score_token"}
    result = dict(payload)
    for field in _LOCATOR_FIELDS:
        if field in result:
            raise SummaryValidationError(_pointer((field,)))
    if code_fact is not None:
        if not isinstance(code_fact, VerifiedCodeFact):
            raise ValueError("code location requires verified code facts")
        result.update(serialize_locator(code_fact))

    def visit(node: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_fields:
                    if not minter.owns(value):
                        raise SummaryValidationError(_pointer(path + (key,)))
                else:
                    visit(value, path + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, path + (str(index),))

    visit(result)

    def wire(node: Any) -> Any:
        if isinstance(node, OpaqueToken):
            return node.value
        if isinstance(node, dict):
            return {key: wire(value) for key, value in node.items()}
        if isinstance(node, list):
            return [wire(value) for value in node]
        return node

    return validate_summary(wire(result))
