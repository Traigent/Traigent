"""Nominal types for values that may cross the optimization channel.

Every application-level value an optimization channel carries is one of: opaque
reference, closed enum, bounded number, digest, timestamp, verified code identifier,
or label-service ciphertext (design v2.2 section 1). These ``str``/``dict`` subclasses
make the class part of the type: a plain ``str`` is content until proven otherwise, and
construction validates (fail closed). They are additive; nothing existing is retyped.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from ._codec import unb64u

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_OPAQUE_REF = re.compile(
    rf"^(?:{_UUID}|(?:tenant|project)_[0-9a-f]{{32}}|[a-z]+ref_[0-9a-f]{{32}})$"
)
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_KEY_MATERIAL = re.compile(r"^[A-Za-z0-9_-]+$")
_MODULE = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_CODE_ID = re.compile(
    rf"^({_MODULE}):([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_<>]*)*)$"
)
_REL_PATH = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_./-]{1,255}$")

MAX_KEY_MATERIAL_LEN = 512


class ContentClassError(ValueError):
    """A value does not satisfy its nominal content class. The value is never echoed."""


class _Checked(str):
    _class_name = "value"

    @classmethod
    def _check(cls, value: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __new__(cls, value: str) -> Any:
        if not isinstance(value, str):
            raise ContentClassError(f"{cls.__name__} requires a string")
        cls._check(value)
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<{len(self)} chars>)"


class OpaqueRef(_Checked):
    """Server-issued opaque reference: UUID, ``tenant_``/``project_``, or ``<kind>ref_<32 hex>``."""

    @classmethod
    def _check(cls, value: str) -> None:
        if not _OPAQUE_REF.match(value):
            raise ContentClassError("not an opaque reference")


class Digest(_Checked):
    """Approved digest: 64 lowercase hex, optionally prefixed ``sha256:``."""

    @classmethod
    def _check(cls, value: str) -> None:
        if not _DIGEST.match(value):
            raise ContentClassError("not a sha256 digest")


class KeyMaterial(_Checked):
    """Public key, signature, salt or nonce: bounded unpadded base64url."""

    @classmethod
    def _check(cls, value: str) -> None:
        if (
            not value
            or len(value) > MAX_KEY_MATERIAL_LEN
            or not _KEY_MATERIAL.match(value)
        ):
            raise ContentClassError("not bounded base64url key material")
        unb64u(value)


class CodeIdentifier(_Checked):
    """A code identifier verified against the local code it names.

    Two forms: ``module:qualname`` (resolved by importing the module and walking the
    qualname; the resolved object's own ``__module__``/``__qualname__`` must agree) or a
    repo-relative file path (must exist under ``root``). Use :meth:`verified`; plain
    construction only checks syntax and is for values already verified elsewhere.
    """

    @classmethod
    def _check(cls, value: str) -> None:
        if not (_CODE_ID.match(value) or _REL_PATH.match(value)):
            raise ContentClassError("not a code identifier")

    @classmethod
    def verified(cls, identifier: str, *, root: Path | None = None) -> CodeIdentifier:
        cls._check(identifier)
        match = _CODE_ID.match(identifier)
        if match:
            module_name, qualname = match.groups()
            try:
                obj: Any = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - any import failure is a refusal
                raise ContentClassError(
                    "code identifier module is not importable"
                ) from exc
            for part in qualname.split("."):
                if not hasattr(obj, part):
                    raise ContentClassError("code identifier does not name local code")
                obj = getattr(obj, part)
            owner = getattr(obj, "__module__", None)
            if owner is not None and owner != module_name:
                # Re-exports are fine only if the qualname still agrees.
                if getattr(obj, "__qualname__", qualname) != qualname:
                    raise ContentClassError("code identifier does not match its target")
            return cls(identifier)
        base = (root or Path.cwd()).resolve()
        target = (base / identifier).resolve()
        if base != target and base not in target.parents:
            raise ContentClassError("code identifier escapes the repository")
        if not target.is_file():
            raise ContentClassError("code identifier path does not exist")
        return cls(identifier)

    @classmethod
    def of(cls, obj: Any) -> CodeIdentifier:
        """Identifier of a local function or class (``module:qualname``), verified."""
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if (
            not module
            or not qualname
            or "<locals>" in qualname
            or "<lambda>" in qualname
        ):
            raise ContentClassError("object has no stable code identifier")
        return cls.verified(f"{module}:{qualname}")


class EncryptedBlob(dict):  # type: ignore[type-arg]
    """A label envelope: ``{version, key_epoch, enc, ciphertext}`` (label_envelope_schema)."""

    _KEYS = frozenset({"version", "key_epoch", "enc", "ciphertext"})

    def __init__(self, envelope: dict[str, Any]) -> None:
        if not isinstance(envelope, dict) or set(envelope) != self._KEYS:
            raise ContentClassError("not a label envelope")
        if envelope["version"] != 1 or not isinstance(envelope["key_epoch"], int):
            raise ContentClassError("not a label envelope")
        if envelope["key_epoch"] < 1 or isinstance(envelope["key_epoch"], bool):
            raise ContentClassError("not a label envelope")
        if len(unb64u(envelope["enc"])) != 32:
            raise ContentClassError("bad envelope key share")
        if len(unb64u(envelope["ciphertext"])) not in (48, 80, 144, 272):
            raise ContentClassError("ciphertext is not a padding bucket")
        super().__init__(envelope)

    def __repr__(self) -> str:
        return f"EncryptedBlob(epoch={self['key_epoch']})"


__all__ = [
    "CodeIdentifier",
    "ContentClassError",
    "Digest",
    "EncryptedBlob",
    "KeyMaterial",
    "OpaqueRef",
]
