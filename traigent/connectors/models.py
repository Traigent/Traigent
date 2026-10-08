"""Immutable customer-side connector records.

These records may contain source content and are intentionally local values;
the connector package defines no backend transport or exporter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import isfunction, ismethod
from pathlib import Path
import re
from types import FunctionType, MethodType
from typing import Any
from weakref import WeakValueDictionary


_VERIFIED_CODE_FACTS: WeakValueDictionary[int, VerifiedCodeFact] = WeakValueDictionary()
_TOKEN_RE = re.compile(r"^tk_[0-9a-hjkmnp-tv-z]{26}$")


def _content_length(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


@dataclass(frozen=True, slots=True, repr=False)
class Observation:
    input: Any
    output: Any
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            "Observation("
            f"input_length={_content_length(self.input)}, "
            f"output_length={_content_length(self.output)}, "
            f"metadata_length={_content_length(self.metadata)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class Score:
    name: str
    value: float | bool | str
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            "Score("
            f"name_length={_content_length(self.name)}, "
            f"value_length={_content_length(self.value)}, "
            f"metadata_length={_content_length(self.metadata)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class DatasetItem:
    input: Any
    expected_output: Any
    scores: tuple[Score, ...] = ()

    def __repr__(self) -> str:
        return (
            "DatasetItem("
            f"input_length={_content_length(self.input)}, "
            f"expected_output_length={_content_length(self.expected_output)}, "
            f"scores_length={_content_length(self.scores)})"
        )


@dataclass(frozen=True, slots=True)
class ExternalRef:
    kind: str
    identifier: str
    connection_token: str
    platform_id: str

    def __post_init__(self) -> None:
        if type(self.connection_token) is not str or not _TOKEN_RE.fullmatch(
            self.connection_token
        ):
            raise ValueError("external reference requires a validated connection token")
        if type(self.platform_id) is not str:
            raise TypeError("platform identifier must be an opaque string")


@dataclass(frozen=True, slots=True)
class ConnectionRef:
    kind: str


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class VerifiedCodeFact:
    """Code location derived from a live Python object, not arbitrary text."""

    function_ref: str
    file_path: str

    @classmethod
    def from_callable(
        cls, value: Any, *, repository_root: str | Path
    ) -> VerifiedCodeFact:
        if not (
            isfunction(value)
            or ismethod(value)
            or isinstance(value, (FunctionType, MethodType))
        ):
            raise ValueError("code fact requires a Python callable")
        code = value.__code__
        module = value.__module__
        qualname = value.__qualname__
        function_ref = f"{module}:{qualname}"
        try:
            if not isinstance(repository_root, (str, Path)) or not repository_root:
                raise ValueError("code fact requires a declared repository root")
            root = Path(repository_root).resolve(strict=True)
            source = Path(code.co_filename).resolve(strict=True)
            if not root.is_dir() or not source.is_file():
                raise ValueError("callable source must exist inside repository root")
            file_path = source.relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError):
            raise ValueError(
                "callable source must exist inside repository root"
            ) from None
        fact = object.__new__(cls)
        object.__setattr__(fact, "function_ref", function_ref)
        object.__setattr__(fact, "file_path", file_path)
        _validate_locator(function_ref, file_path)
        _VERIFIED_CODE_FACTS[id(fact)] = fact
        return fact


_FUNCTION_REF = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?::(?:<locals>|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:<locals>|[A-Za-z_][A-Za-z0-9_]*))*)?$"
)
_FILE_PATH = re.compile(
    r"^(?:(?!\.{1,2}/)[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|json|yaml|yml)$"
)


def _validate_locator(function_ref: str, file_path: str) -> None:
    if len(function_ref) > 512 or not _FUNCTION_REF.fullmatch(function_ref):
        raise ValueError("invalid code function reference")
    if len(file_path) > 512 or not _FILE_PATH.fullmatch(file_path):
        raise ValueError("invalid repository-relative code path")


def serialize_locator(value: VerifiedCodeFact) -> dict[str, str]:
    if (
        not isinstance(value, VerifiedCodeFact)
        or _VERIFIED_CODE_FACTS.get(id(value)) is not value
    ):
        raise ValueError("locator requires a verified code fact")
    _validate_locator(value.function_ref, value.file_path)
    return {
        "agent_function_ref": value.function_ref,
        "agent_file_path": value.file_path,
    }


def validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a closed connector summary without importing privacy at startup."""
    from .privacy import validate_summary as _validate_summary

    return _validate_summary(payload)
