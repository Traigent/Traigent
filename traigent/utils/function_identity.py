"""Utilities for deriving stable identifiers for user functions."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import hashlib
import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "FunctionDescriptor",
    "resolve_function_descriptor",
    "sanitize_identifier",
]


@dataclass(frozen=True)
class FunctionDescriptor:
    """Describes a callable's identity for tracking and persistence."""

    identifier: str
    """Fully-qualified identifier combining relative path, module, and qualname."""

    display_name: str
    """Human-friendly name (typically the callable's ``__qualname__``)."""

    module: str
    """Module path where the callable is defined."""

    relative_path: str
    """Best-effort relative path to the source file, using forward slashes."""

    slug: str
    """Filesystem-friendly slug derived from the identifier (includes hash)."""


_IDENTIFIER_SANITIZER = re.compile(r"[^0-9A-Za-z]+")
_MODULE_SANITIZER = re.compile(r"[^0-9A-Za-z_\.]+")


def sanitize_identifier(value: str, *, max_length: int = 120) -> str:
    """Convert arbitrary identifier text into a filesystem-friendly slug.

    Args:
        value: Original identifier string.
        max_length: Maximum length of the sanitized portion (hash suffix excluded).

    Returns:
        A lowercase slug safe for filenames/lock names.
    """
    if not value:
        return "unnamed"

    normalized = _IDENTIFIER_SANITIZER.sub("_", value).strip("_").lower()
    if not normalized:
        normalized = "identifier"

    digest = hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:8]

    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("_")

    if normalized.endswith(digest):
        return normalized

    if normalized:
        return f"{normalized}_{digest}"
    return digest


def resolve_function_descriptor(
    func: Callable[..., Any],
    *,
    base_dir: str | Path | None = None,
) -> FunctionDescriptor:
    """Derive a stable descriptor for a callable.

    The identifier combines the relative source path, module, and qualname to
    distinguish similarly named callables defined in different modules.

    Args:
        func: Callable to analyse.
        base_dir: Optional base directory for relative path computation. If not
            provided, the current working directory is used as the primary base.

    Returns:
        ``FunctionDescriptor`` capturing identifier, display name, and slug.
    """
    original = inspect.unwrap(func)

    module_name = getattr(original, "__module__", "unknown_module")
    qualname = getattr(
        original, "__qualname__", getattr(original, "__name__", "unknown")
    )

    display_name_obj = getattr(original, "__qualname__", None) or getattr(
        original, "__name__", None
    )
    display_name = (
        display_name_obj if isinstance(display_name_obj, str) else str(original)
    )

    source_path = _get_source_path(original) or _get_source_path(func)
    relative_path = (
        _to_relative_path(source_path, base_dir)
        if source_path is not None
        else module_name.replace(".", "/")
    )

    module_component = _derive_module_component(module_name, relative_path)

    identifier = _build_identifier(relative_path, module_component, qualname)
    slug_source = f"{relative_path}|{module_component}|{qualname}"
    slug = sanitize_identifier(slug_source)

    return FunctionDescriptor(
        identifier=identifier,
        display_name=display_name,
        module=module_component,
        relative_path=relative_path,
        slug=slug,
    )


def _get_source_path(func: Callable[..., Any]) -> Path | None:
    try:
        filename = inspect.getsourcefile(func) or inspect.getfile(func)
    except (TypeError, OSError):
        filename = None

    if not filename:
        return None

    try:
        return Path(filename).resolve()
    except OSError:
        return None


def _to_relative_path(path: Path | None, base_dir: str | Path | None) -> str:
    if path is None:
        return "unknown_path"

    candidates: list[Path] = []
    if base_dir is not None:
        try:
            candidates.append(Path(base_dir).resolve())
        except OSError:
            pass

    try:
        candidates.append(Path.cwd().resolve())
    except OSError:
        pass

    # Include the package root containing this module as a fallback
    try:
        package_root = Path(__file__).resolve().parents[1]
        candidates.append(package_root)
    except (OSError, IndexError):
        pass

    for candidate in candidates:
        try:
            relative = path.relative_to(candidate)
            return relative.as_posix()
        except ValueError:
            continue

    return path.as_posix()


def _build_identifier(relative_path: str, module_name: str, qualname: str) -> str:
    # Use underscores to match expected ``relative_path_module_class_function`` style
    relative_component = relative_path.replace(os.sep, "/")
    module_component = module_name
    qual_component = qualname.replace(".", "_")

    parts = [relative_component, module_component, qual_component]
    return "_".join(part for part in parts if part)


def _derive_module_component(module_name: str, relative_path: str) -> str:
    if module_name and module_name not in {"__main__", "<module>"}:
        return module_name

    candidate = relative_path
    if candidate.endswith(".py") or candidate.endswith(".pyc"):
        candidate = candidate.rsplit(".", 1)[0]
    candidate = candidate.replace(os.sep, ".")
    candidate = _MODULE_SANITIZER.sub("_", candidate)
    candidate = candidate.strip(".")

    return candidate or (module_name or "unknown_module")
