"""Static, no-import extraction of a cold-start system specification."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ColdStartConfigurationError,
    ColdStartInputContractError,
    ColdStartOptions,
    DiscoveryGap,
    FileDigest,
    ParameterSpec,
    SystemSpec,
)


_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "site-packages",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
    }
)
_SUPPORTED_INPUT_ANNOTATIONS = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "optional[str]",
        "optional[int]",
        "optional[float]",
        "optional[bool]",
        "str|none",
        "none|str",
        "int|none",
        "none|int",
        "float|none",
        "none|float",
        "bool|none",
        "none|bool",
    }
)


@dataclass(frozen=True, slots=True)
class _SelectedFiles:
    """The bounded, non-sensitive outcome of repository file selection."""

    files: tuple[FileDigest, ...]
    inspection_truncated: bool
    skipped_file_count: int


def _target_callable(func: Any) -> Any:
    """Return an optimized wrapper's stored callable without invoking either one."""
    return getattr(func, "func", func)


def _relative_source_path(source_path: Path, repo_root: Path) -> Path:
    try:
        return source_path.resolve().relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise ColdStartConfigurationError(
            "The callable source file must be inside repo_root."
        ) from exc


def _matches_include_glob(relative_path: Path, pattern: str) -> bool:
    """Match a repository-relative path with ``Path.rglob``-compatible defaults."""
    return relative_path.match(pattern) or (
        pattern.startswith("**/") and relative_path.match(pattern.removeprefix("**/"))
    )


def _candidate_paths(repo_root: Path, options: ColdStartOptions) -> tuple[Path, ...]:
    """Collect unique matching files while pruning known non-source directories."""
    candidates: set[Path] = set()
    for current_root, directory_names, file_names in os.walk(
        repo_root, followlinks=False
    ):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in _EXCLUDED_DIRECTORY_NAMES
            and not (Path(current_root) / name).is_symlink()
        ]
        current = Path(current_root)
        for name in file_names:
            path = current / name
            try:
                relative_path = path.relative_to(repo_root)
            except ValueError:
                continue
            if any(
                _matches_include_glob(relative_path, pattern)
                for pattern in options.include_globs
            ):
                candidates.add(relative_path)
    return tuple(sorted(candidates, key=lambda path: path.as_posix()))


def _source_bytes(source_path: Path, options: ColdStartOptions) -> bytes:
    """Read a callable source only after enforcing its source-file safety bounds."""
    if source_path.is_symlink():
        raise ColdStartConfigurationError(
            "Static inspection refuses a callable source symbolic link."
        )
    try:
        source_stat = source_path.stat()
    except OSError as exc:
        raise ColdStartConfigurationError(
            "Static inspection could not stat the callable source file."
        ) from exc
    if not source_path.is_file():
        raise ColdStartConfigurationError(
            "Static inspection requires the callable source to be a regular file."
        )
    if source_stat.st_size > options.max_file_bytes:
        raise ColdStartConfigurationError(
            "The callable source file exceeds max_file_bytes."
        )
    try:
        return source_path.read_bytes()
    except OSError as exc:
        raise ColdStartConfigurationError(
            "Static inspection could not read the callable source file."
        ) from exc


def _allowlisted_files(
    repo_root: Path,
    options: ColdStartOptions,
    *,
    source_relative_path: Path,
    source_payload: bytes,
) -> _SelectedFiles:
    """Hash a source-first bounded file selection without importing its contents.

    The inspection result records every matching non-source file omitted because
    it is unsafe, unreadable, outside the repository, or beyond the configured
    selection cap. Excluded directory trees are pruned before enumeration.
    """
    candidates = _candidate_paths(repo_root, options)
    if source_relative_path not in candidates:
        raise ColdStartConfigurationError(
            "The callable source is not covered by include_globs."
        )

    files = [
        FileDigest(
            path=source_relative_path,
            sha256=hashlib.sha256(source_payload).hexdigest(),
            size_bytes=len(source_payload),
        )
    ]
    skipped_file_count = 0
    for relative_path in candidates:
        if relative_path == source_relative_path:
            continue
        path = repo_root / relative_path
        if len(files) >= options.max_files:
            skipped_file_count += 1
            continue
        try:
            if path.is_symlink():
                skipped_file_count += 1
                continue
            path.resolve().relative_to(repo_root)
            if not path.is_file():
                skipped_file_count += 1
                continue
            payload = path.read_bytes()
        except (OSError, ValueError):
            skipped_file_count += 1
            continue
        if len(payload) > options.max_file_bytes:
            skipped_file_count += 1
            continue
        files.append(
            FileDigest(
                path=relative_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    return _SelectedFiles(
        files=tuple(files),
        inspection_truncated=skipped_file_count > 0,
        skipped_file_count=skipped_file_count,
    )


def _find_function(
    tree: ast.Module, callable_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Find the named function definition without resolving or executing symbols."""
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == callable_name
    ]
    if len(matches) != 1:
        raise ColdStartConfigurationError(
            "Static inspection requires exactly one source definition for "
            f"{callable_name!r}."
        )
    return matches[0]


def _annotation(annotation: ast.expr | None) -> str | None:
    return ast.unparse(annotation) if annotation is not None else None


def _is_supported_input_annotation(annotation: str) -> bool:
    normalized = annotation.replace("typing.", "").replace(" ", "").lower()
    return normalized in _SUPPORTED_INPUT_ANNOTATIONS


def _parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ParameterSpec, ...]:
    """Extract the deliberately small, typed input contract supported in v1."""
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise ColdStartInputContractError(
            "Static cold-start input contracts do not support variadic parameters.",
            gap=DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
        )

    positional = [*node.args.posonlyargs, *node.args.args]
    required_at = len(positional) - len(node.args.defaults)
    parameters: list[ParameterSpec] = []
    for index, argument in enumerate(positional):
        annotation = _annotation(argument.annotation)
        if annotation is None:
            raise ColdStartInputContractError(
                f"Input parameter {argument.arg!r} has no type annotation."
            )
        if not _is_supported_input_annotation(annotation):
            raise ColdStartInputContractError(
                f"Input parameter {argument.arg!r} uses unsupported annotation "
                f"{annotation!r}.",
                gap=DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
            )
        parameters.append(
            ParameterSpec(
                name=argument.arg,
                annotation=annotation,
                required=index < required_at,
            )
        )
    for argument, default in zip(
        node.args.kwonlyargs, node.args.kw_defaults, strict=True
    ):
        annotation = _annotation(argument.annotation)
        if annotation is None:
            raise ColdStartInputContractError(
                f"Input parameter {argument.arg!r} has no type annotation."
            )
        if not _is_supported_input_annotation(annotation):
            raise ColdStartInputContractError(
                f"Input parameter {argument.arg!r} uses unsupported annotation "
                f"{annotation!r}.",
                gap=DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
            )
        parameters.append(
            ParameterSpec(
                name=argument.arg,
                annotation=annotation,
                required=default is None,
            )
        )
    if not parameters:
        raise ColdStartInputContractError(
            "Static cold-start input contracts require at least one parameter.",
            gap=DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
        )
    return tuple(parameters)


def _fingerprint(
    *,
    callable_name: str,
    module_name: str | None,
    parameters: tuple[ParameterSpec, ...],
    return_annotation: str | None,
    files: tuple[FileDigest, ...],
) -> str:
    """Build a stable content identity without using a callable's runtime behavior."""
    parts = [callable_name, module_name or "", return_annotation or ""]
    parts.extend(
        f"{parameter.name}:{parameter.annotation}:{parameter.required}"
        for parameter in parameters
    )
    parts.extend(
        f"{file.path.as_posix()}:{file.sha256}:{file.size_bytes}" for file in files
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def extract_system_spec(
    func: Any,
    *,
    repo_root: str | Path,
    options: ColdStartOptions | None = None,
) -> SystemSpec:
    """Extract a typed callable contract from local source without importing it.

    The callable is used solely as metadata pointing at its source file. This
    function never imports the module, invokes the callable, or evaluates its
    annotations; parsing is limited to local AST and bounded allowlisted bytes.
    """
    effective_options = options or ColdStartOptions()
    try:
        root = Path(repo_root).resolve()
    except OSError as exc:
        raise ColdStartConfigurationError("repo_root could not be resolved.") from exc
    if not root.is_dir():
        raise ColdStartConfigurationError("repo_root must be an existing directory.")

    target = _target_callable(func)
    callable_name = getattr(target, "__name__", None)
    if not isinstance(callable_name, str) or not callable_name:
        raise ColdStartConfigurationError(
            "Static inspection requires a callable with a non-empty __name__."
        )
    try:
        source_name = inspect.getsourcefile(target)
    except (OSError, TypeError) as exc:
        raise ColdStartConfigurationError(
            "Static inspection could not locate the callable source file."
        ) from exc
    if source_name is None:
        raise ColdStartConfigurationError(
            "Static inspection could not locate the callable source file."
        )
    source_path = Path(source_name)
    if source_path.is_symlink():
        raise ColdStartConfigurationError(
            "Static inspection refuses a callable source symbolic link."
        )
    relative_source = _relative_source_path(source_path, root)
    source_payload = _source_bytes(source_path, effective_options)
    try:
        source_text = source_payload.decode("utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ColdStartConfigurationError(
            "Static inspection could not parse the callable source file."
        ) from exc

    node = _find_function(tree, callable_name)
    parameters = _parameters(node)
    selected_files = _allowlisted_files(
        root,
        effective_options,
        source_relative_path=relative_source,
        source_payload=source_payload,
    )
    module_name = getattr(target, "__module__", None)
    if not isinstance(module_name, str):
        module_name = None
    return_annotation = _annotation(node.returns)
    return SystemSpec(
        callable_name=callable_name,
        module_name=module_name,
        parameters=parameters,
        return_annotation=return_annotation,
        files=selected_files.files,
        fingerprint=_fingerprint(
            callable_name=callable_name,
            module_name=module_name,
            parameters=parameters,
            return_annotation=return_annotation,
            files=selected_files.files,
        ),
        inspection_truncated=selected_files.inspection_truncated,
        skipped_file_count=selected_files.skipped_file_count,
    )


__all__ = ["extract_system_spec"]
