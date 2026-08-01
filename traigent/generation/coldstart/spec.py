"""Static, no-import extraction of a cold-start system specification."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from typing import Any

from .contracts import (
    ColdStartConfigurationError,
    ColdStartOptions,
    FileDigest,
    ParameterSpec,
    SystemSpec,
)


def _target_callable(func: Any) -> Any:
    """Return an optimized wrapper's stored callable without invoking either one."""
    return getattr(func, "func", func)


def _relative_source_path(source_path: Path, repo_root: Path) -> Path:
    try:
        return source_path.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise ColdStartConfigurationError(
            "The callable source file must be inside repo_root."
        ) from exc


def _allowlisted_files(
    repo_root: Path, options: ColdStartOptions
) -> tuple[FileDigest, ...]:
    """Hash selected local files without importing or executing their contents."""
    selected: set[Path] = set()
    for pattern in options.include_globs:
        selected.update(path for path in repo_root.rglob(pattern) if path.is_file())

    files: list[FileDigest] = []
    for path in sorted(selected):
        try:
            relative = path.resolve().relative_to(repo_root)
        except ValueError:
            continue
        if len(files) >= options.max_files:
            raise ColdStartConfigurationError(
                f"Static inspection exceeds max_files={options.max_files}."
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ColdStartConfigurationError(
                f"Could not read allowlisted file {relative.as_posix()!r}."
            ) from exc
        if len(payload) > options.max_file_bytes:
            raise ColdStartConfigurationError(
                f"Allowlisted file {relative.as_posix()!r} exceeds max_file_bytes."
            )
        files.append(
            FileDigest(
                path=relative,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    return tuple(files)


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


def _parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ParameterSpec, ...]:
    """Extract ordinary typed inputs; variadics are deliberately unsupported."""
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise ColdStartConfigurationError(
            "Static cold-start input contracts do not support variadic parameters."
        )

    positional = [*node.args.posonlyargs, *node.args.args]
    required_at = len(positional) - len(node.args.defaults)
    parameters: list[ParameterSpec] = []
    for index, argument in enumerate(positional):
        annotation = _annotation(argument.annotation)
        if annotation is None:
            raise ColdStartConfigurationError(
                f"Input parameter {argument.arg!r} has no type annotation."
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
            raise ColdStartConfigurationError(
                f"Input parameter {argument.arg!r} has no type annotation."
            )
        parameters.append(
            ParameterSpec(
                name=argument.arg,
                annotation=annotation,
                required=default is None,
            )
        )
    if not parameters:
        raise ColdStartConfigurationError(
            "Static cold-start input contracts require at least one parameter."
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

    The callable is used solely as metadata pointing at its source file.  This
    function never imports the module, invokes the callable, or evaluates its
    annotations; parsing is limited to the local AST and allowlisted file bytes.
    """
    effective_options = options or ColdStartOptions()
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ColdStartConfigurationError("repo_root must be an existing directory.")

    target = _target_callable(func)
    callable_name = getattr(target, "__name__", None)
    if not isinstance(callable_name, str) or not callable_name:
        raise ColdStartConfigurationError(
            "Static inspection requires a callable with a non-empty __name__."
        )
    source_name = inspect.getsourcefile(target)
    if source_name is None:
        raise ColdStartConfigurationError(
            "Static inspection could not locate the callable source file."
        )
    source_path = Path(source_name)
    relative_source = _relative_source_path(source_path, root)
    try:
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ColdStartConfigurationError(
            "Static inspection could not parse the callable source file."
        ) from exc

    node = _find_function(tree, callable_name)
    parameters = _parameters(node)
    files = _allowlisted_files(root, effective_options)
    if relative_source not in {file.path for file in files}:
        raise ColdStartConfigurationError(
            "The callable source is not covered by include_globs."
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
        files=files,
        fingerprint=_fingerprint(
            callable_name=callable_name,
            module_name=module_name,
            parameters=parameters,
            return_annotation=return_annotation,
            files=files,
        ),
    )


__all__ = ["extract_system_spec"]
