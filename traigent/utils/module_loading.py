"""Shared loader and discovery helpers for user modules.

``traigent check``, ``traigent optimize`` and the MCP tools all load a user's
Python file and look for ``@traigent.optimize`` functions in it. They used to
carry three diverging copies; this module is the single implementation.

Loading rules:

* A file inside a package (its directory has ``__init__.py``) is imported by its
  qualified name with the package root on ``sys.path``, so relative imports work
  exactly as they do for ``python -c "import pkg.mod"``.
* Any other file is loaded from its path. In both cases the module is registered
  in ``sys.modules`` *before* it executes (``dataclasses`` resolves string
  annotations through ``sys.modules``) and is unregistered again if execution
  fails.
"""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Collection, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_MODULE_NAME = "user_module"

_DOTTED_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
_RELATIVE_IMPORT_MARKER = "attempted relative import"


def looks_like_dotted_module(spec: str) -> bool:
    """Return True for ``pkg.mod`` style names (not file paths)."""
    return bool(_DOTTED_MODULE_RE.match(spec)) and not spec.endswith(".py")


def package_parts(path: Path) -> tuple[Path, list[str]]:
    """Return ``(package_root_dir, [package, ..., submodule])`` for ``path``.

    Walks up while the directory holds an ``__init__.py``. When ``path`` is not
    inside a package the list is empty and the root is the file's directory.
    """
    path = path.resolve()
    parts: list[str] = []
    directory = path.parent
    while (directory / "__init__.py").is_file() and directory.parent != directory:
        parts.append(directory.name)
        directory = directory.parent
    if not parts:
        return path.parent, []
    parts.reverse()
    if path.name != "__init__.py":
        parts.append(path.stem)
    return directory, parts


@contextlib.contextmanager
def _sys_path_entry(directory: Path, *, keep: bool) -> Iterator[None]:
    entry = str(directory)
    inserted = entry not in sys.path
    if inserted:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if inserted and not keep and entry in sys.path:
            sys.path.remove(entry)


def _explain_relative_import(path: Path, exc: ImportError) -> ImportError:
    if _RELATIVE_IMPORT_MARKER not in str(exc):
        return exc
    return ImportError(
        f"{path.name} uses a relative import but is not inside a Python package "
        "(its directory has no __init__.py). Add an __init__.py next to it, or "
        "use absolute imports."
    )


def resolve_dotted_module(name: str, search_dir: Path | None = None) -> Path:
    """Resolve ``pkg.mod`` to its source file, searching ``search_dir`` (cwd)."""
    base = (search_dir or Path.cwd()).resolve()
    with _sys_path_entry(base, keep=False):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
    origin = getattr(spec, "origin", None)
    if spec is None or not origin or not origin.endswith(".py"):
        raise FileNotFoundError(
            f"Module not found: {name}. Pass a path to a .py file, or a dotted "
            f"module path importable from {base}."
        )
    return Path(origin)


def load_user_module(
    path: Path,
    module_name: str = DEFAULT_MODULE_NAME,
    *,
    keep_sys_path: bool = False,
) -> ModuleType:
    """Import the Python file at ``path`` and return the module.

    Raises ``ImportError`` (or whatever the module raises) on failure; nothing
    stays registered in ``sys.modules`` when the module fails to execute.
    """
    path = path.resolve()
    root, parts = package_parts(path)
    with _sys_path_entry(root, keep=keep_sys_path):
        if parts:
            return _import_packaged(path, ".".join(parts))
        return _import_from_path(path, module_name)


def _import_packaged(path: Path, qualified_name: str) -> ModuleType:
    # Drop a stale copy of the leaf so the file is executed afresh (a package
    # ``__init__`` is left alone: its submodules are registered under it).
    if path.name != "__init__.py":
        sys.modules.pop(qualified_name, None)
    try:
        module = importlib.import_module(qualified_name)
    except ImportError as exc:
        sys.modules.pop(qualified_name, None)
        raise _explain_relative_import(path, exc) from exc
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def _import_from_path(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        _restore_module(module_name, previous)
        raise _explain_relative_import(path, exc) from exc
    except BaseException:
        _restore_module(module_name, previous)
        raise
    return module


def _restore_module(name: str, previous: ModuleType | None) -> None:
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def is_optimized_function(obj: Any) -> bool:
    """True for objects produced by ``@traigent.optimize``.

    That is an ``OptimizedFunction`` instance, or a plain function carrying a
    callable ``optimize`` attribute (the older decorated-function shape). Classes
    and other objects that merely have an ``optimize`` method are not decorated
    functions and are ignored.
    """
    from traigent.core.optimized_function import OptimizedFunction

    if isinstance(obj, OptimizedFunction):
        return True
    return inspect.isfunction(obj) and callable(getattr(obj, "optimize", None))


def find_optimized_functions(
    module: ModuleType, names: Collection[str] | None = None
) -> list[tuple[str, Any]]:
    """Return ``(name, function)`` pairs for decorated functions in ``module``.

    Aliases and re-imports of one decorated function collapse to a single entry
    (deduplicated by object identity). When several names point at one function
    the function's own name wins, then the alphabetically first. With ``names``
    given, an entry is kept only if one of its names was requested, and it is
    reported under the requested name.
    """
    groups: dict[int, list[tuple[str, Any]]] = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        if is_optimized_function(obj):
            groups.setdefault(id(obj), []).append((name, obj))

    found: list[tuple[str, Any]] = []
    for aliases in groups.values():
        obj = aliases[0][1]
        if names is not None:
            requested = [alias for alias, _ in aliases if alias in names]
            if not requested:
                continue
            found.append((requested[0], obj))
            continue
        own_name = getattr(obj, "__name__", None)
        preferred = next((alias for alias, _ in aliases if alias == own_name), None)
        found.append((preferred or aliases[0][0], obj))
    return found
