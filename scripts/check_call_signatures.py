#!/usr/bin/env python3
"""General-purpose call signature mismatch detector.

Finds mismatches between function calls and their definitions.
Sound but incomplete: only flags what can be verified statically.

Usage:
    python scripts/check_call_signatures.py [path]
    python scripts/check_call_signatures.py traigent/ --json
    python scripts/check_call_signatures.py --strict  # Exit 1 on ERROR

Examples:
    # Check entire codebase
    python scripts/check_call_signatures.py

    # Check specific directory
    python scripts/check_call_signatures.py traigent/core/

    # JSON output for CI
    python scripts/check_call_signatures.py --json --strict

    # Show all findings including INFO
    python scripts/check_call_signatures.py --min-severity info

    # Exclude directories
    python scripts/check_call_signatures.py --exclude .venv --exclude build

Suppression:
    Add '# noqa: sigcheck' to suppress a specific line:
        result = func(wrong_args)  # noqa: sigcheck

    Or use '# noqa' to suppress all checks on that line.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path

# Pattern to match noqa comments
NOQA_PATTERN = re.compile(r"#\s*noqa(?::\s*sigcheck)?", re.IGNORECASE)


class Severity(IntEnum):
    """Severity levels for findings."""

    INFO = 0  # Cannot resolve / unverifiable
    WARNING = 1  # Suspicious (decorated, heuristic)
    ERROR = 2  # Definite mismatch


@dataclass
class ParamInfo:
    """Parsed parameter information from ast.arguments."""

    names: list[str]  # All named params (posonly + regular + kwonly)
    posonly_count: int  # Number of positional-only params
    required_positional: int  # Required positional args (before defaults)
    kwonly_names: set[str]  # Keyword-only param names
    kwonly_required: set[str]  # Required keyword-only (no default)
    has_var_positional: bool  # *args present
    has_var_keyword: bool  # **kwargs present

    @classmethod
    def from_ast_arguments(
        cls, args: ast.arguments, skip_first: bool = False
    ) -> ParamInfo:
        """Parse ast.arguments into ParamInfo.

        Args:
            args: The ast.arguments node
            skip_first: If True, skip first param (for self/cls in methods)
        """
        # Collect all positional params (posonly + regular)
        posonly = [a.arg for a in args.posonlyargs]
        regular = [a.arg for a in args.args]

        if skip_first:
            if posonly:
                posonly = posonly[1:]
            elif regular:
                regular = regular[1:]

        all_positional = posonly + regular
        posonly_count = len(posonly)

        # Calculate required positional count
        # defaults apply to rightmost regular args
        num_defaults = len(args.defaults)
        total_positional = len(all_positional)
        required_positional = total_positional - num_defaults

        # Keyword-only params
        kwonly_names = {a.arg for a in args.kwonlyargs}

        # kw_defaults has same length as kwonlyargs, None means no default
        kwonly_required = set()
        for i, kwarg in enumerate(args.kwonlyargs):
            if i < len(args.kw_defaults) and args.kw_defaults[i] is None:
                kwonly_required.add(kwarg.arg)
            elif i >= len(args.kw_defaults):
                kwonly_required.add(kwarg.arg)

        return cls(
            names=all_positional + list(kwonly_names),
            posonly_count=posonly_count,
            required_positional=max(0, required_positional),
            kwonly_names=kwonly_names,
            kwonly_required=kwonly_required,
            has_var_positional=args.vararg is not None,
            has_var_keyword=args.kwarg is not None,
        )


@dataclass
class FunctionDef:
    """Indexed function/method definition."""

    module_path: str  # e.g., "traigent.core.foo"
    qualname: str  # e.g., "MyClass.method" or just "func"
    file: str  # File path as string for JSON serialization
    line: int
    params: ParamInfo
    decorators: list[str]  # ["staticmethod", "dataclass", ...]
    is_method: bool  # Inside a class
    has_explicit_init: bool = True  # For classes, whether __init__ exists


@dataclass
class CallSite:
    """A function/method call in source code."""

    file: str  # File path as string
    line: int
    col: int
    callee_expr: str  # Raw expression, e.g., "module.func"
    positional_count: int  # Number of positional args
    keyword_names: set[str]  # Names of keyword args
    has_star_args: bool  # Call uses *args
    has_star_kwargs: bool  # Call uses **kwargs


@dataclass
class Mismatch:
    """A detected signature mismatch."""

    severity: Severity
    file: str
    line: int
    callee: str
    message: str
    definition_file: str | None = None
    definition_line: int | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "severity": self.severity.name,
            "file": self.file,
            "line": self.line,
            "callee": self.callee,
            "message": self.message,
            "definition_file": self.definition_file,
            "definition_line": self.definition_line,
        }


@dataclass
class ImportMap:
    """Per-file import alias resolution."""

    aliases: dict[str, str] = field(default_factory=dict)  # local_name -> module.path
    file_module: str = ""  # Module path of the file itself

    def add_import(self, node: ast.Import) -> None:
        """Process 'import x' or 'import x as y'."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.aliases[name] = alias.name

    def add_import_from(self, node: ast.ImportFrom) -> None:
        """Process 'from x import y' or 'from . import y'."""
        if node.module is None:
            # from . import y
            base_module = self._resolve_relative(node.level)
        else:
            if node.level > 0:
                # from .x import y or from ..x import y
                base_module = self._resolve_relative(node.level)
                if base_module:
                    base_module = f"{base_module}.{node.module}"
                else:
                    base_module = node.module
            else:
                base_module = node.module

        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if alias.name == "*":
                continue  # Can't track star imports
            self.aliases[name] = f"{base_module}.{alias.name}"

    def _resolve_relative(self, level: int) -> str:
        """Resolve relative import level to parent module."""
        if not self.file_module:
            return ""
        parts = self.file_module.split(".")
        if level > len(parts):
            return ""
        return ".".join(parts[:-level]) if level > 0 else self.file_module

    def resolve(self, name: str) -> str | None:
        """Resolve a local name to its full module path."""
        # Direct alias lookup
        if name in self.aliases:
            return self.aliases[name]

        # Try dotted name (module.func)
        if "." in name:
            parts = name.split(".", 1)
            if parts[0] in self.aliases:
                return f"{self.aliases[parts[0]]}.{parts[1]}"

        return None


class SymbolTable:
    """Index of all function/class definitions keyed by (module_path, qualname)."""

    def __init__(self):
        self.definitions: dict[tuple[str, str], FunctionDef] = {}
        # Also index by simple name for same-file lookups
        self.by_file: dict[str, dict[str, FunctionDef]] = {}

    def add_definition(self, defn: FunctionDef) -> None:
        """Add a function definition to the index."""
        key = (defn.module_path, defn.qualname)
        self.definitions[key] = defn

        # Also index by file + simple name
        if defn.file not in self.by_file:
            self.by_file[defn.file] = {}
        self.by_file[defn.file][defn.qualname] = defn

    def resolve(
        self, callee: str, import_map: ImportMap, from_file: str
    ) -> FunctionDef | None:
        """Resolve a callee string to a definition using import context."""
        # First try same-file lookup
        if from_file in self.by_file:
            if callee in self.by_file[from_file]:
                return self.by_file[from_file][callee]

        # Try via import map
        resolved = import_map.resolve(callee)
        if resolved:
            result = self._lookup_resolved(resolved)
            if result:
                return result

        # Try resolving as module_path.qualname directly
        # Handles: module.func() where module is imported
        if "." in callee:
            parts = callee.rsplit(".", 1)
            module_alias, name = parts
            resolved_module = import_map.resolve(module_alias)
            if resolved_module:
                # Direct function: (resolved_module, name)
                if (resolved_module, name) in self.definitions:
                    return self.definitions[(resolved_module, name)]

                # Class.method pattern: module.Class.method()
                result = self._lookup_resolved(f"{resolved_module}.{name}")
                if result:
                    return result

        return None

    def _lookup_resolved(self, resolved: str) -> FunctionDef | None:
        """Try to find a definition for a resolved module path."""
        # Extract module and name
        parts = resolved.rsplit(".", 1)
        if len(parts) != 2:
            return None

        module, name = parts

        # Try exact match (module, name)
        if (module, name) in self.definitions:
            return self.definitions[(module, name)]

        # Try as Class.method pattern:
        # resolved = "pkg.Class.method" -> try (pkg, Class.method)
        module_parts = module.rsplit(".", 1)
        if len(module_parts) == 2:
            parent_module, class_name = module_parts
            qualname = f"{class_name}.{name}"
            if (parent_module, qualname) in self.definitions:
                return self.definitions[(parent_module, qualname)]

            # Also try with just the last component of parent_module
            # This handles when definitions use relative paths but imports use full paths
            # e.g., def is ("foo", "Class.method") but import resolves to "pkg.foo.Class.method"
            # Guard: only use if there's exactly ONE match to avoid ambiguous resolutions
            last_module = parent_module.rsplit(".", 1)[-1]
            candidate = (last_module, qualname)
            if candidate in self.definitions:
                # Check for ambiguity: count how many modules end with last_module
                matches = [
                    k
                    for k in self.definitions
                    if k[1] == qualname
                    and (k[0] == last_module or k[0].endswith(f".{last_module}"))
                ]
                if len(matches) == 1:
                    return self.definitions[candidate]
                # Ambiguous: multiple modules have same last component, skip fallback

        # Also try with just the last component of module (for functions)
        # e.g., def is ("foo", "func") but import resolves to "pkg.foo.func"
        # Guard: only use if there's exactly ONE match to avoid ambiguous resolutions
        last_module = module.rsplit(".", 1)[-1]
        candidate = (last_module, name)
        if candidate in self.definitions:
            # Check for ambiguity: count how many modules end with last_module
            matches = [
                k
                for k in self.definitions
                if k[1] == name
                and (k[0] == last_module or k[0].endswith(f".{last_module}"))
            ]
            if len(matches) == 1:
                return self.definitions[candidate]
            # Ambiguous: multiple modules have same last component, skip fallback

        return None


class DefinitionCollector(ast.NodeVisitor):
    """Collect all function/class definitions from a file."""

    def __init__(self, file_path: Path, module_path: str):
        self.file_path = file_path
        self.module_path = module_path
        self.definitions: list[FunctionDef] = []
        self._class_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check for problematic class decorators
        class_decorators = self._get_decorator_names(node)

        # Look for __init__ method
        init_method = None
        has_explicit_init = False
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                has_explicit_init = True
                break

        # Determine qualname
        if self._class_stack:
            qualname = ".".join(self._class_stack) + "." + node.name
        else:
            qualname = node.name

        # Add class constructor definition
        if init_method:
            params = ParamInfo.from_ast_arguments(
                init_method.args, skip_first=True  # Skip self
            )
            decorators = class_decorators + self._get_decorator_names(init_method)
        else:
            # No __init__, assume no required args (or inherited - WARNING case)
            params = ParamInfo(
                names=[],
                posonly_count=0,
                required_positional=0,
                kwonly_names=set(),
                kwonly_required=set(),
                has_var_positional=False,
                has_var_keyword=False,
            )
            decorators = class_decorators

        defn = FunctionDef(
            module_path=self.module_path,
            qualname=qualname,
            file=str(self.file_path),
            line=node.lineno,
            params=params,
            decorators=decorators,
            is_method=False,  # Constructor call, not method
            has_explicit_init=has_explicit_init,
        )
        self.definitions.append(defn)

        # Visit class body
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function/method definition."""
        decorators = self._get_decorator_names(node)

        # Determine if this is a method
        is_method = bool(self._class_stack)

        # Build qualname
        if self._class_stack:
            qualname = ".".join(self._class_stack) + "." + node.name
        else:
            qualname = node.name

        # Skip __init__ as it's handled in visit_ClassDef
        if node.name == "__init__" and is_method:
            return

        # Determine if we should skip first param (self/cls)
        skip_first = is_method and "staticmethod" not in decorators

        params = ParamInfo.from_ast_arguments(node.args, skip_first=skip_first)

        defn = FunctionDef(
            module_path=self.module_path,
            qualname=qualname,
            file=str(self.file_path),
            line=node.lineno,
            params=params,
            decorators=decorators,
            is_method=is_method,
        )
        self.definitions.append(defn)

    def _get_decorator_names(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> list[str]:
        """Extract decorator names from a node."""
        names = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                names.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    names.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    names.append(dec.func.attr)
        return names


class CallCollector(ast.NodeVisitor):
    """Collect all call sites from a file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.calls: list[CallSite] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Process a function call."""
        callee_expr = self._get_callee_expr(node.func)

        # Skip self.method() calls - V1 doesn't support instance method tracking
        if callee_expr and callee_expr.startswith("self."):
            self.generic_visit(node)
            return

        if callee_expr:
            # Count positional args (excluding *args)
            positional_count = sum(
                1 for arg in node.args if not isinstance(arg, ast.Starred)
            )

            # Check for *args in call
            has_star_args = any(isinstance(arg, ast.Starred) for arg in node.args)

            # Collect keyword arg names (excluding **kwargs)
            keyword_names = set()
            has_star_kwargs = False
            for kw in node.keywords:
                if kw.arg is None:
                    has_star_kwargs = True
                else:
                    keyword_names.add(kw.arg)

            call = CallSite(
                file=str(self.file_path),
                line=node.lineno,
                col=node.col_offset,
                callee_expr=callee_expr,
                positional_count=positional_count,
                keyword_names=keyword_names,
                has_star_args=has_star_args,
                has_star_kwargs=has_star_kwargs,
            )
            self.calls.append(call)

        self.generic_visit(node)

    def _get_callee_expr(self, node: ast.expr) -> str | None:
        """Extract the callee expression as a string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_callee_expr(node.value)
            if value:
                return f"{value}.{node.attr}"
        return None


def check_call(call: CallSite, defn: FunctionDef) -> Mismatch | None:
    """Check if a call matches its definition. Returns Mismatch or None."""

    # If call uses *args or **kwargs, we can't verify -> INFO
    if call.has_star_args or call.has_star_kwargs:
        return Mismatch(
            severity=Severity.INFO,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{call.callee_expr}() uses *args/**kwargs, cannot verify",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # If definition has problematic decorators -> WARNING
    problematic_decorators = {"dataclass", "overload"}
    found_problematic = problematic_decorators & set(defn.decorators)
    if found_problematic:
        return Mismatch(
            severity=Severity.WARNING,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{defn.qualname} has @{list(found_problematic)[0]}, signature may differ",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # If class without explicit __init__ -> WARNING
    if not defn.is_method and not defn.has_explicit_init:
        return Mismatch(
            severity=Severity.WARNING,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{defn.qualname} has no explicit __init__, may inherit from parent",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # If any non-trivial decorators (other than staticmethod/classmethod) -> WARNING
    safe_decorators = {"staticmethod", "classmethod", "property", "abstractmethod"}
    other_decorators = set(defn.decorators) - safe_decorators - problematic_decorators
    if other_decorators:
        return Mismatch(
            severity=Severity.WARNING,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{defn.qualname} has custom decorator(s) {other_decorators}, signature may be altered",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # Check positional arg count - too few
    # Keyword args can satisfy required positional params (except posonly)
    # Required params that must be positional = posonly_count (they can't be kwargs)
    # Required params that can be either = required_positional - posonly_count
    posonly_required = min(defn.params.posonly_count, defn.params.required_positional)
    regular_required = defn.params.required_positional - posonly_required

    # Get all positional param names for later checks
    all_positional_names = defn.params.names[
        : len(defn.params.names) - len(defn.params.kwonly_names)
    ]
    posonly_names = set(all_positional_names[: defn.params.posonly_count])

    # Check if any posonly params were passed by keyword (invalid in Python)
    # This check must happen BEFORE missing args check because passing posonly
    # as keyword is a distinct error even if all required args are provided
    posonly_as_kwarg = call.keyword_names & posonly_names
    if posonly_as_kwarg:
        return Mismatch(
            severity=Severity.ERROR,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{call.callee_expr}() got positional-only arg(s) as keyword: {posonly_as_kwarg}",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # Posonly params MUST be positional
    if call.positional_count < posonly_required:
        missing = posonly_required - call.positional_count
        return Mismatch(
            severity=Severity.ERROR,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{call.callee_expr}() missing {missing} required positional-only arg(s)",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # Regular required params can be satisfied by either positional or keyword args
    # Get the names of regular required params (non-posonly, non-kwonly, no default)
    regular_param_names = all_positional_names[defn.params.posonly_count :]
    regular_required_names = set(regular_param_names[:regular_required])

    # Count how many regular required params are provided (positional or keyword)
    positional_covering_regular = max(0, call.positional_count - posonly_required)
    covered_by_positional = set(regular_param_names[:positional_covering_regular])
    covered_by_keyword = call.keyword_names & regular_required_names
    total_covered = covered_by_positional | covered_by_keyword

    missing_required = regular_required_names - total_covered
    if missing_required:
        return Mismatch(
            severity=Severity.ERROR,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{call.callee_expr}() missing required arg(s): {missing_required}",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # Check positional arg count - too many
    max_positional = len(defn.params.names) - len(defn.params.kwonly_names)
    if not defn.params.has_var_positional and call.positional_count > max_positional:
        return Mismatch(
            severity=Severity.ERROR,
            file=call.file,
            line=call.line,
            callee=call.callee_expr,
            message=f"{call.callee_expr}() takes {max_positional} positional arg(s), got {call.positional_count}",
            definition_file=defn.file,
            definition_line=defn.line,
        )

    # Check keyword args - unknown kwargs
    if not defn.params.has_var_keyword:
        # Positional-only params cannot be passed as kwargs (already checked above)
        valid_kwargs = set(defn.params.names) - posonly_names
        unknown = call.keyword_names - valid_kwargs

        if unknown:
            return Mismatch(
                severity=Severity.ERROR,
                file=call.file,
                line=call.line,
                callee=call.callee_expr,
                message=f"{call.callee_expr}() got unexpected keyword arg(s): {unknown}",
                definition_file=defn.file,
                definition_line=defn.line,
            )

    # Check required keyword-only args
    if defn.params.kwonly_required:
        missing_kwonly = defn.params.kwonly_required - call.keyword_names
        if missing_kwonly:
            return Mismatch(
                severity=Severity.ERROR,
                file=call.file,
                line=call.line,
                callee=call.callee_expr,
                message=f"{call.callee_expr}() missing required keyword-only arg(s): {missing_kwonly}",
                definition_file=defn.file,
                definition_line=defn.line,
            )

    return None


def path_to_module(path: Path, root: Path) -> str:
    """Convert a file path to a module path."""
    # Handle case where root is a file (single file mode)
    if root.is_file():
        # Use just the filename without extension
        return path.stem

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    # Handle edge case where relative is '.' (same as root)
    if str(relative) == ".":
        return path.stem

    # Remove .py extension and convert / to .
    parts = relative.with_suffix("").parts

    # Handle empty parts
    if not parts:
        return path.stem

    # Handle __init__.py
    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts) if parts else path.stem


def build_import_map(tree: ast.AST, file_module: str) -> ImportMap:
    """Build an ImportMap from an AST."""
    import_map = ImportMap(file_module=file_module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_map.add_import(node)
        elif isinstance(node, ast.ImportFrom):
            import_map.add_import_from(node)

    return import_map


@dataclass
class ScanStats:
    """Statistics from a codebase scan."""

    files_scanned: int = 0
    files_skipped: int = 0
    definitions_found: int = 0
    calls_checked: int = 0
    duration_seconds: float = 0.0
    errors: int = 0
    warnings: int = 0
    infos: int = 0


def has_noqa_comment(file_content: str, line_number: int) -> bool:
    """Check if a line has a noqa comment suppressing sigcheck."""
    lines = file_content.splitlines()
    if 0 < line_number <= len(lines):
        line = lines[line_number - 1]
        return bool(NOQA_PATTERN.search(line))
    return False


def scan_codebase(
    root: Path,
    min_severity: Severity = Severity.WARNING,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[Mismatch], ScanStats]:
    """Scan a codebase for call signature mismatches."""
    mismatches: list[Mismatch] = []
    stats = ScanStats()
    start_time = time.time()

    # Increase recursion limit for deeply nested ASTs
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 5000))

    # Collect all Python files (handle single file or directory)
    if root.is_file() and root.suffix == ".py":
        py_files = [root]
    else:
        py_files = list(root.rglob("*.py"))

    # Apply exclusion patterns
    if exclude_patterns:
        filtered_files = []
        for py_file in py_files:
            excluded = False
            for pattern in exclude_patterns:
                # Check if any part of the path matches the pattern
                if fnmatch.fnmatch(str(py_file), f"*{pattern}*"):
                    excluded = True
                    break
                # Also check individual path components
                for part in py_file.parts:
                    if fnmatch.fnmatch(part, pattern):
                        excluded = True
                        break
                if excluded:
                    break
            if not excluded:
                filtered_files.append(py_file)
            else:
                stats.files_skipped += 1
        py_files = filtered_files

    # Phase 1: Build symbol table
    symbol_table = SymbolTable()
    file_imports: dict[str, ImportMap] = {}
    file_trees: dict[str, ast.AST] = {}
    file_contents: dict[str, str] = {}  # For noqa checking

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed
            stats.files_skipped += 1
            continue

        stats.files_scanned += 1
        module_path = path_to_module(py_file, root)
        file_str = str(py_file)
        file_trees[file_str] = tree
        file_contents[file_str] = content
        file_imports[file_str] = build_import_map(tree, module_path)

        # Collect definitions
        collector = DefinitionCollector(py_file, module_path)
        collector.visit(tree)
        for defn in collector.definitions:
            symbol_table.add_definition(defn)
            stats.definitions_found += 1

    # Phase 2: Collect calls and check
    for py_file in py_files:
        file_str = str(py_file)
        if file_str not in file_trees:
            continue

        tree = file_trees[file_str]
        content = file_contents[file_str]
        call_collector = CallCollector(py_file)
        call_collector.visit(tree)

        import_map = file_imports[file_str]

        for call in call_collector.calls:
            stats.calls_checked += 1

            # Check for noqa comment
            if has_noqa_comment(content, call.line):
                continue

            defn = symbol_table.resolve(call.callee_expr, import_map, file_str)
            if defn:
                mismatch = check_call(call, defn)
                if mismatch and mismatch.severity >= min_severity:
                    mismatches.append(mismatch)
            else:
                # Unresolved call - emit INFO per documented behavior
                # This includes external modules, dynamic calls, etc.
                if Severity.INFO >= min_severity:
                    mismatches.append(
                        Mismatch(
                            severity=Severity.INFO,
                            file=call.file,
                            line=call.line,
                            callee=call.callee_expr,
                            message=f"{call.callee_expr}() - cannot resolve (external or dynamic)",
                        )
                    )

    # Restore recursion limit
    sys.setrecursionlimit(old_limit)

    # Calculate final stats
    stats.duration_seconds = time.time() - start_time
    for m in mismatches:
        if m.severity == Severity.ERROR:
            stats.errors += 1
        elif m.severity == Severity.WARNING:
            stats.warnings += 1
        else:
            stats.infos += 1

    return mismatches, stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="General-purpose call signature mismatch detector"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to scan (default: current directory)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any ERROR found",
    )
    parser.add_argument(
        "--min-severity",
        choices=["info", "warning", "error"],
        default="warning",
        help="Minimum severity to report (default: warning)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Exclude paths matching pattern (can be used multiple times)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show scan statistics",
    )
    args = parser.parse_args()

    min_severity = Severity[args.min_severity.upper()]
    root = Path(args.path).resolve()

    if not root.exists():
        print(f"Error: Path not found: {root}", file=sys.stderr)
        return 1

    mismatches, stats = scan_codebase(root, min_severity, args.exclude or None)

    # Output results
    if args.json:
        output = {
            "mismatches": [m.to_dict() for m in mismatches],
            "stats": asdict(stats),
        }
        print(json.dumps(output, indent=2))
    else:
        if not mismatches:
            print(f"No signature mismatches found in {root}")
        else:
            print(f"Found {len(mismatches)} signature mismatch(es) in {root}\n")

            # Group by severity
            for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
                severity_matches = [m for m in mismatches if m.severity == severity]
                if severity_matches and severity >= min_severity:
                    print(f"=== {severity.name} ({len(severity_matches)}) ===\n")
                    for m in severity_matches:
                        print(f"{m.file}:{m.line}")
                        print(f"  {m.message}")
                        if m.definition_file:
                            print(
                                f"  Defined at: {m.definition_file}:{m.definition_line}"
                            )
                        print()

        # Show stats if requested
        if args.stats:
            print("--- Statistics ---")
            print(f"Files scanned: {stats.files_scanned}")
            print(f"Files skipped: {stats.files_skipped}")
            print(f"Definitions found: {stats.definitions_found}")
            print(f"Calls checked: {stats.calls_checked}")
            print(f"Errors: {stats.errors}")
            print(f"Warnings: {stats.warnings}")
            print(f"Infos: {stats.infos}")
            print(f"Duration: {stats.duration_seconds:.2f}s")

    # Exit code
    if args.strict and stats.errors > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
