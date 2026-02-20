"""Apply generated config as a decorator on user source files.

Uses AST for analysis (finding functions, existing decorators, imports)
and line-based string operations for insertion to preserve user formatting.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

from traigent.config_generator.types import AutoConfigResult


def apply_config(
    file_path: Path,
    result: AutoConfigResult,
    function_name: str | None = None,
    *,
    backup: bool = True,
) -> Path:
    """Insert or update ``@traigent.optimize(...)`` on a target function.

    Parameters
    ----------
    file_path:
        Path to the Python source file to modify.
    result:
        The generated config to apply as a decorator.
    function_name:
        Which function to decorate. Required.
    backup:
        If ``True``, create a ``.py.bak`` copy before modifying.

    Returns
    -------
    Path
        The path to the modified file.

    Raises
    ------
    ValueError
        If ``function_name`` is not found in the source.
    """
    if not function_name:
        raise ValueError("function_name is required for apply_config")

    source = file_path.read_text()
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse {file_path}: {exc}") from exc

    # Find the target function
    func_node = _find_function(tree, function_name)
    if func_node is None:
        raise ValueError(f"Function '{function_name}' not found in {file_path}")

    # Generate the decorator code
    decorator_code = result.to_python_code()

    # Determine the indentation of the function definition line
    func_line = lines[func_node.lineno - 1]
    indent = _get_leading_whitespace(func_line)

    # Indent the decorator to match the function
    indented_decorator = _indent_decorator(decorator_code, indent)

    # Check for existing optimize decorator and replace or insert
    existing = _find_optimize_decorator(func_node)
    if existing is not None:
        # Replace the existing decorator lines
        start_line = existing.lineno - 1  # 0-indexed
        end_line = existing.end_lineno  # end_lineno is 1-indexed, exclusive after slice
        lines[start_line:end_line] = [indented_decorator + "\n"]
    else:
        # Insert above the first decorator or the function def itself
        insert_line = func_node.lineno - 1
        if func_node.decorator_list:
            insert_line = func_node.decorator_list[0].lineno - 1
        lines.insert(insert_line, indented_decorator + "\n")

    # Ensure required imports exist
    import_lines = _collect_needed_imports(decorator_code, tree)
    if import_lines:
        insert_pos = _find_import_insertion_point(tree)
        for i, imp_line in enumerate(import_lines):
            lines.insert(insert_pos + i, imp_line + "\n")

    # Write back
    if backup:
        shutil.copy2(file_path, file_path.with_suffix(".py.bak"))

    file_path.write_text("".join(lines))
    return file_path


def _find_function(
    tree: ast.Module, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a top-level or class-level function by name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _find_optimize_decorator(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.expr | None:
    """Find an existing @traigent.optimize(...) decorator on the function.

    Checks for:
    - @traigent.optimize(...)
    - @optimize(...)
    - @traigent.api.decorators.optimize(...)
    """
    for dec in func_node.decorator_list:
        if _is_optimize_call(dec):
            return dec
    return None


def _is_optimize_call(node: ast.expr) -> bool:
    """Check if an AST node is a call to optimize in any import style."""
    # Unwrap Call → get the func
    call_func = node.func if isinstance(node, ast.Call) else node

    # @optimize(...)
    if isinstance(call_func, ast.Name) and call_func.id == "optimize":
        return True

    # @traigent.optimize(...)
    if (
        isinstance(call_func, ast.Attribute)
        and call_func.attr == "optimize"
        and isinstance(call_func.value, ast.Name)
        and call_func.value.id == "traigent"
    ):
        return True

    # @traigent.api.decorators.optimize(...)
    if isinstance(call_func, ast.Attribute) and call_func.attr == "optimize":
        # Walk the attribute chain
        parts = _unpack_attribute(call_func.value)
        if parts == ["traigent", "api", "decorators"]:
            return True

    return False


def _unpack_attribute(node: ast.expr) -> list[str]:
    """Unpack a.b.c into ["a", "b", "c"]."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _unpack_attribute(node.value) + [node.attr]
    return []


def _get_leading_whitespace(line: str) -> str:
    """Extract the exact leading whitespace from a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""


def _indent_decorator(decorator_code: str, indent: str) -> str:
    """Indent all lines of the decorator to the given level."""
    lines = decorator_code.splitlines()
    return "\n".join(indent + line for line in lines)


# Symbols that can appear in generated decorator code and their imports.
_RANGE_SYMBOLS = {"Range", "IntRange", "LogRange", "Choices"}
_SAFETY_SYMBOLS = {
    "faithfulness",
    "hallucination_rate",
    "toxicity_score",
    "bias_score",
    "safety_score",
}


def _collect_needed_imports(decorator_code: str, tree: ast.Module) -> list[str]:
    """Determine which import lines are missing and need to be added."""
    existing = _get_existing_imports(tree)
    needed: list[str] = []

    # Always need `import traigent` for @traigent.optimize(...)
    if "traigent" not in existing:
        needed.append("import traigent")

    # Check for Range/IntRange/LogRange/Choices usage
    range_used = {s for s in _RANGE_SYMBOLS if re.search(rf"\b{s}\(", decorator_code)}
    range_missing = range_used - existing
    if range_missing:
        symbols = ", ".join(sorted(range_missing))
        needed.append(f"from traigent import {symbols}")

    # Check for safety metric usage
    safety_used = {
        s for s in _SAFETY_SYMBOLS if re.search(rf"\b{s}[.(]", decorator_code)
    }
    safety_missing = safety_used - existing
    if safety_missing:
        symbols = ", ".join(sorted(safety_missing))
        needed.append(f"from traigent.api.safety import {symbols}")

    return needed


def _get_existing_imports(tree: ast.Module) -> set[str]:
    """Collect all imported names from module-level statements only.

    Ignores imports nested inside functions/classes to avoid treating
    locally-scoped names as globally available.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _find_import_insertion_point(tree: ast.Module) -> int:
    """Find the line number (0-indexed) to insert new imports.

    Only considers module-level import statements to avoid inserting
    imports inside indented blocks. Returns 0 if there are no
    top-level imports.
    """
    last_import_line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = getattr(node, "end_lineno", node.lineno)
            if end > last_import_line:
                last_import_line = end
    return last_import_line
