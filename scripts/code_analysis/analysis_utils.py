"""Shared helpers for code analysis scripts."""

from __future__ import annotations

import ast
import csv
import json
import statistics
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".ini": "INI",
    ".cfg": "INI",
    ".toml": "TOML",
    ".md": "Markdown",
    ".txt": "Text",
}

PY_EXTENSIONS = {".py", ".pyi"}


@dataclass
class FunctionInfo:
    """Static information for a function or method."""

    qualified_name: str
    lineno: int
    end_lineno: int
    cyclomatic_complexity: int
    cognitive_complexity: int

    @property
    def length(self) -> int:
        return max(0, self.end_lineno - self.lineno + 1)


def detect_language(path: Path) -> str:
    return LANGUAGE_MAP.get(path.suffix.lower(), "Unknown")


def count_sloc(path: Path) -> int:
    if not path.is_file():
        return 0
    sloc = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if path.suffix in PY_EXTENSIONS and line.startswith("#"):
                    continue
                sloc += 1
    except OSError:
        return 0
    return sloc


def to_module_name(source_root: Path, path: Path) -> str:
    rel = path.relative_to(source_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    prefix = source_root.name
    module_suffix = ".".join(parts)
    if module_suffix:
        return f"{prefix}.{module_suffix}"
    return prefix


def load_ast(path: Path) -> ast.AST | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    except OSError:
        return None
    try:
        return ast.parse(content, filename=str(path))
    except SyntaxError:
        return None


class CyclomaticVisitor(ast.NodeVisitor):
    BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.IfExp,
        ast.BoolOp,
        ast.ExceptHandler,
        ast.Assert,
        ast.comprehension,
    )

    def __init__(self) -> None:
        self.complexity = 1

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, (ast.And, ast.Or)):
            self.complexity += max(1, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.complexity += len(node.handlers)
        if node.orelse:
            self.complexity += 1
        if node.finalbody:
            self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.ifs:
            self.complexity += len(node.ifs)
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            self.complexity += 1
        elif isinstance(node, ast.IfExp):
            self.complexity += 1
        elif isinstance(node, ast.Assert):
            self.complexity += 1
        super().generic_visit(node)


class CognitiveVisitor(ast.NodeVisitor):
    CONTROL_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
        ast.BoolOp,
        ast.comprehension,
    )

    def __init__(self) -> None:
        self.score = 0
        self.depth = 0

    def visit(self, node: ast.AST) -> None:  # type: ignore[override]
        is_control = isinstance(node, self.CONTROL_NODES)
        if is_control:
            self.score += 1 + self.depth
            self.depth += 1
        if isinstance(node, ast.BoolOp):
            # account for chained bool ops
            self.score += max(0, len(node.values) - 2)
        super().visit(node)
        if is_control:
            self.depth -= 1

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.ifs:
            self.score += len(node.ifs) + self.depth
        self.generic_visit(node)


def iter_functions(module_ast: ast.AST, module_name: str) -> Iterator[FunctionInfo]:
    for node in module_ast.body if isinstance(module_ast, ast.Module) else []:
        yield from _extract_functions(node, module_name, parent_name="")


def _extract_functions(node: ast.AST, module_name: str, parent_name: str) -> Iterator[FunctionInfo]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        qualified_name = (
            f"{module_name}.{node.name}"
            if not parent_name
            else f"{module_name}.{parent_name}.{node.name}"
        )
        cyclo = CyclomaticVisitor()
        cyclo.visit(node)
        cognitive = CognitiveVisitor()
        cognitive.visit(node)
        end_lineno = getattr(node, "end_lineno", node.lineno)
        yield FunctionInfo(
            qualified_name=qualified_name,
            lineno=node.lineno,
            end_lineno=end_lineno,
            cyclomatic_complexity=cyclo.complexity,
            cognitive_complexity=cognitive.score,
        )
        for child in node.body:
            yield from _extract_functions(
                child,
                module_name,
                parent_name=f"{parent_name}.{node.name}" if parent_name else node.name,
            )
    elif isinstance(node, ast.ClassDef):
        class_name = f"{parent_name}.{node.name}" if parent_name else node.name
        for child in node.body:
            yield from _extract_functions(child, module_name, class_name)


def get_public_symbols(module_ast: ast.AST) -> set[str]:
    names: set[str] = set()
    if not isinstance(module_ast, ast.Module):
        return names
    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if path.name.endswith("__init__.py") or path.name.endswith(".py"):
            yield path


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


def run_command(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(  # type: ignore[arg-type]
            args=list(args),
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def quantiles(
    data: Sequence[int],
) -> tuple[float | None, float | None, float | None]:
    if not data:
        return None, None, None
    if len(data) == 1:
        value = float(data[0])
        return value, value, value
    try:
        qs = statistics.quantiles(data, n=4, method="inclusive")
        return float(qs[0]), float(qs[1]), float(qs[2])
    except statistics.StatisticsError:
        value = float(data[0])
        return value, value, value


def load_coverage_map(coverage_xml: Path, project_root: Path) -> dict[str, float]:
    # Requires defusedxml for secure XML parsing
    try:
        import defusedxml.ElementTree as ET
    except ImportError as e:
        raise ImportError(
            "defusedxml is required for XML parsing. "
            "Install with: pip install traigent[security]"
        ) from e

    if not coverage_xml.exists():
        return {}
    try:
        tree = ET.parse(str(coverage_xml))
    except ET.ParseError:
        return {}
    root = tree.getroot()
    coverage: dict[str, float] = {}
    for cls in root.iter("class"):
        filename = cls.attrib.get("filename")
        line_rate = cls.attrib.get("line-rate")
        if not filename or line_rate is None:
            continue
        try:
            rate = float(line_rate)
        except ValueError:
            continue
        rel = safe_relpath(Path(filename).resolve(), project_root)
        coverage[rel] = rate * 100.0
    return coverage


def load_lint_map(lint_json: Path, project_root: Path) -> dict[str, int]:
    if not lint_json.exists():
        return {}
    try:
        data = json.loads(lint_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    results = data if isinstance(data, list) else data.get("files", [])
    counts: dict[str, int] = {}
    for entry in results:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not filename:
            continue
        rel = safe_relpath(Path(filename).resolve(), project_root)
        if isinstance(entry, dict) and "message" in entry:
            counts[rel] = counts.get(rel, 0) + 1
        elif isinstance(entry, dict) and "messages" in entry:
            counts[rel] = len(entry["messages"])
        else:
            counts[rel] = counts.get(rel, 0) + 1
    return counts


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat()
