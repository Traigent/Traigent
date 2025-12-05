"""Compute module complexity and quality metrics."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

try:  # pragma: no cover
    from .analysis_utils import (
        count_sloc,
        get_public_symbols,
        iter_functions,
        load_ast,
        quantiles,
        safe_relpath,
        write_csv,
    )
    from .dependency_graph import ModuleIndex, build_edges
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.code_analysis.analysis_utils import (
        count_sloc,
        get_public_symbols,
        iter_functions,
        load_ast,
        quantiles,
        safe_relpath,
        write_csv,
    )
    from scripts.code_analysis.dependency_graph import ModuleIndex, build_edges


def gather_test_imports(tests_root: Path) -> Dict[str, Tuple[Set[str], int]]:
    mapping: Dict[str, Tuple[Set[str], int]] = {}
    for path in sorted(tests_root.rglob("test_*.py")):
        tree = load_ast(path)
        if tree is None or not isinstance(tree, ast.Module):
            continue
        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_import_variants(imports, alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name:
                    add_import_variants(imports, module_name)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = f"{module_name}.{alias.name}" if module_name else alias.name
                    add_import_variants(imports, target)
        test_count = count_test_functions(tree)
        mapping[str(path)] = (imports, test_count)
    return mapping


def add_import_variants(imports: Set[str], name: str) -> None:
    candidate = name.strip()
    if not candidate:
        return
    normalized = candidate.replace("..", ".").strip(".")
    if not normalized:
        return
    variants = {normalized}
    if normalized.startswith("traigent."):
        variants.add(normalized[len("traigent.") :])
    else:
        variants.add(f"traigent.{normalized}")
    imports.update(variant for variant in variants if variant)


def count_test_functions(tree: ast.Module) -> int:
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            count += 1
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    count += 1
    return count


def match_tests(module: str, imports: Set[str]) -> bool:
    module_aliases = {module}
    if module.startswith("traigent."):
        module_aliases.add(module[len("traigent.") :])
    else:
        module_aliases.add(f"traigent.{module}")
    for imported in imports:
        if imported in module_aliases:
            return True
        if module.startswith(f"{imported}."):
            return True
        if any(imported.startswith(f"{alias}.") for alias in module_aliases):
            return True
    return False


def compute_metrics(
    project_root: Path,
    source_root: Path,
    tests_root: Path,
    output: Path,
) -> None:
    index = ModuleIndex(source_root)
    edges = build_edges(index)
    fan_out: Dict[str, Set[str]] = defaultdict(set)
    fan_in: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        fan_out[edge.source].add(edge.target)
        fan_in[edge.target].add(edge.source)

    test_imports = gather_test_imports(tests_root)

    rows: List[Sequence[object]] = []
    for module, record in sorted(index.modules.items()):
        path = record.path
        rel_path = safe_relpath(path, project_root)
        sloc = count_sloc(path)
        tree = record.ast_tree
        public_symbols = len(get_public_symbols(tree)) if tree is not None else 0
        func_infos = list(iter_functions(tree, module)) if tree is not None else []
        cyclomatic_total = sum(info.cyclomatic_complexity for info in func_infos)
        cognitive_total = sum(info.cognitive_complexity for info in func_infos)
        func_lengths = [info.length for info in func_infos if info.length > 0]
        q1, q2, q3 = quantiles(func_lengths)
        test_count = 0
        for imports, count in test_imports.values():
            if match_tests(module, imports):
                test_count += count
        func_count = len(func_infos) or 1
        rows.append(
            (
                module,
                rel_path,
                sloc,
                cyclomatic_total,
                float(cyclomatic_total) / func_count,
                cognitive_total,
                float(cognitive_total) / func_count,
                len(fan_in.get(module, set())),
                len(fan_out.get(module, set())),
                public_symbols,
                q1 if q1 is not None else "",
                q2 if q2 is not None else "",
                q3 if q3 is not None else "",
                test_count,
            )
        )

    header = [
        "module",
        "path",
        "sloc",
        "cyclomatic_total",
        "cyclomatic_average",
        "cognitive_total",
        "cognitive_average",
        "fan_in",
        "fan_out",
        "public_symbols",
        "function_length_p25",
        "function_length_p50",
        "function_length_p75",
        "test_count",
    ]

    write_csv(output, header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute module metrics.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("traigent"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compute_metrics(
        args.project_root.resolve(),
        (args.project_root / args.source_root).resolve(),
        (args.project_root / args.tests_root).resolve(),
        args.output,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
