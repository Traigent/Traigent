#!/usr/bin/env python3
"""Add explicit `-> None` annotations to void functions.

Targets functions (sync or async) that lack a return annotation and do not
return any non-None value nor yield. This helps reduce mypy `no-untyped-def`
noise without changing runtime behaviour.

Usage:
    python scripts/maintenance/add_none_return_annotations.py [paths...]
If no paths are supplied, the tool scans the `traigent` package.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import libcst as cst
import libcst.matchers as m


class _ReturnAnalyzer(cst.CSTVisitor):
    def __init__(self) -> None:
        self.returns_value = False
        self.is_generator = False

    def visit_Return(self, node: cst.Return) -> None:
        if node.value is None:
            return
        if m.matches(node.value, m.Name("None")):
            return
        self.returns_value = True

    def visit_Yield(self, node: cst.Yield) -> None:  # type: ignore[override]
        self.is_generator = True

    def visit_YieldFrom(self, node: cst.YieldFrom) -> None:  # type: ignore[override]
        self.is_generator = True


class AddNoneReturnTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changed = False

    def _should_skip(self, node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            if m.matches(decorator.decorator, m.Name("overload")):
                return True
        return False

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if updated_node.returns is not None:
            return updated_node
        if self._should_skip(updated_node):
            return updated_node

        analyzer = _ReturnAnalyzer()
        updated_node.body.visit(analyzer)
        if analyzer.returns_value or analyzer.is_generator:
            return updated_node

        self.changed = True
        return updated_node.with_changes(returns=cst.Annotation(annotation=cst.Name("None")))

    def leave_AsyncFunctionDef(
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.AsyncFunctionDef:
        if updated_node.returns is not None:
            return updated_node
        if self._should_skip(updated_node.function):
            return updated_node

        analyzer = _ReturnAnalyzer()
        updated_node.body.visit(analyzer)
        if analyzer.returns_value or analyzer.is_generator:
            return updated_node

        self.changed = True
        return updated_node.with_changes(
            returns=cst.Annotation(annotation=cst.Name("None"))
        )


def iter_python_files(paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    for path in paths:
        if path.is_dir():
            for file in sorted(path.rglob("*.py")):
                parts = set(file.parts)
                if {"venv", "__pycache__"} & parts:
                    continue
                yield file
        elif path.is_file() and path.suffix == ".py":
            yield path


def process_file(path: pathlib.Path) -> bool:
    source = path.read_text()
    module = cst.parse_module(source)
    transformer = AddNoneReturnTransformer()
    updated = module.visit(transformer)
    if transformer.changed and updated != module:
        path.write_text(updated.code)
    return transformer.changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        default=[pathlib.Path("traigent")],
        help="Files or directories to scan",
    )
    args = parser.parse_args()

    changed_files: list[pathlib.Path] = []
    for file in iter_python_files(args.paths):
        if process_file(file):
            changed_files.append(file)

    if changed_files:
        print("Updated", len(changed_files), "file(s):")
        for file in changed_files:
            print(" -", file)
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
