#!/usr/bin/env python3
"""Annotate simple `self` attribute assignments with generic container types.

This codemod converts statements like `self.items = []` into annotated
assignments (`self.items: list[Any] = []`). It focuses on easy wins that help
mypy without attempting deep inference.

Usage:
    python scripts/maintenance/add_self_attr_annotations.py [paths...]
If no paths are supplied, the `traigent` package is processed.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor
from libcst.metadata import ParentNodeProvider

from traigent.utils.secure_path import PathTraversalError, validate_path

_CONTAINER_ANNOTATIONS = {
    "list": "list[Any]",
    "dict": "dict[str, Any]",
    "set": "set[Any]",
}


class SelfAttrAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, context: CodemodContext) -> None:
        self.context = context
        self.in_class = 0
        self.in_function = 0
        self.changed = False

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: D401
        self.in_class += 1

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.in_class -= 1
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: D401
        self.in_function += 1

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.in_function -= 1
        return updated_node

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
        self.in_function += 1

    def leave_AsyncFunctionDef(
        self,
        original_node: cst.AsyncFunctionDef,
        updated_node: cst.AsyncFunctionDef,
    ) -> cst.AsyncFunctionDef:
        self.in_function -= 1
        return updated_node

    def _make_annotation(self, value: cst.BaseExpression) -> cst.Annotation | None:
        if isinstance(value, cst.List):
            annotation = _CONTAINER_ANNOTATIONS["list"]
        elif isinstance(value, cst.Dict):
            annotation = _CONTAINER_ANNOTATIONS["dict"]
        elif isinstance(value, cst.Set):
            annotation = _CONTAINER_ANNOTATIONS["set"]
        elif isinstance(value, cst.Call):
            func = value.func
            func_name = None
            if isinstance(func, cst.Name):
                func_name = func.value
            elif isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name):
                func_name = func.attr.value
            if func_name and func_name in _CONTAINER_ANNOTATIONS:
                annotation = _CONTAINER_ANNOTATIONS[func_name]
            else:
                return None
        else:
            return None

        AddImportsVisitor.add_needed_import(self.context, "typing", "Any")
        return cst.Annotation(annotation=cst.parse_expression(annotation))

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.BaseStatement:
        if not (self.in_class and self.in_function):
            return updated_node
        if len(updated_node.targets) != 1:
            return updated_node

        target = updated_node.targets[0].target
        if not m.matches(target, m.Attribute(value=m.Name("self"), attr=m.Name())):
            return updated_node

        annotation = self._make_annotation(updated_node.value)
        if annotation is None:
            return updated_node

        attr = cst.ensure_type(target, cst.Attribute)
        ann_assign = cst.AnnAssign(
            target=attr,
            annotation=annotation,
            value=updated_node.value,
        )
        self.changed = True
        return ann_assign


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
    context = CodemodContext()
    wrapper = cst.MetadataWrapper(module)
    transformer = SelfAttrAnnotationTransformer(context)
    updated = wrapper.visit(transformer)
    if not transformer.changed:
        return False
    updated = AddImportsVisitor(context).transform_module(updated)
    path.write_text(updated.code)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        default=[pathlib.Path("traigent")],
        help="Files or directories to process",
    )
    args = parser.parse_args()

    base_dir = pathlib.Path.cwd()
    safe_paths: list[pathlib.Path] = []
    for path in args.paths:
        try:
            safe_paths.append(validate_path(path, base_dir, must_exist=True))
        except (PathTraversalError, FileNotFoundError) as exc:
            raise SystemExit(f"Error: {exc}") from exc

    changed_files: list[pathlib.Path] = []
    for file in iter_python_files(safe_paths):
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
