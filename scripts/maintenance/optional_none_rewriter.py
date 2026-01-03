#!/usr/bin/env python3
"""Normalize annotations when default values are None.

Adds `| None` to function parameters and annotated assignments whose default
value is `None` but whose annotation does not already allow `None`.

Usage:
    python scripts/maintenance/optional_none_rewriter.py [paths...]
If no paths are provided, the script processes the `traigent` package.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import libcst as cst
import libcst.matchers as m

from traigent.utils.secure_path import (
    PathTraversalError,
    safe_read_text,
    safe_write_text,
    validate_path,
)
_EMPTY_MODULE = cst.Module(body=[])


def _code_for(node: cst.CSTNode) -> str:
    return _EMPTY_MODULE.code_for_node(node)


def _is_optional_annotation(annotation_src: str) -> bool:
    cleaned = annotation_src.replace(" ", "")
    if "Optional[" in annotation_src:
        return True
    if "Union[" in annotation_src and "None" in annotation_src:
        return True
    if "|None" in cleaned or "None|" in cleaned:
        return True
    if annotation_src.strip() == "None":
        return True
    return False


def _make_optional(annotation: cst.BaseExpression) -> cst.BaseExpression:
    return cst.BinaryOperation(
        left=annotation,
        operator=cst.BitOr(),
        right=cst.Name("None"),
    )


class OptionalDefaultTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changed = False

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.Param:
        annotation = updated_node.annotation
        default = updated_node.default
        if not annotation or not default:
            return updated_node
        if not m.matches(default, m.Name("None")):
            return updated_node

        annotation_src = _code_for(annotation.annotation)
        if _is_optional_annotation(annotation_src):
            return updated_node

        self.changed = True
        return updated_node.with_changes(
            annotation=cst.Annotation(annotation=_make_optional(annotation.annotation))
        )

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.AnnAssign:
        if updated_node.value is None or updated_node.annotation is None:
            return updated_node

        if not m.matches(updated_node.value, m.Name("None")):
            return updated_node

        annotation_src = _code_for(updated_node.annotation.annotation)
        if _is_optional_annotation(annotation_src):
            return updated_node

        self.changed = True
        return updated_node.with_changes(
            annotation=cst.Annotation(
                annotation=_make_optional(updated_node.annotation.annotation)
            )
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


def process_file(path: pathlib.Path, base_dir: pathlib.Path) -> bool:
    safe_path = validate_path(path, base_dir, must_exist=True)
    source = safe_read_text(safe_path, base_dir)
    module = cst.parse_module(source)
    transformer = OptionalDefaultTransformer()
    updated = module.visit(transformer)
    if transformer.changed and updated != module:
        safe_write_text(safe_path, updated.code, base_dir)
    return transformer.changed


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

    changed_files = []
    for file in iter_python_files(safe_paths):
        if process_file(file, base_dir):
            changed_files.append(file)

    if changed_files:
        print("Updated", len(changed_files), "file(s):")
        for file in changed_files:
            print(" -", file)
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
