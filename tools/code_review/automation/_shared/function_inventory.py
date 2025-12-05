#!/usr/bin/env python3
"""Build function inventory for a Python module.

Finds top-level functions and class methods. Nested inner functions are ignored.
Outputs JSON to stdout when --json is passed, else prints text.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def list_functions(module_path: str) -> dict[str, list[str]]:
    p = Path(module_path)
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(p))

    top_level: list[str] = []
    methods: list[str] = []
    classes: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_level.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            top_level.append(node.name)
        elif isinstance(node, ast.ClassDef):
            cls = node.name
            classes.append(cls)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(f"{cls}.{item.name}")
    return {
        "top_level": sorted(top_level),
        "class_methods": sorted(methods),
        "classes": sorted(classes),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Path to Python module")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = ap.parse_args(argv)

    inv = list_functions(args.module)
    if args.json:
        print(json.dumps({"module": args.module, **inv}, indent=2))
    else:
        print(f"Module: {args.module}")
        print("Top-level functions:")
        for f in inv["top_level"]:
            print(f"  - {f}")
        print("Class methods:")
        for m in inv["class_methods"]:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
