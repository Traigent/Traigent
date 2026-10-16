#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
"""Generate ``traigent/labels/content_class_table.json`` from TraigentSchema.

The table is a compact, fully-resolved copy of what the transport gate needs from
the optimization endpoint inventory: per route, the class tree of the request body,
path parameters and query parameters. ``$ref`` s are resolved here so the SDK ships no
schema files. Provenance (Schema commit, sha256 of ``docs/content-class-manifest.json``)
is recorded in the table and pinned by ``tests/unit/labels/test_content_class_table.py``.

    python scripts/generate_content_class_table.py --schema <TraigentSchema worktree> [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

BASE = "https://schemas.traigent.ai/"
ENDPOINTS = "optimization/optimization_endpoints.json"
OUT = (
    Path(__file__).resolve().parents[1]
    / "traigent"
    / "labels"
    / "content_class_table.json"
)


class Compiler:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.docs: dict[str, Any] = {}
        for path in sorted(root.rglob("*.json")):
            try:
                doc = json.loads(path.read_text("utf-8"))
            except ValueError:
                continue
            self.docs[BASE + path.relative_to(root).as_posix()] = doc
            if isinstance(doc, dict) and isinstance(doc.get("$id"), str):
                self.docs[doc["$id"]] = doc
        self.nodes: list[Any] = []
        self.memo: dict[int, int] = {}

    @staticmethod
    def pointer(doc: Any, pointer: str) -> Any:
        cur = doc
        for part in pointer.lstrip("#").split("/"):
            if part == "":
                continue
            cur = cur[part.replace("~1", "/").replace("~0", "~")]
        return cur

    def resolve(self, ref: str, base: str, scopes: tuple[Any, ...], doc: Any):
        if ref.startswith("#"):
            for scope in reversed(scopes):
                try:
                    return self.pointer(scope, ref), base, scopes, doc
                except (KeyError, TypeError):
                    continue
            return self.pointer(doc, ref), base, scopes, doc
        uri = urljoin(base, ref)
        target_doc = self.docs[uri.split("#")[0]]
        frag = uri.split("#", 1)[1] if "#" in uri else ""
        node = self.pointer(target_doc, frag) if frag else target_doc
        nbase = (
            target_doc.get("$id", uri.split("#")[0])
            if isinstance(target_doc, dict)
            else uri
        )
        return node, nbase, (target_doc,), target_doc

    def compile(self, node: Any, base: str, scopes: tuple[Any, ...], doc: Any) -> Any:
        if node is True:
            return {}
        if node is False or not isinstance(node, dict):
            return {"never": 1}
        if isinstance(node.get("$id"), str):
            base = node["$id"]
        if "definitions" in node or "$defs" in node:
            scopes = scopes + (node,)
        out: dict[str, Any] = {}
        if "$ref" in node:
            target, tbase, tscopes, tdoc = self.resolve(node["$ref"], base, scopes, doc)
            out["ref"] = self.shared(target, tbase, tscopes, tdoc)
        simple = {
            "x-content-class": "c",
            "x-content-key-class": "kc",
            "type": "t",
            "pattern": "pt",
            "maxLength": "ml",
            "required": "r",
        }
        for src, dst in simple.items():
            if src in node:
                out[dst] = node[src]
        if "enum" in node:
            out["e"] = node["enum"]
        if "const" in node:
            out["k"] = [node["const"]]
        names = node.get("propertyNames")
        if isinstance(names, dict):
            if "x-content-class" in names:
                out["kc"] = names["x-content-class"]
            elif "$ref" in names:
                t, *_ = self.resolve(names["$ref"], base, scopes, doc)
                if isinstance(t, dict) and "x-content-class" in t:
                    out["kc"] = t["x-content-class"]
        if isinstance(node.get("properties"), dict):
            out["p"] = {
                k: self.compile(v, base, scopes, doc)
                for k, v in node["properties"].items()
            }
        if isinstance(node.get("patternProperties"), dict):
            out["pp"] = {
                k: self.compile(v, base, scopes, doc)
                for k, v in node["patternProperties"].items()
            }
        if "additionalProperties" in node:
            ap = node["additionalProperties"]
            out["ap"] = (
                ap if isinstance(ap, bool) else self.compile(ap, base, scopes, doc)
            )
        if isinstance(node.get("items"), dict):
            out["i"] = self.compile(node["items"], base, scopes, doc)
        for src, dst in (("allOf", "all"), ("oneOf", "one"), ("anyOf", "any")):
            if isinstance(node.get(src), list):
                out[dst] = [self.compile(b, base, scopes, doc) for b in node[src]]
        return out

    def shared(self, target: Any, base: str, scopes: tuple[Any, ...], doc: Any) -> int:
        key = id(target)
        if key in self.memo:
            return self.memo[key]
        idx = len(self.nodes)
        self.memo[key] = idx
        self.nodes.append(None)
        self.nodes[idx] = self.compile(target, base, scopes, doc)
        return idx


def build(schema_root: Path) -> dict[str, Any]:
    root = schema_root / "traigent_schema" / "schemas"
    comp = Compiler(root)
    uri = BASE + ENDPOINTS
    doc = comp.docs[uri]
    routes = []
    for template, ops in doc["paths"].items():
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            params: dict[str, dict[str, Any]] = {"path": {}, "query": {}}
            for p in op.get("parameters") or []:
                if p.get("in") in params:
                    params[p["in"]][p["name"]] = comp.compile(
                        p.get("schema", {}), uri, (), doc
                    )
            content = (op.get("requestBody") or {}).get("content") or {}
            body = None
            if content:
                body = comp.compile(
                    next(iter(content.values())).get("schema", {}), uri, (), doc
                )
            routes.append(
                {
                    "method": method.upper(),
                    "template": template,
                    "path": params["path"],
                    "query": params["query"],
                    "body": body,
                }
            )
    manifest = schema_root / "docs" / "content-class-manifest.json"
    commit = subprocess.run(  # noqa: S603
        ["git", "-C", str(schema_root), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    return {
        "_comment": "Generated by scripts/generate_content_class_table.py; do not hand-edit.",
        "source": {
            "schema_commit": commit,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "nodes": comp.nodes,
        "routes": routes,
    }


def render(schema_root: Path) -> str:
    return json.dumps(build(schema_root), sort_keys=True, separators=(",", ":")) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(args.schema)
    if args.check:
        return 0 if OUT.read_text("utf-8") == text else 1
    OUT.write_text(text, "utf-8")
    print(f"{OUT} {len(text)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
