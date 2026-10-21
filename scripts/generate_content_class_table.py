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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

BASE = "https://schemas.traigent.ai/"
ENDPOINTS = "optimization/optimization_endpoints.json"
#: Content-less token issuance route (Schema 5ab2fd8): its own endpoint inventory file.
TOKEN_REFS_ENDPOINTS = "optimization/token_refs_endpoints.json"
#: Negotiated label-write route (Schema 31e0ddf, POST /api/v1/optimization/labels).
LABEL_WRITE_ENDPOINTS = "optimization/label_write_endpoints.json"
LABEL_IO = "labels/label_io_schema.json"
#: Label-service routes (Backend ``/api/v1/labels/*``): the Schema defines the write
#: request in label_io_schema.json; the read request (a list of object refs) is defined
#: by the Backend contract only, so it is compiled from this inline equivalent.
LABEL_ROUTES = (
    ("GET", "/api/v1/labels/binding-list", None),
    ("PUT", "/api/v1/labels/objects", {"$ref": "#/definitions/LabelWriteRequest"}),
    (
        "POST",
        "/api/v1/labels/objects/read",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["object_refs"],
            "properties": {
                "object_refs": {
                    "type": "array",
                    "items": {
                        "$ref": "label_common_schema.json#/definitions/ObjectRef"
                    },
                }
            },
        },
    ),
)
OUT = (
    Path(__file__).resolve().parents[1]
    / "traigent"
    / "labels"
    / "content_class_table.json"
)


_CONSTRAINT_KEYS = frozenset(
    {"pattern", "maxLength", "minLength", "enum", "const", "description"}
)
_VALUE_KEYS = frozenset({"pattern", "maxLength", "minLength", "enum", "const"})
#: (looser, stricter) pattern pairs where every string the stricter one accepts is also
#: accepted by the looser one (hex digits are a subset of ``[A-Za-z0-9_-]``). The table carries
#: ONE pattern per node, so a conjunction of two is representable losslessly only as the
#: stricter of an implying pair; any other pair fails generation.
_IMPLYING_PATTERNS = frozenset({("^[A-Za-z0-9_-]+$", "^[0-9a-f]{32}$")})
_REFINEMENT_KEYS = frozenset({"properties", "required", "not", "description"})


def _constraint_only(branch: Any) -> bool:
    return (
        isinstance(branch, dict)
        and bool(set(branch) & _VALUE_KEYS)
        and set(branch) <= _CONSTRAINT_KEYS
    )


def _merge_constraints(out: dict[str, Any], branch: dict[str, Any]) -> None:
    """Fold a constraint-only ``allOf`` branch INTO the node's compiled form (conjunction).

    Nothing is dropped: patterns keep the stricter of an implying pair (else generation
    fails), lengths take the tighter bound, ``enum`` intersects, ``const`` must agree.
    """
    if "pattern" in branch:
        new = branch["pattern"]
        old = out.get("pt")
        if old is None or old == new:
            out["pt"] = new
        elif (old, new) in _IMPLYING_PATTERNS:
            out["pt"] = new
        elif (new, old) not in _IMPLYING_PATTERNS:
            raise SystemExit(
                "allOf pattern merge is not lossless: " + old + " & " + new
            )
    if "maxLength" in branch:
        out["ml"] = min(out.get("ml", branch["maxLength"]), branch["maxLength"])
    if "minLength" in branch:
        out["mn"] = max(out.get("mn", branch["minLength"]), branch["minLength"])
    if "enum" in branch:
        allowed = branch["enum"]
        out["e"] = [v for v in out["e"] if v in allowed] if "e" in out else allowed
    if "const" in branch:
        if "k" in out and out["k"][0] != branch["const"]:
            raise SystemExit("allOf const merge is not lossless")
        out["k"] = [branch["const"]]


def _refinement_only(branch: Any, parent_properties: Any) -> bool:
    """A oneOf/anyOf branch that only restricts the parent's already-declared properties."""
    if (
        not isinstance(branch, dict)
        or not isinstance(parent_properties, dict)
        or not set(branch) <= _REFINEMENT_KEYS
    ):
        return False
    declared = branch.get("properties") or {}
    if not isinstance(declared, dict) or not set(declared) <= set(parent_properties):
        return False
    return all(
        isinstance(v, dict) and set(v) <= {"x-content-class", "const", "enum"}
        for v in declared.values()
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
        #: Only the token-refs inventory is compiled with the shape simplifications below,
        #: so no other route's compiled classes can change (checked by prefix equality).
        self.simplify = False
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
        if self.simplify and "minLength" in node:
            out["mn"] = node["minLength"]  # only the token-refs inventory carries it
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
            branches = node.get(src)
            if not isinstance(branches, list):
                continue
            if self.simplify and src == "allOf":
                # a branch that only adds a value constraint (pattern / length) carries no
                # class and no structure: the walker would read it as an unclassified leaf
                for b in branches:
                    if _constraint_only(b):
                        _merge_constraints(out, b)
                branches = [b for b in branches if not _constraint_only(b)]
            elif self.simplify and all(
                _refinement_only(b, node.get("properties")) for b in branches
            ):
                # branches that only refine which of the node's OWN properties are required
                # or forbidden (cross-field rules) add no class information either; the
                # request validator in token_store enforces those rules before sending
                continue
            if branches:
                out[dst] = [self.compile(b, base, scopes, doc) for b in branches]
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


def add_endpoint_routes(comp: Compiler, endpoints_file: str, routes: list) -> None:
    """Compile every route of one OpenAPI-style endpoint inventory file into ``routes``."""
    comp.simplify = endpoints_file == TOKEN_REFS_ENDPOINTS
    try:
        _add_endpoint_routes(comp, endpoints_file, routes)
    finally:
        comp.simplify = False


def _add_endpoint_routes(comp: Compiler, endpoints_file: str, routes: list) -> None:
    uri = BASE + endpoints_file
    doc = comp.docs[uri]
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
            # Every `{name}` in the template is a path parameter. One the Schema declares
            # carries its compiled class; one it does not is marked unknown, which the gate
            # classifies as legacy_content (refused in refuse mode). No class is invented.
            for name in re.findall(r"\{([^}]+)\}", template):
                params["path"].setdefault(name, {"unknown": 1})
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


def _assert_operation_key_lossless(
    comp: Compiler, routes: list[dict[str, Any]]
) -> None:
    """The token-refs ``operation_key`` must compile to exactly 32 lowercase hex characters."""
    for route in routes:
        if route["template"] != "/api/v1/optimization/token-refs" or not route["body"]:
            continue
        node = route["body"]
        while "p" not in node and "ref" in node:
            node = comp.nodes[node["ref"]]
        key = node["p"]["operation_key"]
        if (
            key.get("pt") != "^[0-9a-f]{32}$"
            or key.get("mn") != 32
            or key.get("ml") != 32
        ):
            raise SystemExit("operation_key constraints were not preserved losslessly")
        return
    raise SystemExit("token-refs route missing from the compiled table")


def build(schema_root: Path, commit: str | None = None) -> dict[str, Any]:
    root = schema_root / "traigent_schema" / "schemas"
    comp = Compiler(root)
    routes: list[dict[str, Any]] = []
    add_endpoint_routes(comp, ENDPOINTS, routes)
    label_uri = BASE + LABEL_IO
    label_doc = comp.docs[label_uri]
    for method, template, schema in LABEL_ROUTES:
        routes.append(
            {
                "method": method,
                "template": template,
                "path": {},
                "query": {},
                "body": (
                    None
                    if schema is None
                    else comp.compile(schema, label_uri, (label_doc,), label_doc)
                ),
            }
        )
    # appended after the label routes so every earlier node index stays where it was
    add_endpoint_routes(comp, TOKEN_REFS_ENDPOINTS, routes)
    add_endpoint_routes(comp, LABEL_WRITE_ENDPOINTS, routes)
    _assert_operation_key_lossless(comp, routes)
    manifest = schema_root / "docs" / "content-class-manifest.json"
    commit = (
        commit
        or subprocess.run(  # noqa: S603
            ["git", "-C", str(schema_root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    )
    return {
        "_comment": "Generated by scripts/generate_content_class_table.py; do not hand-edit.",
        "source": {
            "schema_commit": commit,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "nodes": comp.nodes,
        "routes": routes,
    }


def render(schema_root: Path, commit: str | None = None) -> str:
    return (
        json.dumps(build(schema_root, commit), sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--commit", help="Schema commit to record (for an exported tree)")
    args = ap.parse_args()
    text = render(args.schema, args.commit)
    if args.check:
        return 0 if OUT.read_text("utf-8") == text else 1
    OUT.write_text(text, "utf-8")
    print(f"{OUT} {len(text)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
