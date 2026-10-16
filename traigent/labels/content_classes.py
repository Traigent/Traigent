"""Classify a serialized request body against the Schema ``x-content-class`` annotations.

The annotations (TraigentSchema ``docs/content-classes.md``) say which class each value
node of an optimization-channel contract belongs to. This module walks a JSON body next
to the request schema of its route and counts values per class. It returns COUNTS ONLY:
never a value, a map key or a path that could be customer-chosen.

Schemas come from a directory laid out like ``traigent_schema/schemas`` (the bundled
copy in ``traigent/labels/contract_schemas`` by default; override with
``TRAIGENT_SCHEMA_DIR``). Findings that block in "refuse" mode: ``legacy_content`` (the
transitional class), ``unclassified`` (a value or map key no annotation covers) and
``violation`` (a value that breaks its own class: not in the enum, wrong pattern,
unknown key of a closed object).
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

BASE_URI = "https://schemas.traigent.ai/"
ENDPOINTS_DOC = "optimization/optimization_endpoints.json"
SCHEMA_DIR_ENV = "TRAIGENT_SCHEMA_DIR"
BLOCKING = ("legacy_content", "unclassified", "violation")
_WORST = ("violation", "legacy_content", "unclassified")
_MAX_DEPTH = 64
_MAX_NODES = 200_000

_V1BETA = re.compile(r"^/api/v1beta/projects/[^/]+")


class SchemaStoreError(RuntimeError):
    """The contract schemas cannot be loaded."""


def default_schema_dir() -> Path:
    override = os.environ.get(SCHEMA_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).parent / "contract_schemas"


class SchemaStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._docs: dict[str, Any] = {}
        if not root.is_dir():
            raise SchemaStoreError("schema directory not found")
        for path in sorted(root.rglob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rel = path.relative_to(root).as_posix()
            self._docs[BASE_URI + rel] = doc
            if isinstance(doc, dict) and isinstance(doc.get("$id"), str):
                self._docs[doc["$id"]] = doc
        self._routes: list[tuple[re.Pattern[str], str, str, Any]] | None = None

    def document(self, uri: str) -> Any:
        try:
            return self._docs[uri.split("#")[0]]
        except KeyError as exc:
            raise SchemaStoreError("unresolved schema reference") from exc

    def routes(self) -> list[tuple[re.Pattern[str], str, str, Any]]:
        """(path regex, METHOD, route template, request schema) for every body-carrying route."""
        if self._routes is None:
            doc = self.document(BASE_URI + ENDPOINTS_DOC)
            out = []
            for template, ops in doc.get("paths", {}).items():
                regex = re.compile(
                    "^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(template)) + "/?$"
                )
                for method, op in ops.items():
                    if not isinstance(op, dict):
                        continue
                    content = (op.get("requestBody") or {}).get("content") or {}
                    first: dict[str, Any] = next(iter(content.values()), {})
                    schema = first.get("schema") if content else None
                    if schema is not None:
                        out.append((regex, method.upper(), template, schema))
            self._routes = out
        return self._routes

    def find_route(self, method: str, path: str) -> tuple[str, Any] | None:
        path = _V1BETA.sub("/api/v1", path)
        for regex, m, template, schema in self.routes():
            if m == method.upper() and regex.match(path):
                return template, schema
        return None


_STORE_LOCK = threading.Lock()
_STORE: dict[str, SchemaStore] = {}


def get_store(root: Path | None = None) -> SchemaStore:
    key = str(root or default_schema_dir())
    with _STORE_LOCK:
        if key not in _STORE:
            _STORE[key] = SchemaStore(Path(key))
        return _STORE[key]


class _Ctx:
    def __init__(
        self, store: SchemaStore, base: str, scopes: tuple[Any, ...], doc: Any
    ):
        self.store, self.base, self.scopes, self.doc = store, base, scopes, doc

    def with_node(self, node: dict[str, Any]) -> _Ctx:
        base = node["$id"] if isinstance(node.get("$id"), str) else self.base
        scopes = (
            self.scopes + (node,)
            if "definitions" in node or "$defs" in node
            else self.scopes
        )
        return _Ctx(self.store, base, scopes, self.doc)


def _pointer(doc: Any, pointer: str) -> Any:
    cur = doc
    for part in pointer.lstrip("#").lstrip("/").split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(part)
        cur = cur[part]
    return cur


def _resolve(ref: str, ctx: _Ctx) -> tuple[dict[str, Any], _Ctx]:
    if ref.startswith("#"):
        for scope in reversed(ctx.scopes):
            try:
                node = _pointer(scope, ref)
                return node, ctx
            except KeyError:
                continue
        try:
            return _pointer(ctx.doc, ref), ctx
        except KeyError as exc:
            raise SchemaStoreError("unresolved local reference") from exc
    uri = urljoin(ctx.base, ref)
    doc = ctx.store.document(uri)
    frag = uri.split("#", 1)[1] if "#" in uri else ""
    node = _pointer(doc, frag) if frag else doc
    base = (
        doc["$id"]
        if isinstance(doc, dict) and isinstance(doc.get("$id"), str)
        else uri.split("#")[0]
    )
    return node, _Ctx(ctx.store, base, (doc,), doc)


def _type_ok(value: Any, node: dict[str, Any]) -> bool:
    t = node.get("type")
    if t is None:
        return True
    types = t if isinstance(t, list) else [t]
    for name in types:
        if name == "object" and isinstance(value, dict):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if (
            name == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return True
        if name == "null" and value is None:
            return True
    return False


class _Walker:
    def __init__(self) -> None:
        self.findings: dict[tuple[str, ...], str] = {}
        self.nodes = 0

    def record(self, path: tuple[str, ...], cls: str) -> None:
        old = self.findings.get(path)
        if old is None or (
            cls in _WORST
            and (old not in _WORST or _WORST.index(cls) < _WORST.index(old))
        ):
            self.findings[path] = cls

    def walk(
        self, value: Any, node: Any, ctx: _Ctx, path: tuple[str, ...], depth: int = 0
    ) -> None:
        self.nodes += 1
        if depth > _MAX_DEPTH or self.nodes > _MAX_NODES:
            self.record(path, "unclassified")
            return
        if value is None:
            return
        if not isinstance(node, dict):
            # `true` / missing schema: nothing proves the class
            self.record(path, "unclassified")
            return
        ctx = ctx.with_node(node)
        own_class = node.get("x-content-class")
        if "$ref" in node:
            target, tctx = _resolve(node["$ref"], ctx)
            if own_class and isinstance(target, dict):
                target = {**target, "x-content-class": own_class}
            self.walk(value, target, tctx, path, depth + 1)
            return
        handled = False
        if "allOf" in node:
            for branch in node["allOf"]:
                self.walk(value, branch, ctx, path, depth + 1)
            handled = True
        for key in ("oneOf", "anyOf"):
            if key in node:
                self._alternatives(value, node[key], ctx, path, depth)
                handled = True
        if isinstance(value, dict):
            self._object(value, node, ctx, path, depth)
        elif isinstance(value, list):
            items = node.get("items")
            for item in value:
                if isinstance(items, dict):
                    self.walk(item, items, ctx, path + ("[]",), depth + 1)
                elif handled:
                    break
                else:
                    self.record(path + ("[]",), "unclassified")
        else:
            if handled and not own_class:
                return
            self._leaf(value, node, own_class, path)

    def _alternatives(
        self,
        value: Any,
        branches: list[Any],
        ctx: _Ctx,
        path: tuple[str, ...],
        depth: int,
    ) -> None:
        best: _Walker | None = None
        for branch in branches:
            probe = _Walker()
            b = branch
            bctx = ctx
            try:
                while isinstance(b, dict) and "$ref" in b and len(b) == 1:
                    b, bctx = _resolve(b["$ref"], bctx)
                if isinstance(b, dict) and not _type_ok(value, b):
                    continue
                if isinstance(b, dict) and isinstance(value, dict):
                    req = b.get("required") or []
                    if any(r not in value for r in req):
                        continue
                probe.walk(value, branch, ctx, path, depth + 1)
            except (SchemaStoreError, KeyError):
                continue
            score = sum(1 for c in probe.findings.values() if c in BLOCKING)
            if best is None or score < sum(
                1 for c in best.findings.values() if c in BLOCKING
            ):
                best = probe
        if best is None:
            self.record(path, "unclassified")
            return
        for p, c in best.findings.items():
            self.record(p, c)

    def _object(
        self,
        value: dict[str, Any],
        node: dict[str, Any],
        ctx: _Ctx,
        path: tuple[str, ...],
        depth: int,
    ) -> None:
        props = node.get("properties") or {}
        patterns = node.get("patternProperties") or {}
        extra = node.get("additionalProperties", None)
        key_class = node.get("x-content-key-class")
        names = node.get("propertyNames")
        if isinstance(names, dict) and "$ref" not in names:
            key_class = names.get("x-content-class", key_class)
        elif isinstance(names, dict):
            try:
                tgt, _ = _resolve(names["$ref"], ctx)
                key_class = tgt.get("x-content-class", key_class)
            except (SchemaStoreError, KeyError):
                pass
        composed = any(k in node for k in ("allOf", "oneOf", "anyOf"))
        for k, v in value.items():
            if k in props:
                self.walk(v, props[k], ctx, path + (k,), depth + 1)
                continue
            matched = next((s for p, s in patterns.items() if re.search(p, k)), None)
            if matched is not None:
                self._map_key(k, key_class, path)
                self.walk(v, matched, ctx, path + ("{}",), depth + 1)
            elif isinstance(extra, dict):
                self._map_key(k, key_class, path)
                self.walk(v, extra, ctx, path + ("{}",), depth + 1)
            elif extra is False:
                if not composed:
                    self.record(path + ("{unknown}",), "violation")
            elif composed and extra is None:
                continue  # declared by a composition branch that already walked it
            else:
                self._map_key(k, key_class, path)
                self.record(
                    path + ("{}",), node.get("x-content-class") or "unclassified"
                )

    def _map_key(self, key: str, key_class: str | None, path: tuple[str, ...]) -> None:
        self.record(path + ("{key}",), key_class or "unclassified")

    def _leaf(
        self, value: Any, node: dict[str, Any], cls: str | None, path: tuple[str, ...]
    ) -> None:
        if cls is None:
            self.record(path, "unclassified")
            return
        if "const" in node and value != node["const"]:
            self.record(path, "violation")
            return
        if "enum" in node and value not in node["enum"]:
            self.record(path, "violation")
            return
        if isinstance(value, str) and "pattern" in node and cls != "legacy_content":
            try:
                if not re.search(node["pattern"], value):
                    self.record(path, "violation")
                    return
            except re.error:
                pass
        if (
            isinstance(value, str)
            and "maxLength" in node
            and len(value) > node["maxLength"]
            and cls != "legacy_content"
        ):
            self.record(path, "violation")
            return
        self.record(path, cls)


def classify_body(body: Any, schema: Any, store: SchemaStore) -> Counter[str]:
    """Count the values of ``body`` per content class under ``schema`` (the route request schema)."""
    doc = store.document(BASE_URI + ENDPOINTS_DOC)
    ctx = _Ctx(store, BASE_URI + ENDPOINTS_DOC, (), doc)
    walker = _Walker()
    walker.walk(body, schema, ctx, ())
    return Counter(walker.findings.values())
