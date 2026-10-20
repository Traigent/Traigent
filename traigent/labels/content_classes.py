"""Classify a request (path parameters, query string, body) by content class.

``content_class_table.json`` is generated from the TraigentSchema optimization endpoint
inventory (``scripts/generate_content_class_table.py``; provenance pinned by a test).
It holds, per route, the resolved ``x-content-class`` tree of the body, path parameters
and query parameters. This module walks a request next to it and counts values per class.
It returns COUNTS ONLY: never a value, a map key or a path that could be
customer-chosen.

Findings that block in "refuse" mode: ``legacy_content`` (the transitional class),
``unclassified`` (a value, key or parameter no annotation covers) and ``violation`` (a
value that breaks its own class: not in the enum, wrong pattern, unknown key of a closed
object).
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

TABLE_PATH = Path(__file__).with_name("content_class_table.json")
BLOCKING = ("legacy_content", "unclassified", "violation")
_WORST = ("violation", "legacy_content", "unclassified")
_MAX_DEPTH = 64
_MAX_NODES = 200_000
_V1BETA = re.compile(r"^/api/v1beta/projects/[^/]+")


class TableError(RuntimeError):
    """The content-class table cannot be loaded."""


class Table:
    def __init__(self, data: dict[str, Any]) -> None:
        self.nodes: list[Any] = data["nodes"]
        self.source: dict[str, str] = data["source"]
        self.routes: list[tuple[re.Pattern[str], dict[str, Any]]] = []
        for route in data["routes"]:
            names = re.findall(r"\{([^}]+)\}", route["template"])
            rx = re.escape(route["template"])
            for name in names:
                rx = rx.replace(re.escape("{" + name + "}"), f"(?P<{name}>[^/]+)")
            self.routes.append((re.compile("^" + rx + "/?$"), route))

    def find(
        self, method: str, path: str
    ) -> tuple[dict[str, Any], dict[str, str]] | None:
        path = _V1BETA.sub("/api/v1", path)
        for rx, route in self.routes:
            m = rx.match(path)
            if m and route["method"] == method.upper():
                return route, m.groupdict()
        return None


_LOCK = threading.Lock()
_TABLES: dict[str, Table] = {}


def get_table(path: Path | None = None) -> Table:
    p = path or TABLE_PATH
    with _LOCK:
        if str(p) not in _TABLES:
            try:
                _TABLES[str(p)] = Table(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError) as exc:
                raise TableError("content-class table unreadable") from exc
        return _TABLES[str(p)]


def _type_ok(value: Any, node: dict[str, Any]) -> bool:
    t = node.get("t")
    if t is None:
        return True
    for name in t if isinstance(t, list) else [t]:
        if (
            (name == "object" and isinstance(value, dict))
            or (name == "array" and isinstance(value, list))
            or (name == "string" and isinstance(value, str))
            or (name == "boolean" and isinstance(value, bool))
            or (
                name == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
            or (
                name == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            or (name == "null" and value is None)
        ):
            return True
    return False


class _Walker:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.findings: dict[tuple[str, ...], str] = {}
        self.nodes = 0

    def record(self, path: tuple[str, ...], cls: str) -> None:
        old = self.findings.get(path)
        if (
            old is None
            or (cls in _WORST and old not in _WORST)
            or (
                cls in _WORST
                and old in _WORST
                and _WORST.index(cls) < _WORST.index(old)
            )
        ):
            self.findings[path] = cls

    def target(self, node: dict[str, Any]) -> dict[str, Any]:
        if "ref" not in node:
            return node
        base = self.table.nodes[node["ref"]]
        merged = {**base, **{k: v for k, v in node.items() if k != "ref"}}
        return (
            self.target(merged)
            if "ref" in merged and merged["ref"] != node["ref"]
            else merged
        )

    def walk(
        self, value: Any, node: Any, path: tuple[str, ...], depth: int = 0
    ) -> None:
        self.nodes += 1
        if depth > _MAX_DEPTH or self.nodes > _MAX_NODES:
            self.record(path, "unclassified")
            return
        if value is None:
            return
        if not isinstance(node, dict) or node.get("never"):
            self.record(path, "violation")
            return
        node = self.target(node)
        composed = False
        for key in ("all",):
            for branch in node.get(key, ()):
                self.walk(value, branch, path, depth + 1)
                composed = True
        for key in ("one", "any"):
            if key in node:
                self._alternatives(value, node[key], path, depth)
                composed = True
        if isinstance(value, dict):
            self._object(value, node, path, depth, composed)
        elif isinstance(value, list):
            items = node.get("i")
            for item in value:
                if isinstance(items, dict):
                    self.walk(item, items, path + ("[]",), depth + 1)
                elif not composed:
                    self.record(path + ("[]",), "unclassified")
        elif not (composed and "c" not in node):
            self._leaf(value, node, path)

    def _alternatives(
        self, value: Any, branches: list[Any], path: tuple[str, ...], depth: int
    ):
        best: _Walker | None = None
        for branch in branches:
            b = self.target(branch) if isinstance(branch, dict) else branch
            if not isinstance(b, dict) or not _type_ok(value, b):
                continue
            if isinstance(value, dict) and any(r not in value for r in b.get("r", ())):
                continue
            probe = _Walker(self.table)
            probe.walk(value, branch, path, depth + 1)
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

    def _object(self, value, node, path, depth, composed) -> None:
        props = node.get("p") or {}
        patterns = node.get("pp") or {}
        extra = node.get("ap")
        key_class = node.get("kc")
        for k, v in value.items():
            if k in props:
                self.walk(v, props[k], path + (k,), depth + 1)
                continue
            matched = next((s for p, s in patterns.items() if re.search(p, k)), None)
            if matched is not None or isinstance(extra, dict):
                self.record(path + ("{key}",), key_class or "unclassified")
                self.walk(
                    v,
                    matched if matched is not None else extra,
                    path + ("{}",),
                    depth + 1,
                )
            elif extra is False:
                if not composed:
                    self.record(path + ("{unknown}",), "violation")
            elif composed and extra is None:
                continue
            else:
                self.record(path + ("{key}",), key_class or "unclassified")
                self.record(path + ("{}",), node.get("c") or "unclassified")

    def _leaf(self, value: Any, node: dict[str, Any], path: tuple[str, ...]) -> None:
        cls = node.get("c")
        if cls is None:
            self.record(path, "unclassified")
            return
        if "k" in node and value != node["k"][0]:
            self.record(path, "violation")
        elif "e" in node and value not in node["e"]:
            self.record(path, "violation")
        elif (
            isinstance(value, str)
            and cls != "legacy_content"
            and (
                ("pt" in node and not _search(node["pt"], value))
                or ("ml" in node and len(value) > node["ml"])
                or ("mn" in node and len(value) < node["mn"])
            )
        ):
            self.record(path, "violation")
        else:
            self.record(path, cls)


def _search(pattern: str, value: str) -> bool:
    try:
        return re.search(pattern, value) is not None
    except re.error:
        return True


def classify_request(
    table: Table,
    method: str,
    path: str,
    query: str,
    body: Any,
    *,
    has_body: bool,
) -> tuple[str, Counter[str]] | None:
    """Count the values of one request per content class; ``None`` if the route is unknown."""
    found = table.find(method, path)
    if found is None:
        return None
    route, path_values = found
    w = _Walker(table)
    for name, value in path_values.items():
        node = route["path"].get(name)
        if node is None or node.get("unknown"):
            # No Schema class for this path parameter: not provably content-free.
            w.record(("path", name), "legacy_content")
        else:
            w.walk(value, node, ("path", name))
    for name, value in parse_qsl(query, keep_blank_values=True):
        node = route["query"].get(name)
        if node is None:
            w.record(("query", name), "unclassified")
        elif node.get("t") == "array":
            for part in value.split(","):
                w.walk(part, node.get("i", {}), ("query", name, "[]"))
        else:
            w.walk(value, node, ("query", name))
    if has_body:
        if route["body"] is None:
            w.record(("body",), "violation")  # a body on a route that declares none
        else:
            w.walk(body, route["body"], ("body",))
    return route["template"], Counter(w.findings.values())
