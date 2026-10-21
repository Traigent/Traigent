"""AST census of HTTP client construction and module-level HTTP calls.

Resolves import aliases (``import requests as r``, ``from httpx import AsyncClient as X``,
``from requests import post as emit``) and simple assignments (``factory =
httpx.AsyncClient``), then reports every construction not directly wrapped by
``gate_session`` and every module-level call whose enclosing function has no
``preflight_requests`` call.
"""

from __future__ import annotations

import ast

CTORS = {
    "aiohttp.ClientSession",
    "httpx.AsyncClient",
    "httpx.Client",
    "requests.Session",
}
_VERBS = ("get", "post", "put", "patch", "delete", "head", "options", "request")
CALLS = {f"requests.{v}" for v in _VERBS} | {f"httpx.{v}" for v in (*_VERBS, "stream")}
TRACKED = CTORS | CALLS
MODULES = {"aiohttp", "httpx", "requests"}
GATE = "traigent.labels.transport_gate.gate_session"
PREFLIGHT = "traigent.labels.transport_gate.preflight_requests"


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    """name -> canonical dotted target, for imports and simple re-bindings."""
    alias: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                alias[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0]
                )
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                alias[a.asname or a.name] = f"{n.module}.{a.name}"
    for _ in range(3):  # a few passes: chains of simple assignments
        for n in ast.walk(tree):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(n, ast.Assign):
                targets, value = list(n.targets), n.value
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                targets, value = [n.target], n.value
            if value is None:
                continue
            canon = _resolve(value, alias)
            if canon:
                for t in targets:
                    if isinstance(t, ast.Name):
                        alias[t.id] = canon
    return alias


def _resolve(node: ast.AST, alias: dict[str, str]) -> str | None:
    name = _dotted(node)
    if not name:
        return None
    head, _, rest = name.partition(".")
    if head in alias:
        return alias[head] + ("." + rest if rest else "")
    return name


def _preceded_by_preflight(
    stmt: ast.AST, parents: dict[ast.AST, ast.AST], alias: dict[str, str]
) -> bool:
    """The statement just before ``stmt`` in the same block is a preflight_requests call."""
    parent = parents.get(stmt)
    if parent is None:
        return False
    for field in ("body", "orelse", "finalbody", "handlers"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and stmt in block:
            i = block.index(stmt)
            if i == 0:
                return False
            prev = block[i - 1]
            return (
                isinstance(prev, ast.Expr)
                and isinstance(prev.value, ast.Call)
                and _resolve(prev.value.func, alias) == PREFLIGHT
            )
    return False


def census(src: str, rel: str = "<src>") -> list[str]:
    tree = ast.parse(src)
    alias = _aliases(tree)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    problems: list[str] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        canon = _resolve(n.func, alias)
        if canon in CTORS:
            parent = parents.get(n)
            if not (
                isinstance(parent, ast.Call) and _resolve(parent.func, alias) == GATE
            ):
                problems.append(f"{rel}:{n.lineno} {canon} not wrapped in gate_session")
        elif canon in CALLS:
            stmt: ast.AST = n
            while not isinstance(stmt, ast.stmt):
                stmt = parents[stmt]
            if not _preceded_by_preflight(stmt, parents, alias):
                problems.append(
                    f"{rel}:{n.lineno} {canon} not immediately preceded by preflight_requests"
                )
    return problems
