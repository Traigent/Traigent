"""AST scan of every raw HTTP egress construct in ``traigent/`` (a coverage AID).

This scan is an aid, not the guarantee: the guarantee is the default-deny backstop inside the
transport gate, which judges every request that goes through it. The scan exists to catch a
new raw call that bypasses the gate entirely. It resolves aliased imports (``import aiohttp
as aio``, ``from aiohttp import ClientSession as CS``) but cannot prove reachability.

A construct is GATED when it goes through the transport gate (where the optimization_only
backstop lives): a session/client wrapped in ``gate_session(...)``, a module-level
``requests``/``httpx`` call in a function that calls ``preflight_requests``, or a urllib
opener built with the gate handler. Everything else is reported so the test can compare it
with the documented bypass register.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2] / "traigent"
_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}


@dataclass(frozen=True)
class Construct:
    file: str
    line: int
    kind: str
    gated: bool


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):  # e.g. cast(Any, aiohttp).ClientSession
        parts.append(_dotted(node.func) + "()")
    return ".".join(reversed(parts))


def _kind(call: ast.Call, resolve: Any = lambda n: n) -> str | None:
    name = resolve(_dotted(call.func))
    # normalise ``from x import y`` forms to their dotted module spelling
    name = name.replace("aiohttp.client.ClientSession", "aiohttp.ClientSession")
    last = name.rsplit(".", 1)[-1]
    head = name.split(".", 1)[0]
    if last == "ClientSession" and (
        "aiohttp" in name or "aiohttp" in ast.unparse(call.func)
    ):
        return "aiohttp-session"
    if name == "requests.Session":
        return "requests-session"
    if head == "requests" and last in _VERBS and name.count(".") == 1:
        return "requests-call"
    if name in ("httpx.Client", "httpx.AsyncClient"):
        return "httpx-client"
    if head == "httpx" and last in _VERBS and name.count(".") == 1:
        return "httpx-call"
    if last == "urlopen":
        return "urllib-urlopen"
    if last == "build_opener":
        return "urllib-opener"
    if last in ("HTTPConnection", "HTTPSConnection"):
        return "http-connection"
    if name == "socket.create_connection":
        return "socket"
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> fully dotted original, for ``import x as y`` and ``from x import y as z``."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, aliases: dict[str, str] | None = None) -> None:
        self.rel = rel
        self.aliases = aliases or {}
        self.found: list[Construct] = []
        self._parents: list[ast.AST] = []
        self._functions: list[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        self._parents.append(node)
        is_fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_fn:
            self._functions.append(node)
        super().generic_visit(node)
        if is_fn:
            self._functions.pop()
        self._parents.pop()

    def _resolve(self, name: str) -> str:
        head, _, rest = name.partition(".")
        target = self.aliases.get(head)
        if target is None:
            return name
        return target + ("." + rest if rest else "")

    def visit_Call(self, node: ast.Call) -> None:
        kind = _kind(node, self._resolve)
        if kind is not None:
            self.found.append(
                Construct(self.rel, node.lineno, kind, self._gated(node, kind))
            )
        self.generic_visit(node)

    def _preceded_by_preflight(self) -> bool:
        """True iff the statement holding the raw call is immediately preceded, in the SAME
        block, by an unconditional ``preflight_requests(...)`` expression statement (so it
        dominates the call and cannot sit in an always-false branch)."""
        stmts = [n for n in self._parents if isinstance(n, ast.stmt)]
        if not stmts:
            return False
        current = stmts[-1]
        parent = (
            stmts[-2] if len(stmts) > 1 else self._parents[0] if self._parents else None
        )
        if parent is None:
            return False
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(parent, field, None)
            if isinstance(block, list) and current in block:
                index = block.index(current)
                if index == 0:
                    return False
                previous = block[index - 1]
                return (
                    isinstance(previous, ast.Expr)
                    and isinstance(previous.value, ast.Call)
                    and self._resolve(_dotted(previous.value.func)).endswith(
                        "preflight_requests"
                    )
                )
        return False

    def _gated(self, node: ast.Call, kind: str) -> bool:
        parent = self._parents[-1] if self._parents else None
        if kind in ("aiohttp-session", "requests-session", "httpx-client"):
            return (
                isinstance(parent, ast.Call)
                and self._resolve(_dotted(parent.func)).rsplit(".", 1)[-1]
                == "gate_session"
            )
        if kind in ("requests-call", "httpx-call"):
            return self._preceded_by_preflight()
        if kind == "urllib-opener":
            return isinstance(parent, ast.Call) and self._resolve(
                _dotted(parent.func)
            ).rsplit(".", 1)[-1] in ("gate_opener", "_gate_opener")
        return False


def scan() -> list[Construct]:
    found: list[Construct] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel == "labels/transport_gate.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _Visitor(rel, _aliases(tree))
        visitor.visit(tree)
        found.extend(visitor.found)
    return found
