"""AST scan of every raw HTTP egress construct in ``traigent/`` (coverage proof helper).

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


def _kind(call: ast.Call) -> str | None:
    name = _dotted(call.func)
    last = name.rsplit(".", 1)[-1]
    head = name.split(".", 1)[0]
    if last == "ClientSession" and ("aiohttp" in name or name == "ClientSession"):
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


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
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

    def visit_Call(self, node: ast.Call) -> None:
        kind = _kind(node)
        if kind is not None:
            self.found.append(
                Construct(self.rel, node.lineno, kind, self._gated(node, kind))
            )
        self.generic_visit(node)

    def _gated(self, node: ast.Call, kind: str) -> bool:
        parent = self._parents[-1] if self._parents else None
        if kind in ("aiohttp-session", "requests-session", "httpx-client"):
            return (
                isinstance(parent, ast.Call)
                and _dotted(parent.func).rsplit(".", 1)[-1] == "gate_session"
            )
        if kind in ("requests-call", "httpx-call"):
            fn = self._functions[-1] if self._functions else None
            return fn is not None and any(
                isinstance(n, ast.Call)
                and _dotted(n.func).endswith("preflight_requests")
                for n in ast.walk(fn)
            )
        if kind == "urllib-opener":
            return isinstance(parent, ast.Call) and _dotted(parent.func).rsplit(".", 1)[
                -1
            ] in ("gate_opener", "_gate_opener")
        return False


def scan() -> list[Construct]:
    found: list[Construct] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel == "labels/transport_gate.py":
            continue
        visitor = _Visitor(rel)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        found.extend(visitor.found)
    return found
