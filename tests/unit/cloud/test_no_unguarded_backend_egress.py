"""Structural invariant: backend HTTP sends must be offline/no-egress guarded."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CLOUD_ROOT = REPO_ROOT / "traigent" / "cloud"

HTTP_METHODS = {"post", "put", "get", "patch", "delete", "request"}
PROJECT_BACKEND_HELPERS = {"resilient_backend_request"}
BOOLEAN_GUARD_CALL_NAMES = {
    "is_backend_offline",
    "backend_egress_disabled",
    "cloud_backend_egress_disabled",
}
POLICY_GUARD_CALL_NAMES = {
    "backend_egress_disabled",
    "cloud_backend_egress_disabled",
    "raise_if_cloud_egress_disabled",
}

# Explicit exceptions must be genuine bootstrap/auth sites with no resolved
# policy owner. None are currently needed; every backend send in traigent/cloud
# is policy-aware.
ALLOWLISTED_ENV_ONLY_SENDS: dict[tuple[str, str, int], str] = {}


@dataclass(frozen=True)
class CallRef:
    target: str
    lineno: int


@dataclass
class FunctionInfo:
    path: str
    qualname: str
    env_guard_lines: list[int] = field(default_factory=list)
    policy_guard_lines: list[int] = field(default_factory=list)
    env_raising_guard_lines: list[int] = field(default_factory=list)
    policy_raising_guard_lines: list[int] = field(default_factory=list)
    env_predicate_guard_lines: list[int] = field(default_factory=list)
    policy_predicate_guard_lines: list[int] = field(default_factory=list)
    calls: list[CallRef] = field(default_factory=list)
    conditional_calls: list[CallRef] = field(default_factory=list)


@dataclass(frozen=True)
class SendSite:
    path: str
    qualname: str
    lineno: int
    method: str
    source: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.path, self.qualname, self.lineno)


def _attr_chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attr_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        callee = _attr_chain(node.func)
        return f"{callee}()" if callee else "call()"
    if isinstance(node, ast.Subscript):
        parent = _attr_chain(node.value)
        return f"{parent}[]" if parent else "subscript[]"
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _looks_like_session_name(name: str) -> bool:
    return name == "session" or name.endswith("_session")


def _is_session_factory_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    callee = _attr_chain(node.func)
    return callee in {
        "requests.Session",
        "requests.sessions.Session",
        "aiohttp.ClientSession",
        "ClientSession",
    } or callee.endswith(".ClientSession")


def _is_http_receiver(base: str, method: str, session_vars: set[str]) -> bool:
    if base == "requests":
        return True
    if base == "requests.api" or base.startswith("requests.api."):
        return True
    if base == "request" and method == "urlopen":
        return True
    if base in session_vars:
        return True
    if base in {"session", "self.session", "self._session", "self._aio_session"}:
        return True
    if base in {
        "requests.Session()",
        "requests.sessions.Session()",
        "aiohttp.ClientSession()",
        "ClientSession()",
    }:
        return True
    if base.endswith(".ClientSession()"):
        return True
    return False


def _is_backend_send_call(node: ast.Call, session_vars: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in PROJECT_BACKEND_HELPERS
    if not isinstance(func, ast.Attribute):
        return False
    method = func.attr
    if method in PROJECT_BACKEND_HELPERS:
        return True
    if method == "urlopen":
        return _is_http_receiver(_attr_chain(func.value), method, session_vars)
    if method not in HTTP_METHODS:
        return False
    return _is_http_receiver(_attr_chain(func.value), method, session_vars)


def _is_literal_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _call_has_policy_argument(node: ast.Call) -> bool:
    name = _call_name(node.func)
    if name not in POLICY_GUARD_CALL_NAMES:
        return False
    if name == "raise_if_cloud_egress_disabled":
        return any(
            kw.arg == "no_egress" and not _is_literal_false(kw.value)
            for kw in node.keywords
        )
    return bool(node.args) or any(
        kw.arg == "no_egress" and not _is_literal_false(kw.value)
        for kw in node.keywords
    )


def _is_raising_guard_call(node: ast.Call) -> bool:
    name = _call_name(node.func)
    return name in {"raise_if_backend_offline", "raise_if_cloud_egress_disabled"}


def _is_boolean_guard_call(node: ast.Call) -> bool:
    return _call_name(node.func) in BOOLEAN_GUARD_CALL_NAMES


def _is_policy_guard_call(node: ast.Call) -> bool:
    return _call_has_policy_argument(node)


def _iter_positive_condition_calls(node: ast.AST) -> list[ast.Call]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return []
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, ast.BoolOp):
        calls: list[ast.Call] = []
        for value in node.values:
            calls.extend(_iter_positive_condition_calls(value))
        return calls
    if isinstance(node, ast.Compare):
        calls = _iter_positive_condition_calls(node.left)
        for comparator in node.comparators:
            calls.extend(_iter_positive_condition_calls(comparator))
        return calls
    return []


def _condition_guard_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in _iter_positive_condition_calls(node)
        if _is_boolean_guard_call(call)
    ]


def _body_exits(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(stmt, ast.Raise | ast.Return | ast.Break | ast.Continue)
        for stmt in body
    )


def _enclosing_qualnames(qualname: str) -> list[str]:
    parts = qualname.split(".")
    return [".".join(parts[:index]) for index in range(len(parts), 0, -1)]


class BackendEgressAnalyzer(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path.relative_to(REPO_ROOT).as_posix()
        self.source = source
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.session_var_stack: list[set[str]] = []
        self.functions: dict[tuple[str, str], FunctionInfo] = {}
        self.sends: list[SendSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        prefix = [*self.class_stack, *self.function_stack, node.name]
        qualname = ".".join(prefix)
        key = (self.path, qualname)
        self.functions.setdefault(key, FunctionInfo(path=self.path, qualname=qualname))
        self.function_stack.append(node.name)
        self.session_var_stack.append(self._session_arg_names(node))
        self.generic_visit(node)
        self.session_var_stack.pop()
        self.function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_session_factory_call(node.value):
            for target in node.targets:
                self._record_session_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _is_session_factory_call(node.value):
            self._record_session_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._record_context_session_vars(node.items)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._record_context_session_vars(node.items)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        current = self._current_function()
        if current is not None and _body_exits(node.body):
            info = self.functions[current]
            guard_calls = _condition_guard_calls(node.test)
            if guard_calls:
                if any(_is_policy_guard_call(call) for call in guard_calls):
                    info.policy_guard_lines.append(node.lineno)
                    info.policy_predicate_guard_lines.append(node.lineno)
                else:
                    info.env_guard_lines.append(node.lineno)
                    info.env_predicate_guard_lines.append(node.lineno)

            for call in _iter_positive_condition_calls(node.test):
                target = self._resolve_local_call(call.func)
                if target:
                    info.conditional_calls.append(
                        CallRef(target=target, lineno=node.lineno)
                    )

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        current = self._current_function()
        if current is not None and node.value is not None:
            guard_calls = _condition_guard_calls(node.value)
            if guard_calls:
                if any(_is_policy_guard_call(call) for call in guard_calls):
                    self.functions[current].policy_predicate_guard_lines.append(
                        node.lineno
                    )
                else:
                    self.functions[current].env_predicate_guard_lines.append(
                        node.lineno
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        current = self._current_function()
        if current is not None:
            info = self.functions[current]
            if _is_raising_guard_call(node):
                if _is_policy_guard_call(node):
                    info.policy_guard_lines.append(node.lineno)
                    info.policy_raising_guard_lines.append(node.lineno)
                else:
                    info.env_guard_lines.append(node.lineno)
                    info.env_raising_guard_lines.append(node.lineno)

            target = self._resolve_local_call(node.func)
            if target:
                info.calls.append(CallRef(target=target, lineno=node.lineno))

            if _is_backend_send_call(node, self._current_session_vars()):
                source = ast.get_source_segment(self.source, node) or _attr_chain(
                    node.func
                )
                self.sends.append(
                    SendSite(
                        path=self.path,
                        qualname=current[1],
                        lineno=node.lineno,
                        method=_call_name(node.func),
                        source=" ".join(source.split()),
                    )
                )

        self.generic_visit(node)

    def _session_arg_names(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> set[str]:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            args.append(node.args.vararg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg)
        return {arg.arg for arg in args if _looks_like_session_name(arg.arg)}

    def _current_session_vars(self) -> set[str]:
        if not self.session_var_stack:
            return set()
        return self.session_var_stack[-1]

    def _record_session_target(self, target: ast.AST) -> None:
        if not self.session_var_stack:
            return
        if isinstance(target, ast.Name):
            self.session_var_stack[-1].add(target.id)

    def _record_context_session_vars(self, items: list[ast.withitem]) -> None:
        for item in items:
            if item.optional_vars is not None and _is_session_factory_call(
                item.context_expr
            ):
                self._record_session_target(item.optional_vars)

    def _current_function(self) -> tuple[str, str] | None:
        if not self.function_stack:
            return None
        return (
            self.path,
            ".".join([*self.class_stack, *self.function_stack]),
        )

    def _resolve_local_call(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            receiver = _attr_chain(func.value)
            if receiver == "self" and self.class_stack:
                return f"{self.class_stack[-1]}.{func.attr}"
            return func.attr
        return ""


def _analyze_cloud_sources() -> tuple[
    dict[tuple[str, str], FunctionInfo], list[SendSite]
]:
    functions: dict[tuple[str, str], FunctionInfo] = {}
    sends: list[SendSite] = []

    for path in sorted(CLOUD_ROOT.glob("*.py")):
        source = path.read_text()
        analyzer = BackendEgressAnalyzer(path, source)
        analyzer.visit(ast.parse(source, filename=str(path)))
        functions.update(analyzer.functions)
        sends.extend(analyzer.sends)

    return functions, sends


def _target_keys_for_call(
    functions: dict[tuple[str, str], FunctionInfo],
    caller: FunctionInfo,
    target: str,
) -> list[tuple[str, str]]:
    same_module = (caller.path, target)
    if same_module in functions:
        return [same_module]

    matches = [key for key in functions if key[1].endswith(f".{target}")]
    if len(matches) == 1:
        return matches
    return []


def _function_invokes_policy_guard(
    functions: dict[tuple[str, str], FunctionInfo],
    key: tuple[str, str],
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    if key in seen:
        return False
    info = functions.get(key)
    if info is None:
        return False
    if info.policy_raising_guard_lines:
        return True

    next_seen = seen | {key}
    for call in info.calls:
        for target_key in _target_keys_for_call(functions, info, call.target):
            if _function_invokes_policy_guard(functions, target_key, next_seen):
                return True
    return False


def _function_checks_policy_guard_predicate(
    functions: dict[tuple[str, str], FunctionInfo],
    key: tuple[str, str],
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    if key in seen:
        return False
    info = functions.get(key)
    if info is None:
        return False
    if info.policy_predicate_guard_lines:
        return True

    next_seen = seen | {key}
    for call in info.calls:
        for target_key in _target_keys_for_call(functions, info, call.target):
            if _function_checks_policy_guard_predicate(
                functions, target_key, next_seen
            ):
                return True
    return False


def _function_policy_guards_before_send(
    functions: dict[tuple[str, str], FunctionInfo],
    key: tuple[str, str],
    send_line: int,
) -> bool:
    info = functions.get(key)
    if info is None:
        return False
    if any(line < send_line for line in info.policy_guard_lines):
        return True

    for call in info.calls:
        if call.lineno >= send_line:
            continue
        for target_key in _target_keys_for_call(functions, info, call.target):
            if _function_invokes_policy_guard(functions, target_key):
                return True

    for call in info.conditional_calls:
        if call.lineno >= send_line:
            continue
        for target_key in _target_keys_for_call(functions, info, call.target):
            if _function_checks_policy_guard_predicate(
                functions, target_key
            ) or _function_invokes_policy_guard(functions, target_key):
                return True
    return False


def _site_is_policy_guarded(
    functions: dict[tuple[str, str], FunctionInfo], site: SendSite
) -> bool:
    for qualname in _enclosing_qualnames(site.qualname):
        if _function_policy_guards_before_send(
            functions, (site.path, qualname), site.lineno
        ):
            return True
    return False


def _analyze_synthetic_cloud_source(
    source: str,
) -> tuple[dict[tuple[str, str], FunctionInfo], list[SendSite]]:
    path = CLOUD_ROOT / "synthetic_fixture.py"
    normalized = textwrap.dedent(source)
    analyzer = BackendEgressAnalyzer(path, normalized)
    analyzer.visit(ast.parse(normalized, filename=str(path)))
    return analyzer.functions, analyzer.sends


def _unguarded_violations(
    functions: dict[tuple[str, str], FunctionInfo],
    sends: list[SendSite],
) -> list[SendSite]:
    return [site for site in sends if not _site_is_policy_guarded(functions, site)]


def test_detector_flags_unguarded_session_send_patterns() -> None:
    functions, sends = _analyze_synthetic_cloud_source(
        """
        import aiohttp
        import requests

        def requests_session_factory():
            requests.Session().post("https://backend.example/api")

        async def aiohttp_context_manager_session():
            async with aiohttp.ClientSession() as s:
                await s.post("https://backend.example/api")

        def local_http_session_variable(http_session):
            http_session.delete("https://backend.example/api")

        def requests_request_style():
            requests.request("POST", "https://backend.example/api")

        def requests_api_style():
            requests.api.put("https://backend.example/api")
        """
    )

    assert len(sends) == 5
    assert {
        "requests_session_factory",
        "aiohttp_context_manager_session",
        "local_http_session_variable",
        "requests_request_style",
        "requests_api_style",
    } == {site.qualname for site in sends}
    assert _unguarded_violations(functions, sends) == sends


def test_detector_rejects_ignored_boolean_offline_predicate() -> None:
    functions, sends = _analyze_synthetic_cloud_source(
        """
        import requests
        from traigent.utils.env_config import is_backend_offline

        def ignored_boolean_predicate():
            is_backend_offline()
            requests.post("https://backend.example/api")
        """
    )

    assert len(sends) == 1
    assert _unguarded_violations(functions, sends) == sends


def test_detector_accepts_fail_closed_guard_control_flow() -> None:
    functions, sends = _analyze_synthetic_cloud_source(
        """
        import requests
        from traigent.cloud.client import (
            cloud_backend_egress_disabled,
            raise_if_cloud_egress_disabled,
        )

        class PolicyAwareClient:
            def guarded_by_boolean_branch(self):
                if cloud_backend_egress_disabled(self.no_egress):
                    return None
                requests.post("https://backend.example/api")

            def guarded_by_raising_guard(self):
                raise_if_cloud_egress_disabled("send", no_egress=self.no_egress)
                requests.post("https://backend.example/api")
        """
    )

    assert len(sends) == 2
    assert _unguarded_violations(functions, sends) == []


def test_detector_flags_env_only_guard_at_policy_aware_site() -> None:
    functions, sends = _analyze_synthetic_cloud_source(
        """
        import requests
        from traigent.utils.env_config import (
            is_backend_offline,
            raise_if_backend_offline,
        )

        class PolicyAwareClient:
            def boolean_env_only_guard(self):
                if is_backend_offline():
                    return None
                requests.post("https://backend.example/api")

            def raising_env_only_guard(self):
                raise_if_backend_offline("send")
                requests.post("https://backend.example/api")
        """
    )

    assert len(sends) == 2
    assert _unguarded_violations(functions, sends) == sends


def test_no_unguarded_backend_http_sends() -> None:
    assert all(reason.strip() for reason in ALLOWLISTED_ENV_ONLY_SENDS.values())

    functions, sends = _analyze_cloud_sources()
    assert sends, "structural egress invariant found no backend send sites"

    violations = []
    for site in sends:
        if site.key in ALLOWLISTED_ENV_ONLY_SENDS:
            continue
        if site in _unguarded_violations(functions, [site]):
            violations.append(
                f"{site.path}:{site.lineno} in {site.qualname} "
                f"({site.method}): {site.source}"
            )

    assert not violations, (
        "backend sends missing policy no_egress guard:\n" + "\n".join(violations)
    )
