"""Structural invariant: wire/cost calls must not fail silently (Phase D, #1727).

Phase D of the silent-failure audit (#1720). This is a report-only, ratcheted
AST lint: it never blocks on *existing* violations (they live in
``silent_baseline.json``), only on *new* ones, so it can land without a large
cleanup. Set ``SILENT_LINT_STRICT=1`` to ignore the baseline entirely and fail
on every violation currently in scope (used by mutation self-checks below and
by eventual full enforcement once the baseline is worked down to zero).

Detected patterns, scoped to ``traigent/cloud/**``, ``traigent/cli/**``, and
the pricing/cost modules (``traigent/utils/cost_calculator.py``,
``traigent/core/cost_estimator.py``, ``traigent/core/cost_enforcement.py``):

    (a) ``bare_except_pass`` -- ``except <anything>:`` whose body is only
        ``pass`` (docstring-only prefix allowed).
    (b) ``swallow_and_substitute`` -- an except body consisting of zero or
        more ``logger.*``/``print`` statements followed by a terminal
        ``return <default>`` (``None``/``{}``/``[]``/``0``/``False``/``dict()``/
        ``list()``) or ``continue`` -- i.e. the failure is reported (or not
        even that) and then papered over with a default value.
    (c) ``suppress_wire_or_cost`` -- ``contextlib.suppress(...)``/``suppress(...)``
        wrapping a block that contains a call (in these scoped modules, any
        call inside a suppress block is effectively a wire/cost call).

Escape hatch: a ``# silent-ok: <reason>`` comment on the ``except``/``with``
line (or the line directly above) exempts that specific site. The reason must
be non-empty -- ``# silent-ok:`` alone does not exempt anything.

Mirrors the structure/style of
``tests/unit/cloud/test_no_unguarded_backend_egress.py``: a small
AST-walking analyzer plus straightforward assertions, no external linting
framework.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOUD_ROOT = REPO_ROOT / "traigent" / "cloud"
CLI_ROOT = REPO_ROOT / "traigent" / "cli"

SCOPE_DIRS: tuple[Path, ...] = (CLOUD_ROOT, CLI_ROOT)
SCOPE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "traigent" / "utils" / "cost_calculator.py",
    REPO_ROOT / "traigent" / "core" / "cost_estimator.py",
    REPO_ROOT / "traigent" / "core" / "cost_enforcement.py",
)

BASELINE_PATH = Path(__file__).resolve().parent / "silent_baseline.json"
STRICT_ENV_VAR = "SILENT_LINT_STRICT"

CATEGORY_BARE_EXCEPT_PASS = "bare_except_pass"
CATEGORY_SWALLOW_SUBSTITUTE = "swallow_and_substitute"
CATEGORY_SUPPRESS_WIRE_OR_COST = "suppress_wire_or_cost"
ALL_CATEGORIES = frozenset(
    {
        CATEGORY_BARE_EXCEPT_PASS,
        CATEGORY_SWALLOW_SUBSTITUTE,
        CATEGORY_SUPPRESS_WIRE_OR_COST,
    }
)

LOG_ATTR_NAMES = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "log",
}
SILENT_OK_RE = re.compile(r"#\s*silent-ok:\s*(\S.*)")


@dataclass(frozen=True)
class Violation:
    path: str
    qualname: str
    category: str
    lineno: int
    snippet: str

    @property
    def snippet_hash(self) -> str:
        return hashlib.sha256(self.snippet.encode("utf-8")).hexdigest()[:12]

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.path, self.qualname, self.category, self.snippet_hash)

    def to_baseline_dict(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "qualname": self.qualname,
            "category": self.category,
            "snippet_hash": self.snippet_hash,
            "lineno": self.lineno,
            "snippet": self.snippet,
        }


def _attr_chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attr_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        callee = _attr_chain(node.func)
        return f"{callee}()" if callee else "call()"
    return ""


def _is_log_or_print_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "print"
    if isinstance(func, ast.Attribute):
        if func.attr not in LOG_ATTR_NAMES:
            return False
        base = _attr_chain(func.value).lower()
        return "log" in base
    return False


def _is_default_substitute_value(node: ast.AST | None) -> bool:
    """True if ``node`` is a "give up quietly" default (bare return counts)."""
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return True
        if isinstance(value, bool):
            return value is False
        if isinstance(value, int | float) and value == 0:
            return True
        return False
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.List):
        return not node.elts
    if isinstance(node, ast.Call):
        name = _attr_chain(node.func)
        return name in {"dict", "list"} and not node.args and not node.keywords
    return False


def _is_swallow_terminal_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Continue):
        return True
    if isinstance(stmt, ast.Return):
        return _is_default_substitute_value(stmt.value)
    return False


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _classify_except_body(body: list[ast.stmt]) -> str | None:
    stmts = _strip_leading_docstring(body)
    if not stmts:
        return None
    if len(stmts) == 1 and isinstance(stmts[0], ast.Pass):
        return CATEGORY_BARE_EXCEPT_PASS

    *prefix, terminal = stmts
    if _is_swallow_terminal_stmt(terminal) and all(
        isinstance(stmt, ast.Expr) and _is_log_or_print_call(stmt.value)
        for stmt in prefix
    ):
        return CATEGORY_SWALLOW_SUBSTITUTE
    return None


def _is_contextlib_suppress_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return _attr_chain(node.func) in {"contextlib.suppress", "suppress"}


def _body_contains_call(body: list[ast.stmt]) -> bool:
    for stmt in body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call):
                return True
    return False


def _silent_ok_reason(lines: list[str], lineno: int) -> str | None:
    """Non-empty ``# silent-ok: <reason>`` on ``lineno`` or the line above."""
    for candidate in (lineno, lineno - 1):
        if candidate < 1 or candidate > len(lines):
            continue
        match = SILENT_OK_RE.search(lines[candidate - 1])
        if match:
            reason = match.group(1).strip()
            if reason:
                return reason
    return None


def _qualname(class_stack: list[str], function_stack: list[str]) -> str:
    parts = [*class_stack, *function_stack]
    return ".".join(parts) if parts else "<module>"


class SilentFailureAnalyzer(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path.relative_to(REPO_ROOT).as_posix()
        self.source = source
        self.lines = source.splitlines()
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.violations: list[Violation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        category = _classify_except_body(node.body)
        if category is not None:
            self._record(node, category)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_suppress(node, node.body)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._check_suppress(node, node.body)
        self.generic_visit(node)

    def _check_suppress(
        self, node: ast.With | ast.AsyncWith, body: list[ast.stmt]
    ) -> None:
        for item in node.items:
            if _is_contextlib_suppress_call(item.context_expr) and _body_contains_call(
                body
            ):
                self._record(node, CATEGORY_SUPPRESS_WIRE_OR_COST)
                return

    def _record(self, node: ast.AST, category: str) -> None:
        if _silent_ok_reason(self.lines, node.lineno) is not None:
            return
        snippet = ast.get_source_segment(self.source, node) or ""
        self.violations.append(
            Violation(
                path=self.path,
                qualname=_qualname(self.class_stack, self.function_stack),
                category=category,
                lineno=node.lineno,
                snippet=" ".join(snippet.split()),
            )
        )


def _iter_scope_files() -> list[Path]:
    files: list[Path] = []
    for scope_dir in SCOPE_DIRS:
        files.extend(sorted(scope_dir.rglob("*.py")))
    files.extend(path for path in SCOPE_FILES if path.exists())
    return files


def _analyze_file(path: Path) -> list[Violation]:
    source = path.read_text()
    analyzer = SilentFailureAnalyzer(path, source)
    analyzer.visit(ast.parse(source, filename=str(path)))
    return analyzer.violations


def _analyze_scope() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_scope_files():
        violations.extend(_analyze_file(path))
    return violations


def _analyze_source(source: str, path: Path) -> list[Violation]:
    """Analyze a synthetic source string as if it lived at ``path``.

    Mirrors ``test_no_unguarded_backend_egress.py``'s
    ``_analyze_synthetic_cloud_source`` -- the path need not exist on disk,
    it only anchors ``qualname``/scope-relative reporting.
    """
    normalized = textwrap.dedent(source)
    analyzer = SilentFailureAnalyzer(path, normalized)
    analyzer.visit(ast.parse(normalized, filename=str(path)))
    return analyzer.violations


def _load_baseline() -> list[dict[str, str | int]]:
    if not BASELINE_PATH.exists():
        return []
    return json.loads(BASELINE_PATH.read_text())


def _baseline_keys(
    baseline: list[dict[str, str | int]],
) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(entry["path"]),
            str(entry["qualname"]),
            str(entry["category"]),
            str(entry["snippet_hash"]),
        )
        for entry in baseline
    }


def _new_violations(
    violations: list[Violation],
    baseline_keys: set[tuple[str, str, str, str]],
    strict: bool,
) -> list[Violation]:
    """Violations that fail the gate.

    Report-only mode: anything already known to the baseline is accepted, so
    only genuinely new violations fail. Strict mode ignores the baseline
    entirely -- every violation currently in scope fails the gate.
    """
    if strict:
        return list(violations)
    return [v for v in violations if v.key not in baseline_keys]


def _is_strict_mode() -> bool:
    return os.environ.get(STRICT_ENV_VAR, "") == "1"


def test_silent_baseline_is_well_formed() -> None:
    baseline = _load_baseline()
    seen: set[tuple[str, str, str, str]] = set()
    for entry in baseline:
        key = (
            str(entry["path"]),
            str(entry["qualname"]),
            str(entry["category"]),
            str(entry["snippet_hash"]),
        )
        assert key not in seen, f"duplicate baseline entry: {key}"
        seen.add(key)
        assert entry["category"] in ALL_CATEGORIES, (
            f"unknown category in baseline: {entry['category']}"
        )


def test_no_new_silent_wire_failure_violations() -> None:
    """Report-only ratchet: pass unless a *new* silent-failure site appears.

    ``SILENT_LINT_STRICT=1`` ignores the baseline and fails on every
    violation currently in scope -- use that to see (and work down) the
    full count.
    """
    violations = _analyze_scope()
    baseline = _load_baseline()
    baseline_keys = _baseline_keys(baseline)
    strict = _is_strict_mode()

    new = _new_violations(violations, baseline_keys, strict)

    print(f"\n[silent-lint] total violations in scope: {len(violations)}")
    print(f"[silent-lint] baseline size: {len(baseline_keys)}")
    print(f"[silent-lint] strict mode ({STRICT_ENV_VAR}=1): {strict}")
    if new:
        print(f"[silent-lint] {len(new)} violation(s) failing the gate:")
        for v in sorted(new, key=lambda v: (v.path, v.lineno)):
            print(f"  {v.path}:{v.lineno} in {v.qualname} [{v.category}] {v.snippet}")

    assert not new, (
        f"{len(new)} silent-failure violation(s) are not accounted for in "
        f"the baseline ({BASELINE_PATH}). If intentional, add a "
        f"`# silent-ok: <reason>` comment on the except/with line; otherwise "
        f"fix the swallow. Run with {STRICT_ENV_VAR}=1 for the full current "
        f"list."
    )


def test_mutation_bare_except_pass_detected_in_strict_mode() -> None:
    """Self-verification: prove the checker actually catches the pattern.

    Injects a synthetic ``except Exception: pass`` into a scanned-scope AST
    and asserts (1) the analyzer detects it, and (2) strict mode fails the
    gate even when a baseline claims to already know about the exact same
    violation key -- i.e. strict mode cannot be satisfied by a stale
    baseline.
    """
    synthetic = """
        def handle_send():
            try:
                do_wire_call()
            except Exception:
                pass
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    assert len(violations) == 1
    assert violations[0].category == CATEGORY_BARE_EXCEPT_PASS

    baseline_keys = {v.key for v in violations}
    assert _new_violations(violations, baseline_keys, strict=False) == []
    assert _new_violations(violations, baseline_keys, strict=True) == violations
    assert _new_violations(violations, set(), strict=True) == violations


def test_mutation_swallow_and_substitute_detected() -> None:
    synthetic = """
        def fetch_cost():
            try:
                return call_cost_backend()
            except Exception:
                logger.warning("cost backend failed")
                return None
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    matches = [v for v in violations if v.category == CATEGORY_SWALLOW_SUBSTITUTE]
    assert len(matches) == 1
    assert _new_violations(matches, set(), strict=True) == matches


def test_mutation_swallow_and_substitute_detects_continue_with_no_log() -> None:
    synthetic = """
        def process_items(items):
            for item in items:
                try:
                    send(item)
                except Exception:
                    continue
        """
    violations = _analyze_source(synthetic, CLI_ROOT / "synthetic_mutation_fixture.py")

    assert len(violations) == 1
    assert violations[0].category == CATEGORY_SWALLOW_SUBSTITUTE


def test_mutation_suppress_over_call_detected() -> None:
    synthetic = """
        import contextlib

        def fire_and_forget_send():
            with contextlib.suppress(Exception):
                requests.post("https://backend.example/api")
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    matches = [v for v in violations if v.category == CATEGORY_SUPPRESS_WIRE_OR_COST]
    assert len(matches) == 1
    assert _new_violations(matches, set(), strict=True) == matches


def test_mutation_suppress_without_call_is_not_flagged() -> None:
    """``with suppress(CancelledError): await task`` (no Call) is not flagged.

    This is the real pattern used for asyncio task-cancellation cleanup in
    this repo (``traigent/cloud/billing.py``, ``traigent/cloud/optimizer_client.py``)
    and is not a wire/cost swallow.
    """
    synthetic = """
        import asyncio
        from contextlib import suppress

        async def stop():
            with suppress(asyncio.CancelledError):
                await some_task
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    assert violations == []


def test_silent_ok_escape_hatch_exempts_site() -> None:
    synthetic = """
        def handle_send():
            try:
                do_wire_call()
            except Exception:  # silent-ok: best-effort telemetry flush, not user-facing
                pass
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    assert violations == []


def test_silent_ok_escape_hatch_requires_nonempty_reason() -> None:
    synthetic = """
        def handle_send():
            try:
                do_wire_call()
            except Exception:  # silent-ok:
                pass
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    assert len(violations) == 1
    assert violations[0].category == CATEGORY_BARE_EXCEPT_PASS


def test_silent_ok_escape_hatch_on_line_above_also_exempts() -> None:
    synthetic = """
        def handle_send():
            try:
                do_wire_call()
            # silent-ok: legacy best-effort cleanup, tracked in TICKET-123
            except Exception:
                pass
        """
    violations = _analyze_source(
        synthetic, CLOUD_ROOT / "synthetic_mutation_fixture.py"
    )

    assert violations == []
