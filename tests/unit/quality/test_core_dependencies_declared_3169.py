"""#3169: every unguarded top-level third-party import in traigent/ is declared.

A module-level ``import yaml`` that resolves only through another package's
dependencies breaks silently when that upstream drops it. Imports guarded by
``try/except ImportError`` (optional features) or ``if TYPE_CHECKING`` are exempt.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "traigent"

# import name -> distribution name, where they differ
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "jwt": "pyjwt",
    "bm25": "rank-bm25",
    "traigent_schema": "traigent-schema",
}

# Top-level imports deliberately not in [project].dependencies. Keep this tight:
# add an entry only with a reason.
ALLOWLIST = {
    "traigent": "this package",
    "traigent_validation": "in-repo top-level package (traigent_validation/)",
}

# Pre-existing undeclared imports outside the scope of #3169 (yaml, packaging,
# urllib3). They resolve transitively today (numpy/httpx/dotenv via litellm,
# opentelemetry and langchain_* via extras). Each needs its own decision on
# declaring versus guarding. Scoped to files so a NEW unguarded import of the
# same package elsewhere still fails; do not grow this table.
KNOWN_UNDECLARED_FOLLOW_UP: dict[str, set[str]] = {
    "numpy": {
        "traigent/cloud/subset_selection.py",
        "traigent/metrics/ragas_metrics.py",
        "traigent/utils/optimization_logger.py",
        "traigent/utils/optimization_analyzer.py",
    },
    "httpx": {"traigent/connectors/http.py", "traigent/hybrid/http_transport.py"},
    "dotenv": {"traigent/utils/env_config.py"},
    # imports are wrapped in try/except ImportError that re-raises with a friendlier message
    "pandas": {"traigent/utils/optimization_analyzer.py"},
    "langchain_anthropic": {"traigent/examples/providers/anthropic.py"},
    "langchain_openai": {
        "traigent/examples/providers/nous.py",
        "traigent/examples/providers/openai.py",
        "traigent/examples/providers/openrouter.py",
    },
    "opentelemetry": {
        f"traigent/observability/otel/{n}.py"
        for n in ("api", "exporter", "lineage", "policy", "processor", "sampling")
    },
}


def _core_dists() -> set[str]:
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    return {
        re.split(r"[\s\[<>=!~;]", d, maxsplit=1)[0].lower().replace("_", "-")
        for d in deps
    }


def _optional_import_guard(node: ast.Try) -> bool:
    """True if the try swallows ImportError (a handler that does not re-raise)."""
    for h in node.handlers:
        if any(isinstance(n, ast.Raise) for n in h.body):
            continue  # handler unconditionally raises: import is required
        if h.type is None:
            return True
        elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        names = {
            e.id if isinstance(e, ast.Name) else getattr(e, "attr", "") for e in elts
        }
        if {"ImportError", "ModuleNotFoundError", "Exception"} & names:
            return True
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _is_not_type_checking(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _is_type_checking(test.operand)
    )


def _toplevel_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()

    def visit(stmts: list[ast.stmt]) -> None:
        for s in stmts:
            if isinstance(s, ast.Import):
                found.update(a.name.split(".")[0] for a in s.names)
            elif isinstance(s, ast.ImportFrom):
                if s.level == 0 and s.module:
                    found.add(s.module.split(".")[0])
            elif isinstance(s, ast.Try):
                if not _optional_import_guard(s):
                    visit(s.body)
                    for h in s.handlers:
                        visit(h.body)
                visit(s.orelse)
                visit(s.finalbody)
            elif isinstance(s, ast.If):
                if _is_type_checking(s.test):
                    visit(s.orelse)
                elif _is_not_type_checking(s.test):
                    visit(s.body)
                else:
                    visit(s.body)
                    visit(s.orelse)
            elif isinstance(s, ast.With):
                visit(s.body)

    visit(tree.body)
    return found


def _undeclared() -> dict[str, list[str]]:
    declared = _core_dists()
    stdlib = set(sys.stdlib_module_names)
    missing: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for mod in _toplevel_imports(tree):
            rel = str(path.relative_to(REPO_ROOT))
            if mod in stdlib or mod in ALLOWLIST:
                continue
            if rel in KNOWN_UNDECLARED_FOLLOW_UP.get(mod, ()):
                continue
            dist = IMPORT_TO_DIST.get(mod, mod).lower().replace("_", "-")
            if dist not in declared:
                missing.setdefault(mod, []).append(str(path.relative_to(REPO_ROOT)))
    return missing


def test_unguarded_third_party_imports_are_declared():
    missing = _undeclared()
    assert not missing, (
        "undeclared top-level third-party imports in traigent/: "
        + "; ".join(
            f"{m} ({fs[0]} +{len(fs) - 1})" for m, fs in sorted(missing.items())
        )
    )


def test_issue_3169_packages_are_declared():
    declared = _core_dists()
    assert {"pyyaml", "packaging", "urllib3"} <= declared


def _scan(src: str) -> set[str]:
    return _toplevel_imports(ast.parse(src))


def test_scanner_recognises_guards_and_leaks():
    assert _scan("import yaml") == {"yaml"}
    assert _scan("try:\n import yaml\nexcept ImportError:\n yaml = None") == set()
    assert _scan("if TYPE_CHECKING:\n import yaml") == set()
    # re-raising handler is not an optional guard
    assert _scan("try:\n import yaml\nexcept ImportError:\n raise") == {"yaml"}
    # `not TYPE_CHECKING` body runs at runtime
    assert _scan("if not TYPE_CHECKING:\n import yaml") == {"yaml"}
    # executable handler body of a non-import guard
    assert _scan("try:\n pass\nexcept ValueError:\n import yaml") == {"yaml"}
    # function-local imports are not top-level
    assert _scan("def f():\n import yaml") == set()
