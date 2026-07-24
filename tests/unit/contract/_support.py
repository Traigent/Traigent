"""Shared helpers for the evaluation-contract test suite.

Not a test module (``python_files = ["test_*.py"]`` never collects it). Holds
the loader that *reuses* the real ``tests/fixtures/signature_check`` call-shape
fixtures and a handful of tiny report-inspection helpers so the individual test
files stay focused on assertions.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

from traigent.contract import EvaluationContractReport

# The shared call-shape fixtures. Their module body deliberately raises on
# import (it contains intentionally-wrong example calls), so we cannot ``import``
# it directly -- we parse the source and evaluate ONLY its ``def`` statements,
# which reuses the exact fixture signatures without triggering the demo errors.
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "signature_check"
    / "posonly_kwonly_defaults.py"
)


def load_signature_fixtures() -> dict[str, Callable[..., Any]]:
    """Return the function objects defined in the shared signature fixture.

    Reuses ``tests/fixtures/signature_check/posonly_kwonly_defaults.py`` by
    compiling only its function definitions, so the posonly / kwonly / defaults
    / ``*args`` / ``**kwargs`` call shapes come straight from that single source
    of truth rather than being re-declared here.
    """
    source = _FIXTURE_PATH.read_text()
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    namespace: dict[str, Any] = {}
    exec(compile(tree, str(_FIXTURE_PATH), "exec"), namespace)  # noqa: S102
    return {
        name: obj
        for name, obj in namespace.items()
        if not name.startswith("__") and callable(obj)
    }


def finding_codes(report: EvaluationContractReport) -> list[str]:
    """All finding codes on a report, as plain strings."""
    return [str(finding.code) for finding in report.findings]


def error_codes(report: EvaluationContractReport) -> list[str]:
    """Codes of the ``severity == "error"`` findings only."""
    return [str(f.code) for f in report.findings if f.severity == "error"]


def find_finding(report: EvaluationContractReport, code: str) -> Any:
    """First finding matching ``code`` (compared as a string), or ``None``."""
    for finding in report.findings:
        if str(finding.code) == str(code):
            return finding
    return None


def source_unavailable_function() -> Callable[..., Any]:
    """A real function whose source cannot be introspected.

    Built via ``compile``/``exec`` with a synthetic filename, so
    ``inspect.getsource`` raises ``OSError`` -- the concrete way seamless AST
    injection becomes unavailable for a function.
    """
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        compile(
            "def dynamic_agent(question):\n    model = 'default'\n    return model",
            "<contract-test-dynamic>",
            "exec",
        ),
        namespace,
    )
    return namespace["dynamic_agent"]
