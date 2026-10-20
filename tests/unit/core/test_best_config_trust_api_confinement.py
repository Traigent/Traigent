"""Only ``traigent/core`` may use the diagnostic trust constructors.

``best_config_diagnostics`` decides which exceptions are trusted to carry a fixed,
SDK-written message. The constructors (``_mark_safe`` / ``_safe_error``) and the registry
behind them (``_trust`` / ``_TRUSTED``) are underscore-private. That is a convention, not
a security boundary, so this test makes the convention checkable: any module under
``traigent/`` but outside ``traigent/core`` that imports, calls, reads or names them
(including as a string, e.g. ``getattr(diag, "_mark_safe")``) fails here. Tests are not
scanned.
"""

from __future__ import annotations

import ast
from pathlib import Path

PRIVATE_NAMES = frozenset({"_mark_safe", "_safe_error", "_trust", "_TRUSTED"})

ROOT = Path(__file__).resolve().parents[3] / "traigent"
CORE = ROOT / "core"


def violations(source: str) -> list[str]:
    """Names from :data:`PRIVATE_NAMES` that ``source`` touches, in source order."""

    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in PRIVATE_NAMES:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in PRIVATE_NAMES:
            found.append(node.attr)
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in PRIVATE_NAMES:
            found.append(node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and node.value in PRIVATE_NAMES:
            found.append(str(node.value))
    return found


def _outside_core() -> list[Path]:
    return [p for p in sorted(ROOT.rglob("*.py")) if CORE not in p.parents]


def test_no_module_outside_core_uses_the_trust_constructors() -> None:
    offenders = {
        str(p.relative_to(ROOT.parent)): hits
        for p in _outside_core()
        if (hits := violations(p.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        "only traigent/core may use the diagnostic trust constructors; found: "
        f"{offenders}"
    )


def test_scan_covers_files_outside_core() -> None:
    assert _outside_core(), "scan found nothing outside traigent/core"


def test_scanner_catches_each_shape() -> None:
    shapes = [
        "from traigent.core import best_config_diagnostics as d\nd._mark_safe(e)\n",
        "from traigent.core.best_config_diagnostics import _safe_error\n",
        "from traigent.core.best_config_diagnostics import _safe_error as ok\n",
        "import traigent.core.best_config_diagnostics as d\nd._TRUSTED.clear()\n",
        "x = getattr(d, '_mark_safe')\n",
        "f = d._trust\n",
    ]
    for source in shapes:
        assert violations(source), source


def test_scanner_ignores_the_public_api() -> None:
    assert not violations(
        "d.is_safe(e)\nd.refusal('m', category=c)\nsafe_error_code = 1\n"
    )
