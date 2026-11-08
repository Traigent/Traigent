"""Regression for #3265: missing optional packages give an actionable install hint."""

from __future__ import annotations

import subprocess
import sys

import pytest

_BLOCK = (
    "import sys\n"
    "class _B:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if any(name == b or name.startswith(b + '.') for b in {BLOCKED!r}):\n"
    '            raise ModuleNotFoundError(f"No module named {name!r}", name=name)\n'
    "sys.meta_path.insert(0, _B())\n"
)


@pytest.mark.parametrize(
    ("module", "blocked", "hint"),
    [
        (
            "traigent.observability.otel",
            {"opentelemetry"},
            "pip install 'traigent[observability]'",
        ),
        (
            "traigent.observability.otel",
            {"opentelemetry.exporter"},
            "pip install 'traigent[observability]'",
        ),
        (
            "traigent.utils.optimization_analyzer",
            {"pandas"},
            "pip install 'traigent[analytics]'",
        ),
    ],
)
def test_missing_optional_dependency_raises_actionable_hint(
    module: str, blocked: set[str], hint: str
) -> None:
    code = _BLOCK.replace("{BLOCKED!r}", repr(blocked)) + (
        f"try:\n    import {module}\nexcept ImportError as e:\n"
        "    print(str(e)); print(repr(e.__cause__)); print(e.__suppress_context__)\n"
        "    sys.exit(0)\nsys.exit(3)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert hint in proc.stdout
