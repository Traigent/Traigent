"""Structural guard for the shared CLI scratch parent.

CLI tests write fixture modules under ``tests/unit/cli/temp_modules`` because
the CLI path guard requires files inside the workspace. The parent is shared
between xdist workers, so removing it (or creating it in order to remove it)
races with other workers. Only the ``cli_scratch_dir`` fixture in
``conftest.py`` may touch it, and it removes just its own subdirectory.

The original race does not reproduce on demand, so this test scans the sources
instead: no CLI test module may remove directories or name ``temp_modules``.

The guard is deliberately strict for tests/unit/cli: any directory-removal
call is flagged, including harmless ones such as ``tmp_path.rmdir()``, and
removers imported by name (``from shutil import rmtree as r``) or called as
``os.rmdir``. A CLI test that truly needs one should clean up through a
fixture in ``conftest.py``, which is exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI_TESTS = Path(__file__).resolve().parent
_ALLOWED = {"conftest.py", Path(__file__).name}
_REMOVERS = {"rmdir", "rmtree", "removedirs"}


def _violations(path: Path) -> list[str]:
    return _violations_in_source(path.read_text(encoding="utf-8"), path.name)


def _violations_in_source(source: str, name: str) -> list[str]:
    tree = ast.parse(source, filename=name)
    found: list[str] = []
    aliases: set[str] = set()  # local names bound to a remover via import
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _REMOVERS:
                    aliases.add(alias.asname or alias.name)
                    found.append(f"{name}:{node.lineno} imports {alias.name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _REMOVERS:
                found.append(f"{name}:{node.lineno} calls .{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in aliases:
                found.append(f"{name}:{node.lineno} calls {func.id}()")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "temp_modules" in node.value
        ):
            found.append(f"{name}:{node.lineno} names temp_modules")
    return found


def test_only_the_fixture_touches_the_shared_scratch_parent() -> None:
    files = [p for p in sorted(CLI_TESTS.glob("*.py")) if p.name not in _ALLOWED]
    assert files, "scan found no CLI test modules"
    violations = [v for p in files for v in _violations(p)]
    assert not violations, violations


def test_guard_detects_the_old_pattern() -> None:
    source = (
        "from pathlib import Path\n"
        'D = Path(__file__).parent / "temp_modules"\n'
        "D.mkdir(exist_ok=True)\n"
        "D.rmdir()\n"
    )
    assert len(_violations_in_source(source, "t.py")) == 2


def test_guard_detects_bare_name_and_os_removers() -> None:
    cases = {
        "from shutil import rmtree\nrmtree(x)\n": 2,
        "from shutil import rmtree as r\nr(x)\n": 2,
        "from os import rmdir, removedirs\nrmdir(x)\nremovedirs(y)\n": 4,
        "import os\nos.rmdir(x)\n": 1,
        "import shutil\nshutil.rmtree(x)\n": 1,
        # Strict by design: harmless-looking uses are flagged too.
        "def test(tmp_path):\n    tmp_path.rmdir()\n": 1,
        "x = 1\n": 0,
    }
    for source, expected in cases.items():
        assert len(_violations_in_source(source, "t.py")) == expected, source
