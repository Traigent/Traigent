"""Regression tests for apply_config source rewrite boundaries."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

import traigent.config_generator.apply as _apply_mod
from traigent.config_generator.apply import (
    _find_function,
    _find_import_insertion_point,
    apply_config,
)
from traigent.config_generator.types import AutoConfigResult, ObjectiveSpec, TVarSpec


@pytest.fixture(autouse=True)
def _allow_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_apply_mod, "_SAFE_BASE_DIR", str(tmp_path))


@pytest.fixture()
def result_with_tvars() -> AutoConfigResult:
    return AutoConfigResult(
        tvars=(
            TVarSpec(
                name="temperature",
                range_type="Range",
                range_kwargs={"low": 0.0, "high": 1.0},
            ),
        ),
        objectives=(ObjectiveSpec(name="accuracy"),),
    )


def test_adds_imports_after_module_docstring(
    tmp_path: Path, result_with_tvars: AutoConfigResult
) -> None:
    source = textwrap.dedent('''\
        """Agent module."""

        def my_func():
            pass
    ''')
    src = tmp_path / "agent.py"
    src.write_text(source)

    apply_config(src, result_with_tvars, "my_func", backup=False)

    assert src.read_text().startswith(
        '"""Agent module."""\n' "import traigent\n" "from traigent import Range\n\n"
    )


def test_adds_imports_after_shebang_and_encoding_comment(
    tmp_path: Path, result_with_tvars: AutoConfigResult
) -> None:
    source = textwrap.dedent("""\
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        def my_func():
            pass
    """)
    src = tmp_path / "agent.py"
    src.write_text(source)

    apply_config(src, result_with_tvars, "my_func", backup=False)

    assert src.read_text().startswith(
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "import traigent\n"
        "from traigent import Range\n\n"
    )


def test_adds_imports_after_future_import(
    tmp_path: Path, result_with_tvars: AutoConfigResult
) -> None:
    source = textwrap.dedent("""\
        from __future__ import annotations

        def my_func():
            pass
    """)
    src = tmp_path / "agent.py"
    src.write_text(source)

    apply_config(src, result_with_tvars, "my_func", backup=False)

    assert src.read_text().startswith(
        "from __future__ import annotations\n"
        "import traigent\n"
        "from traigent import Range\n\n"
    )


def test_decorates_top_level_function_when_nested_name_collides(
    tmp_path: Path, result_with_tvars: AutoConfigResult
) -> None:
    source = textwrap.dedent("""\
        def outer():
            def target():
                pass

        def target():
            pass
    """)
    src = tmp_path / "agent.py"
    src.write_text(source)

    apply_config(src, result_with_tvars, "target", backup=False)

    lines = src.read_text().splitlines()
    top_level_target_idx = lines.index("def target():")
    optimize_idx = max(
        i
        for i, line in enumerate(lines[:top_level_target_idx])
        if "@traigent.optimize(" in line
    )
    assert lines[optimize_idx].startswith("@traigent.optimize(")
    assert not any(line.startswith("    @traigent.optimize(") for line in lines)


def test_nested_function_only_is_not_a_valid_target(
    tmp_path: Path, result_with_tvars: AutoConfigResult
) -> None:
    src = tmp_path / "agent.py"
    src.write_text("def outer():\n    def target():\n        pass\n")

    with pytest.raises(ValueError, match="not found"):
        apply_config(src, result_with_tvars, "target", backup=False)


def test_find_function_ignores_nested_functions() -> None:
    source = "def outer():\n    def target(): pass\n"
    assert _find_function(ast.parse(source), "target") is None


def test_find_import_insertion_point_after_module_docstring() -> None:
    source = '"""Module docs."""\n\ndef f(): pass\n'
    pos = _find_import_insertion_point(
        ast.parse(source), source.splitlines(keepends=True)
    )
    assert pos == 1


def test_find_import_insertion_point_after_shebang_and_encoding() -> None:
    source = "#!/usr/bin/env python3\n# coding: utf-8\ndef f(): pass\n"
    pos = _find_import_insertion_point(
        ast.parse(source), source.splitlines(keepends=True)
    )
    assert pos == 2


def test_find_import_insertion_point_after_future_import() -> None:
    source = "from __future__ import annotations\n\ndef f(): pass\n"
    pos = _find_import_insertion_point(
        ast.parse(source), source.splitlines(keepends=True)
    )
    assert pos == 1


def test_find_import_insertion_point_ignores_later_string_expression() -> None:
    source = 'value = 1\n"not a module docstring"\ndef f(): pass\n'
    pos = _find_import_insertion_point(
        ast.parse(source), source.splitlines(keepends=True)
    )
    assert pos == 0
