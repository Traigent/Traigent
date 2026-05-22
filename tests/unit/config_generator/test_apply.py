"""Tests for config_generator.apply — decorator insertion."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import traigent.config_generator.apply as _apply_mod
from traigent.config_generator.apply import (
    _collect_needed_imports,
    _find_function,
    _find_import_insertion_point,
    _find_optimize_decorator,
    _get_existing_imports,
    _get_leading_whitespace,
    _indent_decorator,
    _sanitize_source_path,
    apply_config,
)
from traigent.config_generator.types import (
    AutoConfigResult,
    ObjectiveSpec,
    SafetySpec,
    TVarSpec,
)


@pytest.fixture(autouse=True)
def _allow_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let tests write to tmp_path by setting the safe base dir."""
    monkeypatch.setattr(_apply_mod, "_SAFE_BASE_DIR", str(tmp_path))


@pytest.fixture()
def simple_source() -> str:
    return textwrap.dedent("""\
        import os

        def answer_question(query: str) -> str:
            return "42"
    """)


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


@pytest.fixture()
def result_with_safety() -> AutoConfigResult:
    return AutoConfigResult(
        tvars=(
            TVarSpec(
                name="temperature",
                range_type="Range",
                range_kwargs={"low": 0.0, "high": 1.0},
            ),
        ),
        safety_constraints=(
            SafetySpec(metric_name="faithfulness", operator=">=", threshold=0.85),
        ),
    )


class TestApplyConfig:
    def test_insert_decorator_on_simple_function(
        self, tmp_path: Path, simple_source: str, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text(simple_source)

        apply_config(src, result_with_tvars, "answer_question", backup=False)

        modified = src.read_text()
        assert "@traigent.optimize(" in modified
        assert "Range(low=0.0, high=1.0)" in modified
        assert "def answer_question" in modified

    def test_backup_created(
        self, tmp_path: Path, simple_source: str, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text(simple_source)

        apply_config(src, result_with_tvars, "answer_question", backup=True)

        bak = tmp_path / "agent.py.bak"
        assert bak.exists()
        assert bak.read_text() == simple_source

    def test_no_backup_when_disabled(
        self, tmp_path: Path, simple_source: str, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text(simple_source)

        apply_config(src, result_with_tvars, "answer_question", backup=False)

        bak = tmp_path / "agent.py.bak"
        assert not bak.exists()

    def test_adds_import_traigent(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import os

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert "import traigent" in modified

    def test_adds_range_imports(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import os

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert "from traigent import Range" in modified

    def test_adds_safety_imports(
        self, tmp_path: Path, result_with_safety: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import os

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_safety, "my_func", backup=False)

        modified = src.read_text()
        assert "from traigent.api.safety import faithfulness" in modified

    def test_skips_existing_imports(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import traigent
            from traigent import Range

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        # Should NOT duplicate imports
        assert modified.count("import traigent") == 1
        assert modified.count("from traigent import Range") == 1

    def test_replaces_existing_decorator(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import traigent

            @traigent.optimize(
                configuration_space={"old": (0, 1)},
            )
            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        # Old config gone, new config present
        assert "old" not in modified
        assert "temperature" in modified
        # Only one @traigent.optimize
        assert modified.count("@traigent.optimize(") == 1

    def test_replaces_bare_optimize_decorator(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            from traigent import optimize

            @optimize(configuration_space={"old": (0, 1)})
            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert "old" not in modified
        assert "@traigent.optimize(" in modified

    def test_indentation_preserved_for_method(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            class Agent:
                def my_method(self):
                    pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_method", backup=False)

        modified = src.read_text()
        # Decorator should be indented to match the method
        for line in modified.splitlines():
            if "@traigent.optimize(" in line:
                assert line.startswith("    @traigent.optimize(")
                break
        else:
            pytest.fail("Decorator not found in modified source")

    def test_function_not_found_raises(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text("def other(): pass\n")

        with pytest.raises(ValueError, match="not found"):
            apply_config(src, result_with_tvars, "missing_func", backup=False)

    def test_function_name_required(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text("def f(): pass\n")

        with pytest.raises(ValueError, match="function_name is required"):
            apply_config(src, result_with_tvars, None, backup=False)

    def test_syntax_error_raises(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        src = tmp_path / "agent.py"
        src.write_text("def f(:\n")

        with pytest.raises(ValueError, match="Cannot parse"):
            apply_config(src, result_with_tvars, "f", backup=False)

    def test_inserts_above_existing_decorators(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import os

            @some_other_decorator
            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        lines = modified.splitlines()
        # Decorator should appear before the existing @some_other_decorator
        optimize_idx = next(
            i for i, line in enumerate(lines) if "@traigent.optimize(" in line
        )
        other_idx = next(
            i for i, line in enumerate(lines) if "@some_other_decorator" in line
        )
        assert optimize_idx < other_idx

    def test_replaces_fully_qualified_decorator(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import traigent.api.decorators

            @traigent.api.decorators.optimize(configuration_space={"old": (0, 1)})
            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert "old" not in modified
        assert "@traigent.optimize(" in modified
        assert "temperature" in modified

    def test_async_function(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            import asyncio

            async def my_async_func():
                await asyncio.sleep(0)
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_async_func", backup=False)

        modified = src.read_text()
        assert "@traigent.optimize(" in modified
        assert "async def my_async_func" in modified


class TestSanitizeSourcePath:
    def test_rejects_path_outside_cwd_when_base_dir_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default containment should fall back to the working directory."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        outside = tmp_path / "evil.py"
        outside.write_text("x = 1\n")

        monkeypatch.setattr(_apply_mod, "_SAFE_BASE_DIR", None)
        monkeypatch.chdir(safe_dir)

        with pytest.raises(ValueError, match="outside the allowed base directory"):
            _sanitize_source_path(outside)

    def test_rejects_path_outside_base_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path traversal outside the safe base directory must be rejected."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        monkeypatch.setattr(_apply_mod, "_SAFE_BASE_DIR", str(safe_dir))

        # Create a file outside the safe dir
        outside = tmp_path / "evil.py"
        outside.write_text("x = 1\n")

        with pytest.raises(ValueError, match="outside the allowed base directory"):
            _sanitize_source_path(outside)

    def test_rejects_non_py_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .py file"):
            _sanitize_source_path(txt)

    def test_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.py"
        with pytest.raises((ValueError, OSError)):
            _sanitize_source_path(missing)

    def test_accepts_valid_py_file(self, tmp_path: Path) -> None:
        src = tmp_path / "ok.py"
        src.write_text("x = 1\n")
        result = _sanitize_source_path(src)
        assert result.is_file()
        assert result.suffix == ".py"


class TestHelpers:
    def test_get_leading_whitespace(self) -> None:
        assert _get_leading_whitespace("    def f(): pass") == "    "
        assert _get_leading_whitespace("def f(): pass") == ""
        assert _get_leading_whitespace("\t\tdef f(): pass") == "\t\t"

    def test_indent_decorator(self) -> None:
        code = "@traigent.optimize(\n    config={},\n)"
        result = _indent_decorator(code, "    ")
        assert result.startswith("    @traigent.optimize(")
        assert "\n        config={}," in result

    def test_find_function_in_ast(self) -> None:
        import ast

        tree = ast.parse("def foo(): pass\ndef bar(): pass\n")
        assert _find_function(tree, "foo") is not None
        assert _find_function(tree, "bar") is not None
        assert _find_function(tree, "baz") is None

    def test_find_optimize_decorator_attribute(self) -> None:
        import ast

        source = "@traigent.optimize(x=1)\ndef f(): pass\n"
        tree = ast.parse(source)
        func = _find_function(tree, "f")
        assert func is not None
        assert _find_optimize_decorator(func) is not None

    def test_find_optimize_decorator_bare(self) -> None:
        import ast

        source = "@optimize(x=1)\ndef f(): pass\n"
        tree = ast.parse(source)
        func = _find_function(tree, "f")
        assert func is not None
        assert _find_optimize_decorator(func) is not None

    def test_find_optimize_decorator_none(self) -> None:
        import ast

        source = "@other_decorator\ndef f(): pass\n"
        tree = ast.parse(source)
        func = _find_function(tree, "f")
        assert func is not None
        assert _find_optimize_decorator(func) is None

    def test_collect_needed_imports_basic(self) -> None:
        import ast

        tree = ast.parse("import os\n")
        code = "@traigent.optimize(\n    configuration_space={'t': Range(low=0, high=1)},\n)"
        imports = _collect_needed_imports(code, tree)
        assert any("import traigent" in i for i in imports)
        assert any("Range" in i for i in imports)

    def test_collect_needed_imports_skips_existing(self) -> None:
        import ast

        tree = ast.parse("import traigent\nfrom traigent import Range\n")
        code = "@traigent.optimize(\n    configuration_space={'t': Range(low=0, high=1)},\n)"
        imports = _collect_needed_imports(code, tree)
        # Neither traigent nor Range should be in needed
        assert not any("traigent" in i for i in imports)

    def test_get_existing_imports_ignores_nested(self) -> None:
        import ast

        source = "import os\n\ndef f():\n    from traigent import Range\n"
        tree = ast.parse(source)
        existing = _get_existing_imports(tree)
        # Only module-level 'os' should be found, not nested 'Range'
        assert "os" in existing
        assert "Range" not in existing

    def test_find_import_insertion_point_ignores_nested(self) -> None:
        import ast

        source = "import os\n\ndef f():\n    import json\n"
        tree = ast.parse(source)
        pos = _find_import_insertion_point(tree)
        # Should point after 'import os' (line 1), not after nested 'import json' (line 4)
        assert pos == 1


class TestImportInsertionHeaderHandling:
    """Regression tests for apply_config #873 — header-aware import insertion."""

    def test_inserts_imports_after_module_docstring(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent('''\
            """Module docstring stays first."""

            def my_func():
                pass
        ''')
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        # Docstring remains the first thing in the file
        assert modified.startswith('"""Module docstring stays first."""\n')
        # Both imports are inserted directly after the docstring
        docstring_idx = modified.index('"""Module docstring stays first."""')
        import_idx = modified.index("import traigent")
        assert docstring_idx < import_idx
        # Decorator follows the imports
        assert "@traigent.optimize(" in modified

    def test_inserts_imports_after_shebang(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            #!/usr/bin/env python3

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert modified.startswith("#!/usr/bin/env python3\n")
        # Imports must not be placed before the shebang
        assert modified.index("#!/usr/bin/env python3") < modified.index(
            "import traigent"
        )

    def test_inserts_imports_after_encoding_comment(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            # -*- coding: utf-8 -*-

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        assert modified.startswith("# -*- coding: utf-8 -*-\n")
        assert modified.index("# -*- coding: utf-8 -*-") < modified.index(
            "import traigent"
        )

    def test_inserts_imports_after_shebang_and_encoding(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
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

        modified = src.read_text()
        modified_lines = modified.splitlines()
        assert modified_lines[0] == "#!/usr/bin/env python3"
        assert modified_lines[1] == "# -*- coding: utf-8 -*-"
        # The first import lands strictly after the two-line header.
        first_import = next(
            i for i, line in enumerate(modified_lines) if line.startswith("import ")
        )
        assert first_import >= 2

    def test_inserts_imports_after_future_import(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent("""\
            from __future__ import annotations

            def my_func():
                pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        # __future__ import remains first; new imports follow it
        future_idx = modified.index("from __future__ import annotations")
        traigent_idx = modified.index("import traigent")
        assert future_idx < traigent_idx

    def test_inserts_imports_after_full_header_combination(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        source = textwrap.dedent('''\
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            """Module docstring."""

            from __future__ import annotations

            def my_func():
                pass
        ''')
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified_lines = src.read_text().splitlines()
        # Verify each header element is preserved in order before the first
        # generated import.
        shebang_idx = modified_lines.index("#!/usr/bin/env python3")
        encoding_idx = modified_lines.index("# -*- coding: utf-8 -*-")
        docstring_idx = modified_lines.index('"""Module docstring."""')
        future_idx = modified_lines.index("from __future__ import annotations")
        first_import_idx = next(
            i
            for i, line in enumerate(modified_lines)
            if line.strip() == "import traigent"
        )
        assert (
            shebang_idx < encoding_idx < docstring_idx < future_idx < first_import_idx
        )

    def test_docstring_only_module_no_existing_imports(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        """The base regression case described in issue #873."""
        source = textwrap.dedent('''\
            """Docstring."""

            def my_func():
                pass
        ''')
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "my_func", backup=False)

        modified = src.read_text()
        # Should NOT start with the import — should still start with the docstring.
        assert not modified.startswith("import traigent")
        assert modified.startswith('"""Docstring."""')


class TestFunctionLookupScope:
    """Regression tests for #873 — top-level/class-method-only lookup."""

    def test_nested_function_only_target_not_found(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        """A name that exists only inside another function is not a valid target."""
        source = textwrap.dedent("""\
            def outer():
                def target():
                    return 1
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        with pytest.raises(ValueError, match="not found"):
            apply_config(src, result_with_tvars, "target", backup=False)

    def test_top_level_function_preferred_over_nested_collision(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        """If a name shadows a nested name, decorate the top-level definition."""
        source = textwrap.dedent("""\
            def outer():
                def target():
                    return "nested"

            def target():
                return "top"
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "target", backup=False)

        modified = src.read_text()
        modified_lines = modified.splitlines()
        # No decorator should be placed at indentation deeper than column 0.
        assert not any(
            line.startswith(" ") and "@traigent.optimize(" in line
            for line in modified_lines
        )
        # The single decorator sits immediately above the top-level definition.
        target_idx = modified_lines.index("def target():")
        decorator_idx = next(
            i
            for i, line in enumerate(modified_lines)
            if line.startswith("@traigent.optimize(")
        )
        assert decorator_idx < target_idx
        # And no other top-level "def target():" appears between them.
        between = modified_lines[decorator_idx + 1 : target_idx]
        assert all("def target" not in line for line in between)

    def test_class_methods_still_resolvable(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        """Class-level methods continue to be valid targets."""
        source = textwrap.dedent("""\
            class Agent:
                def predict(self):
                    pass
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        apply_config(src, result_with_tvars, "predict", backup=False)

        modified_lines = src.read_text().splitlines()
        decorator_idx = next(
            i for i, line in enumerate(modified_lines) if "@traigent.optimize(" in line
        )
        # Decorator stays indented to match the method body.
        assert modified_lines[decorator_idx].startswith("    @traigent.optimize(")

    def test_function_inside_class_method_is_not_targeted(
        self, tmp_path: Path, result_with_tvars: AutoConfigResult
    ) -> None:
        """A function nested inside a method must not be selected."""
        source = textwrap.dedent("""\
            class Agent:
                def predict(self):
                    def helper():
                        return 1
                    return helper()
        """)
        src = tmp_path / "agent.py"
        src.write_text(source)

        with pytest.raises(ValueError, match="not found"):
            apply_config(src, result_with_tvars, "helper", backup=False)


class TestImportInsertionHeaderHelpers:
    """Unit-level coverage for the new helpers in apply.py."""

    def test_find_import_insertion_point_after_module_docstring(self) -> None:
        import ast

        source = '"""Module docs."""\n\ndef f(): pass\n'
        pos = _find_import_insertion_point(
            ast.parse(source), source.splitlines(keepends=True)
        )
        assert pos == 1

    def test_find_import_insertion_point_after_shebang_only(self) -> None:
        import ast

        source = "#!/usr/bin/env python3\n\ndef f(): pass\n"
        pos = _find_import_insertion_point(
            ast.parse(source), source.splitlines(keepends=True)
        )
        assert pos == 1

    def test_find_import_insertion_point_after_shebang_and_encoding(self) -> None:
        import ast

        source = "#!/usr/bin/env python3\n# coding: utf-8\ndef f(): pass\n"
        pos = _find_import_insertion_point(
            ast.parse(source), source.splitlines(keepends=True)
        )
        assert pos == 2

    def test_find_import_insertion_point_after_future_import(self) -> None:
        import ast

        source = "from __future__ import annotations\n\ndef f(): pass\n"
        pos = _find_import_insertion_point(
            ast.parse(source), source.splitlines(keepends=True)
        )
        assert pos == 1

    def test_find_import_insertion_point_ignores_later_string_expression(self) -> None:
        """A bare string after code is not a module docstring."""
        import ast

        source = 'value = 1\n"not a module docstring"\ndef f(): pass\n'
        pos = _find_import_insertion_point(
            ast.parse(source), source.splitlines(keepends=True)
        )
        assert pos == 0

    def test_find_import_insertion_point_empty_file(self) -> None:
        import ast

        pos = _find_import_insertion_point(ast.parse(""), [])
        assert pos == 0

    def test_find_function_ignores_nested(self) -> None:
        import ast

        source = "def outer():\n    def target():\n        pass\n"
        assert _find_function(ast.parse(source), "target") is None

    def test_find_function_finds_method_in_class(self) -> None:
        import ast

        source = "class C:\n    def m(self):\n        pass\n"
        node = _find_function(ast.parse(source), "m")
        assert node is not None
        assert node.name == "m"
