"""Characterization tests for CodeIdentifier.verified.

They pin the search-root selection, the order and arguments of the static
lookups, the exception boundaries around parsing, the exact refusal messages
and the path-escape and existence checks.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path, PosixPath
from typing import Any, Self, cast

import pytest

import traigent.labels.types as types_module
from traigent.labels.types import CodeIdentifier, ContentClassError


def _write(base: Path, relative: str, text: str = "def fn():\n    pass\n") -> Path:
    path = base / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _message(call: Any) -> str:
    with pytest.raises(ContentClassError) as excinfo:
        call()
    return str(excinfo.value)


class _Traced(CodeIdentifier):
    """Subclass whose static lookups record their calls and can be scripted."""

    events: list[Any] = []
    find_result: Any = "real"
    defines_result: Any = "real"

    @staticmethod
    def _find_source(module_name: str, search: list[Path]) -> Path | None:
        _Traced.events.append(("find", module_name, list(search)))
        if _Traced.find_result == "real":
            return cast(Path | None, CodeIdentifier._find_source(module_name, search))
        return cast(Path | None, _Traced.find_result)

    @staticmethod
    def _defines(tree: ast.AST, qualname: str) -> bool:
        _Traced.events.append(("defines", type(tree).__name__, qualname))
        if _Traced.defines_result == "real":
            return cast(bool, CodeIdentifier._defines(tree, qualname))
        return _Traced.defines_result is True


@pytest.fixture
def traced() -> Iterator[type[_Traced]]:
    _Traced.events = []
    _Traced.find_result = "real"
    _Traced.defines_result = "real"
    yield _Traced
    _Traced.events = []
    _Traced.find_result = "real"
    _Traced.defines_result = "real"


def test_syntax_check_runs_first_and_refuses_unknown_forms() -> None:
    for bad in ("", "has space", "/abs/path.py", "a/../b.py", "mod:", ":fn", "a:b:c"):
        assert _message(lambda b=bad: CodeIdentifier.verified(b)) == (
            "not a code identifier"
        )


def test_non_string_identifier_is_refused_with_type_error() -> None:
    with pytest.raises(TypeError):
        CodeIdentifier.verified(cast(Any, 5))


def test_module_form_found_under_search_paths(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "class K:\n    def m(self):\n        pass\n")
    for ident in ("pkg.mod:K", "pkg.mod:K.m"):
        result = CodeIdentifier.verified(ident, search_paths=[tmp_path])
        assert result == ident
        assert type(result) is CodeIdentifier


def test_package_init_is_found_when_no_module_file(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "def fn():\n    pass\n")
    assert CodeIdentifier.verified("pkg:fn", search_paths=[tmp_path]) == "pkg:fn"


def test_module_file_wins_over_package_init_and_first_base_wins(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(first, "pkg.py", "def only_in_first():\n    pass\n")
    _write(second, "pkg.py", "def only_in_second():\n    pass\n")
    assert CodeIdentifier.verified("pkg:only_in_first", search_paths=[first, second])
    assert (
        _message(
            lambda: CodeIdentifier.verified(
                "pkg:only_in_second", search_paths=[first, second]
            )
        )
        == "code identifier does not name local code"
    )


def test_module_not_found_message(tmp_path: Path) -> None:
    assert (
        _message(lambda: CodeIdentifier.verified("nope:fn", search_paths=[tmp_path]))
        == "code identifier module source not found"
    )


def test_qualname_not_defined_message(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "def fn():\n    pass\n")
    for ident in ("mod:other", "mod:fn.inner", "mod:fn.x.y"):
        assert (
            _message(
                lambda i=ident: CodeIdentifier.verified(i, search_paths=[tmp_path])
            )
            == "code identifier does not name local code"
        )


def test_search_paths_win_over_root_and_none_differs_from_empty(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "mod.py")
    other = tmp_path / "other"
    other.mkdir()
    assert CodeIdentifier.verified("mod:fn", root=tmp_path)
    assert (
        _message(
            lambda: CodeIdentifier.verified("mod:fn", root=tmp_path, search_paths=[])
        )
        == "code identifier module source not found"
    )
    assert (
        _message(
            lambda: CodeIdentifier.verified(
                "mod:fn", root=tmp_path, search_paths=[other]
            )
        )
        == "code identifier module source not found"
    )
    assert CodeIdentifier.verified("mod:fn", root=other, search_paths=[tmp_path])


def test_search_paths_entries_are_converted_with_path(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py")
    assert CodeIdentifier.verified("mod:fn", search_paths=[str(tmp_path)])
    assert CodeIdentifier.verified("mod:fn", search_paths=(str(tmp_path),))
    assert CodeIdentifier.verified("mod:fn", search_paths=cast(Any, iter([tmp_path])))


def test_root_is_used_unconverted_for_the_module_search(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py")
    with pytest.raises(AttributeError):
        CodeIdentifier.verified("mod:fn", root=str(tmp_path))


def test_default_search_uses_sys_path_entries_with_blank_as_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    assert CodeIdentifier.verified("mod:fn")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", ["", str(elsewhere)])
    assert CodeIdentifier.verified("mod:fn")
    monkeypatch.setattr(sys, "path", [str(elsewhere)])
    assert _message(lambda: CodeIdentifier.verified("mod:fn")) == (
        "code identifier module source not found"
    )


def test_unreadable_source_is_refused_with_cause(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_bytes(b"def fn():\n    return '\xff\xfe'\n")
    with pytest.raises(ContentClassError) as excinfo:
        CodeIdentifier.verified("mod:fn", search_paths=[tmp_path])
    assert str(excinfo.value) == "code identifier source is unreadable"
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_syntax_error_source_is_refused_with_cause(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "def fn(:\n")
    with pytest.raises(ContentClassError) as excinfo:
        CodeIdentifier.verified("mod:fn", search_paths=[tmp_path])
    assert str(excinfo.value) == "code identifier source is unreadable"
    assert isinstance(excinfo.value.__cause__, SyntaxError)


def test_os_error_while_reading_is_refused_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")

    def _boom(self: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(ContentClassError) as excinfo:
        CodeIdentifier.verified("mod:fn", search_paths=[tmp_path])
    assert str(excinfo.value) == "code identifier source is unreadable"
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_other_parse_failures_propagate_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")

    def _boom(source: str) -> ast.AST:
        raise RecursionError("too deep")

    monkeypatch.setattr(types_module.ast, "parse", _boom)
    with pytest.raises(RecursionError):
        CodeIdentifier.verified("mod:fn", search_paths=[tmp_path])


def test_source_is_read_once_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")
    calls: list[Any] = []
    original = Path.read_text

    def _spy(self: Path, *args: Any, **kwargs: Any) -> str:
        calls.append((self.name, args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy)
    CodeIdentifier.verified("mod:fn", search_paths=[tmp_path])
    assert calls == [("mod.py", (), {"encoding": "utf-8"})]


def test_static_lookups_run_in_order_through_the_subclass(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    _write(tmp_path, "pkg/mod.py", "class K:\n    pass\n")
    result = traced.verified("pkg.mod:K", search_paths=[tmp_path])
    assert type(result) is _Traced
    assert result == "pkg.mod:K"
    assert traced.events == [
        ("find", "pkg.mod", [tmp_path.resolve()]),
        ("defines", "Module", "K"),
    ]


def test_defines_is_not_consulted_when_the_source_is_missing(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    traced.find_result = None
    with pytest.raises(ContentClassError):
        traced.verified("pkg.mod:K", search_paths=[tmp_path])
    assert [e[0] for e in traced.events] == ["find"]


def test_defines_is_not_consulted_when_the_source_is_unreadable(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    bad = _write(tmp_path, "bad.py", "def (:\n")
    traced.find_result = bad
    with pytest.raises(ContentClassError):
        traced.verified("bad:fn", search_paths=[tmp_path])
    assert [e[0] for e in traced.events] == ["find"]


def test_scripted_defines_decides_the_outcome(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    src = _write(tmp_path, "m.py", "x = 1\n")
    traced.find_result = src
    traced.defines_result = True
    assert traced.verified("m:anything", search_paths=[tmp_path]) == "m:anything"
    traced.defines_result = False
    assert (
        _message(lambda: traced.verified("m:anything", search_paths=[tmp_path]))
        == "code identifier does not name local code"
    )


def test_resolve_failure_while_building_the_search_list_propagates(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    class _BadPath(PosixPath):
        def resolve(self, strict: bool = False) -> Self:
            raise OSError("cannot resolve")

    with pytest.raises(OSError, match="cannot resolve"):
        traced.verified("pkg:fn", root=_BadPath(tmp_path))
    assert traced.events == []


def test_search_path_resolution_happens_once_per_entry(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py")
    counts = {"resolve": 0}

    class _CountingPath(PosixPath):
        def resolve(self, strict: bool = False) -> Self:
            counts["resolve"] += 1
            return super().resolve(strict)

    assert CodeIdentifier.verified("mod:fn", root=_CountingPath(tmp_path))
    assert counts["resolve"] == 1


def test_relative_path_inside_root_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py")
    result = CodeIdentifier.verified("pkg/m.py", root=tmp_path)
    assert result == "pkg/m.py"
    assert type(result) is CodeIdentifier


def test_relative_path_returns_subclass_instance(
    tmp_path: Path, traced: type[_Traced]
) -> None:
    _write(tmp_path, "m.py")
    assert type(traced.verified("m.py", root=tmp_path)) is _Traced
    assert traced.events == []


def test_missing_and_directory_paths_are_refused(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    for ident in ("nope.py", "adir", "."):
        assert (
            _message(lambda i=ident: CodeIdentifier.verified(i, root=tmp_path))
            == "code identifier path does not exist"
        )


def test_symlink_escaping_the_root_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    _write(outside, "secret.py")
    root.mkdir()
    (root / "link.py").symlink_to(outside / "secret.py")
    assert (
        _message(lambda: CodeIdentifier.verified("link.py", root=root))
        == "code identifier escapes the repository"
    )


def test_symlink_inside_the_root_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "real.py")
    (tmp_path / "alias.py").symlink_to(tmp_path / "real.py")
    assert CodeIdentifier.verified("alias.py", root=tmp_path) == "alias.py"


def test_escape_message_precedes_existence_message(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "dangling.py").symlink_to(tmp_path / "missing_outside.py")
    assert (
        _message(lambda: CodeIdentifier.verified("dangling.py", root=root))
        == "code identifier escapes the repository"
    )


def test_missing_root_uses_cwd_and_reads_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "m.py")
    calls = {"cwd": 0}
    original = Path.cwd

    def _cwd() -> Path:
        calls["cwd"] += 1
        return original()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "cwd", staticmethod(_cwd))
    assert CodeIdentifier.verified("m.py")
    assert calls["cwd"] == 1
    assert CodeIdentifier.verified("m.py", root=tmp_path)
    assert calls["cwd"] == 1


def test_falsy_string_root_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "m.py")
    monkeypatch.chdir(tmp_path)
    assert CodeIdentifier.verified("m.py", root="")


def test_path_form_root_is_resolved_and_target_resolved_once_each(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "m.py")
    counts = {"resolve": 0}

    class _CountingPath(PosixPath):
        def resolve(self, strict: bool = False) -> Self:
            counts["resolve"] += 1
            return super().resolve(strict)

    assert CodeIdentifier.verified("m.py", root=_CountingPath(tmp_path))
    assert counts["resolve"] >= 2


class _FalsyPath(PosixPath):
    """A path whose truth value is false."""

    def __bool__(self) -> bool:
        return False


def test_module_search_tests_root_against_none_not_truthiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(sys, "path", [str(elsewhere)])
    assert CodeIdentifier.verified("mod:fn", root=_FalsyPath(tmp_path)) == "mod:fn"


def test_path_form_tests_root_truthiness_and_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inside = tmp_path / "inside"
    _write(inside, "m.py")
    monkeypatch.chdir(inside)
    other = tmp_path / "other"
    other.mkdir()
    assert CodeIdentifier.verified("m.py", root=_FalsyPath(other)) == "m.py"


def test_none_entry_in_sys_path_is_read_as_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "mod.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [None])
    assert CodeIdentifier.verified("mod:fn") == "mod:fn"
