"""Tests for no-execution static system-spec extraction."""

from __future__ import annotations

from os import stat_result
from pathlib import Path

import pytest

from traigent.generation.coldstart.contracts import (
    ColdStartConfigurationError,
    ColdStartInputContractError,
    ColdStartOptions,
    DiscoveryGap,
)
from traigent.generation.coldstart.spec import extract_system_spec


class _SourcePointer:
    __name__ = "answer"
    __module__ = "customer.module"


def _point_at(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "traigent.generation.coldstart.spec.inspect.getsourcefile",
        lambda target: str(source),
    )


def test_extracts_typed_local_ast_without_importing_or_executing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text(
        "raise RuntimeError('this module must never be imported')\n"
        "def answer(question: str, retries: int = 2, *, verbose: bool = False) -> str:\n"
        "    raise RuntimeError('this function must never run')\n"
    )
    _point_at(source, monkeypatch)

    spec = extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    assert spec.callable_name == "answer"
    assert spec.module_name == "customer.module"
    assert [(p.name, p.annotation, p.required) for p in spec.parameters] == [
        ("question", "str", True),
        ("retries", "int", False),
        ("verbose", "bool", False),
    ]
    assert spec.files[0].path == Path("customer.py")
    assert not spec.inspection_truncated
    assert spec.skipped_file_count == 0
    assert len(spec.fingerprint) == 64


def test_default_recursive_selection_is_source_first_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "zz_customer.py"
    source.write_text("def answer(question: str) -> str:\n    return question\n")
    for index in range(5):
        (tmp_path / f"module_{index}.py").write_text(f"VALUE = {index}\n")
    _point_at(source, monkeypatch)

    spec = extract_system_spec(
        _SourcePointer(),
        repo_root=tmp_path,
        options=ColdStartOptions(max_files=3),
    )

    assert [file.path for file in spec.files] == [
        Path("zz_customer.py"),
        Path("module_0.py"),
        Path("module_1.py"),
    ]
    assert spec.inspection_truncated
    assert spec.skipped_file_count == 3


def test_skips_excluded_and_unsafe_non_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text("def answer(question: str) -> str:\n    return question\n")
    (tmp_path / "ordinary.py").write_text("VALUE = 'kept'\n")
    (tmp_path / "oversized.py").write_bytes(b"x" * 256)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "ignored.py").write_text("VALUE = 'ignored'\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("VALUE = 'ignored'\n")
    symlink = tmp_path / "symlinked.py"
    symlink.symlink_to(tmp_path / "ordinary.py")
    _point_at(source, monkeypatch)

    spec = extract_system_spec(
        _SourcePointer(),
        repo_root=tmp_path,
        options=ColdStartOptions(max_files=3, max_file_bytes=128),
    )

    assert [file.path for file in spec.files] == [
        Path("customer.py"),
        Path("ordinary.py"),
    ]
    assert spec.inspection_truncated
    assert spec.skipped_file_count == 2


def test_skips_oversized_non_source_file_before_reading_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text("def answer(question: str) -> str:\n    return question\n")
    (tmp_path / "ordinary.py").write_text("VALUE = 'kept'\n")
    oversized = tmp_path / "oversized.py"
    oversized.write_text("VALUE = 'stat-only'\n")
    _point_at(source, monkeypatch)

    original_stat = Path.stat
    original_read_bytes = Path.read_bytes
    read_paths: list[Path] = []

    def stat_with_oversized_file(path: Path, *, follow_symlinks: bool = True):
        result = original_stat(path, follow_symlinks=follow_symlinks)
        if path == oversized:
            return stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    129,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    def spy_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        if path == oversized:
            raise AssertionError("oversized non-source file must not be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "stat", stat_with_oversized_file)
    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)

    spec = extract_system_spec(
        _SourcePointer(),
        repo_root=tmp_path,
        options=ColdStartOptions(max_files=3, max_file_bytes=128),
    )

    assert [file.path for file in spec.files] == [
        Path("customer.py"),
        Path("ordinary.py"),
    ]
    assert spec.skipped_file_count == 1
    assert oversized not in read_paths


@pytest.mark.parametrize(
    "source_text, message, expected_gap",
    [
        (
            "def answer(question):\n    return question\n",
            "no type annotation",
            DiscoveryGap.UNTYPED_INPUT_CONTRACT,
        ),
        (
            "def answer(records: list[str]) -> str:\n    return str(records)\n",
            "unsupported annotation",
            DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
        ),
        (
            "def answer(question: str, *parts: str) -> str:\n    return question\n",
            "variadic",
            DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
        ),
        (
            "def answer() -> str:\n    return 'no inputs'\n",
            "at least one parameter",
            DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,
        ),
    ],
)
def test_untyped_or_unsupported_parameters_raise_input_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_text: str,
    message: str,
    expected_gap: DiscoveryGap,
) -> None:
    source = tmp_path / "customer.py"
    source.write_text(source_text)
    _point_at(source, monkeypatch)

    with pytest.raises(ColdStartInputContractError, match=message) as error:
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    assert error.value.gap is expected_gap


def test_source_configuration_failures_remain_configuration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text("def answer(question: str) -> str:\n    return question\n")
    _point_at(source, monkeypatch)

    with pytest.raises(ColdStartConfigurationError, match="not covered") as unmatched:
        extract_system_spec(
            _SourcePointer(),
            repo_root=tmp_path,
            options=ColdStartOptions(include_globs=("other.py",)),
        )
    assert type(unmatched.value) is ColdStartConfigurationError

    oversized = tmp_path / "oversized_source.py"
    oversized.write_text("def answer(question: str) -> str:\n    return question\n")
    _point_at(oversized, monkeypatch)
    with pytest.raises(ColdStartConfigurationError, match="exceeds max_file_bytes"):
        extract_system_spec(
            _SourcePointer(),
            repo_root=tmp_path,
            options=ColdStartOptions(max_file_bytes=1),
        )

    symlink = tmp_path / "source_link.py"
    symlink.symlink_to(source)
    _point_at(symlink, monkeypatch)
    with pytest.raises(ColdStartConfigurationError, match="symbolic link"):
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    _point_at(source, monkeypatch)
    original_read_bytes = Path.read_bytes

    def reject_source(path: Path) -> bytes:
        if path == source:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_source)
    with pytest.raises(ColdStartConfigurationError, match="could not read"):
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)


def test_rejects_out_of_scope_source_as_static_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def answer(question: str) -> str:\n    return question\n")
    _point_at(outside, monkeypatch)

    with pytest.raises(ColdStartConfigurationError, match="inside repo_root") as error:
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    assert type(error.value) is ColdStartConfigurationError
