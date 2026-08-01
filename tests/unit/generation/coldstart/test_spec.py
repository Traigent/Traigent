"""Tests for no-execution static system-spec extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.generation.coldstart.contracts import ColdStartConfigurationError
from traigent.generation.coldstart.spec import extract_system_spec


class _SourcePointer:
    __name__ = "answer"
    __module__ = "customer.module"


def test_extracts_typed_local_ast_without_importing_or_executing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text(
        "raise RuntimeError('this module must never be imported')\n"
        "def answer(question: str, retries: int = 2, *, verbose: bool = False) -> str:\n"
        "    raise RuntimeError('this function must never run')\n"
    )
    monkeypatch.setattr(
        "traigent.generation.coldstart.spec.inspect.getsourcefile",
        lambda target: str(source),
    )

    spec = extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    assert spec.callable_name == "answer"
    assert spec.module_name == "customer.module"
    assert [(p.name, p.annotation, p.required) for p in spec.parameters] == [
        ("question", "str", True),
        ("retries", "int", False),
        ("verbose", "bool", False),
    ]
    assert spec.files[0].path == Path("customer.py")
    assert len(spec.fingerprint) == 64


def test_rejects_untyped_or_out_of_scope_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "customer.py"
    source.write_text("def answer(question):\n    return question\n")
    monkeypatch.setattr(
        "traigent.generation.coldstart.spec.inspect.getsourcefile",
        lambda target: str(source),
    )

    with pytest.raises(ColdStartConfigurationError, match="no type annotation"):
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)

    outside = tmp_path.parent / "outside.py"
    outside.write_text("def answer(question: str) -> str:\n    return question\n")
    monkeypatch.setattr(
        "traigent.generation.coldstart.spec.inspect.getsourcefile",
        lambda target: str(outside),
    )
    with pytest.raises(ColdStartConfigurationError, match="inside repo_root"):
        extract_system_spec(_SourcePointer(), repo_root=tmp_path)
