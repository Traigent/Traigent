"""Tests for ``Dataset.to_jsonl`` (Traigent#2565).

The writer is the inverse of ``Dataset.from_jsonl``: one
``{"input", "expected_output", "metadata"}`` row per example, UTF-8 with real
(non-escaped) Unicode, no ``default=str`` coercion, and an atomic replace of
the destination so a failed write never leaves a half-written or clobbered
file. An empty dataset is rejected because ``from_jsonl`` cannot load one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Dataset.from_jsonl only reads under the configured dataset root.
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def _dataset(rows: list[tuple[Any, Any, dict[str, Any] | None]]) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data=inp, expected_output=exp, metadata=meta)
            for inp, exp, meta in rows
        ],
        name="writer_source",
    )


def _contents(dataset: Dataset) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [(ex.input_data, ex.expected_output, ex.metadata) for ex in dataset]


ROWS: list[tuple[Any, Any, dict[str, Any] | None]] = [
    (
        {"question": "Quelle est la capitale ? 首都は？", "ctx": {"k": [1, 2.5, None]}},
        "Paris — パリ",
        {"example_id": "q-1", "tags": ["geo", "ünïcode"], "nested": {"a": {"b": 1}}},
    ),
    ({"question": "no answer"}, None, None),
    ({"question": "structured"}, {"label": "B", "score": 0.75}, {"source": "s"}),
    ({"question": "list"}, ["x", 1, True, None], {}),
    ("plain string input", "out", None),
]


class TestRoundTrip:
    def test_round_trips_through_from_jsonl(self, tmp_path: Path) -> None:
        source = _dataset(ROWS)
        out = tmp_path / "out.jsonl"

        source.to_jsonl(out)

        assert _contents(Dataset.from_jsonl(str(out))) == _contents(source)

    def test_accepts_a_str_path(self, tmp_path: Path) -> None:
        source = _dataset(ROWS[:2])
        out = tmp_path / "out.jsonl"

        source.to_jsonl(str(out))

        assert _contents(Dataset.from_jsonl(str(out))) == _contents(source)

    def test_file_shape_is_one_canonical_row_per_example(self, tmp_path: Path) -> None:
        source = _dataset(ROWS)
        out = tmp_path / "out.jsonl"

        source.to_jsonl(out)

        raw = out.read_bytes().decode("utf-8")
        assert raw.endswith("\n")
        lines = raw.splitlines()
        assert len(lines) == len(ROWS)
        first = json.loads(lines[0])
        assert first["input"] == ROWS[0][0]
        assert first["expected_output"] == ROWS[0][1]
        assert first["metadata"] == ROWS[0][2]
        # Explicit null expected output is written, not dropped.
        second = json.loads(lines[1])
        assert "expected_output" in second
        assert second["expected_output"] is None
        # Unicode is written as UTF-8, not \\u-escaped.
        assert "首都は" in raw
        assert "\\u" not in raw


class TestFailureModes:
    def test_empty_dataset_is_rejected_without_creating_a_file(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError):
            Dataset(examples=[]).to_jsonl(out)
        assert not out.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "row",
        [
            ({"q": "set"}, "A", {"bad": {1, 2}}),
            ({"q": object()}, "A", None),
            ({"q": "obj-expected"}, object(), None),
        ],
        ids=["set-in-metadata", "object-in-input", "object-expected"],
    )
    def test_unsupported_value_raises_instead_of_coercing(
        self, tmp_path: Path, row: tuple[Any, Any, dict[str, Any] | None]
    ) -> None:
        source = _dataset([({"q": "fine"}, "ok", None), row])
        out = tmp_path / "out.jsonl"
        with pytest.raises((TypeError, ValueError)):
            source.to_jsonl(out)
        assert not out.exists()

    def test_failed_write_leaves_existing_destination_untouched(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        original = b'{"input": {"q": "keep me"}, "expected_output": "K"}\n'
        out.write_bytes(original)
        # The bad value is in the LAST row: every row must be serialized
        # before the destination is touched.
        source = _dataset([({"q": "fine"}, "ok", None), ({"q": "bad"}, {1, 2}, None)])

        with pytest.raises((TypeError, ValueError)):
            source.to_jsonl(out)

        assert out.read_bytes() == original
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl"]

    def test_successful_write_replaces_destination_without_leftovers(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        out.write_text("stale content\n", encoding="utf-8")
        source = _dataset(ROWS[:1])

        source.to_jsonl(out)

        assert _contents(Dataset.from_jsonl(str(out))) == _contents(source)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl"]

    @pytest.mark.parametrize("failing_call", ["write", "replace"])
    def test_io_failure_mid_write_preserves_destination_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: str
    ) -> None:
        # The failure happens after serialization, inside the filesystem
        # write or the rename, so only a temp-file-then-rename writer passes.
        import os

        out = tmp_path / "out.jsonl"
        original = b"previous export\n"
        out.write_bytes(original)
        real_fdopen = os.fdopen

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        class _PartialWriter:
            """Writes some bytes, then fails like a full disk."""

            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def __enter__(self) -> _PartialWriter:
                return self

            def __exit__(self, *_exc: object) -> None:
                self._handle.close()

            def writelines(self, _lines: Any) -> None:
                self._handle.write("partial")
                self._handle.flush()
                boom()

        if failing_call == "write":
            monkeypatch.setattr(
                os, "fdopen", lambda *a, **k: _PartialWriter(real_fdopen(*a, **k))
            )
        else:
            monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(OSError):
            _dataset(ROWS[:1]).to_jsonl(out)

        assert out.read_bytes() == original
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl"]

    def test_new_file_permissions_follow_the_umask(self, tmp_path: Path) -> None:
        import os
        import stat

        previous = os.umask(0o022)
        try:
            out = tmp_path / "out.jsonl"
            _dataset(ROWS[:1]).to_jsonl(out)
        finally:
            os.umask(previous)
        assert stat.S_IMODE(out.stat().st_mode) == 0o644

    @pytest.mark.parametrize("existing_mode", [0o600, 0o640])
    def test_replacing_an_existing_file_keeps_its_permissions(
        self, tmp_path: Path, existing_mode: int
    ) -> None:
        # A plain ``open(..., "w")`` keeps an existing file's mode; the atomic
        # replace must too, so re-exporting over an owner-only file never
        # widens access to the raw examples it holds.
        import os
        import stat

        out = tmp_path / "out.jsonl"
        out.write_text("previous export\n", encoding="utf-8")
        out.chmod(existing_mode)
        previous = os.umask(0o022)
        try:
            _dataset(ROWS[:1]).to_jsonl(out)
        finally:
            os.umask(previous)
        assert stat.S_IMODE(out.stat().st_mode) == existing_mode
        assert _contents(Dataset.from_jsonl(str(out))) == _contents(_dataset(ROWS[:1]))
