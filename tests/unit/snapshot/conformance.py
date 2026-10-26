"""Shared conformance suite every snapshot store backend must pass.

Subclass :class:`StoreConformance`, implement ``make_store`` and ``break_backend``.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import pytest

from traigent.snapshot import (
    CreateResult,
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
    verify_snapshot,
)

CANARY = "CANARY-SECRET-7f3a9"


class StoreConformance:
    def make_store(self, tmp_path: Any) -> Any:
        raise NotImplementedError

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make every later backend call fail with a provider error carrying CANARY."""
        raise NotImplementedError

    @pytest.fixture
    def store(self, tmp_path: Any) -> Any:
        return self.make_store(tmp_path)

    def test_create_existed_conflict(self, store: Any) -> None:
        assert store.create_if_absent("blobs/x", b"one") is CreateResult.CREATED
        assert (
            store.create_if_absent("blobs/x", b"one") is CreateResult.EXISTED_IDENTICAL
        )
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/x", b"two")
        assert caught.value.code is SnapshotErrorCode.CONFLICT
        assert store.read("blobs/x") == b"one"  # the first writer won

    def test_read_none_only_for_not_found(self, store: Any) -> None:
        assert store.read("blobs/absent") is None

    @pytest.mark.parametrize(
        "key", ["", "../x", "a//b", "/abs", "a/./b", "a\\b", "a b"]
    )
    def test_invalid_keys_refused(self, store: Any, key: str) -> None:
        with pytest.raises(ValueError):
            store.read(key)
        with pytest.raises(ValueError):
            store.create_if_absent(key, b"x")

    def test_publish_verify_roundtrip(self, store: Any) -> None:
        snapshot = build_execution_snapshot(
            dataset=[{"input": {"q": "a"}, "expected": "b"}], config={"t": 0.5}
        )
        ref = publish_snapshot(store, snapshot)
        assert ref.locator == store.locator()
        report = verify_snapshot(store, ref.manifest_digest)
        assert report.integrity == "ok"

    def test_size_preflight_writes_nothing(self, store: Any) -> None:
        snapshot = build_execution_snapshot(
            dataset=[{"input": "x" * 200, "expected": "y"}], config={"a": 1}
        )
        store.max_object_bytes = 100
        with pytest.raises(SnapshotStoreError) as caught:
            publish_snapshot(store, snapshot)
        assert caught.value.code is SnapshotErrorCode.TOO_LARGE
        assert caught.value.part == "dataset/examples.jsonl"
        store.max_object_bytes = None
        for _, entry in snapshot.present_parts():
            assert store.read(f"blobs/sha256/{entry['sha256'][7:]}") is None

    def test_repr_is_locator(self, store: Any) -> None:
        assert repr(store) == store.locator()
        locator = urlsplit(store.locator())
        assert "?" not in store.locator() and locator.username is None
        assert locator.password is None

    def test_error_hygiene(
        self,
        store: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.break_backend(store, monkeypatch)
        operations: list[Callable[[], Any]] = [
            lambda: store.read("blobs/x"),
            lambda: store.create_if_absent("blobs/y", b"data"),
        ]
        for operation in operations:
            with caplog.at_level(logging.DEBUG):
                with pytest.raises(SnapshotStoreError) as caught:
                    operation()
            error = caught.value
            assert error.code in {
                SnapshotErrorCode.BACKEND_UNAVAILABLE,
                SnapshotErrorCode.AUTH,
            }
            assert error.__cause__ is None
            assert error.__suppress_context__ is True
            text = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            assert CANARY not in text and CANARY not in repr(error)
            assert CANARY not in str(vars(error))
        assert CANARY not in caplog.text
