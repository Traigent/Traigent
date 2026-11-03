"""Run the shared conformance suite against each backend with its fake."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.unit.snapshot.conformance import CANARY, StoreConformance
from tests.unit.snapshot.fakes import FakeS3Client, client_error
from traigent.snapshot import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.backends import LocalDirStore, S3Store


class TestLocalDirStoreConformance(StoreConformance):
    def make_store(self, tmp_path: Any) -> LocalDirStore:
        return LocalDirStore(tmp_path / "store")

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(CANARY)

        monkeypatch.setattr(Path, "read_bytes", boom)
        monkeypatch.setattr(Path, "mkdir", boom)

    def test_locator_format(self, tmp_path: Path) -> None:
        store = LocalDirStore(tmp_path / "s")
        assert store.locator() == f"file://{(tmp_path / 's').resolve()}"

    def test_atomic_no_clobber_and_no_temp_leftovers(self, tmp_path: Path) -> None:
        store = LocalDirStore(tmp_path / "s")
        store.create_if_absent("blobs/a", b"first")
        store.create_if_absent("blobs/a", b"first")
        names = sorted(p.name for p in (tmp_path / "s" / "blobs").iterdir())
        assert names == ["a"]  # temp names unlinked, final name written via link

    def test_final_name_never_written_directly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash after the temp write must leave no final-name object."""
        store = LocalDirStore(tmp_path / "s")

        def crash(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("crash before link")

        monkeypatch.setattr(os, "link", crash)
        with pytest.raises(SnapshotStoreError):
            store.create_if_absent("blobs/a", b"data")
        assert store.read("blobs/a") is None

    def test_symlink_escape_refused(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "s"
        store = LocalDirStore(root)
        (root / "blobs").symlink_to(outside, target_is_directory=True)
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/a", b"x")
        assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION
        assert not (outside / "a").exists()


class TestS3StoreConformance(StoreConformance):
    def make_store(self, tmp_path: Any) -> S3Store:
        return S3Store("my-bucket", "snap/prefix", client=FakeS3Client())

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        store._client.fail_with = client_error("InternalError", 500, CANARY)

    def test_locator_format(self) -> None:
        assert S3Store("my-bucket", "a/b/", client=FakeS3Client()).locator() == (
            "s3://my-bucket/a/b"
        )
        assert S3Store("my-bucket", client=FakeS3Client()).locator() == "s3://my-bucket"

    def test_conditional_write_header_sent(self) -> None:
        client = FakeS3Client()
        S3Store("my-bucket", "p", client=client).create_if_absent("blobs/x", b"1")
        assert client.put_calls[0]["IfNoneMatch"] == "*"
        assert client.put_calls[0]["Key"] == "p/blobs/x"

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("AccessDenied", SnapshotErrorCode.AUTH),
            ("ExpiredToken", SnapshotErrorCode.AUTH),
            ("NotImplemented", SnapshotErrorCode.UNSUPPORTED_BACKEND),
            ("SlowDown", SnapshotErrorCode.BACKEND_UNAVAILABLE),
        ],
    )
    def test_error_mapping_raises_never_none(
        self, code: str, expected: SnapshotErrorCode
    ) -> None:
        client = FakeS3Client()
        store = S3Store("my-bucket", client=client)
        client.fail_with = client_error(code, 403)
        with pytest.raises(SnapshotStoreError) as caught:
            store.read("blobs/x")  # an auth/transport failure is never "missing"
        assert caught.value.code is expected

    def test_no_such_bucket_raises_not_found(self) -> None:
        client = FakeS3Client()
        store = S3Store("my-bucket", client=client)
        client.fail_with = client_error("NoSuchBucket", 404)
        with pytest.raises(SnapshotStoreError) as caught:
            store.read("blobs/x")
        assert caught.value.code is SnapshotErrorCode.NOT_FOUND

    def test_missing_boto3_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def no_boto3(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "boto3":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_boto3)
        with pytest.raises(ImportError, match=r"traigent\[snapshot-s3\]"):
            S3Store("my-bucket")
