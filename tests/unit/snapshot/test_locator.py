"""Locator -> snapshot store factory: closed schemes, no network, no echo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from traigent.snapshot import backends
from traigent.snapshot.backends import GitHubStore, LocalDirStore
from traigent.snapshot.locator import (
    LocatorErrorCode,
    SnapshotLocatorError,
    open_store_from_locator,
)


class _Rec:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs
        self.via = "init"

    @classmethod
    def from_account(cls, *args: Any, **kwargs: Any) -> _Rec:
        made = cls(*args, **kwargs)
        made.via = "from_account"
        return made


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("S3Store", "GCSStore", "AzureBlobStore"):
        monkeypatch.setattr(backends, name, type(name, (_Rec,), {}))


def test_s3_with_prefix(recorders: None) -> None:
    store = open_store_from_locator("s3://my-bucket/a/b")
    assert type(store).__name__ == "S3Store"
    assert store.args == ("my-bucket", "a/b")


def test_s3_without_prefix(recorders: None) -> None:
    assert open_store_from_locator("s3://my-bucket").args == ("my-bucket", "")


def test_gcs(recorders: None) -> None:
    store = open_store_from_locator("gs://bkt/p")
    assert type(store).__name__ == "GCSStore"
    assert store.args == ("bkt", "p")


def test_azure(recorders: None) -> None:
    store = open_store_from_locator("azblob://acct.blob.core.windows.net/cont/pre/fix")
    assert type(store).__name__ == "AzureBlobStore"
    assert store.via == "from_account"
    assert store.args[0] == "https://acct.blob.core.windows.net"
    assert store.args[1] == "cont"
    assert store.args[3] == "pre/fix"


def test_github() -> None:
    store = open_store_from_locator("github://owner/repo@main/snaps")
    assert isinstance(store, GitHubStore)
    assert store.locator() == "github://owner/repo@main/snaps"


def test_local_path_and_file_scheme(tmp_path: Path) -> None:
    assert isinstance(open_store_from_locator(str(tmp_path / "a")), LocalDirStore)
    assert isinstance(
        open_store_from_locator(f"file://{tmp_path / 'b'}"), LocalDirStore
    )


@pytest.mark.parametrize(
    "bad", ["ftp://SECRET-host/x", "http://SECRET/x", "weird+scheme://SECRET"]
)
def test_unknown_scheme_refused_without_echo(bad: str) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(bad)
    assert info.value.code == LocatorErrorCode.UNKNOWN_SCHEME
    assert "SECRET" not in str(info.value)


@pytest.mark.parametrize(
    "bad", ["s3://SECRET_UPPER", "github://SECRETnoat", "azblob://SECRETonly", ""]
)
def test_malformed_refused_without_echo(bad: str) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(bad)
    assert info.value.code == LocatorErrorCode.MALFORMED
    assert "SECRET" not in str(info.value)


def test_missing_extra_names_the_pip_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise ImportError("no boto3")

    monkeypatch.setattr(backends, "S3Store", boom)
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator("s3://bkt")
    assert info.value.code == LocatorErrorCode.MISSING_EXTRA
    assert "traigent[snapshot-s3]" in str(info.value)
