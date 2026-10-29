"""Locator -> snapshot store factory: closed schemes, no network, no echo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from traigent.snapshot import backends
from traigent.snapshot import locator as locator_module
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
    monkeypatch.setattr(locator_module, "_azure_credential", lambda: "CRED")


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


@pytest.mark.parametrize(
    "bad",
    [
        "azblob://acct.blob.core.windows.net?sig=SECRET/cont",
        "azblob://acct.blob.core.windows.net/cont?sig=SECRET",
        "azblob://user" + ":SECRET" + "@acct.blob.core.windows.net/cont",
        "azblob://acct.blob.core.windows.net/cont#SECRET",
        "s3://key" + ":SECRET" + "@bucket/p",
        "s3://bucket/p?X-Amz-Signature=SECRET",
        "gs://bucket#SECRET",
        "github://owner/repo@main/p?token=SECRET",
        "github://user" + ":SECRET" + "@owner/repo@main",
        "file://user@host/SECRET",
        "/tmp/SECRET?x",
    ],
)
def test_credential_syntax_refused_before_credential_lookup(
    recorders: None, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    def lookup() -> None:
        raise AssertionError("credential lookup must not happen")

    monkeypatch.setattr(locator_module, "_azure_credential", lookup)
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(bad)
    assert info.value.code == LocatorErrorCode.CREDENTIALS_IN_LOCATOR
    assert "SECRET" not in str(info.value)


@pytest.mark.parametrize(
    "branch,prefix",
    [("main", ""), ("main", "snaps/x"), ("feature/fix", "snaps"), ("a/b/c", "")],
)
def test_github_emitter_factory_round_trip(branch: str, prefix: str) -> None:
    original = GitHubStore("owner/repo", branch, prefix)
    emitted = original.locator()
    assert "@" + branch.replace("/", "%2F") in emitted
    reopened = open_store_from_locator(emitted)
    assert isinstance(reopened, GitHubStore)
    assert reopened.locator() == emitted
    assert reopened._branch == branch and reopened._prefix == prefix


def test_github_unencoded_slash_means_prefix() -> None:
    store = open_store_from_locator("github://owner/repo@feature/fix/snaps")
    assert store._branch == "feature" and store._prefix == "fix/snaps"


@pytest.mark.parametrize("bad", ["main%2e", "ma%41in", "a%252Fb", "main%"])
def test_github_other_percent_escapes_refused(bad: str) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(f"github://owner/repo@{bad}/p")
    assert info.value.code == LocatorErrorCode.MALFORMED


def test_secure_by_default_and_opt_in(recorders: None) -> None:
    default = open_store_from_locator("azblob://127.0.0.1:10000/cont")
    assert default.args[0] == "https://127.0.0.1:10000"
    assert default.kwargs["allow_insecure_local"] is False
    assert open_store_from_locator("s3://b").kwargs["allow_insecure_local"] is False
    assert open_store_from_locator("gs://b").kwargs["allow_insecure_local"] is False
    opted = open_store_from_locator(
        "azblob://127.0.0.1:10000/cont", allow_insecure_local=True
    )
    assert opted.args[0] == "http://127.0.0.1:10000"
    assert opted.kwargs["allow_insecure_local"] is True
    remote = open_store_from_locator(
        "azblob://acct.blob.core.windows.net/cont", allow_insecure_local=True
    )
    assert remote.args[0] == "https://acct.blob.core.windows.net"
    assert (
        open_store_from_locator("s3://b", allow_insecure_local=True).kwargs[
            "allow_insecure_local"
        ]
        is True
    )


def test_azure_without_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name.startswith("azure.identity"):
            raise ImportError("no azure-identity")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator("azblob://acct.blob.core.windows.net/cont")
    assert info.value.code == LocatorErrorCode.MISSING_EXTRA
    assert "azure-identity" in str(info.value)


def test_file_locator_constructor_failure_is_closed(tmp_path: Path) -> None:
    blocker = tmp_path / "SECRETFILE"
    blocker.write_text("x")
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(f"file://{blocker}/sub")
    assert info.value.code == LocatorErrorCode.STORE_UNAVAILABLE
    assert "SECRETFILE" not in str(info.value)


def test_file_locator_traversal_and_symlink_contained(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "k").write_bytes(b"leak")
    store = open_store_from_locator(f"file://{root}")
    (root / "link").symlink_to(outside)
    for key in ("../outside/k", "link/k", "/etc/passwd"):
        with pytest.raises(Exception) as info:
            store.read(key)
        assert "leak" not in str(info.value)
    assert store.read("missing") is None
