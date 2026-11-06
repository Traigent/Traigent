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
    store = open_store_from_locator("github+v2://owner/repo@main/snaps")
    assert isinstance(store, GitHubStore)
    assert store.locator() == "github+v2://owner/repo@main/snaps"


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


@pytest.mark.parametrize(
    "legacy",
    ["github://owner/repo@feature/fix/snaps", "github://owner/repo@main/snaps"],
)
def test_github_unversioned_prefixed_locator_refused(legacy: str) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(legacy)
    assert info.value.code == LocatorErrorCode.AMBIGUOUS_BRANCH
    assert "fix" not in str(info.value) and "snaps" not in str(info.value)
    assert "%2F" in str(info.value)


def test_github_unversioned_without_prefix_accepted() -> None:
    store = open_store_from_locator("github://owner/repo@main")
    assert store._branch == "main" and store._prefix == ""


def test_github_v2_raw_slash_is_branch_then_prefix() -> None:
    store = open_store_from_locator("github+v2://owner/repo@feature%2Ffix/snaps")
    assert store._branch == "feature/fix" and store._prefix == "snaps"


@pytest.mark.parametrize("bad", ["main%2e", "ma%41in", "a%252Fb", "main%"])
def test_github_other_percent_escapes_refused(bad: str) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(f"github+v2://owner/repo@{bad}/p")
    assert info.value.code == LocatorErrorCode.MALFORMED


@pytest.mark.parametrize(
    "host", ["localhost:10000", "[::1]:10000", "127.0.0.1:10000", "localhost"]
)
def test_azblob_loopback_hosts_allow_plain_http_only_when_opted_in(
    recorders: None, host: str
) -> None:
    locator = f"azblob://{host}/cont"
    assert open_store_from_locator(locator).args[0] == f"https://{host}"
    opted = open_store_from_locator(locator, allow_insecure_local=True)
    assert opted.args[0] == f"http://{host}"


@pytest.mark.parametrize("host", ["acct.blob.core.windows.net", "localhost.evil.com"])
def test_azblob_non_loopback_hosts_stay_https_even_when_opted_in(
    recorders: None, host: str
) -> None:
    opted = open_store_from_locator(f"azblob://{host}/cont", allow_insecure_local=True)
    assert opted.args[0] == f"https://{host}"


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


def test_azure_extra_provides_default_credential() -> None:
    """The advertised snapshot-azure extra carries what the factory imports."""
    pytest.importorskip("azure.identity")
    real = locator_module._azure_credential()
    assert type(real).__name__ == "DefaultAzureCredential"
    import tomllib

    root = Path(__file__).resolve().parents[3]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["snapshot-azure"]
    assert any(e.startswith("azure-identity") for e in extras)


_GHE = "https://ghe.example.com/api/v3"


@pytest.fixture
def ghe_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_GITHUB_API_URL", _GHE)


def test_ghe_locator_round_trips_with_api_url(ghe_configured: None) -> None:
    original = GitHubStore("owner/repo", "feature/x", "snaps", api_url=_GHE)
    loc = original.locator()
    assert loc.startswith("github+v2://https%3A%2F%2Fghe.example.com%2Fapi%2Fv3/")
    reopened = open_store_from_locator(loc)
    assert isinstance(reopened, GitHubStore)
    assert reopened._api_url == _GHE
    assert reopened.locator() == loc


def test_github_com_locator_unchanged_and_default_api() -> None:
    store = GitHubStore("owner/repo", "main", "p")
    assert store.locator() == "github+v2://owner/repo@main/p"
    assert open_store_from_locator(store.locator())._api_url == (
        "https://api.github.com"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "github+v2://https%3A%2F%2Fuser%3ASECRET%40ghe.example.com/owner/repo@main",
        "github+v2://https%3A%2F%2Fghe.example.com%3Ftoken%3DSECRET/owner/repo@main",
        "github+v2://https://user" + ":SECRET" + "@ghe.example.com/owner/repo@main",
        "github+v2://SECRET@https%3A%2F%2Fghe.example.com/owner/repo@main",
    ],
)
def test_ghe_locator_credentials_refused(bad: str, ghe_configured: None) -> None:
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(bad)
    assert info.value.code == LocatorErrorCode.CREDENTIALS_IN_LOCATOR
    assert "SECRET" not in str(info.value)


def test_ghe_locator_refused_when_endpoint_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_GITHUB_API_URL", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "TOKEN-SECRET")

    def no_request(*a: Any, **k: Any) -> None:
        raise AssertionError("no request may be made")

    monkeypatch.setattr("requests.Session.request", no_request)
    loc = GitHubStore("owner/repo", "main", api_url=_GHE).locator()
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(loc)
    assert info.value.code == LocatorErrorCode.API_URL_NOT_CONFIGURED
    assert "ghe.example.com" not in str(info.value)


def test_ghe_locator_refused_on_host_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_GITHUB_API_URL", "https://other.example.com/api/v3")
    loc = GitHubStore("owner/repo", "main", api_url=_GHE).locator()
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(loc)
    assert info.value.code == LocatorErrorCode.API_URL_NOT_CONFIGURED


@pytest.mark.parametrize(
    "url",
    [
        "https://ghe.example.com/api%3Ftoken%3DSECRET",
        "https://ghe.example.com/other",
        "https://ghe.example.com/api/v3/x",
        "https://ghe.example.com/%61pi/v3",
    ],
)
def test_github_store_rejects_odd_api_url(url: str) -> None:
    with pytest.raises(Exception) as info:
        GitHubStore("owner/repo", "main", api_url=url)
    assert "UNSAFE_DESTINATION" in str(getattr(info.value, "code", info.value)).upper()


def test_ghe_locator_with_odd_api_url_refused_at_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    odd = "https://ghe.example.com/api%3Ftoken%3DSECRET"
    monkeypatch.setenv("TRAIGENT_GITHUB_API_URL", odd)
    loc = (
        "github+v2://https%3A%2F%2Fghe.example.com%2Fapi%253Ftoken%253DSECRET/o/r@main"
    )
    with pytest.raises(SnapshotLocatorError) as info:
        open_store_from_locator(loc)
    assert info.value.code == LocatorErrorCode.STORE_UNAVAILABLE
    assert "SECRET" not in str(info.value)


@pytest.mark.parametrize("name", ["a?b", "a#b", "x?y#z", "100%", "a%3Fb"])
def test_local_dir_special_names_round_trip(tmp_path: Path, name: str) -> None:
    store = LocalDirStore(tmp_path / name)
    reopened = open_store_from_locator(store.locator())
    assert isinstance(reopened, LocalDirStore)
    assert reopened._root == (tmp_path / name).resolve()


def test_old_file_locator_is_literal_not_decoded(tmp_path: Path) -> None:
    store = open_store_from_locator(f"file://{tmp_path}/a%2Fb")
    assert store._root == (tmp_path / "a%2Fb").resolve()
    assert (tmp_path / "a%2Fb").is_dir()


def test_plain_path_and_raw_file_locator_accept_query_chars(tmp_path: Path) -> None:
    plain = tmp_path / "q?x#y"
    assert open_store_from_locator(str(plain))._root == plain.resolve()
    raw = tmp_path / "r?x#y"
    assert open_store_from_locator(f"file://{raw}")._root == raw.resolve()
