"""Fix round 1 (stage B/C review): GitHub redirects, rate limits, budget, constructors."""

from __future__ import annotations

import io
import json
import traceback
from collections.abc import Callable
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3.util import parse_url

from tests.unit.snapshot.fakes_github import (
    OWNER_REPO,
    FakeGitHubSession,
    FakeResponse,
)
from traigent.snapshot import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.backends import AzureBlobStore, GCSStore, GitHubStore

CANARY = "CANARY-SECRET-r1-9d1b"
TOKEN = "ghp_" + "R1TOKENCANARY" + "0123456789abcdefgh"
API_HOST = "api.github.com"


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def store_over(
    session: Any, clock: Clock | None = None, **kwargs: Any
) -> tuple[GitHubStore, Clock]:
    clock = clock or Clock()
    return (
        GitHubStore(
            OWNER_REPO,
            "main",
            "snaps",
            session=session,
            sleep=clock.sleep,
            clock=clock.time,
            **kwargs,
        ),
        clock,
    )


def code_of(operation: Callable[[], Any]) -> SnapshotErrorCode:
    with pytest.raises(SnapshotStoreError) as caught:
        operation()
    return caught.value.code


# ------------------------------------- BC1 finding 1: redirect userinfo/backslash


class RecordingAdapter(BaseAdapter):
    """A transport that records every PreparedRequest Requests hands it."""

    def __init__(self, evil_location: str) -> None:
        super().__init__()
        self.seen: list[requests.PreparedRequest] = []
        self.evil_location = evil_location

    def send(self, request: requests.PreparedRequest, **_kwargs: Any) -> Any:
        self.seen.append(request)
        url = parse_url(request.url or "")
        path = url.path or ""
        headers: dict[str, str] = {}
        body = b""
        status = 200
        if url.host != API_HOST:
            body = b"stolen"
        elif path == f"/repos/{OWNER_REPO}":
            body = json.dumps({"private": True}).encode("utf-8")
        elif path.startswith(f"/repos/{OWNER_REPO}/branches/"):
            body = b"{}"
        elif "/contents/" in path and path.endswith("/blobs/x"):
            status = 302
            headers["Location"] = self.evil_location
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response._content = body
        response._content_consumed = True
        response.raw = io.BytesIO(body)
        response.url = request.url or ""
        response.request = request
        return response

    def close(self) -> None:
        return None


BAD_LOCATIONS = [
    "https://evil.example\\@api.github.com/x",
    "https://evil.example\\@api.github.com:443/x",
    "https://" + "user" + ":" + "pw" + "@api.github.com/x",
    "https://api.github.com\\@evil.example/x",
    "https://api.github.com\\x",
    "https://api.github.com@evil.example/x",
    "//evil.example\\@api.github.com/x",
    "/x\\@evil.example",
]


@pytest.mark.parametrize("location", BAD_LOCATIONS)
def test_bc1_f1_redirect_parser_differential_never_reaches_another_host(
    location: str,
) -> None:
    adapter = RecordingAdapter(location)
    session = requests.Session()
    session.mount("https://", adapter)
    store, _ = store_over(session)
    assert code_of(lambda: store.read("blobs/x")) is (
        SnapshotErrorCode.UNSAFE_DESTINATION
    )
    hosts = {parse_url(r.url or "").host for r in adapter.seen}
    assert hosts == {API_HOST}  # every request that left used the API host
    for sent in adapter.seen:
        assert "@" not in (sent.url or "").split("/")[2]  # no userinfo on the wire


def test_bc1_f1_valid_same_host_redirect_is_sent_exactly_as_validated() -> None:
    adapter = RecordingAdapter("https://api.github.com/repos/acme/snaps/contents/ok")
    session = requests.Session()
    session.mount("https://", adapter)
    store, _ = store_over(session)
    assert store.read("blobs/x") == b""  # the redirect target answers 200
    last = adapter.seen[-1]
    assert parse_url(last.url or "").host == API_HOST
    assert (last.url or "").startswith(
        "https://api.github.com/repos/acme/snaps/contents/ok"
    )
    assert last.headers["Authorization"] == f"Bearer {TOKEN}"


class DifferentialSession(requests.Session):
    """Preparation that lands on another host: a stand-in for a parser differential."""

    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        prepared = super().prepare_request(request)
        prepared.url = (prepared.url or "").replace(API_HOST, "evil.example", 1)
        return prepared


def test_bc1_f1_the_prepared_url_is_what_gets_validated_and_nothing_is_sent() -> None:
    adapter = RecordingAdapter("https://api.github.com/unused")
    session = DifferentialSession()
    session.mount("https://", adapter)
    store, _ = store_over(session)
    assert code_of(lambda: store.read("blobs/x")) is (
        SnapshotErrorCode.UNSAFE_DESTINATION
    )
    assert adapter.seen == []


# ------------------------------------------- BC2 finding 2: secondary limits

SECONDARY = (
    b'{"message":"You have exceeded a secondary rate limit. Please wait a few '
    b'minutes before you try again."}'
)


def warmed(session: FakeGitHubSession | None = None) -> Any:
    session = session or FakeGitHubSession()
    store, clock = store_over(session)
    store.read("blobs/warm")
    return store, session, clock


def test_bc2_f2_headerless_secondary_403_waits_the_documented_minute() -> None:
    store, session, clock = warmed()
    session.files[("main", "snaps/blobs/x")] = b"data"
    session.queue.append(FakeResponse(403, SECONDARY))
    assert store.read("blobs/x") == b"data"
    assert clock.slept == [60.0]


def test_bc2_f2_headerless_429_waits_the_documented_minute() -> None:
    store, session, clock = warmed()
    session.files[("main", "snaps/blobs/x")] = b"data"
    session.queue.append(FakeResponse(429))
    assert store.read("blobs/x") == b"data"
    assert clock.slept == [60.0]


def test_bc2_f2_second_secondary_limit_exceeds_the_budget_and_fails() -> None:
    store, session, clock = warmed()
    session.queue.extend([FakeResponse(403, SECONDARY), FakeResponse(403, SECONDARY)])
    assert code_of(lambda: store.read("blobs/x")) is (
        SnapshotErrorCode.BACKEND_UNAVAILABLE
    )
    assert sum(clock.slept) <= 60


def test_bc2_f2_ordinary_permission_403_is_still_auth() -> None:
    store, session, clock = warmed()
    session.queue.append(
        FakeResponse(403, b'{"message":"Resource not accessible by personal token"}')
    )
    assert code_of(lambda: store.read("blobs/x")) is SnapshotErrorCode.AUTH
    assert clock.slept == []


# ------------------------------------------ BC3 finding 3: shared wait budget


class LimitFirstUse(FakeGitHubSession):
    """First request of each (method, kind) in ``limited`` answers 429."""

    def __init__(self, limited: set[tuple[str, str]], retry_after: int) -> None:
        super().__init__()
        self.limited = set(limited)
        self.retry_after = retry_after

    @staticmethod
    def kind(path: str) -> str:
        if "/contents/" in path:
            return "contents"
        return "branch" if "/branches/" in path else "repo"

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        marker = (method, self.kind(url))
        if marker in self.limited:
            self.limited.discard(marker)
            return FakeResponse(429, headers={"Retry-After": str(self.retry_after)})
        return super().request(method, url, **kwargs)


def test_bc3_f3_cold_read_shares_one_wait_budget_across_its_requests() -> None:
    session = LimitFirstUse(
        {("GET", "repo"), ("GET", "branch"), ("GET", "contents")}, 40
    )
    store, clock = store_over(session)
    assert code_of(lambda: store.read("blobs/x")) is (
        SnapshotErrorCode.BACKEND_UNAVAILABLE
    )
    assert sum(clock.slept) <= 60


def test_bc3_f3_create_with_conflict_reread_shares_the_budget() -> None:
    session = LimitFirstUse({("GET", "repo"), ("GET", "contents")}, 40)
    session.files[("main", "snaps/blobs/x")] = b"other"
    store, clock = store_over(session)
    assert code_of(lambda: store.create_if_absent("blobs/x", b"mine")) is (
        SnapshotErrorCode.BACKEND_UNAVAILABLE
    )
    assert sum(clock.slept) <= 60


def test_bc3_f3_conflict_backoff_counts_against_the_budget() -> None:
    session = LimitFirstUse({("GET", "repo")}, 55)
    session.put_conflicts = 3  # backoff 1 + 2 + 4 would take the total to 62
    store, clock = store_over(session)
    assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
        SnapshotErrorCode.BACKEND_UNAVAILABLE
    )
    assert sum(clock.slept) <= 60


def test_bc3_f3_budget_is_per_operation_not_per_store() -> None:
    session = LimitFirstUse({("GET", "repo")}, 40)
    store, clock = store_over(session)
    store.read("blobs/a")  # 40s spent in this operation
    session.limited.add(("GET", "contents"))
    session.retry_after = 40
    store.read("blobs/b")  # a new operation gets a fresh budget
    assert clock.slept == [40.0, 40.0]


# ------------------------------------- BC4 finding 4: constructor sanitising


def _assert_hygienic(error: SnapshotStoreError, backend: str) -> None:
    assert error.backend == backend
    assert error.__cause__ is None and error.__suppress_context__ is True
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert CANARY not in text


def test_bc4_f4_gcs_default_client_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = pytest.importorskip("google.cloud.storage")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"credentials file /home/u/{CANARY}.json malformed")

    monkeypatch.setattr(storage, "Client", boom)
    with pytest.raises(SnapshotStoreError) as caught:
        GCSStore("my-bucket")
    assert caught.value.code is SnapshotErrorCode.BACKEND_UNAVAILABLE
    _assert_hygienic(caught.value, "gcs")


def test_bc4_f4_gcs_missing_credentials_map_to_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = pytest.importorskip("google.cloud.storage")
    auth = pytest.importorskip("google.auth.exceptions")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise auth.DefaultCredentialsError(f"no creds {CANARY}")

    monkeypatch.setattr(storage, "Client", boom)
    with pytest.raises(SnapshotStoreError) as caught:
        GCSStore("my-bucket")
    assert caught.value.code is SnapshotErrorCode.AUTH
    _assert_hygienic(caught.value, "gcs")


def test_bc4_f4_azure_container_client_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = pytest.importorskip("azure.storage.blob")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(f"conflicting SAS credentials ?sig={CANARY}")

    monkeypatch.setattr(blob, "ContainerClient", boom)
    with pytest.raises(SnapshotStoreError) as caught:
        AzureBlobStore.from_account("https://acct.blob.core.windows.net", "snaps")
    assert caught.value.code is SnapshotErrorCode.BACKEND_UNAVAILABLE
    _assert_hygienic(caught.value, "azure_blob")


def test_bc4_f4_missing_extra_hints_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("google.cloud", "azure.storage")):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"snapshot-gcs"):
        GCSStore("my-bucket")
    with pytest.raises(ImportError, match=r"snapshot-azure"):
        AzureBlobStore.from_account("https://acct.blob.core.windows.net", "snaps")


# ------------------------------- BC5 finding 5: 404 after cached readiness


def test_bc5_f5_deleted_branch_after_warm_read_is_not_found_not_missing() -> None:
    store, session, _ = warmed()
    session.branches.clear()
    assert code_of(lambda: store.read("blobs/absent")) is SnapshotErrorCode.NOT_FOUND


def test_bc5_f5_lost_repository_access_is_not_found_not_missing() -> None:
    store, session, _ = warmed()
    session.repo_exists = False
    assert code_of(lambda: store.read("blobs/absent")) is SnapshotErrorCode.NOT_FOUND


def test_bc5_f5_missing_file_on_a_live_branch_is_still_none() -> None:
    store, session, _ = warmed()
    assert store.read("blobs/absent") is None
    assert "main" in session.branches


def test_bc5_f5_store_recovers_after_the_branch_returns() -> None:
    store, session, _ = warmed()
    session.branches.clear()
    code_of(lambda: store.read("blobs/absent"))
    session.branches.add("main")
    session.files[("main", "snaps/blobs/x")] = b"back"
    assert store.read("blobs/x") == b"back"


# ------------------------------------ FIX R2 P2: IPv6 literals in prepared URLs


class IPv6Adapter(BaseAdapter):
    """Answers any host with the repo and branch, optionally redirects contents."""

    def __init__(self, redirect_to: str | None = None) -> None:
        super().__init__()
        self.seen: list[requests.PreparedRequest] = []
        self.redirect_to = redirect_to

    def send(self, request: requests.PreparedRequest, **_kwargs: Any) -> Any:
        self.seen.append(request)
        path = parse_url(request.url or "").path or ""
        status, body, headers = 404, b"", {}
        if path == f"/repos/{OWNER_REPO}":
            status, body = 200, json.dumps({"private": True}).encode("utf-8")
        elif path.startswith(f"/repos/{OWNER_REPO}/branches/"):
            status, body = 200, b"{}"
        elif self.redirect_to is not None and "/contents/" in path:
            status, headers = 302, {"Location": self.redirect_to}
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response._content = body
        response._content_consumed = True
        response.raw = io.BytesIO(body)
        response.url = request.url or ""
        response.request = request
        return response

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("api_url", "insecure_local", "bracketed"),
    [
        ("http://[::1]:8080", True, "[::1]"),
        ("https://[2001:db8::1]", False, "[2001:db8::1]"),
    ],
)
def test_fix_r2_p2_ipv6_api_origin_is_sent_not_refused(
    api_url: str, insecure_local: bool, bracketed: str
) -> None:
    adapter = IPv6Adapter()
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    store, _ = store_over(session, api_url=api_url, allow_insecure_local=insecure_local)
    assert store.read("blobs/x") is None  # the 404 is an honest absence
    assert adapter.seen, "the IPv6 origin must reach the transport"
    assert {parse_url(r.url or "").host for r in adapter.seen} == {bracketed}


@pytest.mark.parametrize(
    "location",
    [
        "http://user@[::1]:8080/x",
        "http://[::1]:8080\\@evil.example/x",
    ],
)
def test_fix_r2_p2_ipv6_origin_still_refuses_userinfo_and_backslash(
    location: str,
) -> None:
    adapter = IPv6Adapter(redirect_to=location)
    session = requests.Session()
    session.mount("http://", adapter)
    store, _ = store_over(
        session, api_url="http://[::1]:8080", allow_insecure_local=True
    )
    assert code_of(lambda: store.read("blobs/x")) is (
        SnapshotErrorCode.UNSAFE_DESTINATION
    )
    assert "evil.example" not in {parse_url(r.url or "").host for r in adapter.seen}
