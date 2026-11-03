"""GitHub backend: conformance plus the safety rules of the Contents API store."""

from __future__ import annotations

import base64
import logging
import traceback
from typing import Any

import pytest

from tests.unit.snapshot.conformance import CANARY, StoreConformance
from tests.unit.snapshot.fakes_github import (
    OWNER_REPO,
    FakeGitHubSession,
    FakeResponse,
    connection_error,
    json_response,
)
from traigent.snapshot import (
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
)
from traigent.snapshot.backends import GitHubStore

# Built at runtime so no token-shaped literal sits in the source.
TOKEN = "ghp_" + "TOKENCANARY" + "0123456789abcdefghij"


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make(
    session: FakeGitHubSession | None = None,
    clock: Clock | None = None,
    prefix: str = "snaps",
    **kwargs: Any,
) -> tuple[GitHubStore, FakeGitHubSession, Clock]:
    session = session or FakeGitHubSession()
    clock = clock or Clock()
    store = GitHubStore(
        OWNER_REPO,
        "main",
        prefix,
        session=session,
        sleep=clock.sleep,
        clock=clock.time,
        **kwargs,
    )
    return store, session, clock


def code_of(operation: Any) -> SnapshotErrorCode:
    with pytest.raises(SnapshotStoreError) as caught:
        operation()
    return caught.value.code


class TestGitHubStoreConformance(StoreConformance):
    def make_store(self, tmp_path: Any) -> GitHubStore:
        return make()[0]

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        store._session.always = FakeResponse(500, CANARY.encode())

    def test_locator_format(self) -> None:
        assert make(prefix="a/b/")[0].locator() == f"github://{OWNER_REPO}@main/a/b"
        assert make(prefix="")[0].locator() == f"github://{OWNER_REPO}@main"

    def test_put_has_explicit_branch_and_no_sha(self) -> None:
        store, session, _ = make()
        store.create_if_absent("blobs/x", b"payload")
        (put,) = session.puts()
        assert put["json"]["branch"] == "main" and "sha" not in put["json"]
        assert base64.b64decode(put["json"]["content"]) == b"payload"
        assert put["path"] == f"/repos/{OWNER_REPO}/contents/snaps/blobs/x"

    def test_every_request_is_bounded_and_does_not_auto_redirect(self) -> None:
        store, session, _ = make()
        store.create_if_absent("blobs/x", b"1")
        store.read("blobs/x")
        assert session.calls
        for call in session.calls:
            assert call["timeout"] == 30
            assert call["allow_redirects"] is False
            assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


class TestDestinationSafety:
    def test_missing_branch_is_not_found_and_writes_nothing(self) -> None:
        store, session, _ = make(FakeGitHubSession(branches=("other",)))
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.NOT_FOUND
        )
        assert session.puts() == []

    def test_empty_repository_is_not_found(self) -> None:
        session = FakeGitHubSession(branches=())
        store, _, _ = make(session)
        assert code_of(lambda: store.read("blobs/x")) is SnapshotErrorCode.NOT_FOUND
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.NOT_FOUND
        )
        assert session.puts() == []

    def test_missing_repository_is_not_found(self) -> None:
        store, _, _ = make(FakeGitHubSession(repo_exists=False))
        assert code_of(lambda: store.read("blobs/x")) is SnapshotErrorCode.NOT_FOUND

    def test_public_repository_refused_by_default_and_writes_nothing(self) -> None:
        store, session, _ = make(FakeGitHubSession(private=False))
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )
        assert session.puts() == []

    def test_public_repository_allowed_with_explicit_flag(self) -> None:
        store, _, _ = make(FakeGitHubSession(private=False), allow_public=True)
        assert store.create_if_absent("blobs/x", b"1") is not None

    def test_visibility_without_private_field_fails_closed(self) -> None:
        session = FakeGitHubSession()
        session.queue.append(json_response(200, {"name": "snaps"}))
        store, _, _ = make(session)
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )

    def test_visibility_and_branch_checked_once(self) -> None:
        store, session, _ = make()
        store.create_if_absent("blobs/x", b"1")
        store.read("blobs/x")
        store.create_if_absent("blobs/y", b"2")
        repo_gets = [
            c
            for c in session.calls
            if c["path"]
            in (f"/repos/{OWNER_REPO}", f"/repos/{OWNER_REPO}/branches/main")
        ]
        assert len(repo_gets) == 2  # one visibility + one branch check, total

    def test_missing_token_is_auth_and_sends_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_TOKEN")
        store, session, _ = make()
        assert code_of(lambda: store.read("blobs/x")) is SnapshotErrorCode.AUTH
        assert session.calls == []

    def test_token_read_at_call_time_from_named_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_SNAP_TOKEN", "first")
        store, session, _ = make(token_env="MY_SNAP_TOKEN")
        monkeypatch.setenv("MY_SNAP_TOKEN", "second")
        store.read("blobs/x")
        assert session.calls[0]["headers"]["Authorization"] == "Bearer second"

    @pytest.mark.parametrize(
        "url", ["http://api.example.com", "ftp://x", "api.example.com", "https://"]
    )
    def test_insecure_api_url_refused(self, url: str) -> None:
        assert code_of(lambda: GitHubStore(OWNER_REPO, "main", api_url=url)) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )

    def test_enterprise_https_api_url_and_loopback_flag(self) -> None:
        store = GitHubStore(
            OWNER_REPO, "main", api_url="https://ghe.example.com/api/v3"
        )
        assert store.locator() == f"github://{OWNER_REPO}@main"
        local = GitHubStore(
            OWNER_REPO,
            "main",
            api_url="http://127.0.0.1:8080",
            allow_insecure_local=True,
        )
        assert local.locator() == f"github://{OWNER_REPO}@main"

    def test_credentials_in_api_url_refused(self) -> None:
        assert code_of(
            lambda: GitHubStore(
                OWNER_REPO, "main", api_url="https://u:p@api.example.com"
            )
        ) is (SnapshotErrorCode.UNSAFE_DESTINATION)

    @pytest.mark.parametrize("repo", ["", "noslash", "a/b/c", "a/ b", "../x"])
    def test_invalid_repo_refused(self, repo: str) -> None:
        with pytest.raises(ValueError):
            GitHubStore(repo, "main")

    @pytest.mark.parametrize(
        "branch", ["", "-x", "a..b", "a b", "x.lock", "/x", "a//b"]
    )
    def test_invalid_branch_refused(self, branch: str) -> None:
        with pytest.raises(ValueError):
            GitHubStore(OWNER_REPO, branch)

    def test_invalid_token_env_refused(self) -> None:
        with pytest.raises(ValueError):
            GitHubStore(OWNER_REPO, "main", token_env="not a name")


class TestWritesAndReads:
    def test_existing_file_identical_and_different(self) -> None:
        store, session, _ = make()
        store.create_if_absent("blobs/x", b"one")
        puts = len(session.puts())
        assert store.create_if_absent("blobs/x", b"one").value == "existed_identical"
        assert code_of(lambda: store.create_if_absent("blobs/x", b"two")) is (
            SnapshotErrorCode.CONFLICT
        )
        assert session.files[("main", "snaps/blobs/x")] == b"one"
        assert len(session.puts()) == puts + 2  # attempts, never an update with sha
        assert all("sha" not in p["json"] for p in session.puts())

    def test_branch_advance_conflict_retries_then_succeeds(self) -> None:
        session = FakeGitHubSession()
        session.put_conflicts = 2
        store, _, clock = make(session)
        assert store.create_if_absent("blobs/x", b"1").value == "created"
        assert len(session.puts()) == 3
        assert clock.slept == [1.0, 2.0]

    def test_branch_advance_retry_is_bounded(self) -> None:
        session = FakeGitHubSession()
        session.put_conflicts = 100
        store, _, clock = make(session)
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.BACKEND_UNAVAILABLE
        )
        assert len(session.puts()) == 4  # first attempt + 3 retries
        assert clock.slept == [1.0, 2.0, 4.0]
        assert ("main", "snaps/blobs/x") not in session.files

    def test_conflict_where_another_writer_won_is_compared_not_retried(self) -> None:
        session = FakeGitHubSession()
        store, _, clock = make(session)
        store.create_if_absent("blobs/ready", b"x")  # warm the preflight
        session.queue.append(FakeResponse(409))  # our PUT collides ...
        session.files[("main", "snaps/blobs/y")] = b"theirs"  # ... with this file
        assert code_of(lambda: store.create_if_absent("blobs/y", b"mine")) is (
            SnapshotErrorCode.CONFLICT
        )
        assert clock.slept == []

    def test_large_raw_read_uses_raw_media_type_and_explicit_ref(self) -> None:
        store, session, _ = make()
        big = b"z" * (2 * 1024 * 1024)
        session.files[("main", "snaps/blobs/big")] = big
        assert store.read("blobs/big") == big
        call = session.calls[-1]
        assert call["headers"]["Accept"] == "application/vnd.github.raw+json"
        assert call["params"] == {"ref": "main"}

    def test_read_of_missing_file_is_none_not_an_error(self) -> None:
        assert make()[0].read("blobs/absent") is None

    def test_put_to_deleted_branch_is_not_found(self) -> None:
        store, session, _ = make()
        store.read("blobs/warm")
        session.branches.clear()
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.NOT_FOUND
        )

    def test_413_is_too_large(self) -> None:
        store, session, _ = make()
        store.read("blobs/warm")
        session.queue.append(FakeResponse(413))
        assert code_of(lambda: store.create_if_absent("blobs/x", b"1")) is (
            SnapshotErrorCode.TOO_LARGE
        )

    def test_writes_are_serialized(self) -> None:
        import threading

        class Probe(FakeGitHubSession):
            def __init__(self) -> None:
                super().__init__()
                self.active = 0
                self.peak = 0
                self.guard = threading.Lock()

            def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
                if method != "PUT":
                    return super().request(method, url, **kwargs)
                with self.guard:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                threading.Event().wait(0.01)
                try:
                    return super().request(method, url, **kwargs)
                finally:
                    with self.guard:
                        self.active -= 1

        probe = Probe()
        store, _, _ = make(probe)
        threads = [
            threading.Thread(target=store.create_if_absent, args=(f"blobs/k{i}", b"v"))
            for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert len(probe.files) == 8 and probe.peak == 1


class TestSizeCap:
    def test_publish_preflights_all_parts_before_any_request(self) -> None:
        store, session, _ = make()
        snapshot = build_execution_snapshot(
            dataset=[{"input": "x" * (51 * 1024 * 1024), "expected": "y"}],
            config={"a": 1},
        )
        assert code_of(lambda: publish_snapshot(store, snapshot)) is (
            SnapshotErrorCode.TOO_LARGE
        )
        assert session.calls == []  # not even the visibility check ran

    def test_create_if_absent_refuses_over_cap_directly(self) -> None:
        store, session, _ = make()
        assert code_of(
            lambda: store.create_if_absent("blobs/x", b"a" * (50 * 1024**2 + 1))
        ) is (SnapshotErrorCode.TOO_LARGE)
        assert session.puts() == []


class TestRateLimits:
    def test_retry_after_is_honoured(self) -> None:
        session = FakeGitHubSession()
        session.queue.append(FakeResponse(403, headers={"Retry-After": "5"}))
        store, _, clock = make(session)
        assert store.read("blobs/x") is None
        assert clock.slept == [5.0]

    def test_ratelimit_reset_is_honoured(self) -> None:
        session = FakeGitHubSession()
        clock = Clock()
        session.queue.append(
            FakeResponse(
                403,
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(clock.now) + 10),
                },
            )
        )
        store, _, _ = make(session, clock)
        assert store.read("blobs/x") is None
        assert clock.slept == [11.0]

    def test_wait_longer_than_the_budget_fails_without_sleeping(self) -> None:
        session = FakeGitHubSession()
        session.always = FakeResponse(429, headers={"Retry-After": "3600"})
        store, _, clock = make(session)
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.BACKEND_UNAVAILABLE
        )
        assert clock.slept == []

    def test_repeated_rate_limit_is_bounded_in_calls_and_total_wait(self) -> None:
        session = FakeGitHubSession()
        session.always = FakeResponse(429, headers={"Retry-After": "20"})
        store, _, clock = make(session)
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.BACKEND_UNAVAILABLE
        )
        assert len(session.calls) <= 4 and sum(clock.slept) <= 60

    def test_plain_403_is_auth_not_a_rate_limit(self) -> None:
        session = FakeGitHubSession()
        session.always = FakeResponse(403)
        store, _, clock = make(session)
        assert code_of(lambda: store.read("blobs/x")) is SnapshotErrorCode.AUTH
        assert clock.slept == [] and len(session.calls) == 1

    def test_401_is_auth(self) -> None:
        session = FakeGitHubSession()
        session.always = FakeResponse(401)
        assert code_of(lambda: make(session)[0].read("blobs/x")) is (
            SnapshotErrorCode.AUTH
        )


class TestRedirects:
    def test_cross_host_redirect_is_not_followed_and_token_not_sent(self) -> None:
        store, session, _ = make()
        store.read("blobs/warm")
        target = "https://evil.example.com/steal"
        source = f"https://api.github.com/repos/{OWNER_REPO}/contents/snaps/blobs/x"
        session.redirects[source] = target
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )
        assert all("evil.example.com" not in call["url"] for call in session.calls)

    def test_scheme_downgrade_redirect_refused(self) -> None:
        store, session, _ = make()
        store.read("blobs/warm")
        source = f"https://api.github.com/repos/{OWNER_REPO}/contents/snaps/blobs/x"
        session.redirects[source] = "http://api.github.com/steal"
        assert code_of(lambda: store.read("blobs/x")) is (
            SnapshotErrorCode.UNSAFE_DESTINATION
        )
        assert all(call["url"].startswith("https://") for call in session.calls)

    def test_same_host_redirect_is_followed(self) -> None:
        store, session, _ = make()
        store.read("blobs/warm")
        session.files[("main", "snaps/blobs/x")] = b"data"
        source = f"https://api.github.com/repos/{OWNER_REPO}/contents/snaps/blobs/x"
        session.redirects[source] = source + "?moved=1"
        assert store.read("blobs/x") == b"data"


class TestTokenHygiene:
    def test_token_absent_from_repr_errors_tracebacks_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        failures: list[FakeResponse | Exception] = [
            connection_error(f"boom {TOKEN} {CANARY}"),
            FakeResponse(500, f"{TOKEN} {CANARY}".encode()),
            FakeResponse(401, f"{TOKEN}".encode()),
            FakeResponse(403, TOKEN.encode()),
            FakeResponse(422, TOKEN.encode()),
        ]
        for failure in failures:
            session = FakeGitHubSession()
            session.always = failure
            store, _, _ = make(session)
            for operation in (
                lambda store=store: store.read("blobs/x"),
                lambda store=store: store.create_if_absent("blobs/x", b"d"),
            ):
                with caplog.at_level(logging.DEBUG):
                    with pytest.raises(SnapshotStoreError) as caught:
                        operation()
                error = caught.value
                text = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
                assert error.__cause__ is None and error.__suppress_context__ is True
                for haystack in (text, repr(error), str(vars(error)), repr(store)):
                    assert TOKEN not in haystack and CANARY not in haystack
                assert TOKEN not in str(vars(store))
        assert TOKEN not in caplog.text and CANARY not in caplog.text

    def test_token_is_not_stored_on_the_store(self) -> None:
        store, _, _ = make()
        store.read("blobs/x")
        assert TOKEN not in repr(vars(store))
