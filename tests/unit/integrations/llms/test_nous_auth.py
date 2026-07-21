"""Offline unit tests for the Nous Portal (Hermes) JWT-refresh auth helper.

Everything here is offline/mock: the single network seam
(:func:`traigent.integrations.llms.nous_auth._request_new_token`) and the lower
``requests.post`` layer are monkeypatched, credential files live under
``tmp_path``, and the wall clock (``nous_auth._now``) is injected. No real
network, no real credentials, no spend.

Coverage map (mirrors the #1978 test plan):

* Credential resolution order — ``NOUS_API_KEY`` short-circuits with ZERO HTTP;
  ``NOUS_REFRESH_TOKEN`` / ``NOUS_PORTAL_REFRESH_TOKEN``; ``~/.hermes/auth.json``.
* Missing / malformed credentials -> :class:`NousAuthError` naming the sources.
* Mocked token-endpoint mint; within-TTL reuse does zero HTTP; a fake-clock
  advance re-mints (the offline proxy for "survives mid-run JWT expiry").
* ``expires_in`` absent -> the JWT ``exp`` claim is read; no expiry anywhere ->
  raise. 4xx/5xx and network errors -> :class:`NousAuthError`.
* Expired cache + failed refresh never serves the stale token.
* ``force_refresh=True`` bypasses a still-valid cache; ``clear_nous_auth_cache``.
* Thread-safety smoke: one mint under contention.
"""

from __future__ import annotations

import base64
import json
import threading

import pytest

from traigent.integrations.llms import nous_auth
from traigent.integrations.llms.nous_auth import (
    NOUS_BASE_URL,
    NousAuthError,
    clear_nous_auth_cache,
    get_nous_api_key,
    has_nous_credentials,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _make_jwt(claims: dict) -> str:
    """Build an unsigned 3-segment JWT whose payload carries ``claims``."""

    def _seg(obj: dict) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    return f"{_seg({'alg': 'none'})}.{_seg(claims)}.sig"


class _Clock:
    """A tiny injectable clock so expiry logic is deterministic offline."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _FakePost:
    """Records the last ``requests.post`` call and returns a canned response."""

    def __init__(self, *, status_code: int = 200, body: object | None = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.calls: list[dict] = []

    def __call__(self, url, *, json=None, headers=None, timeout=None):  # noqa: A002
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return _FakeResponse(self.status_code, self._body)


class _FakeResponse:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        return self._body


@pytest.fixture(autouse=True)
def _isolate_nous_env(monkeypatch, tmp_path):
    """Strip every Nous credential from the env, pin the auth file to an absent
    path (so a real ~/.hermes/auth.json on the dev box can never leak in), and
    clear the module JWT cache around each test."""
    for var in (
        "NOUS_API_KEY",
        "NOUS_REFRESH_TOKEN",
        "NOUS_PORTAL_REFRESH_TOKEN",
        "TRAIGENT_NOUS_TOKEN_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(tmp_path / "absent-auth.json"))
    clear_nous_auth_cache()
    yield
    clear_nous_auth_cache()


# --------------------------------------------------------------------------- #
# Base URL sanity                                                             #
# --------------------------------------------------------------------------- #
def test_base_url_is_the_nous_inference_v1_surface():
    # The single source of the base_url shared by discovery + the example.
    assert NOUS_BASE_URL == nous_auth._DEFAULT_NOUS_BASE_URL
    assert NOUS_BASE_URL.endswith("/v1")
    assert "nousresearch.com" in NOUS_BASE_URL


# --------------------------------------------------------------------------- #
# 1. Resolution order                                                         #
# --------------------------------------------------------------------------- #
def test_static_key_short_circuits_with_zero_http(monkeypatch):
    monkeypatch.setenv("NOUS_API_KEY", "pre-minted-jwt")  # pragma: allowlist secret

    def _boom(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("NOUS_API_KEY path must not perform any token mint")

    monkeypatch.setattr(nous_auth, "_request_new_token", _boom)

    assert get_nous_api_key() == "pre-minted-jwt"
    assert has_nous_credentials() is True


def test_static_key_returned_as_is_even_with_force_refresh(monkeypatch):
    # The escape hatch is never refreshed — returned verbatim regardless.
    monkeypatch.setenv("NOUS_API_KEY", "static-token")  # pragma: allowlist secret
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not mint")),
    )
    assert get_nous_api_key(force_refresh=True) == "static-token"


def test_refresh_token_env_mints_and_caches(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "refresh-abc")  # pragma: allowlist secret

    seen: list[str] = []

    def _mint(refresh_token, *, token_url, **_k):
        seen.append(refresh_token)
        return {"access_token": "minted-jwt", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    assert get_nous_api_key() == "minted-jwt"
    assert seen == ["refresh-abc"]  # the refresh token flowed to the mint call
    # Cached with an absolute expiry = now + expires_in.
    assert nous_auth._token_state is not None
    assert nous_auth._token_state.expires_at == pytest.approx(1000.0 + 3600)


def test_refresh_token_env_order_prefers_primary(monkeypatch):
    # NOUS_REFRESH_TOKEN wins over the NOUS_PORTAL_REFRESH_TOKEN alias.
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "primary")  # pragma: allowlist secret
    monkeypatch.setenv(
        "NOUS_PORTAL_REFRESH_TOKEN", "secondary"
    )  # pragma: allowlist secret
    seen: list[str] = []
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: (
            seen.append(rt) or {"access_token": "j", "expires_in": 60}
        ),
    )
    get_nous_api_key()
    assert seen == ["primary"]


def test_portal_refresh_token_alias_is_accepted(monkeypatch):
    monkeypatch.setenv(
        "NOUS_PORTAL_REFRESH_TOKEN", "portal-refresh"
    )  # pragma: allowlist secret
    seen: list[str] = []
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: (
            seen.append(rt) or {"access_token": "aliased-jwt", "expires_in": 120}
        ),
    )
    assert get_nous_api_key() == "aliased-jwt"
    assert seen == ["portal-refresh"]
    assert has_nous_credentials() is True


@pytest.mark.parametrize("key", ["refresh_token", "refreshToken"])
def test_auth_file_refresh_token_is_parsed(monkeypatch, tmp_path, key):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({key: "file-refresh"}), encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth))

    seen: list[str] = []
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: (
            seen.append(rt) or {"access_token": "from-file", "expires_in": 300}
        ),
    )

    assert has_nous_credentials() is True
    assert get_nous_api_key() == "from-file"
    assert seen == ["file-refresh"]


# --------------------------------------------------------------------------- #
# 2. Missing / malformed credentials -> NousAuthError naming the sources       #
# --------------------------------------------------------------------------- #
def test_missing_credentials_raises_naming_all_sources(monkeypatch, tmp_path):
    absent = tmp_path / "nope.json"
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(absent))

    assert has_nous_credentials() is False
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    msg = str(exc.value)
    assert "NOUS_API_KEY" in msg
    assert "NOUS_REFRESH_TOKEN" in msg
    assert "NOUS_PORTAL_REFRESH_TOKEN" in msg
    assert str(absent) in msg


def test_malformed_auth_file_not_json_raises(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text("this is not json {", encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth))
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert str(auth) in str(exc.value)
    assert "not valid JSON" in str(exc.value)


def test_auth_file_not_object_raises(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth))
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert str(auth) in str(exc.value)
    # Names the expected refresh-token keys.
    assert "refresh_token" in str(exc.value)


def test_auth_file_missing_refresh_key_raises(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"unrelated": "value"}), encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth))
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "no refresh token" in str(exc.value)
    assert str(auth) in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. Mint / cache / expiry (fake clock, zero HTTP)                             #
# --------------------------------------------------------------------------- #
def test_within_ttl_reuse_does_zero_http(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    mints = {"n": 0}

    def _mint(rt, *, token_url, **_k):
        mints["n"] += 1
        return {"access_token": f"jwt-{mints['n']}", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    first = get_nous_api_key()
    clock.t = 1200.0  # 3400s of life left, well above the 300s skew
    second = get_nous_api_key()

    assert first == second == "jwt-1"
    assert mints["n"] == 1  # cached — no second mint, no HTTP


def test_fake_clock_advance_remints_before_expiry(monkeypatch):
    # Offline proxy for "a long run survives mid-run JWT expiry": once the cached
    # token drops below the min_ttl skew, the next call re-mints transparently.
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    mints = {"n": 0}

    def _mint(rt, *, token_url, **_k):
        mints["n"] += 1
        return {"access_token": f"jwt-{mints['n']}", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    assert get_nous_api_key() == "jwt-1"  # expires_at = 4600
    clock.t = 4500.0  # only 100s left, below the 300s skew -> re-mint
    assert get_nous_api_key() == "jwt-2"
    assert mints["n"] == 2


def test_expires_in_absent_reads_jwt_exp(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    jwt = _make_jwt({"exp": 5000})
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: {"access_token": jwt},  # no expires_in
    )

    assert get_nous_api_key() == jwt
    # Absolute expiry came from the JWT exp claim, not now+expires_in.
    assert nous_auth._token_state is not None
    assert nous_auth._token_state.expires_at == 5000.0


def test_no_expiry_anywhere_raises(monkeypatch):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: {"access_token": "opaque-not-a-jwt"},
    )
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "expires_in" in str(exc.value)


def test_missing_access_token_in_response_raises(monkeypatch):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: {"expires_in": 60},  # no token
    )
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "no access token" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3a. Never mint an already-expired token / honor the SHORTER expiry (#1978 #1) #
# --------------------------------------------------------------------------- #
def test_minted_token_already_expired_by_jwt_exp_raises(monkeypatch):
    # A mint whose only expiry signal (the JWT exp) is already in the past must
    # fail loud — never cache/serve an already-expired token.
    clock = _Clock(10_000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    past_jwt = _make_jwt({"exp": 9_000})  # 1000s in the PAST vs the clock
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: {"access_token": past_jwt},  # no expires_in
    )

    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "expired" in str(exc.value).lower()
    # The stale token never reached the cache.
    assert nous_auth._token_state is None


def test_expires_in_overstating_shorter_jwt_exp_uses_shorter_exp(monkeypatch):
    # Both signals present: expires_in claims a long life the JWT's own exp does
    # not back. The SHORTER deadline (the JWT exp) must win so the cached expiry
    # can't be pushed past the token's real deadline.
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    jwt = _make_jwt({"exp": 1500})  # real deadline: 500s of life
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        # expires_in would compute now+3600 = 4600; the JWT exp caps it at 1500.
        lambda rt, *, token_url, **_k: {"access_token": jwt, "expires_in": 3600},
    )

    assert get_nous_api_key() == jwt
    assert nous_auth._token_state is not None
    assert nous_auth._token_state.expires_at == 1500.0  # min(4600, 1500)


def test_min_expiry_wins_when_expires_in_is_the_shorter_signal(monkeypatch):
    # Symmetric to the above: when expires_in is the shorter of the two, it wins.
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    jwt = _make_jwt({"exp": 9000})  # 8000s of life per the JWT
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        # expires_in -> now+100 = 1100, shorter than the JWT exp of 9000.
        lambda rt, *, token_url, **_k: {"access_token": jwt, "expires_in": 100},
    )

    assert get_nous_api_key() == jwt
    assert nous_auth._token_state is not None
    assert nous_auth._token_state.expires_at == 1100.0  # min(1100, 9000)


# --------------------------------------------------------------------------- #
# 3b. Robust JWT decode — malformed payloads fail clean, not raw (#1978 #3)     #
# --------------------------------------------------------------------------- #
def test_malformed_jwt_payload_returns_none_not_raw_error():
    # A 3-segment token whose middle segment cannot base64-decode must return
    # None (clean "no exp"), never escape as a raw binascii.Error / ValueError.
    assert nous_auth._decode_jwt_exp("header.x.signature") is None
    # Decodes fine but the bytes are not JSON.
    seg = base64.urlsafe_b64encode(b"not-json-at-all").rstrip(b"=").decode("ascii")
    assert nous_auth._decode_jwt_exp(f"header.{seg}.signature") is None
    # Decodes to JSON that is not an object (no dict -> no exp).
    arr = base64.urlsafe_b64encode(b"[1, 2, 3]").rstrip(b"=").decode("ascii")
    assert nous_auth._decode_jwt_exp(f"header.{arr}.signature") is None


def test_minted_malformed_jwt_no_expiry_raises_clean_error(monkeypatch):
    # End-to-end: a mint returning a malformed JWT with no expires_in must fail
    # loud as a clean NousAuthError (unknown expiry), not a raw decode exception.
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setattr(
        nous_auth,
        "_request_new_token",
        lambda rt, *, token_url, **_k: {"access_token": "header.x.signature"},
    )
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "expires_in" in str(exc.value)  # the clean unknown-expiry message
    assert nous_auth._token_state is None


# --------------------------------------------------------------------------- #
# 4. HTTP status / network errors -> NousAuthError (exercises _request_new_token)#
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_token_endpoint_non_200_raises(monkeypatch, status):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setenv("TRAIGENT_NOUS_TOKEN_URL", "https://token.example/refresh")
    fake_post = _FakePost(status_code=status, body={"error": "nope"})
    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert str(status) in str(exc.value)
    assert "https://token.example/refresh" in str(exc.value)
    assert fake_post.calls, "the token endpoint should have been called"


def test_non_object_json_body_raises(monkeypatch):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setattr(
        "requests.post", _FakePost(status_code=200, body=["not", "obj"])
    )
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "non-object JSON" in str(exc.value)


def test_network_exception_is_wrapped(monkeypatch):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt-net")  # pragma: allowlist secret

    def _raise(*_a, **_k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("requests.post", _raise)

    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    msg = str(exc.value)
    assert "failed" in msg
    # Names the credential source but never echoes the token value itself.
    assert "NOUS_REFRESH_TOKEN" in msg
    assert "rt-net" not in msg


def test_token_value_never_appears_in_error(monkeypatch):
    # The response body may carry a token; it must not leak into the message.
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret
    monkeypatch.setattr(
        "requests.post",
        _FakePost(status_code=500, body={"access_token": "SECRET-LEAK"}),
    )
    with pytest.raises(NousAuthError) as exc:
        get_nous_api_key()
    assert "SECRET-LEAK" not in str(exc.value)


# --------------------------------------------------------------------------- #
# 5. Never serve stale; force_refresh; clear cache                             #
# --------------------------------------------------------------------------- #
def test_expired_cache_with_failed_refresh_never_serves_stale(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    state = {"fail": False}

    def _mint(rt, *, token_url, **_k):
        if state["fail"]:
            raise NousAuthError("refresh endpoint down")
        return {"access_token": "good-jwt", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    assert get_nous_api_key() == "good-jwt"  # cached, expires_at = 4600
    # Advance past the skew AND make refresh fail: must raise, never return stale.
    clock.t = 4500.0
    state["fail"] = True
    with pytest.raises(NousAuthError):
        get_nous_api_key()


def test_force_refresh_bypasses_valid_cache(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(nous_auth, "_now", clock)
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    mints = {"n": 0}

    def _mint(rt, *, token_url, **_k):
        mints["n"] += 1
        return {"access_token": f"jwt-{mints['n']}", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    assert get_nous_api_key() == "jwt-1"
    clock.t = 1100.0  # cache still valid
    assert get_nous_api_key(force_refresh=True) == "jwt-2"
    assert mints["n"] == 2


def test_clear_cache_forces_remint(monkeypatch):
    monkeypatch.setattr(nous_auth, "_now", _Clock(1000.0))
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    mints = {"n": 0}

    def _mint(rt, *, token_url, **_k):
        mints["n"] += 1
        return {"access_token": f"jwt-{mints['n']}", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    get_nous_api_key()
    clear_nous_auth_cache()
    get_nous_api_key()
    assert mints["n"] == 2


# --------------------------------------------------------------------------- #
# 6. has_nous_credentials variants (cheap, no network)                         #
# --------------------------------------------------------------------------- #
def test_has_nous_credentials_false_when_nothing_present():
    assert has_nous_credentials() is False


@pytest.mark.parametrize(
    "var", ["NOUS_API_KEY", "NOUS_REFRESH_TOKEN", "NOUS_PORTAL_REFRESH_TOKEN"]
)
def test_has_nous_credentials_true_for_each_env_source(monkeypatch, var):
    monkeypatch.setenv(var, "x")  # pragma: allowlist secret
    # Prove it does zero network even when a token endpoint would blow up.
    monkeypatch.setattr(
        "requests.post",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no network")),
    )
    assert has_nous_credentials() is True


def test_has_nous_credentials_true_when_auth_file_exists(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"refresh_token": "x"}), encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth))
    assert has_nous_credentials() is True


# --------------------------------------------------------------------------- #
# 7. Thread-safety smoke: one mint under contention                            #
# --------------------------------------------------------------------------- #
def test_concurrent_callers_mint_exactly_once(monkeypatch):
    monkeypatch.setattr(nous_auth, "_now", _Clock(1000.0))
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "rt")  # pragma: allowlist secret

    mints = {"n": 0}
    count_lock = threading.Lock()

    def _mint(rt, *, token_url, **_k):
        with count_lock:
            mints["n"] += 1
        # Widen the race window; the module lock must still serialize callers.
        threading.Event().wait(0.02)
        return {"access_token": "shared-jwt", "expires_in": 3600}

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)

    n_threads = 12
    barrier = threading.Barrier(n_threads)
    results: list[str] = []
    results_lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        token = get_nous_api_key()
        with results_lock:
            results.append(token)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mints["n"] == 1  # the lock collapsed the stampede to a single mint
    assert results == ["shared-jwt"] * n_threads
