"""OIDC provider token cache honours the token's own expiry."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("jwt")

from traigent.security.auth import oidc as oidc_module  # noqa: E402
from traigent.security.auth.oidc import OIDCAuthProvider  # noqa: E402
from traigent.utils.exceptions import AuthenticationError  # noqa: E402

NOW = 1_800_000_000.0
TOKEN = "header.payload.signature"
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clk = _Clock(NOW)
    monkeypatch.setattr(oidc_module, "time", SimpleNamespace(time=clk.time))
    return clk


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> OIDCAuthProvider:
    monkeypatch.setattr(oidc_module, "PyJWKClient", MagicMock())
    return OIDCAuthProvider(
        {
            "client_id": "client",
            "issuer": "https://issuer.example.com",
            "jwks_uri": "https://issuer.example.com/jwks.json",
        }
    )


def _claims(exp: float) -> dict:
    return {"sub": "user-1", "iat": NOW, "exp": exp, "iss": "i", "roles": ["user"]}


@pytest.fixture
def decode(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_jwt = MagicMock()
    mock_jwt.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mock_jwt.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    monkeypatch.setattr(oidc_module, "jwt", mock_jwt)
    return mock_jwt.decode


class TestCacheHonoursTokenExpiry:
    def test_within_expiry_served_from_cache(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 100)
        assert provider.authenticate_oidc(TOKEN) is not None
        clock.now = NOW + 99
        assert provider.authenticate_oidc(TOKEN) is not None
        assert decode.call_count == 1

    @pytest.mark.parametrize("advance", [100, 101, 5000])
    def test_at_or_after_expiry_denied_from_cache(
        self, provider, clock, decode, advance
    ):
        decode.return_value = _claims(NOW + 100)
        assert provider.authenticate_oidc(TOKEN) is not None
        clock.now = NOW + advance
        with pytest.raises(AuthenticationError):
            provider.authenticate_oidc(TOKEN)
        assert decode.call_count == 1
        assert TOKEN_HASH not in provider._token_cache

    def test_cache_lifetime_capped_by_token_expiry(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 100)
        provider.authenticate_oidc(TOKEN)
        remaining = provider._token_cache[TOKEN_HASH]["expires_at"] - datetime.now(UTC)
        assert timedelta(seconds=98) < remaining <= timedelta(seconds=100)

    def test_cache_lifetime_capped_at_three_hundred_seconds(
        self, provider, clock, decode
    ):
        decode.return_value = _claims(NOW + 3000)
        provider.authenticate_oidc(TOKEN)
        remaining = provider._token_cache[TOKEN_HASH]["expires_at"] - datetime.now(UTC)
        assert timedelta(seconds=298) < remaining <= timedelta(seconds=300)

    def test_expiry_is_stored_with_claims(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 100)
        provider.authenticate_oidc(TOKEN)
        assert provider._token_cache[TOKEN_HASH]["exp"] == NOW + 100

    def test_lifetime_expired_entry_denied_explicitly(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 3000)
        provider.authenticate_oidc(TOKEN)
        provider._token_cache[TOKEN_HASH]["expires_at"] = datetime.now(UTC) - timedelta(
            seconds=1
        )
        with pytest.raises(AuthenticationError) as exc:
            provider.authenticate_oidc(TOKEN)
        assert str(exc.value) == (
            "OIDC authentication failed: cached token entry expired"
        )
        assert TOKEN_HASH not in provider._token_cache
        assert decode.call_count == 1

    def test_other_tokens_unaffected_by_stale_entry(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 100)
        provider.authenticate_oidc(TOKEN)
        clock.now = NOW + 200
        decode.return_value = _claims(NOW + 1000)
        assert provider.authenticate_oidc("other.token.value") is not None


class TestCacheUsesIntegerExpiryCutoff:
    def test_cached_hit_rejected_at_integer_cutoff(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 10.5)
        assert provider.authenticate_oidc(TOKEN) is not None
        clock.now = NOW + 10.25
        with pytest.raises(AuthenticationError):
            provider.authenticate_oidc(TOKEN)
        assert decode.call_count == 1

    def test_cached_hit_accepted_before_integer_cutoff(self, provider, clock, decode):
        decode.return_value = _claims(NOW + 10.5)
        assert provider.authenticate_oidc(TOKEN) is not None
        clock.now = NOW + 9.9
        assert provider.authenticate_oidc(TOKEN) is not None
        assert decode.call_count == 1
