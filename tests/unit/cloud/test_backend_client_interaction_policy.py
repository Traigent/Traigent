"""Unit tests for BackendIntegratedClient.get_interaction_policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from traigent.cloud.backend_client import STATIC_POLICY_TEXT, BackendIntegratedClient
from traigent.testing import _reset_for_tests, enable_mock_mode_for_quickstart

FAKE_API_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        payload: object | None = None,
        json_exc: Exception | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._payload = payload
        self._json_exc = json_exc
        self._text = text

    async def json(self) -> object | None:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload

    async def text(self) -> str:
        return self._text


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    def __init__(
        self,
        *,
        response: _FakeResponse | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._exc = exc
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return _FakeRequestContext(self._response)


class _FakeClientSessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def _reset_interaction_policy_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    _reset_for_tests()
    yield
    _reset_for_tests()


def _make_client(api_key: str | None = None) -> BackendIntegratedClient:
    return BackendIntegratedClient(
        api_key=api_key,
        base_url="https://api.example.test",
        timeout=12.0,
    )


def _assert_static_policy(result: dict[str, object]) -> None:
    assert result == {
        "schema_version": "traigent.agent_interaction.response.v1",
        "profile": {
            "control": "guided",
            "expertise": "se",
            "pace": "balanced",
            "source": "default",
            "confidence": 0.0,
            "schema_version": "traigent.interaction_policy.v1",
        },
        "policy_text": STATIC_POLICY_TEXT,
        "question_budget": 2,
        "options_max": 3,
        "jargon_level": "plain",
        "next_skill_hint": None,
        "fallback_policy": "static_v1",
    }


@pytest.mark.asyncio
async def test_get_interaction_policy_returns_backend_payload(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    client = _make_client(api_key=FAKE_API_KEY)
    payload = {
        "schema_version": "traigent.agent_interaction.response.v1",
        "profile": {
            "control": "inspect",
            "expertise": "ds",
            "pace": "explore",
            "source": "backend",
            "confidence": 0.92,
            "schema_version": "traigent.interaction_policy.v1",
        },
        "policy_text": "backend policy text",
        "question_budget": 1,
        "options_max": 3,
        "jargon_level": "compact",
        "next_skill_hint": "spine:feature",
        "fallback_policy": "backend_v1",
    }
    response = _FakeResponse(
        status=200,
        payload=payload,
        headers={"ETag": '"policy-v1"'},
    )
    fake_session = _FakeSession(response=response)
    fake_aiohttp = SimpleNamespace(
        ClientSession=Mock(return_value=_FakeClientSessionContext(fake_session)),
        ClientTimeout=Mock(side_effect=lambda total=None: {"total": total}),
        ClientError=type("FakeClientError", (Exception,), {}),
    )

    with (
        patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.backend_client.aiohttp", fake_aiohttp),
    ):
        result = await client.get_interaction_policy(
            harness="codex",
            skill="python",
            signals={
                "pace": "execute",
                "control": "delegate",
                "private_note": "raw-secret-signal",
            },
        )

    assert result == payload
    assert len(fake_session.calls) == 1
    url, kwargs = fake_session.calls[0]
    assert url == "https://api.example.test/api/v1/auth/me/interaction-policy"
    assert kwargs["headers"]["X-API-Key"] == FAKE_API_KEY
    assert kwargs["headers"]["User-Agent"]
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["params"] == {
        "harness": "codex",
        "skill": "python",
        "signals": (
            '{"control":"delegate","pace":"execute","private_note":"raw-secret-signal"}'
        ),
    }
    assert kwargs["timeout"] == {"total": 12.0}
    cache_entry = client._interaction_policy_cache[("codex", "python")]
    assert set(cache_entry) == {"etag", "policy_text", "profile"}
    assert cache_entry["policy_text"] == "backend policy text"
    assert cache_entry["etag"] == '"policy-v1"'
    assert cache_entry["profile"] == payload["profile"]
    assert "raw-secret-signal" not in repr(cache_entry)
    assert "signals" not in repr(cache_entry)


@pytest.mark.asyncio
async def test_get_interaction_policy_offline_returns_static_without_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    client = _make_client(api_key=FAKE_API_KEY)

    with patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_session:
        result = await client.get_interaction_policy()

    _assert_static_policy(result)
    mock_session.assert_not_called()


@pytest.mark.asyncio
async def test_get_interaction_policy_without_api_key_returns_static_without_network() -> (
    None
):
    client = _make_client()

    with patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_session:
        result = await client.get_interaction_policy()

    _assert_static_policy(result)
    mock_session.assert_not_called()


@pytest.mark.asyncio
async def test_get_interaction_policy_mock_mode_returns_static_without_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    enable_mock_mode_for_quickstart()
    client = _make_client(api_key=FAKE_API_KEY)

    with patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_session:
        result = await client.get_interaction_policy()

    _assert_static_policy(result)
    mock_session.assert_not_called()


@pytest.mark.asyncio
async def test_get_interaction_policy_legacy_mock_llm_returns_static_without_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    client = _make_client(api_key=FAKE_API_KEY)

    with (
        patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.backend_client.aiohttp.ClientSession") as mock_session,
    ):
        result = await client.get_interaction_policy(
            signals={"private_note": "raw-secret-signal"}
        )

    _assert_static_policy(result)
    mock_session.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "transport_error"),
    [
        (_FakeResponse(status=503, text="backend unavailable"), None),
        (None, TimeoutError("timed out")),
        (_FakeResponse(status=200, json_exc=ValueError("not json")), None),
        (
            _FakeResponse(
                status=200,
                payload={
                    "schema_version": "traigent.agent_interaction.response.v1",
                    "profile": {
                        "control": "inspect",
                        "expertise": "ds",
                        "pace": "explore",
                        "source": "backend",
                        "confidence": 0.92,
                        "schema_version": "traigent.interaction_policy.v1",
                    },
                    "question_budget": 1,
                    "options_max": 3,
                    "jargon_level": "compact",
                    "next_skill_hint": "spine:feature",
                    "fallback_policy": "backend_v1",
                },
            ),
            None,
        ),
    ],
)
async def test_get_interaction_policy_backend_failures_return_static(
    monkeypatch,
    response: _FakeResponse | None,
    transport_error: Exception | None,
) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    client = _make_client(api_key=FAKE_API_KEY)
    fake_session = _FakeSession(response=response, exc=transport_error)
    fake_aiohttp = SimpleNamespace(
        ClientSession=Mock(return_value=_FakeClientSessionContext(fake_session)),
        ClientTimeout=Mock(side_effect=lambda total=None: {"total": total}),
        ClientError=type("FakeClientError", (Exception,), {}),
    )

    with (
        patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.backend_client.aiohttp", fake_aiohttp),
    ):
        result = await client.get_interaction_policy(harness="codex")

    _assert_static_policy(result)
    assert len(fake_session.calls) == 1


# ---------------------------------------------------------------------------
# Regression: unreachable/invalid backend URL must not raise from __init__,
# and get_interaction_policy() must return the static default without a
# network call.  (Bug: ValueError from validate_cloud_base_url was raised
# in __init__ before get_interaction_policy()'s try/except could catch it.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unreachable_url",
    [
        # Unresolvable hostname (DNS lookup raises socket.gaierror →
        # CloudUrlUnreachableError). This is the ONLY fallback-eligible case:
        # a backend that is simply not reachable, not an unsafe origin.
        "https://does-not-exist.traigent.invalid",
        "https://api.unreachable.traigent.invalid",
    ],
)
@pytest.mark.asyncio
async def test_unreachable_url_init_does_not_raise_and_returns_static_policy(
    monkeypatch,
    unreachable_url: str,
) -> None:
    """An UNREACHABLE backend must degrade to the static policy, not raise.

    get_interaction_policy() must return the static default (guided/se/balanced)
    when the backend host could not be resolved during __init__ — it must never
    propagate the error to the caller.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)

    # __init__ must succeed when the host is merely unreachable
    client = BackendIntegratedClient(api_key=FAKE_API_KEY, base_url=unreachable_url)

    assert client._url_invalid is True, (
        "Expected _url_invalid=True for an unreachable URL"
    )

    # get_interaction_policy() must return the static fallback without any
    # network call
    with patch(
        "traigent.cloud.backend_client.aiohttp.ClientSession"
    ) as mock_session:
        result = await client.get_interaction_policy()

    _assert_static_policy(result)
    mock_session.assert_not_called()


@pytest.mark.parametrize(
    "unsafe_url",
    [
        # Cloud metadata endpoint (SSRF target) — must NEVER be swallowed
        "https://169.254.169.254",
        # Loopback / private-network targets blocked in production
        "https://127.0.0.1:5000",
        "https://192.168.1.1",
        # Non-http scheme
        "ftp://bad-scheme.example.com",
        # URL with embedded credentials (explicitly forbidden)
        "https://user:pass@api.example.com",  # pragma: allowlist secret
    ],
)
@pytest.mark.asyncio
async def test_unsafe_url_still_fails_loud_and_is_not_swallowed(
    monkeypatch,
    unsafe_url: str,
) -> None:
    """UNSAFE origins must keep failing loud — the fallback must not relax SSRF.

    The interaction-policy fallback is scoped strictly to unreachable hosts; an
    unsafe origin (metadata/loopback/private IP, bad scheme, credentialed URL)
    must still raise from __init__ exactly as before the fallback was added.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)

    with pytest.raises(ValueError):
        BackendIntegratedClient(api_key=FAKE_API_KEY, base_url=unsafe_url)


@pytest.mark.asyncio
async def test_valid_url_does_not_set_url_invalid(monkeypatch) -> None:
    """A well-formed, routable URL must leave _url_invalid=False."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)

    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        base_url="https://api.example.test",
    )

    assert client._url_invalid is False, (
        "Expected _url_invalid=False for a valid URL"
    )


# ---------------------------------------------------------------------------
# Regression (fail-CLOSED counterpart): an unusable backend URL must NOT make
# real cloud operations silently target the inert placeholder. Every cloud op
# must fail closed with CloudEgressBlockedError and emit ZERO transport calls.
# (Policy reads fall back to static; everything else is denied.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_url_cloud_ops_fail_closed_with_zero_transport(
    monkeypatch,
) -> None:
    from traigent.cloud.client import CloudEgressBlockedError

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)

    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        base_url="https://does-not-exist.traigent.invalid",
    )
    assert client._url_invalid is True

    # Each component's fail-closed chokepoint must deny when the URL is invalid.
    for guard_owner in (
        client,
        client._api_ops,
        client._session_ops,
        client._trial_ops,
    ):
        with pytest.raises(CloudEgressBlockedError):
            guard_owner._raise_if_backend_egress_disabled("probe")

    # Representative high-level cloud ops must fail closed end-to-end with no
    # network session ever constructed.
    with patch(
        "traigent.cloud.backend_client.aiohttp.ClientSession"
    ) as mock_session:
        with pytest.raises(CloudEgressBlockedError):
            await client.create_hybrid_session(
                "guarded_problem",
                {"temperature": [0.1]},
                {"objectives": ["accuracy"], "max_trials": 1},
            )
        with pytest.raises(CloudEgressBlockedError):
            await client.request_trial_slot("sess-invalid-url")

    mock_session.assert_not_called()
