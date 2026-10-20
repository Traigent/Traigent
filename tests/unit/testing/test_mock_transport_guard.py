"""Transport-level guard: mock mode blocks known LLM provider hosts (#2418).

Every test uses an ``httpx.MockTransport`` (or a recording stub) underneath,
so no test touches the network. If the guard were absent, a request to a
provider host would reach the recording transport -- that is the fail-before
signal.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import httpx
import pytest

from traigent import testing as traigent_testing
from traigent.utils.mock_transport_guard import (
    MockModeTransportBlockedError,
    is_known_llm_provider_host,
)

URL = "https://api.openai.com/v1/chat/completions?secret_q=SENTINEL_QUERY"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


def _recording() -> tuple[list[httpx.Request], httpx.MockTransport]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return seen, httpx.MockTransport(handler)


def test_sync_httpx_provider_host_blocked_in_mock_mode() -> None:
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    with httpx.Client(transport=transport) as client:
        with pytest.raises(MockModeTransportBlockedError) as exc:
            client.post(
                URL,
                headers={"Authorization": "Bearer SENTINEL_KEY"},
                content=b"SENTINEL_BODY",
            )
    assert seen == []
    msg = str(exc.value)
    assert "api.openai.com" in msg
    for leaked in ("SENTINEL_QUERY", "SENTINEL_KEY", "SENTINEL_BODY", "/v1/chat"):
        assert leaked not in msg


def test_async_httpx_provider_host_blocked_in_mock_mode() -> None:
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(URL)

    with pytest.raises(MockModeTransportBlockedError):
        asyncio.run(run())
    assert seen == []


def test_non_provider_host_passes_through_in_mock_mode() -> None:
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    with httpx.Client(transport=transport) as client:
        assert client.get("http://127.0.0.1:9/health").status_code == 200
    assert len(seen) == 1


def test_provider_host_passes_through_when_mock_off() -> None:
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()  # installs the guard
    traigent_testing.disable_mock_mode()
    with httpx.Client(transport=transport) as client:
        assert client.post(URL).status_code == 200
    assert len(seen) == 1


def test_env_var_mock_mode_checked_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    traigent_testing.disable_mock_mode()
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    with httpx.Client(transport=transport) as client:
        with pytest.raises(MockModeTransportBlockedError):
            client.post(URL)
    assert seen == []


def test_install_is_idempotent() -> None:
    traigent_testing.enable_mock_mode_for_quickstart()
    send = httpx.Client.send
    traigent_testing.disable_mock_mode()
    traigent_testing.enable_mock_mode_for_quickstart()
    assert httpx.Client.send is send


def test_openai_client_blocked_in_mock_mode() -> None:
    openai = pytest.importorskip("openai")
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    client = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )
    with pytest.raises(Exception) as exc:
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
    assert seen == []
    # openai wraps transport errors in APIConnectionError; the guard error is the cause.
    chain = []
    cur: BaseException | None = exc.value
    while cur is not None:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    assert any(isinstance(e, MockModeTransportBlockedError) for e in chain)


def _anthropic_uses_httpx2() -> bool:
    import importlib.util

    return importlib.util.find_spec("httpx2") is not None and bool(
        getattr(importlib.import_module("anthropic._base_client"), "httpx2", None)
    )


def test_httpx2_client_blocked_in_mock_mode() -> None:
    httpx2 = pytest.importorskip("httpx2")
    traigent_testing.enable_mock_mode_for_quickstart()
    reached: list[object] = []
    transport = httpx2.MockTransport(
        lambda r: reached.append(r) or httpx2.Response(200)
    )
    with httpx2.Client(transport=transport) as client:
        with pytest.raises(MockModeTransportBlockedError):
            client.get("https://api.anthropic.com/v1/messages")
    assert reached == []


def test_anthropic_client_blocked_in_mock_mode() -> None:
    anthropic = pytest.importorskip("anthropic")
    # Newer anthropic releases require httpx2 clients; older ones require httpx.
    http_pkg = pytest.importorskip("httpx2") if _anthropic_uses_httpx2() else httpx
    seen: list[object] = []

    def handler(request: object) -> object:
        seen.append(request)
        return http_pkg.Response(200, json={"ok": True})

    transport = http_pkg.MockTransport(handler)
    traigent_testing.enable_mock_mode_for_quickstart()
    client = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=http_pkg.Client(transport=transport),
    )
    with pytest.raises(Exception) as exc:
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert seen == []
    chain = []
    cur: BaseException | None = exc.value
    while cur is not None:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    assert any(isinstance(e, MockModeTransportBlockedError) for e in chain)


def test_litellm_with_caller_logging_obj_blocked_in_mock_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    litellm = pytest.importorskip("litellm")
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    client = httpx.Client(transport=transport)
    # Caller-supplied logging object skips function_setup (the existing guard);
    # the transport guard is the remaining line of defence.
    import litellm.utils as lu

    original = getattr(lu.function_setup, "_traigent_original", lu.function_setup)
    monkeypatch.setattr(lu, "function_setup", original)  # simulate the skipped guard
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler = HTTPHandler(client=client)
    with pytest.raises(Exception) as exc:
        handler.post(URL, data="{}")
    cur: BaseException | None = exc.value
    found = False
    while cur is not None:
        found = found or isinstance(cur, MockModeTransportBlockedError)
        cur = cur.__cause__ or cur.__context__
    assert found
    assert seen == []
    assert litellm is not None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("api.openai.com", True),
        ("API.Anthropic.com", True),
        ("generativelanguage.googleapis.com", True),
        ("api.cohere.ai", True),
        ("api.cohere.com", True),
        ("openrouter.ai", True),
        ("bedrock-runtime.us-east-1.amazonaws.com", True),
        ("myres.openai.azure.com", True),
        ("evil-api.openai.com.example.org", False),
        ("notapi.openai.com", False),
        ("127.0.0.1", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_host_matching(host: str, expected: bool) -> None:
    assert is_known_llm_provider_host(host) is expected


def test_aiohttp_session_provider_host_blocked_in_mock_mode() -> None:
    aiohttp = pytest.importorskip("aiohttp")
    traigent_testing.enable_mock_mode_for_quickstart()

    async def run() -> None:
        # Blocked before any connection is attempted; no network involved.
        async with aiohttp.ClientSession() as session:
            await session.post("https://api.openai.com/v1/chat/completions")

    with pytest.raises(MockModeTransportBlockedError):
        asyncio.run(run())


def test_requests_session_provider_host_blocked_in_mock_mode() -> None:
    requests = pytest.importorskip("requests")
    traigent_testing.enable_mock_mode_for_quickstart()
    with pytest.raises(MockModeTransportBlockedError):
        requests.post("https://api.anthropic.com/v1/messages", timeout=1)


def test_install_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import traigent.utils.mock_transport_guard as guard

    def boom() -> None:
        raise OSError("patch failed")

    monkeypatch.setattr(guard, "install_mock_transport_guard", boom)
    with pytest.raises(RuntimeError):
        traigent_testing.enable_mock_mode_for_quickstart()
    assert traigent_testing.is_mock_mode_enabled() is False
