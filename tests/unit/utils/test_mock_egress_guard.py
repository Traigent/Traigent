"""Mock mode must never reach a model provider (issues #2418, #2515).

Transport-level: ``socket.getaddrinfo``/``connect`` are replaced with spies that
refuse everything, so no packet leaves the machine even on a failing build.
Keys are synthetic.
"""

import asyncio
import contextlib
import socket

import pytest

import traigent.testing as traigent_testing
from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture


@pytest.fixture
def egress_spy(monkeypatch):
    """Record every DNS lookup / connect attempt and refuse it."""
    attempts: list[str] = []

    def fake_getaddrinfo(host, *args, **kwargs):
        attempts.append(str(host))
        raise socket.gaierror("blocked by test")

    def fake_connect(self, addr):
        attempts.append(str(addr))
        raise OSError("blocked by test")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", fake_connect)
    return attempts


@pytest.fixture
def mock_mode(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    for key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-synthetic-not-real")
    yield
    traigent_testing.disable_mock_mode()


@contextlib.contextmanager
def _expect_blocked():
    """The call must fail; SDKs may wrap the guard error, so only require failure."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - any failure is acceptable here
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return
    pytest.fail("call succeeded instead of failing closed")


def _provider_hits(attempts: list[str]) -> list[str]:
    return [a for a in attempts if "openai" in a or "anthropic" in a]


def test_raw_openai_client_fails_closed_in_mock_mode(mock_mode, egress_spy):
    """#2418: a raw openai client call must not reach the provider."""
    import openai

    from traigent.utils.mock_egress_guard import MockEgressBlockedError

    patch_litellm_for_metadata_capture()
    client = openai.OpenAI(max_retries=0)
    with pytest.raises(Exception) as excinfo:
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
    assert _provider_hits(egress_spy) == []
    assert isinstance(excinfo.value, MockEgressBlockedError) or isinstance(
        excinfo.value.__cause__, MockEgressBlockedError
    )
    assert (
        "mock" in str(excinfo.value).lower()
        or "mock" in str(excinfo.value.__cause__).lower()
    )


def test_raw_anthropic_client_fails_closed_in_mock_mode(mock_mode, egress_spy):
    """#2418: same for the anthropic client."""
    import anthropic

    patch_litellm_for_metadata_capture()
    client = anthropic.Anthropic(max_retries=0)
    with _expect_blocked():
        client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert _provider_hits(egress_spy) == []


def test_raw_async_openai_client_fails_closed(mock_mode, egress_spy):
    import openai

    patch_litellm_for_metadata_capture()

    async def go():
        client = openai.AsyncOpenAI(max_retries=0)
        await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )

    with _expect_blocked():
        asyncio.run(go())
    assert _provider_hits(egress_spy) == []


def test_from_import_litellm_completion_makes_no_provider_request(
    mock_mode, egress_spy
):
    """#2515: ``from litellm import completion`` bound BEFORE the patch."""
    import litellm

    # ``litellm.main.completion`` is the function ``from litellm import completion``
    # binds when the import runs before Traigent patches the module attribute.
    original = litellm.main.completion
    patch_litellm_for_metadata_capture()
    with _expect_blocked():
        original(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=0,
        )
    assert _provider_hits(egress_spy) == []


def test_guard_inactive_when_mock_mode_off(monkeypatch, egress_spy):
    """With mock off the guard must not interfere: the request is attempted."""
    import httpx

    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    patch_litellm_for_metadata_capture()
    with pytest.raises(httpx.ConnectError):
        httpx.Client().get("https://api.openai.com/v1/models")
    assert any("api.openai.com" in a for a in egress_spy)


def test_guard_allows_local_and_non_provider_hosts_in_mock_mode(mock_mode, egress_spy):
    import httpx

    patch_litellm_for_metadata_capture()
    with pytest.raises(httpx.ConnectError):
        httpx.Client().get("http://127.0.0.1:9/x")
    with pytest.raises(httpx.ConnectError):
        httpx.Client().get("https://portal.traigent.ai/health")
