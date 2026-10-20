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
        ("api.openai.com.", True),
        ("api\u3002openai\u3002com", True),
        ("evil.example\u3002com", False),
        ("127.0.0.1", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_host_matching(host: str, expected: bool) -> None:
    assert is_known_llm_provider_host(host) is expected


class _RecordingAiohttpConnector:
    """Factory for a TCPConnector that never opens a non-loopback socket.

    ``_create_connection`` is below ``BaseConnector.connect`` (where the guard
    sits). A loopback destination connects normally (local test server only);
    anything else is *recorded* and aborted with a sentinel, so a broken guard
    reaches the stub, never the internet.
    """

    class Reached(Exception):
        pass

    @staticmethod
    def make(aiohttp: object, reached: list[str]) -> object:
        class Connector(aiohttp.TCPConnector):  # type: ignore[name-defined, misc]
            async def _create_connection(self, req, traces, timeout):  # type: ignore[no-untyped-def]
                if req.url.host in ("127.0.0.1", "localhost"):
                    return await super()._create_connection(req, traces, timeout)
                reached.append(str(req.url.host))
                raise _RecordingAiohttpConnector.Reached

        return Connector(force_close=True)


@pytest.mark.parametrize("kind", ["str", "yarl", "relative", "unicode_sep"])
def test_aiohttp_provider_destination_blocked_in_mock_mode(kind: str) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    import yarl

    traigent_testing.enable_mock_mode_for_quickstart()
    reached: list[str] = []

    async def run() -> None:
        connector = _RecordingAiohttpConnector.make(aiohttp, reached)
        if kind == "relative":
            session = aiohttp.ClientSession(
                base_url="https://api.anthropic.com", connector=connector
            )
            target: object = "/v1/messages"
        else:
            session = aiohttp.ClientSession(connector=connector)
            target = {
                "str": "https://api.openai.com/v1/chat/completions",
                "yarl": yarl.URL("https://api.openai.com/v1/chat/completions"),
                # IDEOGRAPHIC FULL STOP separators; yarl normalizes to ASCII dots.
                "unicode_sep": "https://api\u3002openai\u3002com/v1/chat/completions",
            }[kind]
        async with session:
            await session.post(target)  # type: ignore[arg-type]

    with pytest.raises(MockModeTransportBlockedError):
        asyncio.run(run())
    assert reached == []


def test_aiohttp_redirect_hop_to_provider_blocked_in_mock_mode() -> None:
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    traigent_testing.enable_mock_mode_for_quickstart()
    reached: list[str] = []
    hits: list[str] = []

    async def redirect(request: web.Request) -> web.Response:
        hits.append(request.path)
        raise web.HTTPTemporaryRedirect("https://api.anthropic.com/v1/messages")

    async def run() -> None:
        app = web.Application()
        app.router.add_post("/go", redirect)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        try:
            connector = _RecordingAiohttpConnector.make(aiohttp, reached)
            async with aiohttp.ClientSession(connector=connector) as session:
                await session.post(f"http://127.0.0.1:{port}/go", allow_redirects=True)
        finally:
            await runner.cleanup()

    with pytest.raises(MockModeTransportBlockedError):
        asyncio.run(run())
    assert hits == ["/go"]  # first (loopback) hop was allowed
    assert reached == []  # provider hop never reached the connector


class _RecordingAdapter:
    def __new__(cls, seen: list[str], redirect: bool):  # type: ignore[no-untyped-def]
        import requests

        class Adapter(requests.adapters.BaseAdapter):
            def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
                seen.append(request.url)
                resp = requests.Response()
                resp.request = request
                resp.url = request.url
                if redirect and "127.0.0.1" in request.url:
                    resp.status_code = 307
                    resp.headers["Location"] = "https://api.anthropic.com/v1/messages"
                else:
                    resp.status_code = 200
                resp._content = b"{}"
                return resp

            def close(self) -> None:
                return None

        return Adapter()


def test_requests_session_provider_host_blocked_in_mock_mode() -> None:
    requests = pytest.importorskip("requests")
    traigent_testing.enable_mock_mode_for_quickstart()
    seen: list[str] = []
    with requests.Session() as session:
        session.mount("https://", _RecordingAdapter(seen, redirect=False))
        with pytest.raises(MockModeTransportBlockedError):
            session.post("https://api.anthropic.com/v1/messages", timeout=1)
    assert seen == []


def test_requests_redirect_hop_to_provider_blocked_in_mock_mode() -> None:
    requests = pytest.importorskip("requests")
    traigent_testing.enable_mock_mode_for_quickstart()
    seen: list[str] = []
    with requests.Session() as session:
        session.mount("http://", _RecordingAdapter(seen, redirect=True))
        session.mount("https://", _RecordingAdapter(seen, redirect=True))
        with pytest.raises(MockModeTransportBlockedError):
            session.post("http://127.0.0.1:9/go", allow_redirects=True, timeout=1)
    assert seen == ["http://127.0.0.1:9/go"]


def _redirect_handler(mod: object, seen: list[str]):  # type: ignore[no-untyped-def]
    def handler(request):  # type: ignore[no-untyped-def]
        seen.append(str(request.url))
        if request.url.host == "127.0.0.1":
            return mod.Response(  # type: ignore[attr-defined]
                307, headers={"Location": "https://api.anthropic.com/v1/messages"}
            )
        return mod.Response(200, json={"ok": True})  # type: ignore[attr-defined]

    return handler


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_sync_redirect_hop_to_provider_blocked_in_mock_mode(module_name: str) -> None:
    mod = pytest.importorskip(module_name)
    traigent_testing.enable_mock_mode_for_quickstart()
    seen: list[str] = []
    with mod.Client(
        transport=mod.MockTransport(_redirect_handler(mod, seen)),
        follow_redirects=True,
    ) as client:
        with pytest.raises(MockModeTransportBlockedError):
            client.post("http://127.0.0.1/go", content=b"x")
    assert seen == ["http://127.0.0.1/go"]


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_async_redirect_hop_to_provider_blocked_in_mock_mode(module_name: str) -> None:
    mod = pytest.importorskip(module_name)
    traigent_testing.enable_mock_mode_for_quickstart()
    seen: list[str] = []

    async def run() -> None:
        async with mod.AsyncClient(
            transport=mod.MockTransport(_redirect_handler(mod, seen)),
            follow_redirects=True,
        ) as client:
            await client.post("http://127.0.0.1/go", content=b"x")

    with pytest.raises(MockModeTransportBlockedError):
        asyncio.run(run())
    assert seen == ["http://127.0.0.1/go"]


def test_env_only_activation_installs_guard_at_import() -> None:
    """TRAIGENT_MOCK_LLM=true + ``import traigent`` and NO in-code enable call."""
    import os
    import subprocess
    import sys

    code = (
        "import httpx, traigent\n"
        "from traigent.utils.mock_transport_guard import "
        "MockModeTransportBlockedError\n"
        "seen = []\n"
        "t = httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200))\n"
        "try:\n"
        "    with httpx.Client(transport=t) as c:\n"
        "        c.post('https://api.openai.com/v1/chat/completions')\n"
        "except MockModeTransportBlockedError:\n"
        "    print('BLOCKED', len(seen))\n"
        "else:\n"
        "    print('NOT_BLOCKED', len(seen))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAIGENT_")}
    env.update(
        TRAIGENT_MOCK_LLM="true",
        ENVIRONMENT="development",
        TRAIGENT_SKIP_DOTENV="true",
        PYTHONPATH=os.getcwd(),
    )
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-W", "ignore", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert "BLOCKED 0" in proc.stdout, (proc.stdout[-500:], proc.stderr[-500:])


def test_env_only_install_failure_fails_closed_at_import() -> None:
    """If the guard cannot be installed, ``import traigent`` itself raises."""
    import os
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['httpx'] = None  # makes the guard's httpx import fail\n"
        "try:\n"
        "    import traigent\n"
        "except RuntimeError:\n"
        "    print('FAILED_CLOSED')\n"
        "else:\n"
        "    print('IMPORTED')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAIGENT_")}
    env.update(
        TRAIGENT_MOCK_LLM="true",
        ENVIRONMENT="development",
        TRAIGENT_SKIP_DOTENV="true",
        PYTHONPATH=os.getcwd(),
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "ignore", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert "FAILED_CLOSED" in proc.stdout, (proc.stdout[-300:], proc.stderr[-300:])


def test_flag_set_after_import_installs_guard_via_env_config() -> None:
    """Flag arriving after ``import traigent`` (e.g. from .env) installs the guard."""
    import os
    import subprocess
    import sys

    code = (
        "import os, httpx, traigent\n"
        "os.environ['TRAIGENT_MOCK_LLM'] = 'true'\n"
        "from traigent.utils import env_config\n"
        "env_config._check_mock_llm_prod_guard()\n"
        "from traigent.utils.mock_transport_guard import "
        "MockModeTransportBlockedError\n"
        "seen = []\n"
        "t = httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200))\n"
        "try:\n"
        "    with httpx.Client(transport=t) as c:\n"
        "        c.get('https://api.anthropic.com/v1/messages')\n"
        "except MockModeTransportBlockedError:\n"
        "    print('BLOCKED', len(seen))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAIGENT_")}
    env.update(
        ENVIRONMENT="development", TRAIGENT_SKIP_DOTENV="true", PYTHONPATH=os.getcwd()
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "ignore", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert "BLOCKED 0" in proc.stdout, (proc.stdout[-300:], proc.stderr[-300:])


def test_env_only_production_does_not_install_guard() -> None:
    """Production keeps its existing behaviour: mock refused, guard not installed."""
    import os
    import subprocess
    import sys

    code = (
        "import traigent, sys\n"
        "print('HTTPX_LOADED' if 'httpx' in sys.modules else 'UNTOUCHED')\n"
        "try:\n"
        "    import traigent.utils.env_config\n"
        "except OSError:\n"
        "    print('PROD_BLOCKED')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAIGENT_")}
    env.update(
        TRAIGENT_MOCK_LLM="true",
        ENVIRONMENT="production",
        TRAIGENT_SKIP_DOTENV="true",
        PYTHONPATH=os.getcwd(),
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "ignore", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert "UNTOUCHED" in proc.stdout, (proc.stdout[-300:], proc.stderr[-300:])
    assert "PROD_BLOCKED" in proc.stdout


def test_litellm_completion_with_caller_logging_obj_blocked_in_mock_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    litellm = pytest.importorskip("litellm")
    seen, transport = _recording()
    traigent_testing.enable_mock_mode_for_quickstart()
    monkeypatch.setattr(litellm, "client_session", httpx.Client(transport=transport))
    monkeypatch.setattr(
        litellm, "aclient_session", httpx.AsyncClient(transport=transport)
    )
    from litellm.litellm_core_utils.litellm_logging import Logging

    def make_logging() -> object:
        return Logging(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-fn",
        )

    def chain(exc: BaseException) -> bool:
        cur: BaseException | None = exc
        while cur is not None:
            if isinstance(cur, MockModeTransportBlockedError):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    # Call the unwrapped functions: Traigent's own mock wrapper would otherwise
    # answer with a canned response before any HTTP is attempted.
    completion = getattr(litellm.completion, "_traigent_original", litellm.completion)
    acompletion = getattr(
        litellm.acompletion, "_traigent_original", litellm.acompletion
    )
    with pytest.raises(Exception) as exc:
        completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            litellm_logging_obj=make_logging(),
            max_retries=0,
            num_retries=0,
        )
    assert chain(exc.value)
    assert seen == []

    async def run() -> None:
        await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            litellm_logging_obj=make_logging(),
            max_retries=0,
            num_retries=0,
        )

    with pytest.raises(Exception) as aexc:
        asyncio.run(run())
    assert chain(aexc.value)
    assert seen == []


def test_install_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import traigent.utils.mock_transport_guard as guard

    def boom() -> None:
        raise OSError("patch failed")

    monkeypatch.setattr(guard, "install_mock_transport_guard", boom)
    with pytest.raises(RuntimeError):
        traigent_testing.enable_mock_mode_for_quickstart()
    assert traigent_testing.is_mock_mode_enabled() is False
