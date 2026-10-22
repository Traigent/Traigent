"""Redirects are refused for the private self-hosted backend (TRAIG-76).

The project-management client and the gated HTTP funnels must never follow a
redirect from the credential-bearing origin admitted by the opt-in. Every test
uses a mocked handler or transport; no socket is opened.
"""

from __future__ import annotations

import email.message
import io
import socket
import urllib.request
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.response import addinfourl

import pytest

from traigent.labels import transport_gate as gate
from traigent.labels.transport_gate import TransportGateRefusal
from traigent.projects import ProjectManagementClient
from traigent.projects import client as project_client
from traigent.projects.config import ProjectManagementConfig
from traigent.utils.exceptions import ClientError

SELF_HOSTED = "https://traigent.internal.example:8443"
OPT_IN_ENV = {
    "TRAIGENT_BACKEND_URL": SELF_HOSTED,
    "TRAIGENT_ALLOW_PRIVATE_BACKEND": "1",
}
REDIRECT_TARGETS = [
    "http://169.254.169.254/latest/meta-data/",
    "https://10.99.0.5:8443/api/v1beta/projects/",
]


def _addr_info(ip: str) -> list[tuple]:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]


class _RecordingRedirectHandler(
    urllib.request.HTTPSHandler, urllib.request.HTTPHandler
):
    """Answers every http(s) request with a 302 to ``location`` and records URLs.

    Subclassing the default handlers makes ``build_opener`` drop them, so no
    real connection can be attempted.
    """

    def __init__(self, location: str) -> None:
        super().__init__()
        self.location = location
        self.seen: list[str] = []

    def _answer(self, req: urllib.request.Request):
        self.seen.append(req.full_url)
        headers = email.message.Message()
        headers["Location"] = self.location
        response: Any = addinfourl(io.BytesIO(b""), headers, req.full_url, code=302)
        response.msg = "Found"
        return response

    def http_open(self, req: urllib.request.Request):
        return self._answer(req)

    def https_open(self, req: urllib.request.Request):
        return self._answer(req)


@pytest.mark.parametrize("location", REDIRECT_TARGETS)
def test_project_client_refuses_redirect_from_private_backend(location: str) -> None:
    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            config = ProjectManagementConfig(backend_origin=SELF_HOSTED, api_key="k")
        handler = _RecordingRedirectHandler(location)
        with patch.object(
            project_client,
            "_build_opener",
            lambda: urllib.request.build_opener(
                project_client._RefuseRedirectHandler, handler
            ),
        ):
            with pytest.raises(ClientError, match="refused HTTP redirect"):
                ProjectManagementClient(config).get_project("project_alpha")
    # Exactly one request left the process: the redirect target was never requested.
    assert len(handler.seen) == 1
    assert handler.seen[0].startswith(SELF_HOSTED)


@pytest.mark.parametrize("location", REDIRECT_TARGETS)
def test_gated_httpx_refuses_redirect_from_private_backend(location: str) -> None:
    httpx = pytest.importorskip("httpx")
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": location})

    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        client = gate.gate_session(
            httpx.Client(transport=httpx.MockTransport(handler)),
            origins=[SELF_HOSTED],
        )
        with pytest.raises(TransportGateRefusal):
            client.get(f"{SELF_HOSTED}/api/v1beta/projects/")
    assert len(seen) == 1


def test_gated_httpx_leaves_redirects_alone_outside_opt_in() -> None:
    # Outside the opt-in origin the gate leaves redirect behaviour unchanged.
    httpx = pytest.importorskip("httpx")
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.append(str(request.url))
        return httpx.Response(200, json={})

    with patch.dict("os.environ", {}, clear=True):
        client = gate.gate_session(
            httpx.Client(transport=httpx.MockTransport(handler)),
            origins=["https://api.example.com"],
        )
        response = client.get("https://api.example.com/api/v1beta/projects/")
    assert response.status_code == 200
    assert len(seen) == 1


def test_workflow_traces_ingest_does_not_follow_redirect_under_opt_in() -> None:
    from unittest.mock import MagicMock

    from traigent.integrations.observability import workflow_traces

    response = MagicMock(status_code=302)
    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            client = workflow_traces.WorkflowTracesClient(SELF_HOSTED)
        with patch.object(
            workflow_traces.requests, "post", return_value=response
        ) as post:
            graph = MagicMock(to_dict=MagicMock(return_value={"nodes": []}))
            result = client.ingest_traces(graph=graph)
    assert post.call_args.kwargs["allow_redirects"] is False
    assert result.success is False
    assert "Redirect refused" in (result.error or "")


# --- ad-hoc credentialed backend calls (sol r2) ----------------------------------

_METADATA_LOCATION = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture
def recorded_redirect(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every requests send at the adapter: each 3xx answer names the metadata service.

    Patching ``HTTPAdapter.send`` keeps requests' own redirect loop, so a request that
    is allowed to follow a redirect shows up here a second time.
    """
    import requests

    seen: list[str] = []

    def fake_send(self: Any, request: Any, **kwargs: Any) -> requests.Response:
        seen.append(request.url)
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = _METADATA_LOCATION
        response._content = b""
        response.url = request.url
        response.request = request
        response.encoding = "utf-8"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    return seen


def _backend_client() -> Any:
    from traigent.cloud.backend_client import BackendIntegratedClient

    client = BackendIntegratedClient(api_key="k", base_url=SELF_HOSTED)
    # No key validation over the network: the auth header is stubbed.
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"Authorization": "Bearer test-token"}
    )
    return client


@pytest.mark.parametrize("operation", ["publish", "fetch"])
def test_best_config_call_refuses_redirect_to_metadata(
    operation: str, recorded_redirect: list[str]
) -> None:
    from traigent.cloud.backend_client import CloudServiceError

    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            client = _backend_client()
            with pytest.raises(CloudServiceError, match="refused HTTP redirect"):
                if operation == "publish":
                    client.publish_best_config_sync({"spec": "x"}, environment="dev")
                else:
                    client.fetch_best_config_sync("cfg-1", environment="dev")
    assert len(recorded_redirect) == 1
    assert recorded_redirect[0].startswith(SELF_HOSTED)


@pytest.mark.parametrize("operation", ["publish", "fetch"])
def test_best_config_call_refused_when_transport_gate_is_off(
    operation: str, recorded_redirect: list[str]
) -> None:
    env = {**OPT_IN_ENV, "TRAIGENT_TRANSPORT_GATE": "off"}
    with patch.dict("os.environ", env, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            with pytest.raises(Exception) as excinfo:
                client = _backend_client()
                if operation == "publish":
                    client.publish_best_config_sync({"spec": "x"}, environment="dev")
                else:
                    client.fetch_best_config_sync("cfg-1", environment="dev")
    assert "TRAIGENT_ALLOW_PRIVATE_BACKEND requires the transport gate" in str(
        excinfo.value
    )
    assert recorded_redirect == []


@pytest.fixture
def answer_status(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Factory: every requests send at the adapter answers ``status`` (optional Location)."""
    import requests

    def install(status: int, location: str | None = None) -> list[str]:
        seen: list[str] = []

        def fake_send(self: Any, request: Any, **kwargs: Any) -> requests.Response:
            seen.append(request.url)
            response = requests.Response()
            response.status_code = status
            if location is not None:
                response.headers["Location"] = location
            response._content = b""
            response.url = request.url
            response.request = request
            response.encoding = "utf-8"
            return response

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
        return seen

    return install


def test_fetch_best_config_304_not_modified_returns_none(answer_status: Any) -> None:
    # Conditional fetch: the backend answers 304 Not Modified. That is the documented
    # None answer, not a refused redirect.
    seen = answer_status(304)
    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            client = _backend_client()
            result = client.fetch_best_config_sync(
                "cfg-1", environment="dev", etag='"v1"'
            )
    assert result is None
    assert len(seen) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_fetch_best_config_real_redirect_with_etag_still_refused(
    status: int, answer_status: Any
) -> None:
    from traigent.cloud.backend_client import CloudServiceError

    seen = answer_status(status, location=_METADATA_LOCATION)
    with patch.dict("os.environ", OPT_IN_ENV, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.20.30.40")):
            client = _backend_client()
            with pytest.raises(CloudServiceError, match="refused HTTP redirect"):
                client.fetch_best_config_sync("cfg-1", environment="dev", etag='"v1"')
    assert len(seen) == 1
    assert seen[0].startswith(SELF_HOSTED)
