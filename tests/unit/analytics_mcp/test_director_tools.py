"""Unit tests for the Director v0 (advisory-only) MCP tool wrappers.

Director v0 is governed by the FROZEN contract at
``~/.claude/plans/director-v0-contract/`` (rev 2). These tools
(``director_start_tool`` / ``director_turn_tool`` / ``director_state_tool``)
are thin pass-throughs to ``BackendAnalyticsClient.director_start`` /
``director_turn`` / ``director_state`` -- copying the same three-layer
pattern, mocking, and failure-normalization conventions as
``test_analytics_mcp_tools.py`` (e.g. ``TestDecisionBriefTool``,
``TestBackendErrorTaxonomy``). The backend client is mocked at the
tools-module boundary; no live backend is contacted.

C1 (frozen contract): no tool here exposes a free-text parameter a client
agent composes. ``director_turn_tool`` accepts only session identity, the
closed ``intent`` enum, and the typed ``report`` block.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest


def _install_fake_client(
    monkeypatch,
    reader: AsyncMock,
    *,
    backend_url: str = "https://backend.example.com",
):
    """Patch ``_new_analytics_client`` to yield ``reader`` as an async ctx mgr.

    Mirrors ``test_analytics_mcp_tools.py``'s helper of the same name: the
    tools open the client with ``async with await _new_analytics_client() as
    reader``, so this exercises each tool's real call path.
    """
    from traigent.analytics_mcp import tools as tools_mod

    class _FakeClient:
        def __init__(self) -> None:
            self.backend_url = backend_url

        async def __aenter__(self):
            return reader

        async def __aexit__(self, *exc):
            return False

    async def _fake_new_analytics_client():
        return _FakeClient()

    monkeypatch.setattr(tools_mod, "_new_analytics_client", _fake_new_analytics_client)
    return reader


def _http_status_error(status_code: int, payload: dict[str, object]):
    import httpx

    request = httpx.Request(
        "POST",
        "https://secret-host.invalid/internal/director?api_key=raw-secret",
    )
    response = httpx.Response(status_code, json=payload, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code} raw-response-body https://secret-host.invalid",
        request=request,
        response=response,
    )


class TestDirectorStartTool:
    @pytest.mark.asyncio
    async def test_calls_client_and_wraps_payload(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_start_tool

        reader = AsyncMock()
        reader.director_start.return_value = {"session_id": "ds_" + "a" * 32}
        _install_fake_client(monkeypatch, reader)

        result = await director_start_tool("proj_abc", run_ids=["run_1"])

        assert result["ok"] is True
        assert result["director_session"] == {"session_id": "ds_" + "a" * 32}
        reader.director_start.assert_awaited_once_with(
            "proj_abc",
            workflow_kind="optimization_run_advisory",
            run_ids=["run_1"],
        )

    @pytest.mark.asyncio
    async def test_missing_project_id_rejected_without_call(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_start_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_start_tool("")

        assert result["ok"] is False
        assert "project_id" in result["message"]
        reader.director_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_unsupported_workflow_kind_without_call(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp.tools import director_start_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_start_tool("proj_abc", workflow_kind="bogus")

        assert result["ok"] is False
        reader.director_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_failure_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_start_tool

        reader = AsyncMock()
        reader.director_start.side_effect = RuntimeError("https://secret-host leaked")
        _install_fake_client(monkeypatch, reader)

        result = await director_start_tool("proj_abc")

        assert result["ok"] is False
        assert result["code"] == "backend_unavailable"
        assert "secret-host" not in result["message"]


class TestDirectorTurnTool:
    @pytest.mark.asyncio
    async def test_calls_client_with_intent_and_report(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        reader.director_turn.return_value = {"turn_id": "dt_" + "b" * 32}
        _install_fake_client(monkeypatch, reader)

        session_id = "ds_" + "a" * 32
        report = {"instruction_id": "di_" + "c" * 32, "status": "done"}
        result = await director_turn_tool(
            session_id, 3, intent="report_progress", report=report
        )

        assert result["ok"] is True
        assert result["director_turn"] == {"turn_id": "dt_" + "b" * 32}
        reader.director_turn.assert_awaited_once_with(
            session_id, 3, intent="report_progress", report=report
        )

    @pytest.mark.asyncio
    async def test_defaults_intent_to_ask_next_step(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        reader.director_turn.return_value = {"turn_id": "dt_" + "b" * 32}
        _install_fake_client(monkeypatch, reader)

        await director_turn_tool("ds_" + "a" * 32, 1)

        kwargs = reader.director_turn.await_args.kwargs
        assert kwargs["intent"] == "ask_next_step"
        assert kwargs["report"] is None

    @pytest.mark.asyncio
    async def test_rejects_unsupported_intent_without_call(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool("ds_" + "a" * 32, 1, intent="bogus")

        assert result["ok"] is False
        reader.director_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_malformed_report_without_call(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool(
            "ds_" + "a" * 32,
            1,
            report={"instruction_id": "not-a-valid-id", "status": "done"},
        )

        assert result["ok"] is False
        reader.director_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_unsupported_report_status_without_call(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool(
            "ds_" + "a" * 32,
            1,
            report={"instruction_id": "di_" + "c" * 32, "status": "bogus"},
        )

        assert result["ok"] is False
        reader.director_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_revision_is_structured_recoverable_result(
        self, monkeypatch
    ) -> None:
        """A 409 stale_revision must never surface as a generic failure --
        the caller needs the current revision so it can re-read state."""
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        reader.director_turn.side_effect = _http_status_error(
            409,
            {
                "error": {
                    "code": "stale_revision",
                    "message": "session_revision is stale.",
                    "current_revision": 9,
                }
            },
        )
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool("ds_" + "a" * 32, 5)

        assert result["ok"] is False
        assert result["code"] == "stale_revision"
        assert result["current_revision"] == 9
        assert result["http_status"] == 409

    @pytest.mark.asyncio
    async def test_backend_404_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        reader.director_turn.side_effect = _http_status_error(
            404, {"error": {"code": "session_not_found", "message": "Not found."}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool("ds_" + "a" * 32, 1)

        assert result["ok"] is False
        assert result["code"] == "not_found"
        assert result["http_status"] == 404

    @pytest.mark.asyncio
    async def test_malformed_response_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool
        from traigent.cloud.analytics_client import AnalyticsClientError

        reader = AsyncMock()
        reader.director_turn.side_effect = AnalyticsClientError("bad shape")
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool("ds_" + "a" * 32, 1)

        assert result["ok"] is False
        assert result["code"] == "malformed_response"

    @pytest.mark.asyncio
    async def test_backend_failure_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_turn_tool

        reader = AsyncMock()
        reader.director_turn.side_effect = RuntimeError("https://secret-host leaked")
        _install_fake_client(monkeypatch, reader)

        result = await director_turn_tool("ds_" + "a" * 32, 1)

        assert result["ok"] is False
        assert result["code"] == "backend_unavailable"
        assert "secret-host" not in result["message"]


class TestDirectorStateTool:
    @pytest.mark.asyncio
    async def test_calls_client_and_wraps_payload(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_state_tool

        reader = AsyncMock()
        reader.director_state.return_value = {"session_id": "ds_" + "a" * 32}
        _install_fake_client(monkeypatch, reader)

        session_id = "ds_" + "a" * 32
        result = await director_state_tool(session_id)

        assert result["ok"] is True
        assert result["director_state"] == {"session_id": session_id}
        reader.director_state.assert_awaited_once_with(session_id)

    @pytest.mark.asyncio
    async def test_missing_session_id_rejected_without_call(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_state_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await director_state_tool("")

        assert result["ok"] is False
        reader.director_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_404_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import director_state_tool

        reader = AsyncMock()
        reader.director_state.side_effect = _http_status_error(
            404, {"error": {"code": "session_not_found", "message": "Not found."}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await director_state_tool("ds_" + "a" * 32)

        assert result["ok"] is False
        assert result["code"] == "not_found"


class TestDirectorToolsNoFreeText:
    """C1: no tool exposes a free-text (client-composed prose) parameter."""

    _BANNED_SUBSTRINGS = ("message", "notes", "goal", "detail", "text", "prompt")

    def test_no_director_tool_exposes_free_text_or_tenant_params(self) -> None:
        from traigent.analytics_mcp import tools as tools_mod

        tool_callables = [
            tools_mod.director_start_tool,
            tools_mod.director_turn_tool,
            tools_mod.director_state_tool,
        ]
        for fn in tool_callables:
            params = set(inspect.signature(fn).parameters)
            assert not any(
                "tenant" in p.lower() for p in params
            ), f"{fn.__name__} must not accept a tenant parameter"
            for banned in self._BANNED_SUBSTRINGS:
                assert not any(banned in p.lower() for p in params), (
                    f"{fn.__name__} must not accept a free-text-shaped "
                    f"parameter ({banned!r} found in {params})"
                )

    def test_director_turn_signature_is_exactly_intent_and_report_plus_identity(
        self,
    ) -> None:
        """C1: director_turn takes only session identity, intent (enum), and
        the typed report block -- no other content field."""
        from traigent.analytics_mcp.tools import director_turn_tool

        params = set(inspect.signature(director_turn_tool).parameters)
        assert params == {"session_id", "session_revision", "intent", "report"}

    def test_director_start_signature_has_no_extra_content_field(self) -> None:
        from traigent.analytics_mcp.tools import director_start_tool

        params = set(inspect.signature(director_start_tool).parameters)
        assert params == {"project_id", "workflow_kind", "run_ids"}

    def test_director_state_signature_is_session_id_only(self) -> None:
        from traigent.analytics_mcp.tools import director_state_tool

        params = set(inspect.signature(director_state_tool).parameters)
        assert params == {"session_id"}


class TestDirectorToolsRegisteredOnServer:
    """Guards against a *_tool defined but never wired into create_server()."""

    @pytest.mark.asyncio
    async def test_director_tools_are_registered(self) -> None:
        pytest.importorskip("mcp")
        from traigent.analytics_mcp.server import create_server

        server = create_server()
        registered = {tool.name for tool in await server.list_tools()}
        assert {"director_start", "director_turn", "director_state"} <= registered
