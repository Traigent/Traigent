"""Unit tests for the agent decision (host-prompt) MCP tools.

The backend client is mocked at the tools-module boundary (tool tests) or at the HTTP
client (request-shape tests); no live backend is contacted.
"""

from __future__ import annotations

import inspect
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

DECISION_ID = "dr_" + "a" * 32
DIGEST = "sha256:" + "b" * 64
NONCE = "rn_" + "c" * 32
AGENT = "agent_123"


def _decision(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decision_id": DECISION_ID,
        "kind": "PREPARE_BASELINE_EVALUATION",
        "prompt": {
            "question": "Prepare the baseline evaluation for this agent now?",
            "options": [
                {"code": "YES", "label": "Yes, prepare it"},
                {"code": "NOT_NOW", "label": "Not now"},
            ],
        },
        "options_version": "1",
        "policy_version": "1",
        "basis_digest": DIGEST,
        "status": "OPEN",
        "answer": None,
        "answered_at": None,
        "answered_via": None,
        "answered_principal_type": None,
        "human_responder": None,
        "execution_status": "NOT_APPLICABLE",
        "execution_ref": None,
        "next_action": "ASK_BUILDER",
        "issued_at": "2026-10-27T10:00:00Z",
        "expires_at": "2026-10-28T10:00:00Z",
    }
    base.update(overrides)
    return base


def _install_fake_client(monkeypatch, reader: AsyncMock):
    from traigent.analytics_mcp import tools as tools_mod

    class _FakeClient:
        backend_url = "https://backend.example.com"

        async def __aenter__(self):
            return reader

        async def __aexit__(self, *exc):
            return False

    async def _fake():
        return _FakeClient()

    monkeypatch.setattr(tools_mod, "_new_analytics_client", _fake)
    monkeypatch.setattr(
        "traigent.config.backend_config.BackendConfig.get_cloud_web_url",
        classmethod(lambda cls: "https://portal.example.com"),
    )
    return reader


def _http_status_error(status_code: int, payload: object):
    import httpx

    request = httpx.Request("POST", "https://secret-host.invalid/x?api_key=raw-secret")
    response = httpx.Response(status_code, json=payload, request=request)
    return httpx.HTTPStatusError(
        "HTTP raw-response-body https://secret-host.invalid",
        request=request,
        response=response,
    )


class TestSignaturesAndRegistration:
    def test_signatures_have_no_free_text_or_tenant(self) -> None:
        from traigent.analytics_mcp import tools as t

        assert set(inspect.signature(t.agent_decision_ensure_tool).parameters) == {
            "agent_id",
            "project_id",
        }
        assert set(inspect.signature(t.agent_decision_latest_tool).parameters) == {
            "agent_id",
            "project_id",
        }
        assert set(inspect.signature(t.agent_decision_answer_tool).parameters) == {
            "decision_id",
            "answer",
            "options_version",
            "policy_version",
            "basis_digest",
            "retry_nonce",
            "project_id",
        }

    @pytest.mark.asyncio
    async def test_tools_registered_with_guidance(self) -> None:
        pytest.importorskip("mcp")
        from traigent.analytics_mcp.server import create_server

        tools = {tool.name: tool for tool in await create_server().list_tools()}
        for name in (
            "agent_decision_ensure",
            "agent_decision_latest",
            "agent_decision_answer",
        ):
            assert name in tools
        answer_doc = tools["agent_decision_answer"].description
        assert "never choose for the human" in answer_doc
        assert "never authorises spend" in answer_doc
        assert "question UI" in tools["agent_decision_ensure"].description

    @pytest.mark.asyncio
    async def test_answer_description_says_how_to_retry(self) -> None:
        pytest.importorskip("mcp")
        from traigent.analytics_mcp.server import create_server

        tools = {tool.name: tool for tool in await create_server().list_tools()}
        text = " ".join(tools["agent_decision_answer"].description.split())
        assert "RATE_LIMITED" in text
        assert "retry_after_seconds" in text
        assert "retry with the returned retry_nonce" in text
        assert "never change the answer on a retry" in text


class TestEnsureAndLatest:
    @pytest.mark.asyncio
    async def test_ensure_returns_backend_fields_and_project_link(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp.tools import agent_decision_ensure_tool

        reader = AsyncMock()
        reader.agent_decision_ensure.return_value = {"decision": _decision()}
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_ensure_tool(AGENT, "proj_1")

        assert result["ok"] is True
        assert result["agent_decision"] == {"decision": _decision()}
        assert result["agent_url"] == (
            "https://portal.example.com/agents/view/agent_123?projectId=proj_1"
        )
        reader.agent_decision_ensure.assert_awaited_once_with(
            AGENT, project_id="proj_1"
        )

    @pytest.mark.asyncio
    async def test_latest_without_project_has_no_link(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import agent_decision_latest_tool

        reader = AsyncMock()
        reader.agent_decision_latest.return_value = {"decision": None}
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_latest_tool(AGENT)

        assert result == {"ok": True, "agent_decision": {"decision": None}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad", ["", "  ", " agent ", ".", "..", "x" * 256, "a\nb", "a\x00b", "a\x7fb"]
    )
    async def test_bad_agent_id_rejected_locally(self, monkeypatch, bad) -> None:
        from traigent.analytics_mcp.tools import agent_decision_ensure_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_ensure_tool(bad)

        assert result["ok"] is False
        assert result["message"] == "agent_id is not a valid agent identifier."
        reader.agent_decision_ensure.assert_not_called()


class TestAnswer:
    @pytest.mark.asyncio
    async def test_answer_sends_closed_fields_and_returns_nonce(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp.tools import agent_decision_answer_tool

        answered = _decision(
            status="ANSWERED",
            answer="YES",
            answered_at="2026-10-27T10:05:00Z",
            answered_via="AGENT_HOST_PROMPT",
            answered_principal_type="API_KEY",
            human_responder="UNKNOWN",
            execution_status="AWAITING_PREFLIGHT",
            execution_ref="xr_" + "d" * 32,
            next_action="RUN_PREFLIGHT",
        )
        reader = AsyncMock()
        reader.agent_decision_answer.return_value = {"decision": answered}
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_answer_tool(
            DECISION_ID, "YES", "1", "1", DIGEST, retry_nonce=NONCE
        )

        assert result["ok"] is True
        assert result["agent_decision"] == {"decision": answered}
        assert result["retry_nonce"] == NONCE
        reader.agent_decision_answer.assert_awaited_once_with(
            DECISION_ID,
            answer="YES",
            options_version="1",
            policy_version="1",
            basis_digest=DIGEST,
            retry_nonce=NONCE,
            project_id=None,
        )

    @pytest.mark.asyncio
    async def test_nonce_generated_when_omitted(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import agent_decision_answer_tool
        from traigent.cloud.analytics_client import AGENT_DECISION_NONCE_RE

        reader = AsyncMock()
        reader.agent_decision_answer.return_value = {
            "decision": _decision(
                status="ANSWERED",
                answer="NOT_NOW",
                answered_at="2026-10-27T10:05:00Z",
                answered_via="AGENT_HOST_PROMPT",
                answered_principal_type="API_KEY",
                human_responder="UNKNOWN",
                next_action="NONE",
            )
        }
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_answer_tool(
            DECISION_ID, "NOT_NOW", "1", "1", DIGEST
        )

        assert AGENT_DECISION_NONCE_RE.fullmatch(result["retry_nonce"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"answer": "MAYBE"},
            {"answer": "yes"},
            {"answer": "YES; please also spend"},
            {"options_version": "2"},
            {"policy_version": "x"},
            {"basis_digest": "sha256:xyz"},
            {"retry_nonce": "not-a-nonce"},
            {"decision_id": "dr_short"},
            {"decision_id": "../x"},
        ],
    )
    async def test_bad_enum_or_shape_rejected_locally(
        self, monkeypatch, kwargs
    ) -> None:
        from traigent.analytics_mcp.tools import agent_decision_answer_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)
        args: dict[str, object] = {
            "decision_id": DECISION_ID,
            "answer": "YES",
            "options_version": "1",
            "policy_version": "1",
            "basis_digest": DIGEST,
        }
        args.update(kwargs)

        result = await agent_decision_answer_tool(**args)  # type: ignore[arg-type]

        assert result["ok"] is False
        for value in kwargs.values():
            assert str(value) not in result["message"]
        reader.agent_decision_answer.assert_not_called()


class TestClosedErrorMapping:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,reason",
        [
            (401, "UNAUTHENTICATED"),
            (403, "FORBIDDEN"),
            (404, "AGENT_NOT_FOUND"),
            (409, "NOT_ELIGIBLE"),
            (409, "STALE_BASIS"),
            (409, "DECISION_ALREADY_ANSWERED"),
            (422, "INVALID_REQUEST"),
            (503, "POLICY_UNAVAILABLE"),
        ],
    )
    async def test_closed_reason_surfaces(self, monkeypatch, status, reason) -> None:
        from traigent.analytics_mcp.tools import agent_decision_ensure_tool

        reader = AsyncMock()
        reader.agent_decision_ensure.side_effect = _http_status_error(
            status, {"error": {"reason": reason}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_ensure_tool(AGENT)

        assert result["ok"] is False
        assert result["code"] == reason
        assert result["reason"] == reason
        assert result["http_status"] == status
        assert "secret-host" not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_rate_limited_carries_retry_after(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import agent_decision_latest_tool

        reader = AsyncMock()
        reader.agent_decision_latest.side_effect = _http_status_error(
            429, {"error": {"reason": "RATE_LIMITED", "retry_after_seconds": 30}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_latest_tool(AGENT)

        assert result["code"] == "RATE_LIMITED"
        assert result["retry_after_seconds"] == 30

    @pytest.mark.asyncio
    async def test_non_closed_error_body_falls_back_generic(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import agent_decision_ensure_tool

        reader = AsyncMock()
        reader.agent_decision_ensure.side_effect = _http_status_error(
            500, {"error": {"reason": "CANARY_free_text", "extra": "CANARY"}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_ensure_tool(AGENT)

        assert result["ok"] is False
        assert result["code"] == "backend_unavailable"
        assert "CANARY" not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_malformed_response_is_closed_failure(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import agent_decision_ensure_tool
        from traigent.cloud.analytics_client import AnalyticsClientError

        reader = AsyncMock()
        reader.agent_decision_ensure.side_effect = AnalyticsClientError("CANARY body")
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_ensure_tool(AGENT)

        assert result["code"] == "malformed_response"
        assert "CANARY" not in json.dumps(result)


class TestCanaryNeverLogged:
    @pytest.mark.asyncio
    async def test_canary_in_inputs_absent_from_logs_and_output(
        self, monkeypatch, caplog
    ) -> None:
        from traigent.analytics_mcp import tools as t

        canary = "CANARY-Alice_has_diabetes"
        reader = AsyncMock()
        reader.agent_decision_ensure.side_effect = _http_status_error(
            403, {"error": {"reason": "FORBIDDEN"}}
        )
        reader.agent_decision_answer.side_effect = RuntimeError(canary)
        _install_fake_client(monkeypatch, reader)

        with caplog.at_level(logging.DEBUG):
            outputs = [
                await t.agent_decision_ensure_tool(canary),
                await t.agent_decision_ensure_tool(AGENT, canary),
                await t.agent_decision_answer_tool(
                    DECISION_ID, canary, "1", "1", DIGEST
                ),
                await t.agent_decision_answer_tool(
                    DECISION_ID, "YES", "1", "1", DIGEST, NONCE, canary
                ),
                await t.agent_decision_answer_tool(
                    DECISION_ID, "YES", "1", "1", DIGEST, NONCE
                ),
            ]

        assert canary not in json.dumps(outputs)
        assert canary not in caplog.text


@pytest.mark.backend_online
class TestClientRequestShape:
    """Request shape against the pinned TraigentSchema agent_decision contract."""

    @staticmethod
    def _client():
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        return BackendAnalyticsClient(
            backend_url="http://localhost:5000",
            api_key="uk_test_key",  # pragma: allowlist secret
        )

    @staticmethod
    def _http(payload: dict[str, object]):
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        http = AsyncMock()
        http.post.return_value = response
        http.get.return_value = response
        return http

    @pytest.mark.asyncio
    async def test_ensure_posts_only_kind_to_agent_path(self) -> None:
        client = self._client()
        client._client = http = self._http({"decision": None})

        result = await client.agent_decision_ensure(AGENT, project_id="proj_1")

        assert result == {"decision": None}
        args, kwargs = http.post.call_args
        assert args[0] == f"/api/v1beta/agent-decisions/{AGENT}/ensure"
        assert kwargs["json"] == {"kind": "PREPARE_BASELINE_EVALUATION"}
        assert kwargs["headers"]["X-Project-Id"] == "proj_1"

    @pytest.mark.asyncio
    async def test_latest_is_a_get_with_no_body(self) -> None:
        client = self._client()
        client._client = http = self._http({"decision": _decision()})

        await client.agent_decision_latest(AGENT)

        args, kwargs = http.get.call_args
        assert args[0] == f"/api/v1beta/agent-decisions/{AGENT}/latest"
        assert "json" not in kwargs
        http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_body_has_exactly_the_contract_keys_and_no_idempotency_header(
        self,
    ) -> None:
        client = self._client()
        client._client = http = self._http(
            {
                "decision": _decision(
                    status="ANSWERED",
                    answer="NOT_NOW",
                    answered_at="2026-10-27T10:05:00Z",
                    answered_via="AGENT_HOST_PROMPT",
                    answered_principal_type="API_KEY",
                    human_responder="UNKNOWN",
                    next_action="NONE",
                )
            }
        )

        await client.agent_decision_answer(
            DECISION_ID,
            answer="NOT_NOW",
            options_version="1",
            policy_version="1",
            basis_digest=DIGEST,
            retry_nonce=NONCE,
        )

        args, kwargs = http.post.call_args
        assert args[0] == f"/api/v1beta/agent-decisions/{DECISION_ID}/answer"
        assert set(kwargs["json"]) == {
            "answer",
            "options_version",
            "policy_version",
            "basis_digest",
            "retry_nonce",
        }
        assert "Idempotency-Key" not in (kwargs.get("headers") or {})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"decision": None, "extra": 1},
            {"decision": {"decision_id": DECISION_ID}},
            {"decision": {**_decision(), "status": "WEIRD"}},
            {"decision": {**_decision(), "next_action": "SPEND"}},
            {"decision": {**_decision(), "unexpected": "x"}},
        ],
    )
    async def test_response_outside_contract_is_rejected(self, payload) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = self._client()
        client._client = self._http(payload)

        with pytest.raises(AnalyticsClientError) as info:
            await client.agent_decision_ensure(AGENT)
        assert "unexpected" not in str(info.value)


class TestProjectQualifiedLinks:
    def test_agent_url_carries_project_id(self) -> None:
        from traigent.cloud.sync_manager import build_agent_url

        assert (
            build_agent_url("https://portal.traigent.ai/", "agent/1", project_id="p 1")
            == "https://portal.traigent.ai/agents/view/agent%2F1?projectId=p%201"
        )

    def test_agent_url_without_project_is_bare(self) -> None:
        from traigent.cloud.sync_manager import build_agent_url

        assert (
            build_agent_url("https://portal.traigent.ai", "a1", project_id=" ")
            == "https://portal.traigent.ai/agents/view/a1"
        )

    def test_experiment_url_carries_project_id_query(self) -> None:
        from traigent.cloud.sync_manager import build_experiment_url

        url = build_experiment_url("https://p.example", "e1", project_id="proj_1")
        assert "projectId=proj_1" in url


def _answered(**overrides: object) -> dict[str, object]:
    base = _decision(
        status="ANSWERED",
        answer="YES",
        answered_at="2026-10-27T10:05:00.123Z",
        answered_via="AGENT_HOST_PROMPT",
        answered_principal_type="API_KEY",
        human_responder="UNKNOWN",
        execution_status="AWAITING_PREFLIGHT",
        execution_ref="xr_" + "d" * 32,
        next_action="RUN_PREFLIGHT",
    )
    base.update(overrides)
    return base


class TestAgentIdAndProjectIdInputs:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent", ["my agent", "pkg.mod/fn name", "naïve_agent", "a?b=c"]
    )
    async def test_customer_agent_ids_beyond_a_narrow_charset_are_accepted(
        self, monkeypatch, agent
    ) -> None:
        from traigent.analytics_mcp.tools import agent_decision_latest_tool

        reader = AsyncMock()
        reader.agent_decision_latest.return_value = {"decision": None}
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_latest_tool(agent)

        assert result["ok"] is True
        reader.agent_decision_latest.assert_awaited_once()

    @pytest.mark.backend_online
    @pytest.mark.asyncio
    async def test_agent_id_is_one_url_encoded_path_segment(self) -> None:
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        client = BackendAnalyticsClient(
            backend_url="http://localhost:5000",
            api_key="uk_test_key",  # pragma: allowlist secret
        )
        response = MagicMock()
        response.json.return_value = {"decision": None}
        http = AsyncMock()
        http.get.return_value = response
        client._client = http

        await client.agent_decision_latest("a/../b c?x=1")

        path = http.get.call_args.args[0]
        assert path == "/api/v1beta/agent-decisions/a%2F..%2Fb%20c%3Fx%3D1/latest"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "  ",
            "CANARY/../x",
            "CANARY x",
            "CANARY\nx",
            "CANARY?a=1",
            "c" * 129,
            "-x",
        ],
    )
    async def test_bad_project_id_rejected_without_echo(self, monkeypatch, bad) -> None:
        from traigent.analytics_mcp import tools as t

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        outputs = [
            await t.agent_decision_ensure_tool(AGENT, bad),
            await t.agent_decision_latest_tool(AGENT, bad),
            await t.agent_decision_answer_tool(
                DECISION_ID, "YES", "1", "1", DIGEST, NONCE, bad
            ),
        ]

        for result in outputs:
            assert result["ok"] is False
            assert "CANARY" not in json.dumps(result)
        reader.agent_decision_ensure.assert_not_called()
        reader.agent_decision_latest.assert_not_called()
        reader.agent_decision_answer.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("good", ["proj_1", "2f6c1a0e-9d3b-4c1a-8e55-0a1b2c3d4e5f"])
    async def test_good_project_id_accepted(self, monkeypatch, good) -> None:
        from traigent.analytics_mcp.tools import agent_decision_latest_tool

        reader = AsyncMock()
        reader.agent_decision_latest.return_value = {"decision": None}
        _install_fake_client(monkeypatch, reader)

        assert (await agent_decision_latest_tool(AGENT, good))["ok"] is True


class TestMalformedErrorBodiesAreFixedFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429, 500, 503])
    @pytest.mark.parametrize(
        "body",
        [
            {"error": {"message": "CANARY"}},
            {"error": {"reason": "CANARY_free_text", "message": "CANARY"}},
            {"message": "CANARY"},
        ],
    )
    async def test_backend_text_never_reaches_the_result(
        self, monkeypatch, status, body
    ) -> None:
        from traigent.analytics_mcp import tools as t

        reader = AsyncMock()
        reader.agent_decision_ensure.side_effect = _http_status_error(status, body)
        reader.agent_decision_answer.side_effect = _http_status_error(status, body)
        _install_fake_client(monkeypatch, reader)

        outputs = [
            await t.agent_decision_ensure_tool(AGENT),
            await t.agent_decision_answer_tool(DECISION_ID, "YES", "1", "1", DIGEST),
        ]

        for result in outputs:
            assert result["ok"] is False
            assert "CANARY" not in json.dumps(result)
            assert result["code"] in {
                "request_rejected",
                "backend_unavailable",
                "authentication_failed",
                "forbidden",
            }


class TestAnswerErrorMapping:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,reason",
        [(409, "REPLAY_MISMATCH"), (410, "DECISION_EXPIRED")],
    )
    async def test_answer_failure_keeps_reason_and_retry_nonce(
        self, monkeypatch, status, reason
    ) -> None:
        from traigent.analytics_mcp.tools import agent_decision_answer_tool

        reader = AsyncMock()
        reader.agent_decision_answer.side_effect = _http_status_error(
            status, {"error": {"reason": reason}}
        )
        _install_fake_client(monkeypatch, reader)

        result = await agent_decision_answer_tool(
            DECISION_ID, "YES", "1", "1", DIGEST, NONCE
        )

        assert result["ok"] is False
        assert result["code"] == reason
        assert result["reason"] == reason
        assert result["http_status"] == status
        assert result["retry_nonce"] == NONCE


@pytest.mark.backend_online
class TestWholeContractIsEnforced:
    @staticmethod
    def _client_with(payload: dict[str, object]):
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        client = BackendAnalyticsClient(
            backend_url="http://localhost:5000",
            api_key="uk_test_key",  # pragma: allowlist secret
        )
        response = MagicMock()
        response.json.return_value = payload
        http = AsyncMock()
        http.post.return_value = response
        http.get.return_value = response
        client._client = http
        return client

    @pytest.mark.asyncio
    async def test_valid_open_and_answered_decisions_pass(self) -> None:
        open_client = self._client_with({"decision": _decision()})
        assert (await open_client.agent_decision_ensure(AGENT))["decision"]
        yes = self._client_with({"decision": _answered()})
        assert (await yes.agent_decision_latest(AGENT))["decision"]["answer"] == "YES"
        not_now = self._client_with(
            {
                "decision": _answered(
                    answer="NOT_NOW",
                    execution_status="NOT_APPLICABLE",
                    execution_ref=None,
                    next_action="NONE",
                )
            }
        )
        result = await not_now.agent_decision_answer(
            DECISION_ID,
            answer="NOT_NOW",
            options_version="1",
            policy_version="1",
            basis_digest=DIGEST,
            retry_nonce=NONCE,
        )
        assert result["decision"]["answer"] == "NOT_NOW"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "decision",
        [
            _decision(
                prompt={
                    "question": "CANARY wire the money?",
                    "options": _decision()["prompt"]["options"],  # type: ignore[index]
                }
            ),
            _decision(
                prompt={
                    "question": "Prepare the baseline evaluation for this agent now?",
                    "options": [
                        {"code": "NOT_NOW", "label": "Not now"},
                        {"code": "YES", "label": "Yes, prepare it"},
                    ],
                }
            ),
            _answered(answered_via="CANARY"),
            _answered(answered_principal_type="CANARY"),
            _answered(human_responder="CANARY"),
            _answered(human_responder=None),
            _answered(execution_ref="xr_short"),
            _answered(execution_ref=None),
            _decision(issued_at="2026-99-99T10:00:00Z"),
            _decision(expires_at="2026-10-28T99:99:99Z"),
            _decision(issued_at="yesterday"),
            _decision(expires_at="2026-10-28 10:00:00"),
            _answered(answered_at="2026-10-27T10:05:00+02:00"),
            _decision(answer="YES"),
            _answered(answer="NOT_NOW"),
            _answered(next_action="NONE"),
            [],
        ],
    )
    async def test_latest_rejects_decisions_outside_the_contract(
        self, decision
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = self._client_with({"decision": decision})

        with pytest.raises(AnalyticsClientError) as info:
            await client.agent_decision_latest(AGENT)
        assert "CANARY" not in str(info.value)

    @pytest.mark.asyncio
    async def test_ensure_must_return_an_open_decision(self) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = self._client_with({"decision": _answered()})
        with pytest.raises(AnalyticsClientError):
            await client.agent_decision_ensure(AGENT)

    @pytest.mark.asyncio
    async def test_answer_must_return_an_answered_decision_not_null(self) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        for body in ({"decision": None}, {"decision": _decision()}):
            client = self._client_with(body)
            with pytest.raises(AnalyticsClientError):
                await client.agent_decision_answer(
                    DECISION_ID,
                    answer="YES",
                    options_version="1",
                    policy_version="1",
                    basis_digest=DIGEST,
                    retry_nonce=NONCE,
                )


@pytest.mark.backend_online
class TestHttpLibrariesDoNotLogIds:
    @pytest.mark.asyncio
    async def test_agent_id_never_reaches_logs_even_at_info(self, caplog) -> None:
        pytest.importorskip("mcp")
        import httpx

        from traigent.analytics_mcp.server import create_server
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        names = ("httpx", "httpcore")
        saved = {n: logging.getLogger(n).level for n in names}
        try:
            for name in names:
                logging.getLogger(name).setLevel(logging.NOTSET)
            caplog.set_level(logging.INFO)
            create_server()

            canary = "CANARYagent9f3b"
            client = BackendAnalyticsClient(
                backend_url="http://localhost:5000",
                api_key="uk_test_key",  # pragma: allowlist secret
            )
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"decision": None})
                ),
                base_url="http://localhost:5000",
            )
            await client.agent_decision_latest(canary)
            await client._client.aclose()

            assert canary not in caplog.text
        finally:
            for name, level in saved.items():
                logging.getLogger(name).setLevel(level)


@pytest.mark.backend_online
class TestExactAgentIdSegment:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [" agent ", "agent ", " agent", ".", ".."])
    async def test_client_rejects_ids_normalisation_would_change(self, bad) -> None:
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        client = BackendAnalyticsClient(
            backend_url="http://localhost:5000",
            api_key="uk_test_key",  # pragma: allowlist secret
        )
        client._client = http = AsyncMock()

        for call in (client.agent_decision_ensure, client.agent_decision_latest):
            with pytest.raises(ValueError) as info:
                await call(bad)
            assert str(info.value) == "agent_id is not a valid agent identifier."
        http.get.assert_not_called()
        http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_internal_slash_is_one_encoded_segment(self) -> None:
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        client = BackendAnalyticsClient(
            backend_url="http://localhost:5000",
            api_key="uk_test_key",  # pragma: allowlist secret
        )
        response = MagicMock()
        response.json.return_value = {"decision": None}
        client._client = http = AsyncMock()
        http.post.return_value = response

        await client.agent_decision_ensure("pkg/agent")

        assert http.post.call_args.args[0] == (
            "/api/v1beta/agent-decisions/pkg%2Fagent/ensure"
        )
