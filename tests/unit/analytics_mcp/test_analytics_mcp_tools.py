"""Unit tests for the agent-facing analytics MCP tools.

These tests verify the tools REALLY call BackendAnalyticsClient (no fake
completion), enforce explicit project_id, accept no tenant_id, and normalize
backend/transport failures into structured ``ok=False`` payloads. The backend
client is mocked at the tools-module boundary; no live backend is contacted.
"""

from __future__ import annotations

import importlib.util
import inspect
from unittest.mock import AsyncMock

import pytest

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None


def _install_fake_client(monkeypatch, reader: AsyncMock):
    """Patch ``_new_analytics_client`` to yield ``reader`` as an async ctx mgr.

    The tools open the client with ``async with _new_analytics_client() as
    reader``; this returns an object whose async context manager yields the
    provided mock ``reader`` so each tool's real call path is exercised.
    """
    from traigent.analytics_mcp import tools as tools_mod

    class _FakeClient:
        async def __aenter__(self):
            return reader

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(tools_mod, "_new_analytics_client", lambda: _FakeClient())
    return reader


class TestRunReportTool:
    @pytest.mark.asyncio
    async def test_calls_client_and_wraps_payload(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_report_tool

        reader = AsyncMock()
        reader.get_run_report.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_report_tool("proj_abc", "run_123")

        assert result["ok"] is True
        assert result["run_report"] == {"run_id": "run_123"}
        reader.get_run_report.assert_awaited_once_with("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_missing_project_id_rejected_without_call(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_report_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_report_tool("", "run_123")

        assert result["ok"] is False
        assert "project_id" in result["message"]
        reader.get_run_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_failure_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_report_tool

        reader = AsyncMock()
        reader.get_run_report.side_effect = RuntimeError(
            "https://secret-host/internal leaked"
        )
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_report_tool("proj_abc", "run_123")

        assert result["ok"] is False
        assert result["code"] == "backend_unavailable"
        # The raw exception text (which embedded a URL) must NOT leak.
        assert "secret-host" not in result["message"]

    @pytest.mark.asyncio
    async def test_malformed_response_is_structured(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_report_tool
        from traigent.cloud.analytics_client import AnalyticsClientError

        reader = AsyncMock()
        reader.get_run_report.side_effect = AnalyticsClientError("bad shape")
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_report_tool("proj_abc", "run_123")

        assert result["ok"] is False
        assert result["code"] == "malformed_response"


class TestProjectOverviewTool:
    @pytest.mark.asyncio
    async def test_calls_client(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_project_overview_tool

        reader = AsyncMock()
        reader.get_project_overview.return_value = {"project_id": "proj_abc"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_project_overview_tool("proj_abc")

        assert result["ok"] is True
        assert result["project_overview"] == {"project_id": "proj_abc"}
        reader.get_project_overview.assert_awaited_once_with("proj_abc")


class TestCompareRunsTool:
    @pytest.mark.asyncio
    async def test_calls_client_with_cleaned_runs(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_compare_runs_tool

        reader = AsyncMock()
        reader.compare_runs.return_value = {"comparison": []}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_compare_runs_tool("proj_abc", ["run_1", " run_2 ", ""])

        assert result["ok"] is True
        reader.compare_runs.assert_awaited_once_with("proj_abc", ["run_1", "run_2"])

    @pytest.mark.asyncio
    async def test_single_run_rejected(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_compare_runs_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await analytics_compare_runs_tool("proj_abc", ["run_1"])

        assert result["ok"] is False
        assert "at least two" in result["message"]
        reader.compare_runs.assert_not_called()


class TestDecisionBriefTool:
    @pytest.mark.asyncio
    async def test_calls_client_with_intent(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_decision_brief_tool

        reader = AsyncMock()
        reader.get_run_decision_brief.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_decision_brief_tool(
            "proj_abc", "run_123", intent="deploy"
        )

        assert result["ok"] is True
        assert result["decision_brief"] == {"run_id": "run_123"}
        reader.get_run_decision_brief.assert_awaited_once_with(
            "proj_abc", "run_123", "deploy"
        )

    @pytest.mark.asyncio
    async def test_defaults_intent_to_iterate(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_decision_brief_tool

        reader = AsyncMock()
        reader.get_run_decision_brief.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        await analytics_get_run_decision_brief_tool("proj_abc", "run_123")

        args = reader.get_run_decision_brief.await_args.args
        assert args[2] == "iterate"

    @pytest.mark.asyncio
    async def test_rejects_unregistered_intent_without_client_call(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_decision_brief_tool

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_decision_brief_tool(
            "proj_abc", "run_123", intent="promote"
        )

        assert result["ok"] is False
        assert "intent must be one of" in result["message"]
        reader.get_run_decision_brief.assert_not_called()


class TestWave2AnalyticsTools:
    @pytest.mark.asyncio
    async def test_pareto_tool_calls_client(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_single_run_pareto_tool

        reader = AsyncMock()
        reader.get_single_run_pareto.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_single_run_pareto_tool(
            "proj_abc",
            "run_123",
            x_measure="cost",
            y_measure="quality",
            request_count=5,
        )

        assert result["ok"] is True
        assert result["single_run_pareto"] == {"run_id": "run_123"}
        reader.get_single_run_pareto.assert_awaited_once_with(
            "proj_abc",
            "run_123",
            x_measure="cost",
            y_measure="quality",
            request_count=5,
        )

    @pytest.mark.asyncio
    async def test_correlation_tool_calls_client(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_correlation_matrix_tool

        reader = AsyncMock()
        reader.get_correlation_matrix.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_correlation_matrix_tool(
            "proj_abc", "run_123", method="spearman", min_sample=4
        )

        assert result["ok"] is True
        assert result["correlation_matrix"] == {"run_id": "run_123"}
        reader.get_correlation_matrix.assert_awaited_once_with(
            "proj_abc", "run_123", method="spearman", min_sample=4
        )

    @pytest.mark.asyncio
    async def test_leaderboard_tool_calls_client(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_run_leaderboard_tool

        reader = AsyncMock()
        reader.get_run_leaderboard.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_run_leaderboard_tool(
            "proj_abc",
            "run_123",
            objective="weighted",
            weights={"quality": 0.8},
            constraints='{"cost":1.0}',
            request_count=7,
            limit=10,
        )

        assert result["ok"] is True
        assert result["run_leaderboard"] == {"run_id": "run_123"}
        reader.get_run_leaderboard.assert_awaited_once_with(
            "proj_abc",
            "run_123",
            objective="weighted",
            weights={"quality": 0.8},
            constraints='{"cost":1.0}',
            request_count=7,
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_parameter_insights_tool_calls_client(self, monkeypatch) -> None:
        from traigent.analytics_mcp.tools import analytics_get_parameter_insights_tool

        reader = AsyncMock()
        reader.get_parameter_insights.return_value = {"run_id": "run_123"}
        _install_fake_client(monkeypatch, reader)

        result = await analytics_get_parameter_insights_tool(
            "proj_abc",
            "run_123",
            target_measure="accuracy",
            min_trials=12,
            top_k=5,
        )

        assert result["ok"] is True
        assert result["parameter_insights"] == {"run_id": "run_123"}
        reader.get_parameter_insights.assert_awaited_once_with(
            "proj_abc",
            "run_123",
            target_measure="accuracy",
            min_trials=12,
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_missing_project_id_is_rejected_for_new_tools(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp import tools as tools_mod

        reader = AsyncMock()
        _install_fake_client(monkeypatch, reader)
        calls = [
            tools_mod.analytics_get_single_run_pareto_tool("", "run_123"),
            tools_mod.analytics_get_correlation_matrix_tool("", "run_123"),
            tools_mod.analytics_get_run_leaderboard_tool("", "run_123"),
            tools_mod.analytics_get_parameter_insights_tool("", "run_123"),
        ]

        for call in calls:
            result = await call
            assert result["ok"] is False
            assert "project_id" in result["message"]

        reader.get_single_run_pareto.assert_not_called()
        reader.get_correlation_matrix.assert_not_called()
        reader.get_run_leaderboard.assert_not_called()
        reader.get_parameter_insights.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_failures_are_structured_for_new_tools(
        self, monkeypatch
    ) -> None:
        from traigent.analytics_mcp import tools as tools_mod

        reader = AsyncMock()
        reader.get_single_run_pareto.side_effect = RuntimeError(
            "https://secret-host/internal leaked"
        )
        reader.get_correlation_matrix.side_effect = RuntimeError(
            "https://secret-host/internal leaked"
        )
        reader.get_run_leaderboard.side_effect = RuntimeError(
            "https://secret-host/internal leaked"
        )
        reader.get_parameter_insights.side_effect = RuntimeError(
            "https://secret-host/internal leaked"
        )
        _install_fake_client(monkeypatch, reader)

        calls = [
            tools_mod.analytics_get_single_run_pareto_tool("proj_abc", "run_123"),
            tools_mod.analytics_get_correlation_matrix_tool("proj_abc", "run_123"),
            tools_mod.analytics_get_run_leaderboard_tool("proj_abc", "run_123"),
            tools_mod.analytics_get_parameter_insights_tool("proj_abc", "run_123"),
        ]

        for call in calls:
            result = await call
            assert result["ok"] is False
            assert result["code"] == "backend_unavailable"
            assert "secret-host" not in result["message"]


class TestRenderChartTool:
    def test_renders_and_returns_path(self, tmp_path) -> None:
        from traigent.analytics_mcp.tools import analytics_render_chart_tool

        payload = {
            "run_id": "run_123",
            "measures": {},
            "frontier": [
                {"config_id": "cfg_1", "metrics": {"quality": 0.9, "cost": 0.02}}
            ],
            "dominated": [],
        }
        out = tmp_path / "p.png"
        result = analytics_render_chart_tool(payload, "run_pareto", str(out))

        assert result["ok"] is True
        assert result["kind"] == "run_pareto"
        from pathlib import Path

        assert Path(result["chart_path"]).exists()

    def test_unsupported_kind_rejected(self) -> None:
        from traigent.analytics_mcp.tools import analytics_render_chart_tool

        result = analytics_render_chart_tool({}, "scatter")
        assert result["ok"] is False
        assert "kind must be one of" in result["message"]

    def test_non_dict_payload_rejected(self) -> None:
        from traigent.analytics_mcp.tools import analytics_render_chart_tool

        result = analytics_render_chart_tool(["x"], "run_pareto")  # type: ignore[arg-type]
        assert result["ok"] is False


class TestHealthAndAuthTools:
    @pytest.mark.asyncio
    async def test_health_check_makes_no_network_call_and_reports_flags(self) -> None:
        from traigent.analytics_mcp.tools import health_check_tool

        result = await health_check_tool()

        assert result["ok"] is True
        assert result["service"] == "traigent-analytics-mcp"
        assert "httpx_available" in result
        assert "chart_rendering_available" in result
        assert "run_pareto" in result["supported_chart_kinds"]

    @pytest.mark.asyncio
    async def test_auth_status_masks_key(self, monkeypatch) -> None:
        from traigent.analytics_mcp import tools as tools_mod

        monkeypatch.setattr(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            classmethod(
                lambda cls: {
                    "api_key": "uk_abcdef123456",  # pragma: allowlist secret
                    "backend_url": "http://localhost:5000",
                    "source": "environment",
                }
            ),
        )
        result = await tools_mod.auth_status_tool()

        assert result["ok"] is True
        assert result["authenticated"] is True
        assert result["auth_type"] == "api_key"
        # Full key must never appear; only prefix + last4.
        assert result["api_key"]["present"] is True
        assert result["api_key"]["prefix"] == "uk_a"
        assert result["api_key"]["last4"] == "3456"
        assert "uk_abcdef123456" not in str(result)


class TestNoTenantArgument:
    """The MCP must not accept caller-supplied tenancy; the backend owns it."""

    def test_no_tool_exposes_a_tenant_parameter(self) -> None:
        from traigent.analytics_mcp import tools as tools_mod

        tool_callables = [
            tools_mod.analytics_get_run_report_tool,
            tools_mod.analytics_get_project_overview_tool,
            tools_mod.analytics_compare_runs_tool,
            tools_mod.analytics_get_run_decision_brief_tool,
            tools_mod.analytics_get_single_run_pareto_tool,
            tools_mod.analytics_get_correlation_matrix_tool,
            tools_mod.analytics_get_run_leaderboard_tool,
            tools_mod.analytics_get_parameter_insights_tool,
            tools_mod.analytics_render_chart_tool,
        ]
        for fn in tool_callables:
            params = set(inspect.signature(fn).parameters)
            assert not any("tenant" in p.lower() for p in params), (
                f"{fn.__name__} must not accept a tenant parameter"
            )
