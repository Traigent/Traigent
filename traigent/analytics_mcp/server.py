"""Agent-facing stdio MCP server for Traigent optimization-results analytics.

This is a NEW, separate MCP from the local ``traigent.mcp`` server. Where the
local server only sees ``.traigent/`` results on disk, this analytics MCP is an
authenticated **cloud-read** client: its tools call the backend analytics
endpoints (through ``client.analytics`` /
:class:`traigent.cloud.analytics_client.BackendAnalyticsClient`) using the
user's existing SDK credentials, plus a local chart-render helper.

Console entry point: ``traigent-analytics-mcp`` (see ``pyproject.toml``).

Security posture:

* Every cloud tool requires an explicit ``project_id`` (no implicit "latest").
* No tool accepts a ``tenant_id`` — the backend owns tenancy from the
  authenticated principal.
* ``health_check`` / ``auth_status`` never return credential material and make
  no network call.
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-ANALYTICS

from __future__ import annotations

from typing import Any

from traigent.analytics_mcp.tools import (
    analytics_compare_runs_tool,
    analytics_get_correlation_matrix_tool,
    analytics_get_parameter_insights_tool,
    analytics_get_project_overview_tool,
    analytics_get_run_decision_brief_tool,
    analytics_get_run_leaderboard_tool,
    analytics_get_run_report_tool,
    analytics_get_single_run_pareto_tool,
    analytics_render_chart_tool,
    auth_status_tool,
    health_check_tool,
)
from traigent.cloud.analytics_client import SUPPORTED_DECISION_INTENTS

_MCP_INSTALL_MESSAGE = (
    "The optional MCP dependency is not installed. "
    "Install it with: pip install 'traigent[mcp]'"
)


def create_server() -> Any:
    """Create the Traigent analytics MCP server.

    The optional ``mcp`` package is imported lazily so
    ``traigent-analytics-mcp --help`` and import-time checks still work in
    environments that have not installed ``traigent[mcp]``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - covered by CLI/entrypoint test
        raise RuntimeError(_MCP_INSTALL_MESSAGE) from exc

    server = FastMCP(
        "traigent-analytics",
        instructions=(
            "Agent-facing Traigent analytics MCP. Tools read optimization "
            "results from the Traigent backend using your existing SDK "
            "credentials and render charts locally. Every cloud tool requires "
            "an explicit project_id; no tool accepts a tenant_id (the backend "
            "owns tenancy). Credentials are never returned."
        ),
    )

    @server.tool(
        description=(
            "Report analytics-MCP readiness: SDK import status, sanitized "
            "backend URL, and whether chart rendering is available. No network "
            "call; no credentials returned."
        )
    )
    async def health_check() -> dict[str, Any]:
        return await health_check_tool()

    @server.tool(
        description=(
            "Report local Traigent auth posture for the analytics MCP. API key "
            "output is masked to prefix and last4 only. No network call."
        )
    )
    async def auth_status() -> dict[str, Any]:
        return await auth_status_tool()

    @server.tool(
        description=(
            "Fetch the backend's full analytics report for one optimization "
            "run. Requires explicit project_id and run_id and backend auth."
        )
    )
    async def analytics_get_run_report(project_id: str, run_id: str) -> dict[str, Any]:
        return await analytics_get_run_report_tool(project_id, run_id)

    @server.tool(
        description=(
            "Fetch the backend's cross-run overview for a project. Requires an "
            "explicit project_id and backend auth."
        )
    )
    async def analytics_get_project_overview(project_id: str) -> dict[str, Any]:
        return await analytics_get_project_overview_tool(project_id)

    @server.tool(
        description=(
            "Compare two or more runs within a project. Requires an explicit "
            "project_id and a list of at least two run_ids, plus backend auth."
        )
    )
    async def analytics_compare_runs(
        project_id: str, run_ids: list[str]
    ) -> dict[str, Any]:
        return await analytics_compare_runs_tool(project_id, run_ids)

    @server.tool(
        description=(
            "Fetch the backend's decision brief (decision_payload v0) for a "
            "run: a headline, confidence, recommended action, and evidence. "
            "Requires explicit project_id and run_id; optional intent "
            f"({', '.join(SUPPORTED_DECISION_INTENTS)}, default iterate). "
            "Backend auth required."
        )
    )
    async def analytics_get_run_decision_brief(
        project_id: str,
        run_id: str,
        intent: str = "iterate",
    ) -> dict[str, Any]:
        return await analytics_get_run_decision_brief_tool(project_id, run_id, intent)

    @server.tool(
        description=(
            "Fetch the backend's Pareto frontier (run_pareto v0) for one "
            "optimization run. Requires explicit project_id and run_id; "
            "optional x_measure, y_measure, and request_count query params. "
            "Backend auth required."
        )
    )
    async def analytics_get_single_run_pareto(
        project_id: str,
        run_id: str,
        x_measure: str = "cost",
        y_measure: str = "quality",
        request_count: int = 1,
    ) -> dict[str, Any]:
        return await analytics_get_single_run_pareto_tool(
            project_id,
            run_id,
            x_measure,
            y_measure,
            request_count,
        )

    @server.tool(
        description=(
            "Fetch the backend's correlation matrix (run_correlations v0) for "
            "one optimization run. Requires explicit project_id and run_id; "
            "optional method (pearson/spearman) and min_sample query params. "
            "Backend auth required."
        )
    )
    async def analytics_get_correlation_matrix(
        project_id: str,
        run_id: str,
        method: str = "pearson",
        min_sample: int = 3,
    ) -> dict[str, Any]:
        return await analytics_get_correlation_matrix_tool(
            project_id,
            run_id,
            method,
            min_sample,
        )

    @server.tool(
        description=(
            "Fetch the backend's ranked configuration leaderboard "
            "(run_leaderboard v0) for one optimization run. Requires explicit "
            "project_id and run_id; optional objective, weights, constraints, "
            "request_count, and limit query params. weights/constraints are "
            "JSON-object values on the wire. Backend auth required."
        )
    )
    async def analytics_get_run_leaderboard(
        project_id: str,
        run_id: str,
        objective: str = "weighted",
        weights: dict[str, object] | str | None = None,
        constraints: dict[str, object] | str | None = None,
        request_count: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await analytics_get_run_leaderboard_tool(
            project_id,
            run_id,
            objective,
            weights,
            constraints,
            request_count,
            limit,
        )

    @server.tool(
        description=(
            "Fetch the backend's parameter-importance insights "
            "(run_parameter_insights v0) for one optimization run. Requires "
            "explicit project_id and run_id; optional target_measure, "
            "min_trials, and top_k query params. Backend auth required."
        )
    )
    async def analytics_get_parameter_insights(
        project_id: str,
        run_id: str,
        target_measure: str = "quality",
        min_trials: int = 10,
        top_k: int = 10,
    ) -> dict[str, Any]:
        return await analytics_get_parameter_insights_tool(
            project_id,
            run_id,
            target_measure,
            min_trials,
            top_k,
        )

    @server.tool(
        description=(
            "Render a canonical analytics payload (a run_pareto or "
            "run_correlations document from the backend) to a PNG/SVG file on "
            "disk and return the path. Renders pixels from the payload's "
            "numbers; never recomputes analytics. kind must be 'run_pareto' or "
            "'run_correlations'."
        )
    )
    def analytics_render_chart(
        payload: dict[str, Any],
        kind: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        return analytics_render_chart_tool(payload, kind, output_path)

    return server


def run_stdio_server() -> None:
    """Run the Traigent analytics MCP server over stdio."""
    create_server().run("stdio")


def main() -> None:
    """Console entry point for ``traigent-analytics-mcp``."""
    try:
        run_stdio_server()
    except RuntimeError as exc:
        # Surface the optional-dependency hint cleanly rather than a traceback.
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()
