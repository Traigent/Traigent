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
    analytics_get_example_insights_tool,
    analytics_get_experiment_group_tool,
    analytics_get_parameter_insights_tool,
    analytics_get_project_overview_tool,
    analytics_get_run_decision_brief_tool,
    analytics_get_run_leaderboard_tool,
    analytics_get_run_report_tool,
    analytics_get_single_run_pareto_tool,
    analytics_list_experiment_group_configuration_runs_tool,
    analytics_list_experiment_groups_tool,
    analytics_render_chart_tool,
    auth_status_tool,
    health_check_tool,
    observability_compare_cohorts_tool,
    observability_build_change_brief_tool,
    observability_get_analysis_insights_tool,
    observability_get_issue_tool,
    observability_get_related_changes_tool,
    observability_get_tool_analysis_tool,
    observability_get_trace_slice_tool,
    observability_list_issues_tool,
    observability_search_traces_tool,
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
            "Fetch privacy-bounded example insights for one optimization run. "
            "Requires explicit project_id and run_id. The backend returns only "
            "safe-agent-projection data: coarse counts, dataset-quality "
            "buckets, safe example refs, templated recommendations, and "
            "redaction metadata. Backend auth required."
        )
    )
    async def analytics_get_example_insights(
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return await analytics_get_example_insights_tool(project_id, run_id)

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

    @server.tool(
        description=(
            "List experiment-group cohorts for a project — source-preserving "
            "groups of optimization runs keyed by (agent_id + dataset_id). A "
            "group is a browsing/aggregation view over source runs, never a "
            "merged analytics run. Optional agent_id/dataset_id filters. "
            "Backend auth required."
        )
    )
    async def analytics_list_experiment_groups(
        project_id: str,
        agent_id: str | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        return await analytics_list_experiment_groups_tool(
            project_id,
            agent_id,
            dataset_id,
        )

    @server.tool(
        description=(
            "Fetch one experiment group's summary (counts, status rollup, and "
            "source experiments) by group_id. The group is a source-preserving "
            "cohort keyed by (agent_id + dataset_id); join on source ids, "
            "never on config hash. Backend auth required."
        )
    )
    async def analytics_get_experiment_group(
        project_id: str,
        group_id: str,
    ) -> dict[str, Any]:
        return await analytics_get_experiment_group_tool(project_id, group_id)

    @server.tool(
        description=(
            "Fetch the group's aggregated multi-run results table: one row per "
            "configuration-run across the cohort's runs, each carrying "
            "configuration, measures, status, trial_number, and the source "
            "ids (experiment_run_id, configuration_run_id, experiment_id). "
            "Rows remain individual source runs — grouped, not merged or "
            "deduped by config hash. Backend auth required."
        )
    )
    async def analytics_list_experiment_group_configuration_runs(
        project_id: str,
        group_id: str,
    ) -> dict[str, Any]:
        return await analytics_list_experiment_group_configuration_runs_tool(
            project_id,
            group_id,
        )

    @server.tool(
        description=(
            "Search traces in an explicit window of at most 31 days. Returns "
            "only bounded content-free summaries; names, user/session IDs, "
            "tags, metadata, inputs, outputs, comments, and error text are excluded."
        )
    )
    async def observability_search_traces(
        project_id: str,
        start_time: str,
        end_time: str,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
        environment: str | None = None,
        release: str | None = None,
    ) -> dict[str, Any]:
        return await observability_search_traces_tool(
            project_id,
            start_time,
            end_time,
            page,
            per_page,
            status,
            environment,
            release,
        )

    @server.tool(
        description=(
            "List durable recurring observability issues for an explicit project. "
            "Results are paginated to at most 100 aggregate-safe issue records."
        )
    )
    async def observability_list_issues(
        project_id: str,
        page: int = 1,
        per_page: int = 50,
        state: str | None = None,
        detector_family: str | None = None,
        severity: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        return await observability_list_issues_tool(
            project_id,
            page,
            per_page,
            state,
            detector_family,
            severity,
            search,
        )

    @server.tool(
        description=(
            "Fetch one durable issue with bounded occurrence evidence. Evidence "
            "contains immutable trace/span references and closed signal taxonomy, "
            "never copied trace content."
        )
    )
    async def observability_get_issue(
        project_id: str,
        issue_id: str,
        occurrence_page: int = 1,
        occurrences_per_page: int = 50,
    ) -> dict[str, Any]:
        return await observability_get_issue_tool(
            project_id,
            issue_id,
            occurrence_page,
            occurrences_per_page,
        )

    @server.tool(
        description=(
            "Fetch a cursor-bounded content-free trace slice of at most 500 "
            "observations. No raw content option exists in this MCP."
        )
    )
    async def observability_get_trace_slice(
        project_id: str,
        trace_id: str,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return await observability_get_trace_slice_tool(
            project_id, trace_id, cursor, limit
        )

    @server.tool(
        description=(
            "Fetch aggregate tool attempts, failures, retries, fallbacks, latency, "
            "and cost for a window of at most 31 days. This reports execution "
            "outcomes and does not claim the selected tool was semantically correct."
        )
    )
    async def observability_get_tool_analysis(
        project_id: str,
        start_time: str,
        end_time: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await observability_get_tool_analysis_tool(
            project_id, start_time, end_time, limit
        )

    @server.tool(
        description=(
            "Fetch content-free structural conformance facts and deterministic "
            "validation recommendations for a window of at most 31 days. The "
            "baseline is the dominant observed variant, not an intended workflow, "
            "and recommendations are hypotheses to test rather than causal claims."
        )
    )
    async def observability_get_analysis_insights(
        project_id: str,
        start_time: str,
        end_time: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        return await observability_get_analysis_insights_tool(
            project_id, start_time, end_time, limit
        )

    @server.tool(
        description=(
            "Compare bounded reference and comparison trace cohorts over selected "
            "aggregate metrics. Each cohort requires an explicit window of at most "
            "31 days and permits only structured execution-context filters."
        )
    )
    async def observability_compare_cohorts(
        project_id: str,
        reference: dict[str, object],
        comparison: dict[str, object],
        metrics: list[str],
    ) -> dict[str, Any]:
        return await observability_compare_cohorts_tool(
            project_id, reference, comparison, metrics
        )

    @server.tool(
        description=(
            "Fetch content-free platform lineage related to a trace, including "
            "releases, revisions, configurations, experiments, and interventions. "
            "Relationships are provenance links, not causal attribution."
        )
    )
    async def observability_get_related_changes(
        project_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        return await observability_get_related_changes_tool(project_id, trace_id)

    @server.tool(
        description=(
            "Build a deterministic, privacy-bounded change brief for one trace. "
            "Composes structural analysis, a bounded content-free trace slice, "
            "lineage, aggregate tool outcomes, and an optional before/after cohort "
            "comparison. Hypotheses are templated and explicitly non-causal; the "
            "tool never mutates traces, issues, code, or configuration."
        )
    )
    async def observability_build_change_brief(
        project_id: str,
        trace_id: str,
        start_time: str,
        end_time: str,
        reference: dict[str, object] | None = None,
        comparison: dict[str, object] | None = None,
        metrics: list[str] | None = None,
        trace_limit: int = 200,
        tool_limit: int = 50,
        insights_limit: int = 20,
    ) -> dict[str, Any]:
        return await observability_build_change_brief_tool(
            project_id,
            trace_id,
            start_time,
            end_time,
            reference,
            comparison,
            metrics,
            trace_limit,
            tool_limit,
            insights_limit,
        )

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
