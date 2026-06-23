"""Local stdio MCP server for Traigent SDK tools."""

from __future__ import annotations

from typing import Any, Literal

from traigent.mcp.tools import (
    auth_status_tool,
    detect_tvars_tool,
    estimate_cost_tool,
    export_evidence_tool,
    get_optimization_plan_tool,
    get_results_tool,
    list_recommendation_agent_types_tool,
    recommend_configuration_space_tool,
    run_optimization_tool,
    scaffold_eval_tool,
    validate_dataset_tool,
)

_MCP_INSTALL_MESSAGE = (
    "The optional MCP dependency is not installed. "
    "Install it with: pip install 'traigent[mcp]'"
)


def create_server() -> Any:
    """Create the local Traigent MCP server.

    The optional ``mcp`` package is imported lazily so ``traigent mcp --help``
    still works in environments that have not installed ``traigent[mcp]``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - covered by CLI test
        raise RuntimeError(_MCP_INSTALL_MESSAGE) from exc

    server = FastMCP(
        "traigent",
        instructions=(
            "Local-first Traigent SDK MCP server. Tools run against local code "
            "and local auth storage; API keys are never returned."
        ),
    )

    @server.tool(
        description=(
            "Return local Traigent auth status from existing credential storage. "
            "API key output is masked to prefix and last4 only. No network call is "
            "made unless check=true."
        )
    )
    async def auth_status(check: bool = False) -> dict[str, Any]:
        return await auth_status_tool(check=check)

    @server.tool(
        description=(
            "List valid local public catalog agent/task types for configuration "
            "space recommendations. No network call."
        )
    )
    def list_recommendation_agent_types() -> dict[str, Any]:
        return list_recommendation_agent_types_tool()

    @server.tool(
        description=(
            "Return the versioned public catalog recommendation response for an "
            "agent/task type. Optional min_impact and min_confidence filters "
            "accept low, medium, or high. No network call."
        )
    )
    def recommend_configuration_space(
        agent_type: str,
        min_impact: Literal["low", "medium", "high"] | None = None,
        min_confidence: Literal["low", "medium", "high"] | None = None,
    ) -> dict[str, Any]:
        return recommend_configuration_space_tool(
            agent_type=agent_type,
            min_impact=min_impact,
            min_confidence=min_confidence,
        )

    @server.tool(
        description=(
            "Detect tuned variable candidates in a Python file using the local "
            "TunedVariableDetector. Path security: file_path must resolve under "
            "the MCP server's current working directory."
        )
    )
    def detect_tvars(
        file_path: str,
        function_name: str | None = None,
    ) -> dict[str, Any]:
        return detect_tvars_tool(file_path=file_path, function_name=function_name)

    @server.tool(
        description=(
            "Scaffold a reviewable draft JSONL evaluation dataset and a local "
            "Traigent approval manifest. Generated examples are draft evidence; "
            "real optimization refuses them until a human approves or replaces "
            "the dataset."
        )
    )
    def scaffold_eval(
        agent_type: str = "agent",
        output_path: str = "eval.draft.jsonl",
        example_count: int = 3,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return scaffold_eval_tool(
            agent_type=agent_type,
            output_path=output_path,
            example_count=example_count,
            overwrite=overwrite,
        )

    @server.tool(
        description=(
            "Validate a JSON/JSONL dataset using Validators.validate_dataset. "
            "Path security: path must resolve under TRAIGENT_DATASET_ROOT when "
            "set, otherwise under the MCP server's current working directory."
        )
    )
    def validate_dataset(path: str) -> dict[str, Any]:
        return validate_dataset_tool(path=path)

    @server.tool(
        description=(
            "Estimate optimization scale using the dry-run-first arithmetic: "
            "max_trials multiplied by dataset examples. Path security matches "
            "validate_dataset. Optional model pricing uses local SDK estimates."
        )
    )
    def estimate_cost(
        dataset_path: str,
        max_trials: int,
        model: str | None = None,
    ) -> dict[str, Any]:
        return estimate_cost_tool(
            dataset_path=dataset_path,
            max_trials=max_trials,
            model=model,
        )

    @server.tool(
        description=(
            "fetch an aggregated pre-run optimization plan from the Traigent "
            "backend; requires backend auth"
        )
    )
    async def get_optimization_plan(
        task_description: str,
        dataset_size: int,
        dataset_has_holdout: bool,
        objectives: list[str],
        max_trials: int,
        cost_limit_usd: float,
        task_type: str | None = None,
        agent_shape: str | None = None,
        weights: dict[str, float] | None = None,
        offline: bool | None = None,
    ) -> dict[str, Any]:
        return await get_optimization_plan_tool(
            task_description=task_description,
            dataset_size=dataset_size,
            dataset_has_holdout=dataset_has_holdout,
            objectives=objectives,
            max_trials=max_trials,
            cost_limit_usd=cost_limit_usd,
            task_type=task_type,
            agent_shape=agent_shape,
            weights=weights,
            offline=offline,
        )

    @server.tool(
        description=(
            "Run local optimization for @traigent.optimize functions in a Python "
            "script. Defaults to mode='mock' dry-run. Real mode spends provider "
            "tokens/money and refuses unless confirm=true and cost_limit is set. "
            "Real mode also refuses any local draft eval manifest that has not "
            "been human-approved. "
            "Path security: script_path must resolve under the current working "
            "directory. Note: this runs synchronously and blocks the MCP stdio "
            "loop for the full duration of the run (single-agent v1 limitation); "
            "real-run stdout/stderr are redacted from the response."
        )
    )
    def run_optimization(
        script_path: str | None = None,
        mode: Literal["mock", "real"] = "mock",
        confirm: bool = False,
        cost_limit: float | None = None,
        max_trials: int | None = None,
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        return run_optimization_tool(
            script_path=script_path,
            mode=mode,
            confirm=confirm,
            cost_limit=cost_limit,
            max_trials=max_trials,
            algorithm=algorithm,
        )

    @server.tool(
        description=(
            "List local optimization results from .traigent, or show one result "
            "by result_name using the existing PersistenceManager."
        )
    )
    def get_results(result_name: str | None = None) -> dict[str, Any]:
        return get_results_tool(result_name=result_name)

    @server.tool(
        description=(
            "Export a local Traigent evidence bundle as Markdown plus JSON from "
            "saved optimization results. The bundle carries dataset approval "
            "status and labels generated/unapproved evals as draft evidence."
        )
    )
    def export_evidence(
        result_name: str | None = None,
        output_dir: str = ".traigent/evidence",
    ) -> dict[str, Any]:
        return export_evidence_tool(result_name=result_name, output_dir=output_dir)

    return server


def run_stdio_server() -> None:
    """Run the local Traigent MCP server over stdio."""
    create_server().run("stdio")
