"""Query Backend for stored cost records.

This module provides functionality to:
1. Fetch cost records from BE database via API
2. Compare BE stored costs with SDK computed costs
3. Generate evidence reports for verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


@dataclass
class BECostRecord:
    """Cost record from Backend database."""

    experiment_id: str
    run_id: str
    configuration_runs: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    query_timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "configuration_runs": self.configuration_runs,
            "total_cost": self.total_cost,
            "query_timestamp": self.query_timestamp.isoformat(),
        }


def fetch_be_cost_records(
    experiment_id: str,
    run_id: str,
    be_base_url: str = "http://localhost:8000",
    auth_token: str | None = None,
) -> BECostRecord:
    """
    Fetch cost records from BE database via API.

    Args:
        experiment_id: The experiment ID
        run_id: The run ID
        be_base_url: Backend base URL
        auth_token: Optional auth token for API access

    Returns:
        BECostRecord with cost data from Backend
    """
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        response = requests.get(
            f"{be_base_url}/api/v1/experiments/{experiment_id}/runs/{run_id}",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            config_runs = []
            total_cost = 0.0

            for config_run in data.get("configuration_runs", []):
                measures = config_run.get("measures", {})
                cost = measures.get("cost", 0.0)
                total_cost += cost

                config_runs.append(
                    {
                        "id": config_run.get("id"),
                        "measures": measures,
                        "model": config_run.get("configuration", {}).get("model"),
                        "prompt_tokens": measures.get("prompt_tokens"),
                        "completion_tokens": measures.get("completion_tokens"),
                        "cost": cost,
                    }
                )

            return BECostRecord(
                experiment_id=experiment_id,
                run_id=run_id,
                configuration_runs=config_runs,
                total_cost=total_cost,
                raw_response=data,
            )

        return BECostRecord(
            experiment_id=experiment_id,
            run_id=run_id,
            raw_response={
                "error": f"HTTP {response.status_code}",
                "body": response.text,
            },
        )

    except requests.exceptions.RequestException as e:
        return BECostRecord(
            experiment_id=experiment_id,
            run_id=run_id,
            raw_response={"error": str(e)},
        )


def fetch_be_experiment_summary(
    experiment_id: str,
    be_base_url: str = "http://localhost:8000",
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Fetch experiment summary including all runs.

    Args:
        experiment_id: The experiment ID
        be_base_url: Backend base URL
        auth_token: Optional auth token

    Returns:
        Dictionary with experiment summary data
    """
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        response = requests.get(
            f"{be_base_url}/api/v1/experiments/{experiment_id}",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

        return {"error": f"HTTP {response.status_code}", "body": response.text}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def compare_sdk_vs_be_costs(
    sdk_costs: list[dict[str, Any]], be_record: BECostRecord, tolerance: float = 0.001
) -> dict[str, Any]:
    """
    Compare SDK computed costs with BE stored costs.

    Args:
        sdk_costs: List of SDK computed costs
        be_record: Backend cost record
        tolerance: Tolerance for cost comparison (default 0.1%)

    Returns:
        Comparison results with matches and discrepancies
    """
    results = {
        "matches": [],
        "discrepancies": [],
        "sdk_only": [],
        "be_only": [],
        "total_sdk_cost": sum(c.get("cost", 0) for c in sdk_costs),
        "total_be_cost": be_record.total_cost,
    }

    # Create lookup by model for SDK costs
    sdk_by_model = {}
    for c in sdk_costs:
        model = c.get("model", "unknown")
        if model not in sdk_by_model:
            sdk_by_model[model] = []
        sdk_by_model[model].append(c)

    # Create lookup by model for BE costs
    be_by_model = {}
    for c in be_record.configuration_runs:
        model = c.get("model", "unknown")
        if model not in be_by_model:
            be_by_model[model] = []
        be_by_model[model].append(c)

    # Compare costs by model
    all_models = set(sdk_by_model.keys()) | set(be_by_model.keys())
    for model in all_models:
        sdk_model_costs = sdk_by_model.get(model, [])
        be_model_costs = be_by_model.get(model, [])

        if not sdk_model_costs:
            results["be_only"].extend(be_model_costs)
            continue

        if not be_model_costs:
            results["sdk_only"].extend(sdk_model_costs)
            continue

        # Compare totals for this model
        sdk_total = sum(c.get("cost", 0) for c in sdk_model_costs)
        be_total = sum(c.get("cost", 0) for c in be_model_costs)

        if be_total == 0:
            matches = sdk_total == 0
        else:
            matches = abs(sdk_total - be_total) / be_total < tolerance

        comparison = {
            "model": model,
            "sdk_cost": sdk_total,
            "be_cost": be_total,
            "difference": sdk_total - be_total,
            "matches": matches,
        }

        if matches:
            results["matches"].append(comparison)
        else:
            results["discrepancies"].append(comparison)

    # Overall match
    results["all_match"] = (
        len(results["discrepancies"]) == 0 and len(results["sdk_only"]) == 0
    )

    return results


def generate_be_evidence_report(
    records: list[BECostRecord],
    comparisons: list[dict[str, Any]] | None = None,
    output_path: str = "be_evidence.md",
) -> None:
    """
    Generate markdown report with BE cost evidence.

    Args:
        records: List of BE cost records
        comparisons: Optional list of SDK vs BE comparisons
        output_path: Path to save the report
    """
    lines = [
        "# Backend Cost Evidence",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"## Experiment: {record.experiment_id}",
                f"### Run: {record.run_id}",
                "",
                f"**Total Cost:** ${record.total_cost:.6f}",
                f"**Queried:** {record.query_timestamp.isoformat()}",
                "",
            ]
        )

        if record.configuration_runs:
            lines.extend(
                [
                    "| Config ID | Model | Tokens (in/out) | Cost |",
                    "|-----------|-------|-----------------|------|",
                ]
            )
            for config in record.configuration_runs:
                prompt = config.get("prompt_tokens", "N/A")
                completion = config.get("completion_tokens", "N/A")
                lines.append(
                    f"| {config.get('id', 'N/A')[:8]}... | "
                    f"{config.get('model', 'N/A')} | "
                    f"{prompt}/{completion} | "
                    f"${config.get('cost', 0):.6f} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    if comparisons:
        lines.extend(
            [
                "## SDK vs BE Comparisons",
                "",
            ]
        )
        for comp in comparisons:
            status = "✅" if comp.get("all_match") else "❌"
            lines.extend(
                [
                    f"### Comparison {status}",
                    f"- SDK Total: ${comp.get('total_sdk_cost', 0):.6f}",
                    f"- BE Total: ${comp.get('total_be_cost', 0):.6f}",
                    "",
                ]
            )

            if comp.get("discrepancies"):
                lines.append("**Discrepancies:**")
                for d in comp["discrepancies"]:
                    lines.append(
                        f"- {d['model']}: SDK=${d['sdk_cost']:.6f}, "
                        f"BE=${d['be_cost']:.6f} (diff=${d['difference']:.6f})"
                    )
                lines.append("")

    Path(output_path).write_text("\n".join(lines))


def check_be_available(be_base_url: str = "http://localhost:8000") -> bool:
    """Check if BE is available at the given URL."""
    try:
        response = requests.get(f"{be_base_url}/api/v1/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
