"""Capture Frontend evidence of cost logging.

This module provides functionality to:
1. Fetch cost data from FE API endpoints
2. Capture screenshots of cost displays (optional, requires playwright)
3. Generate markdown reports with FE evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


@dataclass
class FECostSnapshot:
    """Snapshot of cost data from Frontend."""

    experiment_id: str
    run_id: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    snapshot_timestamp: datetime = field(default_factory=datetime.utcnow)
    screenshot_path: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "trials": self.trials,
            "total_cost": self.total_cost,
            "snapshot_timestamp": self.snapshot_timestamp.isoformat(),
            "screenshot_path": self.screenshot_path,
        }


def capture_fe_cost_data(
    experiment_id: str,
    run_id: str,
    fe_base_url: str = "http://localhost:3000",
    auth_token: str | None = None,
) -> FECostSnapshot:
    """
    Fetch cost data from FE API endpoint.

    Args:
        experiment_id: The experiment ID
        run_id: The run ID
        fe_base_url: Frontend base URL
        auth_token: Optional auth token for API access

    Returns:
        FECostSnapshot with cost data from Frontend
    """
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        # Try the runs endpoint first
        response = requests.get(
            f"{fe_base_url}/api/experiments/{experiment_id}/runs/{run_id}",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            trials = []
            total_cost = 0.0

            # Extract cost data from configuration runs
            for config_run in data.get("configuration_runs", []):
                measures = config_run.get("measures", {})
                cost = measures.get("cost", 0.0)
                total_cost += cost
                trials.append(
                    {
                        "trial_id": config_run.get("id"),
                        "model": config_run.get("configuration", {}).get("model"),
                        "cost_usd": cost,
                        "timestamp": config_run.get("created_at"),
                    }
                )

            return FECostSnapshot(
                experiment_id=experiment_id,
                run_id=run_id,
                trials=trials,
                total_cost=total_cost,
                raw_response=data,
            )

        # If that fails, try the costs endpoint
        response = requests.get(
            f"{fe_base_url}/api/experiments/{experiment_id}/runs/{run_id}/costs",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            return FECostSnapshot(
                experiment_id=experiment_id,
                run_id=run_id,
                trials=data.get("trials", []),
                total_cost=data.get("total_cost", 0.0),
                raw_response=data,
            )

        return FECostSnapshot(
            experiment_id=experiment_id,
            run_id=run_id,
            raw_response={
                "error": f"HTTP {response.status_code}",
                "body": response.text,
            },
        )

    except requests.exceptions.RequestException as e:
        return FECostSnapshot(
            experiment_id=experiment_id,
            run_id=run_id,
            raw_response={"error": str(e)},
        )


def capture_fe_screenshot(
    experiment_id: str,
    run_id: str,
    fe_base_url: str = "http://localhost:3000",
    output_path: str | None = None,
) -> str | None:
    """
    Capture a screenshot of the FE cost display.

    Args:
        experiment_id: The experiment ID
        run_id: The run ID
        fe_base_url: Frontend base URL
        output_path: Path to save screenshot (default: auto-generated)

    Returns:
        Path to screenshot file, or None if failed
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed - skipping screenshot capture")
        return None

    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"fe_snapshot_{experiment_id}_{run_id}_{timestamp}.png"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the experiment run page
            url = f"{fe_base_url}/experiments/{experiment_id}/runs/{run_id}"
            page.goto(url)

            # Wait for content to load
            page.wait_for_load_state("networkidle")

            # Take screenshot
            page.screenshot(path=output_path, full_page=True)
            browser.close()

            return output_path

    except Exception as e:
        print(f"Failed to capture screenshot: {e}")
        return None


def generate_fe_snapshot_report(
    snapshots: list[FECostSnapshot], output_path: str
) -> None:
    """
    Generate markdown report with FE cost evidence.

    Args:
        snapshots: List of FE cost snapshots
        output_path: Path to save the report
    """
    lines = [
        "# Frontend Cost Snapshots",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]

    for snapshot in snapshots:
        lines.extend(
            [
                f"## Experiment: {snapshot.experiment_id}",
                f"### Run: {snapshot.run_id}",
                "",
                f"**Total Cost:** ${snapshot.total_cost:.6f}",
                f"**Captured:** {snapshot.snapshot_timestamp.isoformat()}",
                "",
            ]
        )

        if snapshot.trials:
            lines.extend(
                [
                    "| Trial ID | Model | Cost |",
                    "|----------|-------|------|",
                ]
            )
            for trial in snapshot.trials:
                lines.append(
                    f"| {trial.get('trial_id', 'N/A')} | "
                    f"{trial.get('model', 'N/A')} | "
                    f"${trial.get('cost_usd', 0):.6f} |"
                )
            lines.append("")

        if snapshot.screenshot_path:
            lines.extend(
                [
                    "### Screenshot",
                    f"![FE Screenshot]({snapshot.screenshot_path})",
                    "",
                ]
            )

        lines.append("---")
        lines.append("")

    Path(output_path).write_text("\n".join(lines))


def check_fe_available(fe_base_url: str = "http://localhost:3000") -> bool:
    """Check if FE is available at the given URL."""
    try:
        response = requests.get(f"{fe_base_url}/api/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
