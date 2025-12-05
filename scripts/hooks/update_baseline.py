#!/usr/bin/env python3
"""
Update performance baseline after verifying improvements.
Run this after confirming that performance improvements are legitimate.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Update baseline with current performance metrics."""

    # Get repository root
    repo_root = Path(__file__).parent.parent.parent
    baseline_path = repo_root / "baselines" / "performance.json"
    backup_path = (
        repo_root
        / "baselines"
        / f"performance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Check if performance report exists
    perf_report_path = repo_root / "performance_report.json"
    if not perf_report_path.exists():
        print("❌ No performance report found. Run optimization first.")
        return 1

    # Load performance report
    with open(perf_report_path) as f:
        report = json.load(f)

    current_metrics = report.get("current_metrics", {})
    if not current_metrics:
        print("❌ No current metrics found in performance report.")
        return 1

    # Create backup of existing baseline
    if baseline_path.exists():
        print(f"📦 Creating backup: {backup_path.name}")
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        with open(backup_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

    # Update baseline
    print("📝 Updating baseline with current metrics:")
    for key, value in current_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Add metadata
    current_metrics["updated_at"] = datetime.now().isoformat()
    current_metrics["updated_by"] = "update_baseline.py"

    # Save new baseline
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(current_metrics, f, indent=2)

    print(f"✅ Baseline updated successfully at {baseline_path}")
    print("   Previous baseline backed up for safety")

    return 0


if __name__ == "__main__":
    sys.exit(main())
