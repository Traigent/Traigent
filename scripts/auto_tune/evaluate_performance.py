#!/usr/bin/env python3
"""
Evaluate optimization performance against baseline.
Compares current results with baseline and generates metrics.
"""

import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_optimization_results(
    path: str = "optimization_results/latest.json",
) -> Dict[str, Any]:
    """Load optimization results from Traigent."""
    if not Path(path).exists():
        print(f"⚠️  No optimization results found at {path}")
        return {}

    with open(path) as f:
        return json.load(f)


def load_baseline(baseline_path: str) -> Optional[Dict[str, Any]]:
    """Load baseline metrics for comparison."""
    path = Path(baseline_path)
    if not path.exists():
        print(f"⚠️  No baseline found at {baseline_path}")
        return None

    with open(path) as f:
        return json.load(f)


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate performance metrics from optimization results.
    """
    metrics = {
        "accuracy": 0.0,
        "avg_latency": 0.0,
        "total_cost": 0.0,
        "tokens_used": 0,
        "trials_completed": 0,
        "best_score": 0.0,
    }

    if not results:
        return metrics

    # Extract trials data
    trials = results.get("trials", [])
    if not trials:
        return metrics

    # Calculate metrics
    scores = []
    latencies = []
    costs = []
    tokens = []

    for trial in trials:
        if "score" in trial:
            scores.append(trial["score"])
        if "latency" in trial:
            latencies.append(trial["latency"])
        if "cost" in trial:
            costs.append(trial["cost"])
        if "tokens" in trial:
            tokens.append(trial["tokens"])

    if scores:
        metrics["accuracy"] = statistics.mean(scores)
        metrics["best_score"] = max(scores)

    if latencies:
        metrics["avg_latency"] = statistics.mean(latencies)

    if costs:
        metrics["total_cost"] = sum(costs)

    if tokens:
        metrics["tokens_used"] = sum(tokens)

    metrics["trials_completed"] = len(trials)

    # Add best configuration info
    if results.get("best_config"):
        metrics["best_model"] = results["best_config"].get("model", "unknown")
        metrics["best_provider"] = results["best_config"].get("provider", "unknown")

    return metrics


def compare_with_baseline(
    current: Dict[str, float], baseline: Dict[str, float], threshold: float
) -> Dict[str, Any]:
    """
    Compare current metrics with baseline.

    Returns:
        Comparison results with regression/improvement flags
    """
    comparison = {
        "metrics": {},
        "has_regression": False,
        "has_improvement": False,
        "summary": [],
    }

    for metric in ["accuracy", "avg_latency", "total_cost"]:
        if metric not in current or metric not in baseline:
            continue

        current_val = current[metric]
        baseline_val = baseline[metric]

        if baseline_val == 0:
            change_pct = 0
        else:
            change_pct = ((current_val - baseline_val) / baseline_val) * 100

        # Determine if it's good or bad based on metric type
        is_improvement = False
        if metric == "accuracy":
            is_improvement = change_pct > threshold * 100
            is_regression = change_pct < -threshold * 100
        else:  # For latency and cost, lower is better
            is_improvement = change_pct < -threshold * 100
            is_regression = change_pct > threshold * 100

        comparison["metrics"][metric] = {
            "baseline": baseline_val,
            "current": current_val,
            "change_pct": change_pct,
            "is_improvement": is_improvement,
            "is_regression": is_regression,
        }

        if is_regression:
            comparison["has_regression"] = True
            comparison["summary"].append(
                f"❌ {metric}: {baseline_val:.3f} → {current_val:.3f} ({change_pct:+.1f}%)"
            )
        elif is_improvement:
            comparison["has_improvement"] = True
            comparison["summary"].append(
                f"✅ {metric}: {baseline_val:.3f} → {current_val:.3f} ({change_pct:+.1f}%)"
            )
        else:
            comparison["summary"].append(
                f"➡️  {metric}: {baseline_val:.3f} → {current_val:.3f} ({change_pct:+.1f}%)"
            )

    return comparison


def generate_plots_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for performance plots."""
    plots_data = {"trials": [], "scores": [], "latencies": [], "costs": []}

    trials = results.get("trials", [])
    for i, trial in enumerate(trials):
        plots_data["trials"].append(i + 1)
        plots_data["scores"].append(trial.get("score", 0))
        plots_data["latencies"].append(trial.get("latency", 0))
        plots_data["costs"].append(trial.get("cost", 0))

    return plots_data


def main():
    """Main evaluation pipeline."""
    print("📊 Evaluating optimization performance...")

    # Load parameters
    params = load_params()
    baseline_path = params["evaluate"]["baseline_path"]
    threshold = params["evaluate"]["threshold"]

    # Load optimization results
    results = load_optimization_results()
    if not results:
        print("❌ No optimization results to evaluate")
        return 1

    # Calculate current metrics
    current_metrics = calculate_metrics(results)
    print(f"Current metrics: {json.dumps(current_metrics, indent=2)}")

    # Load and compare with baseline
    baseline = load_baseline(baseline_path)

    if baseline:
        comparison = compare_with_baseline(current_metrics, baseline, threshold)
        print("\n📈 Performance Comparison:")
        for summary in comparison["summary"]:
            print(f"  {summary}")
    else:
        # No baseline, create one
        print("Creating initial baseline...")
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(current_metrics, f, indent=2)
        comparison = {
            "metrics": {},
            "has_regression": False,
            "has_improvement": False,
            "summary": ["Initial baseline created"],
        }

    # Generate performance report
    report = {
        "current_metrics": current_metrics,
        "baseline": baseline,
        "comparison": comparison if baseline else None,
        "metadata": {
            "timestamp": os.environ.get("CI_COMMIT_TIMESTAMP", "local"),
            "commit": os.environ.get("CI_COMMIT_SHA", "local"),
            "branch": os.environ.get("CI_COMMIT_BRANCH", "local"),
        },
    }

    # Save performance report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate plots data
    plots_dir = Path("performance_plots")
    plots_dir.mkdir(exist_ok=True)

    plots_data = generate_plots_data(results)
    with open(plots_dir / "data.json", "w") as f:
        json.dump(plots_data, f, indent=2)

    print("✅ Performance evaluation complete")

    # Return non-zero if regression detected
    if comparison.get("has_regression"):
        print("⚠️  Performance regression detected!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
