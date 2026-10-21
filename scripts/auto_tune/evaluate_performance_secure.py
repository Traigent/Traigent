#!/usr/bin/env python3
"""
Secure performance evaluation with comprehensive error handling.
Compares optimization results against baseline with security controls.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from security_utils import (
    AuditLogger,
    CostController,
    retry_with_backoff,
    safe_file_read,
    safe_file_write,
    sanitize_input,
    setup_logging,
    timeout,
    validate_json_schema,
    validate_path,
)

# Initialize logging and audit
logger = setup_logging(__name__, "evaluate_performance.log")
audit_logger = AuditLogger("evaluate_performance_audit.jsonl")
cost_controller = CostController()


def load_params() -> dict[str, Any]:
    """Load and validate parameters from params.yaml."""
    try:
        params_path = Path("params.yaml")
        if not validate_path(params_path):
            raise ValueError("Invalid params.yaml path")

        content = safe_file_read(params_path)
        if not content:
            raise FileNotFoundError("params.yaml not found")

        params = yaml.safe_load(content)
        logger.info("Parameters loaded successfully")
        return params

    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise


@retry_with_backoff(max_attempts=3)
def load_optimization_results(
    path: str = "optimization_results/latest.json",
) -> dict[str, Any]:
    """Load optimization results with validation and retry."""
    results_path = Path(path)

    if not validate_path(results_path):
        logger.error(f"Invalid results path: {results_path}")
        return {}

    if not results_path.exists():
        logger.warning(f"No optimization results found at {results_path}")
        return {}

    try:
        with timeout(10):
            content = safe_file_read(results_path)
            if not content:
                return {}

            data = json.loads(content)

            # Validate structure
            if not isinstance(data, dict):
                logger.error("Invalid results format")
                return {}

            logger.info(f"Loaded optimization results from {results_path}")
            return data

    except TimeoutError:
        logger.error("Timeout loading optimization results")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse results JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return {}


def load_baseline(baseline_path: str) -> dict[str, Any] | None:
    """Load baseline metrics with validation."""
    path = Path(baseline_path)

    if not validate_path(path):
        logger.error(f"Invalid baseline path: {path}")
        return None

    if not path.exists():
        logger.info(f"No baseline found at {baseline_path}")
        return None

    try:
        content = safe_file_read(path)
        if not content:
            return None

        data = json.loads(content)

        # Validate baseline structure
        required_keys = ["accuracy", "avg_latency", "cost"]
        if not validate_json_schema(data, required_keys):
            logger.warning("Baseline missing required keys")
            return None

        logger.info("Baseline loaded successfully")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse baseline JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading baseline: {e}")
        return None


def measured_cost(trial: dict[str, Any]) -> float | None:
    """A trial's cost only when it was measured; unknown cost is never 0."""
    status = trial.get("cost_status")
    cost = trial.get("cost")
    if status not in (None, "measured"):
        return None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost)


def known_cost_floor(trial: dict[str, Any]) -> float | None:
    """A positive, finite exported ``cost_lower_bound`` for an unmeasured trial."""
    floor = trial.get("cost_lower_bound")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        return None
    if not math.isfinite(floor) or floor <= 0:
        return None
    return float(floor)


def aggregate_costs(trials: list[Any]) -> dict[str, Any]:
    """Exact total only when every trial is measured; else a lower bound.

    ``total_cost`` is None when any trial is unmeasured. ``cost_lower_bound_total``
    is measured spend plus the known floors of unmeasured trials, or None when
    nothing is known. ``cost_trials_unbounded`` counts unmeasured trials with no
    floor (spend we cannot bound at all).
    """
    measured: list[float] = []
    floors: list[float] = []
    unbounded = 0
    for trial in trials:
        if not isinstance(trial, dict):
            # A malformed entry has unknown cost: unmeasured and unbounded.
            unbounded += 1
            continue
        cost = measured_cost(trial)
        if cost is not None:
            measured.append(cost)
            continue
        floor = known_cost_floor(trial)
        if floor is None:
            unbounded += 1
        else:
            floors.append(floor)
    unmeasured = len(floors) + unbounded
    known = measured + floors
    return {
        "total_cost": sum(measured) if measured and not unmeasured else None,
        "cost_lower_bound_total": sum(known) if known else None,
        "cost_trials_measured": len(measured),
        "cost_trials_unmeasured": unmeasured,
        "cost_trials_unbounded": unbounded,
    }


def budget_status(metrics: dict[str, Any], max_budget: float) -> str:
    """``exceeded`` / ``within`` / ``unverifiable``; never passes on unknown spend."""
    total = metrics.get("total_cost")
    if isinstance(total, (int, float)):
        return "exceeded" if total > max_budget else "within"
    floor = metrics.get("cost_lower_bound_total")
    if isinstance(floor, (int, float)) and floor > max_budget:
        return "exceeded"
    return "unverifiable"


def budget_remaining(metrics: dict[str, Any], max_budget: float) -> float | None:
    """Remaining budget from the snapshot; None unless the total is exact."""
    total = metrics.get("total_cost")
    if isinstance(total, (int, float)):
        return max(0.0, max_budget - total)
    return None


def budget_remaining_at_most(
    metrics: dict[str, Any], max_budget: float
) -> float | None:
    """Upper bound on remaining budget from the known floor, else None."""
    known = metrics.get("total_cost")
    if not isinstance(known, (int, float)):
        known = metrics.get("cost_lower_bound_total")
    if isinstance(known, (int, float)):
        return max(0.0, max_budget - known)
    return None


def calculate_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Calculate performance metrics with validation."""
    metrics = {
        "accuracy": 0.0,
        "avg_latency": 0.0,
        "total_cost": None,  # None = unmeasured, never 0
        "cost_lower_bound_total": None,  # measured + known floors
        "cost_trials_measured": 0,
        "cost_trials_unmeasured": 0,
        "cost_trials_unbounded": 0,
        "tokens_used": 0,
        "trials_completed": 0,
        "best_score": 0.0,
        "confidence": 0.0,
    }

    if not results:
        return metrics

    # Extract trials data
    trials = results.get("trials", [])
    if not trials:
        logger.warning("No trials found in results")
        return metrics

    # Validate and extract metrics
    scores, latencies, tokens = [], [], []

    for i, trial in enumerate(trials):
        try:
            # Validate trial structure
            if not isinstance(trial, dict):
                logger.warning(f"Invalid trial format at index {i}")
                continue

            # Extract with validation
            if "score" in trial and isinstance(trial["score"], (int, float)):
                scores.append(float(trial["score"]))

            if "latency" in trial and isinstance(trial["latency"], (int, float)):
                latencies.append(float(trial["latency"]))

            if "tokens" in trial and isinstance(trial["tokens"], int):
                tokens.append(trial["tokens"])

        except Exception as e:
            logger.warning(f"Error processing trial {i}: {e}")
            continue

    # Calculate aggregated metrics
    if scores:
        metrics["accuracy"] = statistics.mean(scores)
        metrics["best_score"] = max(scores)
        metrics["confidence"] = (
            1.0 - statistics.stdev(scores) if len(scores) > 1 else 0.5
        )

    if latencies:
        metrics["avg_latency"] = statistics.mean(latencies)

    # Read-only accounting: the snapshot is evaluated, never added to persisted
    # spend (CostController.track_spending is additive, so re-evaluating the same
    # results would double-count).
    metrics.update(aggregate_costs(trials))

    if tokens:
        metrics["tokens_used"] = sum(tokens)

    metrics["trials_completed"] = len(trials)

    # Add best configuration info
    if results.get("best_config"):
        metrics["best_model"] = sanitize_input(
            results["best_config"].get("model", "unknown")
        )
        metrics["best_provider"] = sanitize_input(
            results["best_config"].get("provider", "unknown")
        )

    logger.info(f"Calculated metrics for {len(trials)} trials")
    return metrics


def compare_with_baseline(
    current: dict[str, float], baseline: dict[str, float], threshold: float
) -> dict[str, Any]:
    """Compare metrics with baseline including statistical significance."""
    comparison = {
        "metrics": {},
        "has_regression": False,
        "has_improvement": False,
        "summary": [],
        "confidence": 0.0,
    }

    # Validate threshold
    if not 0.01 <= threshold <= 0.5:
        logger.warning(f"Invalid threshold {threshold}, using 0.05")
        threshold = 0.05

    for metric in ["accuracy", "avg_latency", "total_cost"]:
        if metric not in current or metric not in baseline:
            continue

        if current[metric] is None or baseline[metric] is None:
            comparison["summary"].append(f"❔ {metric}: unmeasured, not compared")
            continue
        if metric == "total_cost" and (
            current.get("cost_trials_unmeasured")
            or baseline.get("cost_trials_unmeasured")
        ):
            # A partial sum is a floor, not a total: do not compare it.
            comparison["summary"].append(
                f"❔ {metric}: some trials unmeasured, not compared"
            )
            continue
        current_val = float(current[metric])
        baseline_val = float(baseline[metric])

        # Prevent division by zero
        if baseline_val == 0:
            if current_val == 0:
                change_pct = 0
            else:
                change_pct = 100 if current_val > 0 else -100
        else:
            change_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100

        # Determine improvement/regression based on metric type
        is_improvement = False
        is_regression = False

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
            "significant": abs(change_pct) > threshold * 100,
        }

        # Build summary
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

    # Calculate overall confidence
    if "confidence" in current:
        comparison["confidence"] = current["confidence"]

    return comparison


def generate_plots_data(results: dict[str, Any]) -> dict[str, Any]:
    """Generate data for performance plots with validation."""
    plots_data = {
        "trials": [],
        "scores": [],
        "latencies": [],
        "costs": [],
        "cumulative_cost": [],
    }

    trials = results.get("trials", [])
    cumulative_cost: float | None = 0.0

    for i, trial in enumerate(trials):
        if not isinstance(trial, dict):
            continue

        plots_data["trials"].append(i + 1)
        plots_data["scores"].append(trial.get("score", 0))
        plots_data["latencies"].append(trial.get("latency", 0))
        cost = measured_cost(trial)
        plots_data["costs"].append(cost)

        # Unknown from the first unmeasured trial onward; never a made-up 0.
        if cost is None:
            cumulative_cost = None
        elif cumulative_cost is not None:
            cumulative_cost += cost
        plots_data["cumulative_cost"].append(cumulative_cost)

    return plots_data


def main():
    """Main evaluation pipeline with comprehensive error handling."""
    start_time = datetime.now(UTC)

    try:
        print("📊 Securely evaluating optimization performance...")
        logger.info("Starting performance evaluation")

        # Audit log start
        audit_logger.log_event(
            "evaluate_start",
            {
                "start_time": start_time.isoformat(),
                "remaining_budget": cost_controller.remaining_budget(),
            },
        )

        # Load parameters
        params = load_params()
        baseline_path = params["evaluate"]["baseline_path"]
        threshold = float(params["evaluate"]["threshold"])

        # Validate threshold
        if not 0 < threshold < 1:
            raise ValueError(f"Invalid threshold: {threshold}")

        # Load optimization results
        results = load_optimization_results()
        if not results:
            logger.error("No optimization results to evaluate")
            print("❌ No optimization results to evaluate")
            return 1

        # Calculate current metrics
        current_metrics = calculate_metrics(results)

        # Check budget against the exact total, else the known lower bound
        budget_state = budget_status(current_metrics, cost_controller.max_budget)
        if budget_state == "exceeded":
            spent = current_metrics["total_cost"]
            if spent is None:
                spent = current_metrics["cost_lower_bound_total"]
            logger.critical(f"Budget exceeded: at least ${spent:.2f}")
            audit_logger.log_event(
                "budget_exceeded",
                {
                    "spent_at_least": spent,
                    "exact": current_metrics["total_cost"] is not None,
                    "budget": cost_controller.max_budget,
                },
                success=False,
            )
        elif budget_state == "unverifiable":
            logger.warning("Budget unverifiable: some trial costs are unmeasured")
            print("⚠️  Budget unverifiable: some trial costs are unmeasured")
            audit_logger.log_event(
                "budget_unverifiable",
                {"budget": cost_controller.max_budget},
                success=False,
            )

        logger.info(f"Current metrics: {json.dumps(current_metrics, indent=2)}")

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

            baseline_data = current_metrics.copy()
            baseline_data["created_at"] = datetime.now(UTC).isoformat()

            safe_file_write(
                Path(baseline_path), json.dumps(baseline_data, indent=2), backup=True
            )

            comparison = {
                "metrics": {},
                "has_regression": False,
                "has_improvement": False,
                "summary": ["Initial baseline created"],
                "confidence": 0.5,
            }

        # Generate performance report
        report = {
            "current_metrics": current_metrics,
            "baseline": baseline,
            "comparison": comparison if baseline else None,
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "duration_seconds": (datetime.now(UTC) - start_time).total_seconds(),
                "commit": os.environ.get("GITHUB_SHA", "local"),
                "branch": os.environ.get("GITHUB_REF", "local"),
                "environment": os.environ.get("CI_ENVIRONMENT_NAME", "development"),
            },
            "budget": {
                "spent": current_metrics["total_cost"],
                "spent_at_least": current_metrics["cost_lower_bound_total"],
                "status": budget_state,
                "remaining": budget_remaining(
                    current_metrics, cost_controller.max_budget
                ),
                "remaining_at_most": budget_remaining_at_most(
                    current_metrics, cost_controller.max_budget
                ),
                "limit": cost_controller.max_budget,
            },
        }

        # Save performance report
        safe_file_write(
            Path("performance_report.json"), json.dumps(report, indent=2), backup=False
        )

        # Generate plots data
        plots_dir = Path("performance_plots")
        plots_dir.mkdir(exist_ok=True)

        plots_data = generate_plots_data(results)
        safe_file_write(
            plots_dir / "data.json", json.dumps(plots_data, indent=2), backup=False
        )

        print("✅ Performance evaluation complete")

        # Audit log success
        audit_logger.log_event(
            "evaluate_complete",
            {
                "metrics": current_metrics,
                "has_regression": comparison.get("has_regression", False),
                "duration_seconds": (datetime.now(UTC) - start_time).total_seconds(),
            },
        )

        # A floor already over budget fails the run
        if budget_state == "exceeded":
            return 1

        # Return non-zero if regression detected
        if comparison.get("has_regression"):
            logger.warning("Performance regression detected!")
            print("⚠️  Performance regression detected!")
            return 1

        logger.info("Performance evaluation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}")
        print(f"❌ Performance evaluation failed: {e}")

        # Audit log failure
        audit_logger.log_event(
            "evaluate_failed",
            {
                "error": str(e),
                "type": type(e).__name__,
                "duration_seconds": (datetime.now(UTC) - start_time).total_seconds(),
            },
            success=False,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
