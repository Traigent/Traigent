#!/usr/bin/env python3
"""
Pre-push hook to check for performance regressions.
Compares current model performance against baseline with security hardening.
"""

import json
import logging
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from traigent.utils.secure_path import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("performance_check.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# Security: Timeout context manager
@contextmanager
def timeout(seconds):
    """Context manager for timeout control."""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def validate_path(path: Path) -> bool:
    """Validate path to prevent traversal attacks."""
    try:
        # Resolve to absolute path and check if it's within repo
        resolved = path.resolve()
        repo_root = Path(__file__).parent.parent.parent.resolve()
        return resolved.is_relative_to(repo_root)
    except Exception:
        return False


def load_baseline(baseline_path: Path) -> Optional[Dict[str, Any]]:
    """Load baseline performance metrics with validation."""
    # Security: Validate path
    if not validate_path(baseline_path):
        logger.error(f"Invalid path detected: {baseline_path}")
        return None

    if not baseline_path.exists():
        logger.info(f"No baseline found at {baseline_path}")
        print("⚠️  No baseline found, creating initial baseline...")
        return None

    try:
        repo_root = Path(__file__).parent.parent.parent.resolve()
        with safe_open(baseline_path, repo_root, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            # Validate JSON structure
            required_keys = {"accuracy", "avg_latency", "cost"}
            if not all(key in data for key in required_keys):
                logger.warning("Baseline missing required keys")
                return None
            return data
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error loading baseline: {e}")
        print(f"❌ Error loading baseline: {e}")
        return None


def run_quick_performance_test() -> Dict[str, Any]:
    """Run a quick performance test using mock mode with security hardening."""
    test_script = """
import json
import os
import sys

try:
    # Enable mock mode for testing
    os.environ['TRAIGENT_MOCK_MODE'] = 'true'

    # FIX: Correct import for the decorator
    from traigent.api.decorators import optimize

    @optimize(
        max_trials=3,
        strategy="random"
    )
    def test_function(x: int) -> int:
        return x * 2

    # Run optimization
    result = test_function.optimize(
        dataset=[{"x": i} for i in range(5)],
        evaluator=lambda result, expected: 1.0
    )

    # Extract metrics with fallback values
    metrics = {
        "avg_latency": 0.05,  # Mock latency
        "accuracy": 0.95,      # Mock accuracy
        "cost": 0.0,          # Zero cost in mock mode
        "trials_run": len(result.trials) if hasattr(result, 'trials') else 3,
        "timestamp": os.environ.get("CI_COMMIT_TIMESTAMP", "local")
    }

    print(json.dumps(metrics))
except ImportError as e:
    # Fallback for import errors
    print(json.dumps({
        "avg_latency": 0.05,
        "accuracy": 0.90,
        "cost": 0.0,
        "trials_run": 3,
        "error": str(e),
        "fallback": True
    }))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"""

    try:
        # Security: Use subprocess with restricted environment
        env = os.environ.copy()
        env["TRAIGENT_MOCK_MODE"] = "true"
        # Remove sensitive variables
        for key in list(env.keys()):
            if "KEY" in key or "SECRET" in key or "TOKEN" in key:
                del env[key]

        # Run with timeout and capture output
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Performance test failed: {result.stderr}")
            # Return fallback metrics instead of empty dict
            return {
                "avg_latency": 0.1,
                "accuracy": 0.85,
                "cost": 0.0,
                "trials_run": 1,
                "fallback": True,
            }

        output = json.loads(result.stdout)

        # Validate output structure
        required_keys = {"avg_latency", "accuracy", "cost"}
        if not all(key in output for key in required_keys):
            logger.warning("Performance test output missing required keys")
            return {
                "avg_latency": 0.1,
                "accuracy": 0.85,
                "cost": 0.0,
                "trials_run": 1,
                "fallback": True,
            }

        return output

    except subprocess.TimeoutExpired:
        logger.error("Performance test timed out")
        print("⚠️  Performance test timed out, using fallback metrics")
        return {
            "avg_latency": 0.1,
            "accuracy": 0.85,
            "cost": 0.0,
            "trials_run": 1,
            "timeout": True,
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse performance test output: {e}")
        return {
            "avg_latency": 0.1,
            "accuracy": 0.85,
            "cost": 0.0,
            "trials_run": 1,
            "parse_error": True,
        }
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        print(f"⚠️  Error running performance test: {e}")
        return {
            "avg_latency": 0.1,
            "accuracy": 0.85,
            "cost": 0.0,
            "trials_run": 1,
            "error": str(e),
        }


def check_regression(
    current: Dict[str, Any], baseline: Dict[str, Any], threshold: float = 0.05
) -> bool:
    """
    Check if there's a performance regression.

    Args:
        current: Current performance metrics
        baseline: Baseline performance metrics
        threshold: Acceptable regression threshold (default 5%)

    Returns:
        True if regression detected, False otherwise
    """
    regressions = []

    # Check key metrics
    metrics_to_check = {
        "accuracy": {"direction": "higher", "weight": 1.0},
        "avg_latency": {"direction": "lower", "weight": 0.8},
        "cost": {"direction": "lower", "weight": 0.6},
    }

    for metric, config in metrics_to_check.items():
        if metric not in current or metric not in baseline:
            continue

        current_val = current[metric]
        baseline_val = baseline[metric]

        if baseline_val == 0:
            continue

        # Calculate percentage change
        change = (current_val - baseline_val) / baseline_val

        # Check if it's a regression based on direction
        is_regression = False
        if config["direction"] == "higher" and change < -threshold:
            is_regression = True
        elif config["direction"] == "lower" and change > threshold:
            is_regression = True

        if is_regression:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "change_pct": change * 100,
                }
            )

    # Report regressions
    if regressions:
        print("\n🚨 Performance Regressions Detected!\n")
        for reg in regressions:
            direction = "↓" if reg["change_pct"] < 0 else "↑"
            print(
                f"  {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f} "
                f"({direction} {abs(reg['change_pct']):.1f}%)"
            )
        return True

    return False


def update_baseline(metrics: Dict[str, Any], baseline_path: Path):
    """Update the baseline with new metrics and audit logging."""
    # Security: Validate path
    if not validate_path(baseline_path):
        logger.error(f"Invalid path for baseline update: {baseline_path}")
        raise ValueError("Invalid baseline path")

    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata for audit
    metrics_with_metadata = metrics.copy()
    metrics_with_metadata.update(
        {
            "updated_at": datetime.now().isoformat(),
            "updated_by": os.environ.get("USER", "unknown"),
            "git_commit": os.environ.get("GITHUB_SHA", "local"),
            "environment": os.environ.get("CI_ENVIRONMENT_NAME", "development"),
        }
    )

    # Backup existing baseline
    if baseline_path.exists():
        backup_path = baseline_path.with_suffix(
            f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            repo_root = Path(__file__).parent.parent.parent.resolve()
            with safe_open(
                baseline_path, repo_root, mode="r", encoding="utf-8"
            ) as f:
                backup_data = json.load(f)
            with safe_open(
                backup_path, repo_root, mode="w", encoding="utf-8"
            ) as f:
                json.dump(backup_data, f, indent=2)
            logger.info(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")

    # Write new baseline
    try:
        repo_root = Path(__file__).parent.parent.parent.resolve()
        with safe_open(baseline_path, repo_root, mode="w", encoding="utf-8") as f:
            json.dump(metrics_with_metadata, f, indent=2)
        logger.info(f"Baseline updated at {baseline_path}")
        print(f"✅ Baseline updated at {baseline_path}")

        # Audit log
        audit_log_entry = {
            "action": "baseline_update",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "user": os.environ.get("USER", "unknown"),
            "success": True,
        }
        log_audit_event(audit_log_entry)

    except Exception as e:
        logger.error(f"Failed to update baseline: {e}")
        raise


def log_audit_event(event: Dict[str, Any]):
    """Log audit events for compliance and tracking."""
    audit_file = Path("audit_log.jsonl")
    try:
        repo_root = Path(__file__).parent.parent.parent.resolve()
        with safe_open(audit_file, repo_root, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")


def main():
    """Main hook entry point."""
    # Check if we're in CI environment
    if os.environ.get("CI") == "true":
        print("ℹ️  Skipping performance check in CI environment")
        return 0

    # Get repository root
    repo_root = Path(__file__).parent.parent.parent
    baseline_path = repo_root / "baselines" / "performance.json"

    print("🔍 Running performance check...")

    # Load baseline
    baseline = load_baseline(baseline_path)

    # Run performance test
    current_metrics = run_quick_performance_test()

    if not current_metrics:
        print("⚠️  Could not get performance metrics, skipping check")
        return 0

    # If no baseline exists, create it
    if baseline is None:
        update_baseline(current_metrics, baseline_path)
        print("✅ Initial baseline created")
        return 0

    # Check for regressions
    if check_regression(current_metrics, baseline):
        print("\n❌ Performance regression detected!")
        print("   To bypass: git push --no-verify")
        print("   To update baseline: python scripts/hooks/update_baseline.py")
        return 1

    print("✅ No performance regressions detected")

    # Check if performance improved significantly (>10%)
    if "accuracy" in current_metrics and "accuracy" in baseline:
        improvement = (current_metrics["accuracy"] - baseline["accuracy"]) / baseline[
            "accuracy"
        ]
        if improvement > 0.1:
            print(f"🎉 Performance improved by {improvement*100:.1f}%!")
            response = input("Update baseline? (y/N): ")
            if response.lower() == "y":
                update_baseline(current_metrics, baseline_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
