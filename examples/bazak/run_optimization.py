#!/usr/bin/env python3
"""Run Traigent grid optimization against the Bazak ZAP agent.

Exhaustive grid search: every model in the discovered config space is evaluated
against all 3 hardcoded example IDs. Results are logged to the local Traigent
backend (visible in the frontend) and saved to results.json + REPORT.md.

Usage:
    .venv/bin/python examples/bazak/run_optimization.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import requests

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load .env from this folder (not just repo root) so secrets stay local.
# ---------------------------------------------------------------------------
_ENV_FILE = SCRIPT_DIR / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                # Force-set (not setdefault) so bazak .env overrides root .env
                # values like TRAIGENT_OFFLINE_MODE=true.
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

# Block port-443 during import to avoid litellm hanging on external fetch.
import socket as _socket

_orig_connect = _socket.socket.connect


def _block_443(self: _socket.socket, addr: object) -> None:
    if isinstance(addr, tuple) and len(addr) >= 2 and addr[1] == 443:
        raise ConnectionRefusedError("blocked during import")
    return _orig_connect(self, addr)


_socket.socket.connect = _block_443  # type: ignore[assignment]

try:
    from traigent.api.types import OptimizationResult
    from traigent.config.types import TraigentConfig, resolve_execution_mode
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators import HybridAPIEvaluator
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.utils.env_config import get_env_var
finally:
    _socket.socket.connect = _orig_connect  # type: ignore[assignment]

from traigent.utils.logging import setup_logging

setup_logging(level=os.getenv("TRAIGENT_LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL: Final[str] = get_env_var("BAZAK_BASE_URL", "https://ai.bazak.ai")
AUTH_TOKEN: Final[str] = get_env_var("BAZAK_AUTH_TOKEN", "", mask_in_logs=True)
TUNABLE_ID: Final[str] = get_env_var("BAZAK_TUNABLE_ID", "zap_agent")
_INPUT_IDS_RAW = get_env_var(
    "BAZAK_INPUT_IDS",
    "no-filter-single-search-trashcan-blue,product-search-specific-model,consultant-fridge",
)
INPUT_IDS: Final[list[str]] = [
    x.strip() for x in _INPUT_IDS_RAW.split(",") if x.strip()
]
MAX_COST_USD: Final[float] = float(get_env_var("BAZAK_MAX_COST_USD", "10.0"))
# Optional trial cap — 0 means use full grid size.
_MAX_TRIALS_OVERRIDE = int(get_env_var("BAZAK_MAX_TRIALS", "0"))

REQUEST_HEADERS: Final[dict[str, str]] = {"User-Agent": "Traigent-SDK/1.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_server(url: str) -> bool:
    """Verify the Bazak service is reachable."""
    try:
        resp = requests.get(
            f"{url}/traigent/v1/capabilities",
            timeout=5,
            headers={**REQUEST_HEADERS, "Authorization": AUTH_TOKEN},
        )
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def build_dataset() -> Dataset:
    """Build Dataset from the configured input IDs."""
    return Dataset(
        [EvaluationExample(input_data={"input_id": iid}) for iid in INPUT_IDS]
    )


def serialize_result(result: OptimizationResult) -> dict[str, Any]:
    """Convert OptimizationResult to a JSON-serializable dict."""
    trials = []
    for t in result.trials:
        trials.append(
            {
                "trial_id": t.trial_id,
                "config": t.config,
                "metrics": {
                    k: v
                    for k, v in t.metrics.items()
                    if isinstance(v, (int, float, type(None)))
                },
                "status": (
                    t.status.value if hasattr(t.status, "value") else str(t.status)
                ),
                "is_successful": t.is_successful,
                "duration": t.duration,
            }
        )
    return {
        "algorithm": result.algorithm,
        "stop_reason": result.stop_reason,
        "best_config": result.best_config,
        "best_score": result.best_score,
        "best_metrics": result.best_metrics,
        "total_trials": len(result.trials),
        "success_rate": result.success_rate,
        "duration": result.duration,
        "total_cost": result.total_cost,
        "trials": trials,
    }


def write_report(data: dict[str, Any], path: Path) -> None:
    """Generate a markdown report from serialized results."""
    lines = [
        "# Bazak Grid Optimization Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Algorithm**: {data['algorithm']}",
        f"**Stop reason**: {data['stop_reason']}",
        f"**Total trials**: {data['total_trials']}",
        f"**Success rate**: {data['success_rate']:.0%}",
        f"**Duration**: {data['duration']:.1f}s",
    ]
    if data.get("total_cost") is not None:
        lines.append(f"**Total cost**: ${data['total_cost']:.6f}")

    lines += [
        "",
        "## Best Configuration",
        "",
        "```json",
        json.dumps(data["best_config"], indent=2),
        "```",
        "",
        f"**Best score (tool_accuracy)**: {data['best_score']:.4f}",
        "",
        "### Best Metrics",
        "",
    ]
    for k, v in data["best_metrics"].items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines += [
        "",
        "## All Trials",
        "",
        "| # | Model | tool_accuracy | param_accuracy | text_accuracy | Status |",
        "|---|-------|--------------|---------------|--------------|--------|",
    ]
    for i, t in enumerate(data["trials"]):
        model = t["config"].get("model", "?")
        ta = t["metrics"].get("tool_accuracy", "—")
        pa = t["metrics"].get("param_accuracy", "—")
        xa = t["metrics"].get("text_accuracy", "—")
        ta_str = f"{ta:.4f}" if isinstance(ta, (int, float)) else str(ta)
        pa_str = f"{pa:.4f}" if isinstance(pa, (int, float)) else str(pa)
        xa_str = f"{xa:.4f}" if isinstance(xa, (int, float)) else str(xa)
        lines.append(
            f"| {i+1} | {model} | {ta_str} | {pa_str} | {xa_str} | {t['status']} |"
        )

    lines += ["", "---", "*Generated by `examples/bazak/run_optimization.py`*", ""]
    path.write_text("\n".join(lines))


def _print_results(result: OptimizationResult) -> None:
    """Print optimization results to console."""
    print("\n" + "=" * 70)
    print("  BAZAK GRID OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\n  Algorithm:     {result.algorithm}")
    print(f"  Stop reason:   {result.stop_reason}")
    print(f"  Total trials:  {len(result.trials)}")
    print(f"  Successful:    {len(result.successful_trials)}")
    print(f"  Success rate:  {result.success_rate:.0%}")
    print(f"  Duration:      {result.duration:.1f}s")
    if result.total_cost is not None:
        print(f"  Total cost:    ${result.total_cost:.6f}")

    print("\n  Best config:")
    print(f"    {json.dumps(result.best_config, indent=6)}")
    print(f"\n  Best score (tool_accuracy): {result.best_score:.4f}")

    print("\n  Best metrics:")
    for key, value in result.best_metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    print("\n  All trials:")
    print(f"  {'#':>3}  {'Model':<30}  {'tool_acc':>8}  {'param_acc':>9}  Config")
    print(f"  ---  {'-'*30}  --------  ---------  {'-'*30}")
    for i, trial in enumerate(result.trials):
        status = "ok" if trial.is_successful else "FAIL"
        ta = trial.metrics.get("tool_accuracy", 0)
        pa = trial.metrics.get("param_accuracy", 0)
        model = trial.config.get("model", "?")
        print(f"  {i+1:>3}  {model:<30}  {ta:>8.4f}  {pa:>9.4f}  [{status}]")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_optimization() -> None:
    """Run grid optimization against Bazak and log to backend."""
    if not check_server(SERVER_URL):
        print(f"ERROR: Bazak service not reachable at {SERVER_URL}")
        sys.exit(1)

    print("=" * 70)
    print("  Traigent Grid Optimization — Bazak ZAP Agent")
    print("=" * 70)
    print(f"  Service:    {SERVER_URL}")
    print(f"  Tunable:    {TUNABLE_ID}")
    print(f"  Examples:   {len(INPUT_IDS)} ({', '.join(INPUT_IDS)})")

    # --- Step 1: Create evaluator and discover config space ---
    evaluator = HybridAPIEvaluator(
        api_endpoint=SERVER_URL,
        tunable_id=TUNABLE_ID,
        batch_size=len(INPUT_IDS),
        auto_discover_tvars=True,
        auth_header=AUTH_TOKEN,
    )

    async with evaluator:
        print("\n[1/3] Discovering config space...")
        config_space = await evaluator.discover_config_space()
        print(f"      Found {len(config_space)} tunables:")
        for name, domain in config_space.items():
            print(f"        {name}: {domain}")

        # Count total grid size
        grid_size = 1
        for domain in config_space.values():
            if isinstance(domain, list):
                grid_size *= len(domain)
        # Allow capping trials via env var (useful for smoke-testing backend integration)
        max_trials = _MAX_TRIALS_OVERRIDE if _MAX_TRIALS_OVERRIDE > 0 else grid_size
        print(
            f"      Grid size: {grid_size} configs, max_trials: {max_trials}, examples: {len(INPUT_IDS)} ({max_trials * len(INPUT_IDS)} evaluations)"
        )

        # --- Step 2: Create optimizer + orchestrator ---
        print(f"\n[2/3] Setting up GridSearchOptimizer (max_trials={max_trials})...")
        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["tool_accuracy"],
            max_trials=max_trials,
        )

        raw_mode = get_env_var("TRAIGENT_EXECUTION_MODE", "edge_analytics")
        if raw_mode.strip().lower() == "local":
            raw_mode = "edge_analytics"
        execution_mode = resolve_execution_mode(raw_mode)

        cost_approved = get_env_var(
            "TRAIGENT_COST_APPROVED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

        traigent_config = TraigentConfig(execution_mode=execution_mode.value)

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=max_trials,
            objectives=["tool_accuracy"],
            config=traigent_config,
            cost_limit=MAX_COST_USD,
            cost_approved=cost_approved,
        )

        # --- Step 3: Run optimization ---
        dataset = build_dataset()
        print(f"      Dataset: {len(dataset)} examples")
        print(f"      Execution mode: {execution_mode.value}")
        print(f"      Cost limit: ${MAX_COST_USD:.2f} (approved={cost_approved})")
        print("\n[3/3] Running grid optimization...")
        print("-" * 70)

        async def bazak_agent():
            """Placeholder — execution handled by HybridAPIEvaluator."""

        result = await orchestrator.optimize(
            func=bazak_agent,
            dataset=dataset,
            function_name="bazak_agent",
        )

        _print_results(result)

        # --- Save artifacts ---
        data = serialize_result(result)

        results_path = SCRIPT_DIR / "results.json"
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

        report_path = SCRIPT_DIR / "REPORT.md"
        write_report(data, report_path)
        print(f"  Report saved to {report_path}")

        # --- Backend status ---
        api_key = os.environ.get("TRAIGENT_API_KEY")
        backend_url = os.environ.get("TRAIGENT_API_URL", "http://localhost:5000/api/v1")
        if api_key:
            print(f"\n  Backend: {backend_url}")
            print(f"  API Key: ***{api_key[-4:]}")
            print("  Results should be visible in the Traigent frontend.")
        else:
            print("\n  TRAIGENT_API_KEY not set — results NOT logged to backend.")


if __name__ == "__main__":
    asyncio.run(run_optimization())
