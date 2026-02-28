#!/usr/bin/env python3
"""Run Traigent optimization against the JS-Mastra demo server.

Uses Traigent's OptimizationOrchestrator with RandomSearchOptimizer — NOT a
brute-force grid.  The orchestrator handles:
  - Smart config sampling from the discovered config space
  - Early stopping when max_trials or budget is reached
  - Best-config selection with proper scoring
  - Full trial history and convergence tracking

Flow:
  1. HybridAPIEvaluator discovers config space (TVARs) from the demo server
  2. RandomSearchOptimizer samples configs from that space
  3. OptimizationOrchestrator runs the loop (evaluate, record, stop)
  4. Reports best config, metrics, stop reason

Prerequisites:
  - JS-Mastra demo server running at http://localhost:8080
    (cd JS-Mastra-APIs-Validation && npm run api:dev)
  - OPENAI_API_KEY set in the demo server's .env

Usage:
    .venv/bin/python examples/hybrid_mode_demo/run_mastra_js_optimization.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Final

import requests

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators import HybridAPIEvaluator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.random import RandomSearchOptimizer

SERVER_URL: Final[str] = os.getenv("MASTRA_JS_BASE_URL", "http://localhost:8080")
TUNABLE_ID: Final[str | None] = os.getenv(
    "MASTRA_JS_TUNABLE_ID"
)  # None = auto-select first
DATASET_SIZE: Final[int] = int(
    os.getenv("MASTRA_JS_DATASET_SIZE", "100")
)  # case_001 through case_100
MAX_TRIALS: Final[int] = int(
    os.getenv("MASTRA_JS_MAX_TRIALS", "10")
)  # Let Traigent decide which configs to try
MAX_COST_USD: Final[float] = float(
    os.getenv("MASTRA_JS_MAX_COST_USD", "4.0")
)  # Stop after spending $4
MAX_REASONING_LEVEL: Final[str] = (
    os.getenv("MASTRA_JS_MAX_REASONING_LEVEL", "medium").strip().lower()
)

_REASONING_LEVEL_ORDER: Final[list[str]] = [
    "minimal",
    "low",
    "medium",
    "high",
    "highest",
]
REQUEST_HEADERS: Final[dict[str, str]] = {
    "User-Agent": "Traigent-SDK/1.0",
}


def _resolve_execution_mode(mode: str | None) -> str:
    """Normalize execution mode for this demo.

    Supported values:
    - edge_analytics (local)
    - local (alias for edge_analytics)
    - hybrid
    """

    raw = (mode or "edge_analytics").strip().lower()
    if raw == "local":
        return "edge_analytics"
    if raw in {"edge_analytics", "hybrid"}:
        return raw
    raise ValueError(
        "TRAIGENT_EXECUTION_MODE must be one of: edge_analytics, local, hybrid"
    )


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse common truthy env values."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_reasoning_cap(
    config_space: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Cap reasoning-level choices in discovered config space.

    Looks for categorical parameters with "reason" in the key and values that
    match known reasoning levels. Values above MAX_REASONING_LEVEL are removed.
    """

    if MAX_REASONING_LEVEL not in _REASONING_LEVEL_ORDER:
        return config_space, False

    max_idx = _REASONING_LEVEL_ORDER.index(MAX_REASONING_LEVEL)
    updated = dict(config_space)
    changed = False

    for name, domain in config_space.items():
        if "reason" not in name.lower() or not isinstance(domain, list):
            continue

        normalized: dict[str, object] = {str(v).lower(): v for v in domain}
        known_levels = [lvl for lvl in _REASONING_LEVEL_ORDER if lvl in normalized]
        if not known_levels:
            continue

        allowed = [
            normalized[lvl]
            for lvl in _REASONING_LEVEL_ORDER
            if lvl in normalized and _REASONING_LEVEL_ORDER.index(lvl) <= max_idx
        ]
        if allowed and len(allowed) != len(domain):
            updated[name] = allowed
            changed = True

    return updated, changed


def check_server(url: str) -> bool:
    """Verify demo server is running."""
    try:
        resp = requests.get(
            f"{url}/traigent/v1/health",
            timeout=3,
            headers=REQUEST_HEADERS,
        )
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def discover_tunable_id(url: str) -> str:
    """GET /capabilities and return the selected tunable_id."""
    resp = requests.get(
        f"{url}/traigent/v1/capabilities",
        timeout=5,
        headers=REQUEST_HEADERS,
    )
    resp.raise_for_status()
    tunable_ids = resp.json().get("tunable_ids", [])
    if not tunable_ids:
        raise RuntimeError("Server returned no tunable_ids in /capabilities")
    print(f"      Available tunables: {tunable_ids}")
    if TUNABLE_ID:
        if TUNABLE_ID not in tunable_ids:
            raise RuntimeError(
                f"MASTRA_JS_TUNABLE_ID={TUNABLE_ID!r} not found. "
                f"Available: {tunable_ids}"
            )
        selected = TUNABLE_ID
    else:
        selected = tunable_ids[0]
    print(f"      Selected: {selected}")
    return selected


def build_dataset() -> Dataset:
    """Build the full dataset using the demo server's input_ids."""
    return Dataset(
        [
            EvaluationExample(input_data={"input_id": f"case_{i:03d}"})
            for i in range(1, DATASET_SIZE + 1)
        ]
    )


async def run_optimization() -> None:
    """Run Traigent optimization loop against the demo server."""
    if not check_server(SERVER_URL):
        print(f"ERROR: Demo server not running at {SERVER_URL}")
        print("Start it: cd JS-Mastra-APIs-Validation && npm run api:dev")
        sys.exit(1)

    print("=" * 70)
    print("  Traigent Optimization — JS-Mastra Demo")
    print("=" * 70)

    # --- Step 1: Discover tunable and create evaluator ---
    tunable_id = discover_tunable_id(SERVER_URL)
    evaluator = HybridAPIEvaluator(
        api_endpoint=SERVER_URL,
        tunable_id=tunable_id,
        batch_size=50,
        auto_discover_tvars=True,
    )

    async with evaluator:
        print("\n[1/3] Discovering config space...")
        config_space = await evaluator.discover_config_space()
        config_space, reasoning_capped = _apply_reasoning_cap(config_space)
        print(f"      Found {len(config_space)} tunables:")
        for name, domain in config_space.items():
            print(f"        {name}: {domain}")
        if reasoning_capped:
            print(
                f"      Applied reasoning cap: <= {MAX_REASONING_LEVEL} "
                f"(MASTRA_JS_MAX_REASONING_LEVEL)"
            )

        # --- Step 2: Create optimizer + orchestrator ---
        print(f"\n[2/3] Setting up RandomSearchOptimizer (max_trials={MAX_TRIALS})...")
        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=MAX_TRIALS,
            random_seed=42,
        )

        execution_mode = _resolve_execution_mode(os.getenv("TRAIGENT_EXECUTION_MODE"))
        cost_approved = _parse_bool_env("TRAIGENT_COST_APPROVED", default=False)
        traigent_config = TraigentConfig(execution_mode=execution_mode)

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=MAX_TRIALS,
            objectives=["accuracy"],
            config=traigent_config,
            cost_limit=MAX_COST_USD,
            cost_approved=cost_approved,
        )

        # --- Step 3: Run optimization ---
        dataset = build_dataset()
        print(f"      Dataset: {len(dataset)} examples")
        print(f"      Execution mode: {execution_mode}")
        print(f"      Cost limit: ${MAX_COST_USD:.2f} (approved={cost_approved})")
        print("\n[3/3] Running optimization...")
        print("-" * 70)

        async def hybrid_demo_agent():
            """Placeholder — execution is handled by HybridAPIEvaluator."""

        result = await orchestrator.optimize(
            func=hybrid_demo_agent,
            dataset=dataset,
            function_name="hybrid_demo_agent",
        )

        # --- Report ---
        print("\n" + "=" * 70)
        print("  OPTIMIZATION RESULTS")
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

        print(f"\n  Best score:    {result.best_score:.4f}")
        print("\n  Best metrics:")
        for key, value in result.best_metrics.items():
            if isinstance(value, float):
                if "cost" in key:
                    print(f"    {key}: ${value:.6f}")
                elif "latency" in key:
                    print(f"    {key}: {value:.0f}ms")
                else:
                    print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

        print("\n  All trials:")
        print(f"  {'#':>3}  {'Acc':>7}  {'Cost':>9}  Config")
        print(f"  ---  -------  ---------  {'-'*45}")
        for i, trial in enumerate(result.trials):
            status = "ok" if trial.is_successful else "FAIL"
            acc = trial.metrics.get("accuracy", 0)
            cost = trial.metrics.get("cost", 0)
            print(
                f"  {i+1:>3}  {acc:>7.4f}  ${cost:>8.4f}  "
                f"{json.dumps(trial.config, separators=(',', ':'))}"
                f"  [{status}]"
            )

        print("\n  Convergence info:")
        for k, v in result.convergence_info.items():
            print(f"    {k}: {v}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(run_optimization())
