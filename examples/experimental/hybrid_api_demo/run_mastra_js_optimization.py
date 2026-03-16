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
  - JS-Mastra demo server running (see .env.example for endpoint config)
  - Copy ``examples/experimental/hybrid_api_demo/.env.example`` to ``.env`` in the
    project root and fill in MASTRA_JS_AUTH_TOKEN

Usage:
    .venv/bin/python examples/experimental/hybrid_api_demo/run_mastra_js_optimization.py
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

# A dependency makes a blocking HTTPS call at import time (to
# raw.githubusercontent.com). Temporarily block outgoing port-443
# connections so the import finishes instantly instead of hanging.
import socket as _socket

_orig_connect = _socket.socket.connect


def _block_443(self: _socket.socket, addr: object) -> None:
    if isinstance(addr, tuple) and len(addr) >= 2 and addr[1] == 443:
        raise ConnectionRefusedError("blocked during import")
    return _orig_connect(self, addr)


_socket.socket.connect = _block_443  # type: ignore[assignment]

try:
    # Importing env_config triggers automatic .env loading from the project root.
    from traigent.api.types import OptimizationResult
    from traigent.config.types import TraigentConfig, resolve_execution_mode
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators import HybridAPIEvaluator
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.optimizers.random import RandomSearchOptimizer
    from traigent.utils.env_config import get_env_var
finally:
    # Restore normal socket behavior for the actual HTTP calls.
    _socket.socket.connect = _orig_connect  # type: ignore[assignment]

# Enable SDK logging so per-example results are visible.
from traigent.utils.logging import setup_logging

setup_logging(level=os.getenv("TRAIGENT_LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Configuration — all values come from environment / .env (no hardcoded secrets)
# ---------------------------------------------------------------------------
SERVER_URL: Final[str] = get_env_var(
    "MASTRA_JS_BASE_URL", "https://your-mastra-server.example.com"
)
AUTH_TOKEN: Final[str] = get_env_var("MASTRA_JS_AUTH_TOKEN", "", mask_in_logs=True)
AUTH_HEADERS: Final[dict[str, str]] = {
    "Authorization": AUTH_TOKEN,
    "x-api-key": AUTH_TOKEN,
    "User-Agent": "Traigent-SDK/1.0",
}
TUNABLE_ID: Final[str | None] = get_env_var("MASTRA_JS_TUNABLE_ID")
_EXAMPLE_IDS_ENV = get_env_var("MASTRA_JS_EXAMPLE_IDS", "")
EXAMPLE_IDS_OVERRIDE: Final[list[str]] = (
    [x.strip() for x in _EXAMPLE_IDS_ENV.split(",") if x.strip()]
    if _EXAMPLE_IDS_ENV
    else []  # empty = auto-discover via GET /benchmarks
)
MAX_TRIALS: Final[int] = int(get_env_var("MASTRA_JS_MAX_TRIALS", "10"))
MAX_COST_USD: Final[float] = float(get_env_var("MASTRA_JS_MAX_COST_USD", "4.0"))
MAX_REASONING_LEVEL: Final[str] = (
    get_env_var("MASTRA_JS_MAX_REASONING_LEVEL", "medium").strip().lower()
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


def build_dataset(example_ids: list[str]) -> Dataset:
    """Build the dataset from the given example_ids."""
    return Dataset(
        [EvaluationExample(input_data={"example_id": iid}) for iid in example_ids]
    )


def _print_results(result: OptimizationResult) -> None:
    """Print optimization results report."""
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
        batch_size=1,
        auto_discover_tvars=True,
        auth_header=AUTH_TOKEN,
    )

    async with evaluator:
        print("\n[1/4] Discovering config space...")
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

        # --- Step 2: Discover example IDs (or use override) ---
        if EXAMPLE_IDS_OVERRIDE:
            example_ids = EXAMPLE_IDS_OVERRIDE
            print(
                f"\n[2/4] Using {len(example_ids)} example IDs from MASTRA_JS_EXAMPLE_IDS"
            )
        else:
            print("\n[2/4] Discovering example IDs via GET /benchmarks...")
            example_ids = await evaluator.discover_example_ids()
            print(f"      Discovered {len(example_ids)} example IDs from server")

        # --- Step 3: Create optimizer + orchestrator ---
        print(f"\n[3/4] Setting up RandomSearchOptimizer (max_trials={MAX_TRIALS})...")
        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=MAX_TRIALS,
            random_seed=42,
        )

        # Use Traigent's resolve_execution_mode (handles canonical values).
        # "local" is accepted as a convenience alias for "edge_analytics".
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
            max_trials=MAX_TRIALS,
            objectives=["accuracy"],
            config=traigent_config,
            cost_limit=MAX_COST_USD,
            cost_approved=cost_approved,
        )

        # --- Step 4: Run optimization ---
        dataset = build_dataset(example_ids)
        print(f"      Dataset: {len(dataset)} examples")
        print(f"      Execution mode: {execution_mode.value}")
        print(f"      Cost limit: ${MAX_COST_USD:.2f} (approved={cost_approved})")
        print("\n[4/4] Running optimization...")
        print("-" * 70)

        async def hybrid_demo_agent():
            """Placeholder — execution is handled by HybridAPIEvaluator."""

        result = await orchestrator.optimize(
            func=hybrid_demo_agent,
            dataset=dataset,
            function_name="hybrid_demo_agent",
        )

        _print_results(result)


if __name__ == "__main__":
    asyncio.run(run_optimization())
