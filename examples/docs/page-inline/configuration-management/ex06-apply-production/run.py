#!/usr/bin/env python3
"""Runner for production configuration example (standardized results)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_production import OptimizedChatService

RESULTS_FILE = "results.json"
DATASET_FILE = "production_queries.jsonl"


def _serialize_trials(opt_result: Any, limit: int = 10) -> list[dict[str, Any]]:
    trials_out: list[dict[str, Any]] = []
    trials = getattr(opt_result, "trials", [])
    for t in trials[:limit]:
        trials_out.append(
            {
                "id": getattr(t, "trial_id", ""),
                "metrics": getattr(t, "metrics", {}),
                "status": str(getattr(t, "status", "")),
                "duration": getattr(t, "duration", 0.0),
            }
        )
    return trials_out


def main() -> None:
    print("Running Traigent Production Configuration Example")
    print("=" * 60)

    # Initialize service (uses optimized functions inside)
    service = OptimizedChatService()

    # Optimize the response function using the provided dataset
    # Use asyncio.run here to avoid cross-thread aiohttp issues
    opt_result = asyncio.run(service.generate_response.optimize(max_trials=10))

    out: dict[str, Any] = {
        "example": "configuration-management__apply-production",
        "dataset": os.path.basename(DATASET_FILE),
        "best_config": getattr(opt_result, "best_config", {}),
        "best_score": getattr(opt_result, "best_score", None),
        "objectives": getattr(opt_result, "objectives", []),
        "algorithm": getattr(opt_result, "algorithm", ""),
        "total_cost": getattr(opt_result, "total_cost", None),
        "total_tokens": getattr(opt_result, "total_tokens", None),
        "trials": _serialize_trials(opt_result, limit=10),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"Wrote {os.path.abspath(RESULTS_FILE)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
