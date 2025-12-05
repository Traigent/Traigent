#!/usr/bin/env python3
"""Run optimization for the parameter-custom example and write results.json."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from typing import Any

# Configuration
MODULE_NAME: str = "parameter_custom"
EXAMPLE_ID: str = "configuration-management__parameter-custom"


# Paths
THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
DATASET: str = os.path.join(THIS_DIR, "chat_interactions.jsonl")
RESULTS: str = os.path.join(THIS_DIR, "results.json")


def _serialize_trials(opt_result: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Serialize a small subset of trials in a linter-friendly way."""
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
    """Run example optimization and write a compact results.json."""
    try:
        ex: Any = importlib.import_module(MODULE_NAME)
    except ImportError as exc:
        payload: dict[str, Any] = {
            "example": EXAMPLE_ID,
            "dataset": os.path.basename(DATASET),
            "error": f"Module import failed: {exc}",
        }
        with open(RESULTS, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {RESULTS}")
        raise

    opt_result = asyncio.run(ex.adaptive_chat_bot.optimize(max_trials=10))

    out: dict[str, Any] = {
        "example": EXAMPLE_ID,
        "dataset": os.path.basename(DATASET),
        "best_config": getattr(opt_result, "best_config", {}),
        "best_score": getattr(opt_result, "best_score", None),
        "objectives": getattr(opt_result, "objectives", []),
        "algorithm": getattr(opt_result, "algorithm", ""),
        "total_cost": getattr(opt_result, "total_cost", None),
        "total_tokens": getattr(opt_result, "total_tokens", None),
        "trials": _serialize_trials(opt_result, limit=10),
    }

    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # Print results to screen for quick inspection
    print(json.dumps(out, indent=2))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
