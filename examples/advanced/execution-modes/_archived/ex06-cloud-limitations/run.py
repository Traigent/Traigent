#!/usr/bin/env python3
"""Runner for Execution Modes - Cloud Limitations (demo)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Ensure safe local + mock mode + local results folder
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_EDGE_ANALYTICS_MODE", "true")
os.environ.setdefault("TRAIGENT_RESULTS_FOLDER", os.path.join(HERE, ".traigent"))

from cloud_limitations import simple_classifier  # noqa: E402

RESULTS_FILE = "results.json"
DATASET_FILE = "text_classification.jsonl"


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
    print("Running Execution Modes - Cloud Limitations (simple classifier)")
    print("=" * 60)

    opt_result = asyncio.run(simple_classifier.optimize(max_trials=10))

    out: dict[str, Any] = {
        "example": "execution-modes__cloud-limitations",
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
        raise SystemExit(130) from None
