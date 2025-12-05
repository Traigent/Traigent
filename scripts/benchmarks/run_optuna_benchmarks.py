#!/usr/bin/env python3
"""Run Optuna vs random search benchmarks and persist results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from traigent.optimizers.benchmarking import run_optuna_random_parity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna benchmark suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmarks/optuna/latest.json"),
        help="Path to store benchmark results (JSON)",
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to average")
    parser.add_argument(
        "--seed-offset", type=int, default=0, help="Offset applied to random seeds"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_optuna_random_parity(
        n_trials=args.trials, runs=args.runs, seed_offset=args.seed_offset
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    optuna_avg = results["optuna"]["average_best_value"]
    random_avg = results["random"]["average_best_value"]
    print(
        f"Optuna average best value: {optuna_avg:.4f}\n"
        f"Random average best value: {random_avg:.4f}\n"
        f"Results written to {output_path}"
    )


if __name__ == "__main__":
    main()
