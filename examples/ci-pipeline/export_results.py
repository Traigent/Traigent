"""Export the newest ``traigent optimize`` result to the files dvc.yaml declares.

``python -m traigent optimize --output-dir D`` persists each run under
``D/<function>_<algorithm>_<max_trials>/`` (``metadata.json`` plus
``trials_summary.json``). The downstream DVC stages read a flat
``D/latest.json`` and a ``D/metrics.json``; this step writes them.

Usage: python export_results.py [optimization_results]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _newest_run(base: Path) -> Path:
    runs = [p for p in base.iterdir() if (p / "metadata.json").is_file()]
    if not runs:
        raise SystemExit(
            "no persisted optimization result found; run the optimize step"
        )
    return max(runs, key=lambda p: (p / "metadata.json").stat().st_mtime)


def export(base: Path) -> None:
    run = _newest_run(base)
    meta = json.loads((run / "metadata.json").read_text())
    summary_path = run / "trials_summary.json"
    raw = json.loads(summary_path.read_text()) if summary_path.is_file() else []

    trials = []
    for i, t in enumerate(raw):
        m = t.get("metrics") or {}
        trial = {"trial": i, "config": t.get("config", {})}
        score = t.get("score", m.get("score"))
        if score is not None:
            trial["score"] = score
        if "duration" in t:
            trial["latency"] = t["duration"]
        if "total_tokens" in m:
            trial["tokens"] = m["total_tokens"]
        # Cost is only exported when it was actually measured.
        if m.get("cost_unmeasured") != 1.0 and "total_cost" in m:
            trial["cost"] = m["total_cost"]
        trials.append(trial)

    latest = {
        "function_name": meta.get("function_name"),
        "algorithm": meta.get("algorithm"),
        "best_config": meta.get("best_config"),
        "best_score": meta.get("best_score"),
        "trials": trials,
    }
    (base / "latest.json").write_text(json.dumps(latest, indent=2))
    metrics = {
        "best_score": meta.get("best_score"),
        "total_trials": meta.get("total_trials"),
        "successful_trials": meta.get("successful_trials"),
        "success_rate": meta.get("success_rate"),
        "duration": meta.get("duration"),
    }
    (base / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    export(Path(sys.argv[1] if len(sys.argv) > 1 else "optimization_results"))
