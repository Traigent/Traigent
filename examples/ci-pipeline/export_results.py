"""Export the newest ``traigent optimize`` result to the files dvc.yaml declares.

``python -m traigent optimize --output-dir D`` persists each run under
``D/<function>_<algorithm>_<max_trials>/``. The authoritative per-trial record
is ``trials.json.gz`` (every trial, failed ones included, with status and
metrics); ``trials_summary.json`` is only a readability sample (the first 50
*successful* trials) and is never used here. The downstream DVC stages read a
flat ``D/latest.json`` and a ``D/metrics.json``; this step writes them.

Cost honesty: a trial's ``cost`` is the measured value or ``null``.
``cost_status`` is ``measured``, ``unmeasured`` or ``unpriced``; a known spend
floor, when the record has one, is exported as ``cost_lower_bound``. An unknown
cost is never written as ``0``.

Fails with a non-zero exit (and a content-free message) when the persisted run
is missing or incomplete.

Usage: python export_results.py [optimization_results]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any


def _newest_run(base: Path) -> Path:
    runs = [p for p in base.iterdir() if (p / "metadata.json").is_file()]
    if not runs:
        raise SystemExit(
            "no persisted optimization result found; run the optimize step"
        )
    return max(runs, key=lambda p: (p / "metadata.json").stat().st_mtime)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _flag(metrics: dict[str, Any], key: str) -> bool:
    v = metrics.get(key)
    return v is True or (_num(v) is not None and _num(v) > 0)


def trial_cost(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return ``cost``/``cost_status``/(optional) ``cost_lower_bound`` for a trial."""
    total = _num(metrics.get("total_cost"))
    if total is None:
        total = _num(metrics.get("cost"))
    floor = _num(metrics.get("cost_lower_bound"))
    if _flag(metrics, "cost_unmeasured"):
        out: dict[str, Any] = {"cost": None, "cost_status": "unmeasured"}
    elif _flag(metrics, "cost_unpriced"):
        # A non-strict run records a placeholder $0 for an unpriceable call.
        out = {"cost": None, "cost_status": "unpriced"}
        floor = max(floor or 0.0, total or 0.0) or None
    elif total is not None:
        return {"cost": total, "cost_status": "measured"}
    else:
        out = {"cost": None, "cost_status": "unmeasured"}
    if floor is not None and floor > 0:
        out["cost_lower_bound"] = floor
    return out


def _load_trials(run: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    path = run / "trials.json.gz"
    if not path.is_file():
        raise SystemExit("persisted run is incomplete: trial records are missing")
    try:
        with gzip.open(path, "rt") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        raise SystemExit(
            "persisted run is incomplete: trial records are unreadable"
        ) from None
    expected = meta.get("total_trials")
    if (
        not isinstance(raw, list)
        or not raw
        or (isinstance(expected, int) and len(raw) != expected)
    ):
        raise SystemExit("persisted run is incomplete: trial count mismatch")
    return raw


def export(base: Path) -> None:
    run = _newest_run(base)
    meta = json.loads((run / "metadata.json").read_text())
    raw = _load_trials(run, meta)

    trials = []
    for i, t in enumerate(raw):
        m = t.get("metrics") or {}
        status = str(t.get("status", "unknown")).lower()
        trial: dict[str, Any] = {
            "trial": i,
            "status": status,
            "config": t.get("config", {}),
        }
        score = t.get("score", m.get("score"))
        if _num(score) is not None:
            trial["score"] = score
        if _num(t.get("duration")) is not None:
            trial["latency"] = t["duration"]
        if _num(m.get("total_tokens")) is not None:
            trial["tokens"] = m["total_tokens"]
        trial.update(trial_cost(m))
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
