"""Per-example x per-trial outcome matrix: build, persist, and load.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-STORAGE FUNC-ANALYTICS REQ-STOR-007 SYNC-StorageLogging

Every optimization run already evaluates each eval-dataset example once per
trial (per configuration/replicate) and keeps the per-example outcomes in
``trial.metadata["example_results"]``. That structure is the *outcome matrix* —
example rows by trial/config columns — but until issue #1838 it was only ever
persisted as one JSONL record per trial (``trials/trials_v2.jsonl``), leaving
example-level analysis to spelunk and re-assemble it (or, worse, re-collect it
with fresh LLM calls). This module assembles that already-computed by-product
into a single, self-describing artifact alongside the other run artifacts
(``best_config.json`` etc.) and exposes an accessor to load it back.

It recomputes and re-runs nothing: it is a pure projection of data the run has
already produced.

File format (``outcome_matrix.json`` / ``outcome_matrix_v2.json``)
-----------------------------------------------------------------
A single JSON object::

    {
      "schema_version": "1.0",
      "optimization_id": "opt-123",
      "algorithm": "GridSearchOptimizer",
      "objectives": ["accuracy"],
      "created_at": "2026-07-15T12:00:00+00:00",
      "trial_count": 2,
      "example_count": 3,
      "trials": [                       # the matrix columns, in trial order
        {
          "index": 0,
          "trial_id": "trial_0",
          "config": {"model": "gpt-4o"},
          "config_hash": "a1b2c3d4e5f6"  # sha256[:12] of the sorted config
        },
        ...
      ],
      "examples": [                      # the matrix rows, one per example_id
        {
          "example_id": "ex-1",
          "cells": {                     # keyed by trial_id (a column above)
            "trial_0": {
              "score": 1.0,              # the per-example ``score`` metric (the
                                         # optimization signal), or null
              "accuracy": 1.0,           # the ``accuracy`` metric IFF present,
                                         # else null (never assumed present)
              "metrics": {"f1": 0.8},    # the full per-example metrics map,
                                         # verbatim — whatever the run optimized
                                         # (f1/exact_match/custom) is readable here
              "success": true,           # explicit ``success`` if serialized,
                                         # else ``error is None``
              "tokens": {                # per-example token telemetry, if any
                "input": 120,
                "output": 8,
                "total": 128
              },
              "cost_usd": 0.0004,        # per-example cost, or null
              "execution_time": 0.42,    # seconds (ExampleResult), or null
              "latency_ms": null,        # milliseconds (HybridExampleResult), or null
              "error": null              # error message, if the example failed
            },
            ...
          }
        },
        ...
      ]
    }

Cell fields record only what the run actually produced — no fabricated values:

* ``score``/``accuracy`` are ``null`` when the run did not write that metric key.
  A run optimizing a non-accuracy metric (f1/exact_match/custom) has ``accuracy``
  ``null`` and its real signal in ``metrics`` (and, when the evaluator aligned it,
  ``score``). Never assume a metric is named "accuracy".
* ``metrics`` is the per-example metrics map passed through verbatim, so
  downstream can read whatever metric the run used.
* ``success`` is derived robustly: the explicit ``success`` field when present
  (``ExampleResult``), otherwise ``error is None`` — because
  ``HybridExampleResult.success`` is a ``@property`` that ``asdict`` drops.
* ``tokens`` is ``null`` when no token telemetry was captured. Token/cost values
  are recorded as-is: present-as-0 stays ``0`` (the evaluator defaults some to 0);
  an absent key is ``null``. The distinction is preserved, not flattened.
* ``execution_time`` (seconds) and ``latency_ms`` (milliseconds) come from two
  different result shapes and are kept in their own units, each recorded as-is.
  ``execution_time`` is ``null`` when absent. Note ``HybridExampleResult``
  defaults ``latency_ms`` and ``cost_usd`` to ``0.0`` at the dataclass level, so
  a hybrid example that never set them records ``0.0`` (not ``null``) — the
  projection fabricates nothing and reflects exactly what the run produced.

Downstream consumers (#1880 eval-defect detectors, #1881 defect score) read
``examples`` as the item x config outcome/token matrix.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

#: Bump when the on-disk shape changes incompatibly.
OUTCOME_MATRIX_SCHEMA_VERSION = "1.0"

#: Canonical (legacy) artifact filename; the versioned form is
#: ``outcome_matrix_v{version}.json`` (see file_versioning.FILE_PATTERNS).
OUTCOME_MATRIX_FILE = "outcome_matrix.json"

# Token-telemetry keys as attached per-example by the evaluator into each
# example's ``metrics`` map. Two lanes use different names:
#   - the LLM / hybrid lane writes prompt_tokens / completion_tokens /
#     total_tokens (BaseEvaluator._add_llm_metrics_to_example, base.py:3323-3325);
#   - the local response-metrics lane writes input_tokens / output_tokens /
#     total_tokens (base.py:2448-2450).
# We surface a canonical input/output/total, preferring the real prompt/completion
# names and falling back to the input/output aliases. Kept here so build stays a
# pure projection.
_TOKEN_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("input", ("prompt_tokens", "input_tokens")),
    ("output", ("completion_tokens", "output_tokens")),
    ("total", ("total_tokens",)),
)


def _get(ex: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an example that may be a dict or an ExampleResult."""
    if isinstance(ex, dict):
        return ex.get(key, default)
    return getattr(ex, key, default)


def _extract_tokens(metrics: dict[str, Any]) -> dict[str, int] | None:
    """Pull per-example token counts out of the example's metrics map.

    Returns ``None`` when no token telemetry was captured, so the cell records
    an honest "unknown" rather than a fabricated zero.
    """
    tokens: dict[str, int] = {}
    for out_key, metric_keys in _TOKEN_KEYS:
        for metric_key in metric_keys:
            value = metrics.get(metric_key)
            if isinstance(value, bool):  # guard: bool is an int subclass
                continue
            if isinstance(value, (int, float)):
                tokens[out_key] = int(value)
                break  # first present alias wins (prefer prompt/completion)
    return tokens or None


def _extract_cell(ex: Any) -> dict[str, Any]:
    """Project one per-example result into a compact outcome cell.

    Pure read: no scoring, no recomputation. ``ex`` is either the serialized
    payload stored in ``trial.metadata["example_results"]`` — an
    ``ExampleResult.to_dict()`` (has ``success``/``error_message``/
    ``execution_time``) or a ``HybridExampleResult`` ``asdict`` (has
    ``error``/``latency_ms``/``cost_usd`` and *no* ``success`` key, because
    ``HybridExampleResult.success`` is a ``@property`` ``asdict`` drops) — or an
    in-memory ``ExampleResult`` / ``HybridExampleResult`` (tests / direct
    callers).
    """
    metrics = _get(ex, "metrics", {}) or {}
    if not isinstance(metrics, dict):
        metrics = {}

    # Optimization signal: the per-example ``score`` metric the run wrote — NOT
    # hardcoded to "accuracy". A run optimizing f1/exact_match/custom has its
    # signal under ``score`` (the evaluator's scoring path aligns it) and always
    # under its own name in the ``metrics`` passthrough below. ``accuracy`` is a
    # convenience field populated only when that metric key truly exists.
    score = metrics.get("score")
    accuracy = metrics.get("accuracy")

    # HybridExampleResult uses ``error``; ExampleResult uses ``error_message``.
    error = _get(ex, "error_message")
    if error is None:
        error = _get(ex, "error")

    # Success: ExampleResult serializes an explicit ``success`` bool; the
    # HybridExampleResult ``success`` is a ``@property`` that ``asdict`` omits,
    # so fall back to ``error is None``. Use the explicit key when present so a
    # genuinely-failed example (success=False) is respected.
    explicit_success = _get(ex, "success")
    success = bool(explicit_success) if explicit_success is not None else error is None

    # Cost: evaluator writes ``total_cost`` into metrics; HybridExampleResult
    # carries a top-level ``cost_usd``.
    cost_usd = metrics.get("total_cost")
    if cost_usd is None:
        cost_usd = _get(ex, "cost_usd")

    # Timing kept per-source in its own unit (no unit-mixing): ExampleResult
    # exposes ``execution_time`` (seconds); HybridExampleResult exposes
    # ``latency_ms`` (milliseconds). Each is null when its source didn't set it.
    execution_time = _get(ex, "execution_time")
    latency_ms = _get(ex, "latency_ms")

    cell: dict[str, Any] = {
        "score": score,
        "accuracy": accuracy,
        "metrics": dict(metrics),
        "success": success,
        "tokens": _extract_tokens(metrics),
        "cost_usd": cost_usd,
        "execution_time": execution_time,
        "latency_ms": latency_ms,
        "error": error,
    }
    return cell


def _config_hash(config: Any) -> str:
    """Stable short hash of a trial config for grouping replicate columns."""
    try:
        encoded = json.dumps(config, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = str(config)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def build_outcome_matrix(result: OptimizationResult) -> dict[str, Any]:
    """Assemble the per-example x per-trial outcome matrix from a result.

    Reuses the already-computed ``trial.metadata["example_results"]`` on each
    trial; it does not re-run or re-score anything. Trials with no per-example
    detail contribute a column but no cells. Example rows are emitted in
    first-seen order across trials so the layout is deterministic.

    Args:
        result: A completed ``OptimizationResult`` (or any object exposing a
            ``trials`` iterable of trials with a ``metadata`` mapping).

    Returns:
        The matrix as a JSON-serializable dict (see the module docstring for the
        schema). ``examples`` is empty when no trial carried per-example results.
    """
    trials = list(getattr(result, "trials", []) or [])

    columns: list[dict[str, Any]] = []
    # example_id -> {trial_id: cell}
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for index, trial in enumerate(trials):
        trial_id = getattr(trial, "trial_id", None) or f"trial_{index}"
        config = getattr(trial, "config", {}) or {}
        columns.append(
            {
                "index": index,
                "trial_id": trial_id,
                "config": config,
                "config_hash": _config_hash(config),
            }
        )

        metadata = getattr(trial, "metadata", None) or {}
        example_results = (
            metadata.get("example_results") if isinstance(metadata, dict) else None
        )
        if not isinstance(example_results, list):
            continue

        for ex in example_results:
            example_id = _get(ex, "example_id")
            if example_id is None:
                continue
            example_id = str(example_id)
            if example_id not in rows:
                rows[example_id] = {}
                order.append(example_id)
            rows[example_id][trial_id] = _extract_cell(ex)

    examples = [
        {"example_id": example_id, "cells": rows[example_id]} for example_id in order
    ]

    return {
        "schema_version": OUTCOME_MATRIX_SCHEMA_VERSION,
        "optimization_id": getattr(result, "optimization_id", None),
        "algorithm": getattr(result, "algorithm", None),
        "objectives": list(getattr(result, "objectives", []) or []),
        "created_at": datetime.now(UTC).isoformat(),
        "trial_count": len(columns),
        "example_count": len(examples),
        "trials": columns,
        "examples": examples,
    }


def load_outcome_matrix(run_path: str | Path) -> dict[str, Any] | None:
    """Load a persisted outcome matrix from a run's artifact directory.

    Accepts either the run directory (``.../runs/<run_id>``) or its
    ``artifacts`` subdirectory. Tries the versioned filename first, then the
    legacy unversioned one. Returns ``None`` when no matrix artifact exists,
    so #1880-style consumers can fall back to an in-memory build.

    Args:
        run_path: Path to the run directory or its ``artifacts`` subdirectory.

    Returns:
        The parsed matrix dict, or ``None`` when no artifact is present.

    Raises:
        ValueError: When the artifact exists but is not valid JSON.
    """
    base = Path(run_path)
    artifacts = base if base.name == "artifacts" else base / "artifacts"

    candidates = [
        artifacts / "outcome_matrix_v2.json",
        artifacts / OUTCOME_MATRIX_FILE,
    ]
    # Any other versioned variant (e.g. a future v3), newest first.
    candidates.extend(
        sorted(
            artifacts.glob("outcome_matrix_v*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if artifacts.exists()
        else []
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                return dict(json.load(handle))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt outcome matrix artifact at {candidate}: {exc}"
            ) from exc

    return None
