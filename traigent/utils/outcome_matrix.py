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
          "config_hash": "a1b2c3d4e5f6"  # sha256[:12] of the sorted config  # pragma: allowlist secret
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
              "predicted": "B",          # the model's actual answer for this cell
                                         # (``actual_output`` verbatim); present
                                         # only on NON-correct cells, else null
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
  The one exception is cost: a row flagged ``cost_unmeasured`` (the service
  reported no cost for it) records ``cost_usd: null``, because its ``0.0`` is a
  placeholder, not a measurement.
* ``predicted`` is the model's actual answer for the cell (``actual_output``
  verbatim). It is recorded **only on non-correct cells** (``success is not
  True`` — the wrong and indeterminate cells) and is ``null`` otherwise. That is
  exactly the subset the #1880 cross-family consensus detector (D7) reads to tell
  whether models from different families agreed on the *same* wrong answer;
  clearly-correct cells never anchor consensus, so their (potentially large)
  output is not persisted, keeping the matrix from growing with data no detector
  consumes. Recording it here keeps the eval-defect audit a pure function of the
  persisted matrix. Serialized as-is (the writer's ``json.dump(default=str)``
  coerces any non-JSON value).

Downstream consumers (#1880 eval-defect detectors, #1881 defect score) read
``examples`` as the item x config outcome/token matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

logger = get_logger(__name__)

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


def _cost_flagged_unmeasured(row_flag: Any, metrics: dict[str, Any]) -> bool:
    """True when the row (or its metrics) flags its cost as unmeasured (#2517).

    Mirrors ``traigent.utils.trial_costs.is_cost_unmeasured`` (kept stdlib-only
    here so the matrix stays a pure projection): ``True`` or a finite number
    greater than zero is the flag; ``0`` / ``False`` / absent is not.
    """
    for flag in (row_flag, metrics.get("cost_unmeasured")):
        if flag is True:
            return True
        if (
            isinstance(flag, (int, float))
            and not isinstance(flag, bool)
            and math.isfinite(flag)
            and flag > 0
        ):
            return True
    return False


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
    # carries a top-level ``cost_usd``. A row flagged ``cost_unmeasured`` has no
    # cost: its ``cost_usd`` is a 0.0 placeholder the flag disowns, so the cell
    # records ``null``, never a measured zero (#2517).
    if _cost_flagged_unmeasured(_get(ex, "cost_unmeasured"), metrics):
        cost_usd = None
    else:
        cost_usd = metrics.get("total_cost")
        if cost_usd is None:
            cost_usd = _get(ex, "cost_usd")

    # Timing kept per-source in its own unit (no unit-mixing): ExampleResult
    # exposes ``execution_time`` (seconds); HybridExampleResult exposes
    # ``latency_ms`` (milliseconds). Each is null when its source didn't set it.
    execution_time = _get(ex, "execution_time")
    latency_ms = _get(ex, "latency_ms")

    # The model's actual answer for this cell, recorded verbatim (no coercion) so
    # the #1880 cross-family consensus detector (D7) can compare answers across
    # configs/families straight from the persisted matrix. Stored ONLY when the
    # cell is not clearly correct (``success is not True`` — the wrong and
    # indeterminate cells), which is exactly the subset D7 reads: a clearly-correct
    # cell never anchors consensus, so keeping its full model output would only
    # bloat the persisted matrix (examples x configs x output length) with data no
    # detector reads. ``None`` for clearly-correct cells and when the run captured
    # no output (e.g. a failed example).
    predicted = _get(ex, "actual_output") if success is not True else None

    cell: dict[str, Any] = {
        "score": score,
        "accuracy": accuracy,
        "metrics": dict(metrics),
        "success": success,
        "tokens": _extract_tokens(metrics),
        "cost_usd": cost_usd,
        "execution_time": execution_time,
        "latency_ms": latency_ms,
        "predicted": predicted,
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


def _fill_trial_cells(
    index: int,
    trial_id: str,
    example_results: list[Any],
    rows: dict[str, dict[str, Any]],
    order: list[str],
    merged_cells: dict[int, int],
) -> None:
    """Add one trial's per-example cells to ``rows`` (mutating the accumulators)."""
    for ex in example_results:
        example_id = _get(ex, "example_id")
        if example_id is None:
            continue
        example_id = str(example_id)
        if example_id not in rows:
            rows[example_id] = {}
            order.append(example_id)
        elif trial_id in rows[example_id]:
            merged_cells[index] = merged_cells.get(index, 0) + 1
        rows[example_id][trial_id] = _extract_cell(ex)


def _warn_merged_cells(merged_cells: dict[int, int]) -> None:
    if not merged_cells:
        return
    # Not an error: with content identity, identical rows legitimately
    # share one example id. But a merge must never be silent, because a
    # buggy id scheme (e.g. one id for every example) looks the same.
    # Content-free: column indexes (the matrix ``trials[].index``) and
    # counts only. Never example ids, and never trial ids either: a trial
    # id is a free-form string set by whoever built the result, so it can
    # carry content.
    logger.warning(
        "outcome matrix keeps one cell per example_id per trial; merged "
        "cells that repeated an example_id within the same trial "
        "(by trial column index): %s",
        ", ".join(
            f"column {index}: {count} merged" for index, count in merged_cells.items()
        ),
    )


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
    # column index -> number of cells that landed on an id already filled in
    # that trial (one cell per id is kept, so these were merged).
    merged_cells: dict[int, int] = {}

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

        _fill_trial_cells(index, trial_id, example_results, rows, order, merged_cells)

    _warn_merged_cells(merged_cells)

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

    artifacts_real = artifacts.resolve()
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        # Containment guard: candidates are constructed under ``artifacts``, but
        # a symlink placed inside the artifacts dir could still point anywhere
        # on disk. Refuse to read a resolved path that escapes the artifacts
        # root rather than silently including an arbitrary file.
        candidate_real = candidate.resolve()
        try:
            candidate_real.relative_to(artifacts_real)
        except ValueError:
            raise ValueError(
                f"Refusing to read outcome matrix artifact outside the artifacts "
                f"directory: {candidate} resolves to {candidate_real}, which is "
                f"not under {artifacts_real}"
            ) from None
        try:
            with open(candidate, encoding="utf-8") as handle:
                return dict(json.load(handle))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt outcome matrix artifact at {candidate}: {exc}"
            ) from exc

    return None


#: Selection modes accepted by :func:`select_failing_example_ids`.
FAILING_EXAMPLE_MODES: tuple[str, ...] = ("all_configs", "best_config")

_FAIL = "fail"
_PASS = "pass"
_UNKNOWN = "unknown"


def _classify_cell(cell: Any, threshold: float) -> str:
    """Classify one matrix cell as ``fail``, ``pass`` or ``unknown``.

    A non-``None`` error or an explicit ``success=False`` is a failure whatever
    the score says. Otherwise the cell's ``score`` (falling back to
    ``accuracy`` when ``score`` is absent/``None``) is compared to
    ``threshold``: strictly below fails, equal or above passes. ``success=True``
    alone proves only that execution succeeded, not that the answer was right,
    so a cell with no usable finite score is ``unknown`` — never treated as 0.
    """
    if not isinstance(cell, Mapping):
        return _UNKNOWN
    if cell.get("error") is not None or cell.get("success") is False:
        return _FAIL
    score = cell.get("score")
    if score is None:
        score = cell.get("accuracy")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return _UNKNOWN
    if not math.isfinite(score):
        return _UNKNOWN
    return _FAIL if score < threshold else _PASS


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"threshold must be a finite number, got {type(threshold).__name__}"
        )
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number, got {threshold!r}")
    return float(threshold)


def _matrix_columns(matrix: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return ``(normalized trial_id, config)`` for every matrix column."""
    trials = matrix.get("trials")
    if not isinstance(trials, list):
        raise ValueError("Malformed outcome matrix: 'trials' must be a list")
    columns: list[tuple[str, Any]] = []
    for position, column in enumerate(trials):
        if not isinstance(column, Mapping) or column.get("trial_id") is None:
            raise ValueError(
                f"Malformed outcome matrix: trials[{position}] must be an object "
                "with a non-null 'trial_id'"
            )
        # Normalize: local-session trial ids are ints, while JSON object keys
        # (the cells map) always reload as strings.
        columns.append((str(column["trial_id"]), column.get("config")))
    return columns


def _winner_column_ids(
    columns: list[tuple[str, Any]],
    best_config: Mapping[str, Any] | None,
    winning_trial_ids: Iterable[Any] | None,
) -> list[str]:
    """Resolve the winning configuration's trial columns.

    Prefers the authoritative ``winning_trial_ids`` stamp (#1854): replicate
    grouping projects configs onto the config-space keys, so full-config
    equality can drop replicates. Falls back to config equality with
    ``best_config`` (``None`` means no winner; ``{}`` is a real config).
    """
    stamped = (
        [str(trial_id) for trial_id in winning_trial_ids]
        if winning_trial_ids is not None
        else []
    )
    if stamped:
        wanted = set(stamped)
        selected = [trial_id for trial_id, _ in columns if trial_id in wanted]
        if not selected:
            raise ValueError(
                "No outcome-matrix column matches the winning trial ids "
                f"{sorted(wanted)}; cannot select best_config failures"
            )
        return selected
    if best_config is None:
        raise ValueError(
            "The run has no best configuration; cannot select best_config failures"
        )
    selected = [trial_id for trial_id, config in columns if config == best_config]
    if not selected:
        raise ValueError(
            "No outcome-matrix column ran the best configuration; cannot select "
            "best_config failures"
        )
    return selected


def _row_id_and_cells(position: int, row: Any) -> tuple[str, dict[str, Any]]:
    """Validate one matrix example row; return its id and string-keyed cells."""
    if not isinstance(row, Mapping):
        raise ValueError(
            f"Malformed outcome matrix: examples[{position}] must be an object"
        )
    example_id = row.get("example_id")
    if not isinstance(example_id, str):
        raise ValueError(
            f"Malformed outcome matrix: examples[{position}].example_id must "
            f"be a string, got {example_id!r}"
        )
    cells = row.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError(
            f"Malformed outcome matrix: examples[{position}].cells must be an object"
        )
    return example_id, {str(key): value for key, value in cells.items()}


def select_failing_example_ids(
    matrix: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    mode: str = "all_configs",
    best_config: Mapping[str, Any] | None = None,
    winning_trial_ids: Iterable[Any] | None = None,
) -> list[str]:
    """Select the ids of examples that failed in an outcome matrix.

    Pure function over the matrix built by :func:`build_outcome_matrix` (or
    loaded by :func:`load_outcome_matrix`); it re-scores nothing. A cell fails
    on an error, on ``success=False``, or on a finite score strictly below
    ``threshold`` (see :func:`_classify_cell`). An example is selected only
    when **every** selected column holds a failing cell — a passing, missing or
    unscorable cell excludes it, so the result measures persistent failure, not
    failure frequency.

    Args:
        matrix: The outcome matrix.
        threshold: Finite score cutoff; ``score < threshold`` fails, equality
            passes.
        mode: ``"all_configs"`` (every trial column) or ``"best_config"`` (the
            winning configuration's columns only).
        best_config: The run's best configuration, used for ``best_config``
            mode when ``winning_trial_ids`` is not given.
        winning_trial_ids: The winning configuration's trial ids, preferred
            over config equality for ``best_config`` mode.

    Returns:
        Failing example ids in matrix row order, each once. ``[]`` when the
        matrix carries usable evidence but no example failed everywhere.

    Raises:
        ValueError: On an invalid threshold or mode, a malformed matrix, a
            matrix with no per-example detail, no usable pass/fail signal in
            the selected columns, or no resolvable winner for ``best_config``.
    """
    threshold = _validate_threshold(threshold)
    if mode not in FAILING_EXAMPLE_MODES:
        raise ValueError(
            f"mode must be one of {list(FAILING_EXAMPLE_MODES)}, got {mode!r}"
        )
    if not isinstance(matrix, Mapping):
        raise ValueError("Malformed outcome matrix: expected a JSON object")
    examples = matrix.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Malformed outcome matrix: 'examples' must be a list")
    if not examples:
        raise ValueError(
            "The run captured no per-example detail; cannot select failing examples"
        )

    columns = _matrix_columns(matrix)
    if mode == "best_config":
        selected_ids = _winner_column_ids(columns, best_config, winning_trial_ids)
    else:
        selected_ids = [trial_id for trial_id, _ in columns]
    if not selected_ids:
        raise ValueError("The outcome matrix has no trial columns")

    failing: list[str] = []
    seen: set[str] = set()
    has_signal = False
    for position, row in enumerate(examples):
        example_id, cells_by_id = _row_id_and_cells(position, row)
        verdicts = [
            _classify_cell(cells_by_id.get(trial_id), threshold)
            for trial_id in selected_ids
        ]
        if any(verdict != _UNKNOWN for verdict in verdicts):
            has_signal = True
        if all(verdict == _FAIL for verdict in verdicts) and example_id not in seen:
            seen.add(example_id)
            failing.append(example_id)

    if not has_signal:
        raise ValueError(
            "No cell in the selected columns carries a usable pass/fail signal "
            "(no error, success=False, or finite score/accuracy); cannot select "
            "failing examples"
        )
    return failing
