"""Rich trial results table for optimization output.

Renders a formatted ASCII table with box-drawing characters showing
trial configurations and metrics, highlighting the best trial.

Example output::

    ┌──────────────── Trial Results (6 trials) ─────────────────┐
    │  #   │     model      │ temperature  │ accuracy  │
    ├──────┼────────────────┼──────────────┼───────────┤
    │    1 │  gpt-4o-mini   │     0.0      │   72.3%   │
    │ ★ 2  │    gpt-4o      │     0.0      │   87.4%   │
    └──────┴────────────────┴──────────────┴───────────┘
    Legend: ★ Overall Best
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.core.result import OptimizationResult

BEST_METRIC_REL_TOL = 1e-9
BEST_METRIC_ABS_TOL = 1e-12

# Characters that cannot be encoded on narrow consoles (e.g. Windows cp1252) and
# their ASCII fallbacks.  Applied lazily only when an encode error is detected.
_TABLE_UNICODE_FALLBACKS: dict[str, str] = {
    "─": "-",
    "│": "|",
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",
    "┴": "+",
    "├": "+",
    "┤": "+",
    "┼": "+",
    "★": "*",
    "⚠": "!",
}


def _safe_table_print(text: str, **kwargs: Any) -> None:
    """Print table output, replacing non-encodable chars instead of crashing."""
    encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        for uni, asc in _TABLE_UNICODE_FALLBACKS.items():
            text = text.replace(uni, asc)
        try:
            text.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            text = text.encode(encoding, errors="replace").decode(encoding)
    print(text, **kwargs)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


def _check_color_support() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb"


def _get_colors() -> tuple[str, str, str, str, str, str]:
    """Return (BOLD, GREEN, YELLOW, CYAN, DIM, RESET) per-call based on TTY."""
    if _check_color_support():
        return "\033[1m", "\033[92m", "\033[93m", "\033[96m", "\033[2m", "\033[0m"
    return "", "", "", "", "", ""


class _Colors:
    """ANSI color codes resolved per-call (never permanently mutated)."""

    BOLD = ""
    GREEN = ""
    YELLOW = ""
    CYAN = ""
    DIM = ""
    RESET = ""

    @classmethod
    def refresh(cls) -> None:
        """Refresh color codes based on current TTY state."""
        cls.BOLD, cls.GREEN, cls.YELLOW, cls.CYAN, cls.DIM, cls.RESET = _get_colors()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_config_value(val: Any) -> str:
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _format_metric_value(metric: str, val: float) -> str:
    if metric == "cost":
        return f"${val:.5f}"
    if metric == "latency":
        return f"{val:.3f}s"
    return f"{val:.1%}"


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None


def _example_success_counts(trial: Any) -> tuple[int, int] | None:
    metadata = getattr(trial, "metadata", {}) or {}
    successful = _coerce_int(metadata.get("successful_examples"))
    attempted = _coerce_int(metadata.get("examples_attempted"))
    if attempted is None:
        evaluation_result = metadata.get("evaluation_result")
        attempted = _coerce_int(getattr(evaluation_result, "total_examples", None))
    if successful is None or attempted is None:
        return None
    return successful, attempted


def _get_objective_info(objectives: Any) -> list[tuple[str, str]]:
    """Return [(name, orientation)] from an objectives definition."""
    if hasattr(objectives, "objectives"):
        return [
            (obj.name, getattr(obj, "orientation", "maximize"))
            for obj in objectives.objectives
        ]
    if isinstance(objectives, (list, tuple)):
        result = []
        for o in objectives:
            name = str(o)
            if name in ("cost", "latency"):
                result.append((name, "minimize"))
            else:
                result.append((name, "maximize"))
        return result
    return []


def _find_best_per_objective(
    trials: list,
    metric_info: list[tuple[str, str]],
    metric_overrides: dict[str, list[float]] | None = None,
) -> dict[str, set[int]]:
    """Return {metric_name: {best_trial_indices}} for all tied bests."""
    best_indices: dict[str, set[int]] = {}
    for metric_name, orientation in metric_info:
        best_set: set[int] = set()
        best_val: float | None = None
        is_minimize = orientation == "minimize"
        for i, trial in enumerate(trials):
            if not _trial_is_best_candidate(trial):
                continue
            val = _coerce_float(
                _get_metric_value(trial, metric_name, i, metric_overrides)
            )
            if val is None:
                continue
            if best_val is None:
                best_val = val
                best_set = {i}
                continue
            if math.isclose(
                val,
                best_val,
                rel_tol=BEST_METRIC_REL_TOL,
                abs_tol=BEST_METRIC_ABS_TOL,
            ):
                best_set.add(i)
                continue
            if (val < best_val) if is_minimize else (val > best_val):
                best_val = val
                best_set = {i}
        best_indices[metric_name] = best_set
    return best_indices


def _get_metric_value(
    trial: Any,
    metric_name: str,
    trial_index: int,
    metric_overrides: dict[str, list[float]] | None = None,
) -> Any:
    if metric_overrides and metric_name in metric_overrides:
        override_values = metric_overrides[metric_name]
        if trial_index < len(override_values):
            return override_values[trial_index]
    return getattr(trial, "metrics", {}).get(metric_name)


def _find_best_trial(
    trials: list,
    metric_names: list[str],
    weighted_scores: list[tuple[Any, float]] | None = None,
    metric_info: list[tuple[str, str]] | None = None,
) -> Any:
    """Find the best trial by weighted scores or the primary displayed metric."""
    if weighted_scores:
        return max(weighted_scores, key=lambda item: item[1])[0]

    eligible_trials = [trial for trial in trials if _trial_is_best_candidate(trial)]
    if not eligible_trials:
        return None

    orientation_by_metric = dict(metric_info or [])
    primary_metric = (
        "accuracy"
        if "accuracy" in metric_names
        else (metric_names[0] if metric_names else None)
    )
    if primary_metric is None:
        return eligible_trials[0]

    is_minimize = orientation_by_metric.get(primary_metric) == "minimize"
    chooser = min if is_minimize else max
    missing = float("inf") if is_minimize else float("-inf")

    def primary_score(trial: Any) -> float:
        value = _coerce_float(getattr(trial, "metrics", {}).get(primary_metric))
        return missing if value is None else value

    return chooser(
        eligible_trials,
        key=primary_score,
    )


def _has_positive_quality_metric(trial: Any) -> bool:
    """Return True when the displayed metrics prove at least one scored example."""
    quality_metric_names = {"accuracy", "score", "success_rate"}
    metrics = getattr(trial, "metrics", {}) or {}

    for raw_name, raw_value in metrics.items():
        metric_name = str(raw_name).lower()
        if metric_name not in quality_metric_names and not metric_name.endswith(
            "_accuracy"
        ):
            continue
        try:
            if float(raw_value) > 0:
                return True
        except (TypeError, ValueError):
            continue

    return False


def _trial_is_best_candidate(trial: Any) -> bool:
    if not getattr(trial, "is_successful", False):
        return False
    counts = _example_success_counts(trial)
    if counts is None:
        return True
    successful, _attempted = counts
    return successful > 0


def _trial_identity_index(trials: list, target_trial: Any) -> int | None:
    if target_trial is None:
        return None
    for index, trial in enumerate(trials):
        if trial is target_trial:
            return index
    target_id = getattr(target_trial, "trial_id", None)
    if target_id is None:
        return None
    for index, trial in enumerate(trials):
        if getattr(trial, "trial_id", None) == target_id:
            return index
    return None


def _find_trial_index_by_id(trials: list, trial_id: Any) -> int | None:
    if trial_id is None:
        return None
    for index, trial in enumerate(trials):
        if getattr(trial, "trial_id", None) == trial_id:
            return index
    return None


def _find_best_trial_index(
    results: OptimizationResult,
    trials: list,
    metric_info: list[tuple[str, str]],
    objectives: Any,
) -> int | None:
    metric_names = [name for name, _orientation in metric_info]
    metadata = getattr(results, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    best_trial_id = metadata.get("best_trial_id") or getattr(
        results, "best_trial_id", None
    )
    best_index = _find_trial_index_by_id(trials, best_trial_id)
    if best_index is not None:
        return best_index

    try:
        weighted = results.calculate_weighted_scores(objective_schema=objectives)
        weighted_scores = weighted.get("weighted_scores") or []
        best_trial = _find_best_trial(
            trials, metric_names, weighted_scores, metric_info
        )
        best_index = _trial_identity_index(trials, best_trial)
        if best_index is not None:
            return best_index
    except Exception:
        pass

    best_trial = _find_best_trial(trials, metric_names, metric_info=metric_info)
    best_index = _trial_identity_index(trials, best_trial)
    if best_index is not None:
        return best_index

    best_config = getattr(results, "best_config", None)
    best_score = _coerce_float(getattr(results, "best_score", None))
    primary_metric = metric_names[0] if metric_names else None
    for index, trial in enumerate(trials):
        if getattr(trial, "config", {}) != best_config:
            continue
        if best_score is None or primary_metric is None:
            return index
        trial_score = _coerce_float(getattr(trial, "metrics", {}).get(primary_metric))
        if trial_score is not None and math.isclose(
            trial_score,
            best_score,
            rel_tol=BEST_METRIC_REL_TOL,
            abs_tol=BEST_METRIC_ABS_TOL,
        ):
            return index
    return None


def _trials_all_failed(trials: list) -> bool:
    """True iff every completed trial explicitly recorded zero successful examples.

    "All failed" here means the table should suppress the "Overall Best" framing
    because no winner can be honestly named. When older/external evaluators do
    not report ``metadata["successful_examples"]``, fall back to the completed
    trial status so honest 0.0 quality scores and cost-only runs remain rankable.
    Explicit zero-success metadata is authoritative even if a stale or mocked
    metric payload also contains a positive quality score.
    """
    for trial in trials:
        if not getattr(trial, "is_successful", False):
            continue

        metadata = getattr(trial, "metadata", {}) or {}
        successful = metadata.get("successful_examples")
        if isinstance(successful, bool):
            return False
        if isinstance(successful, (int, float)):
            if successful > 0:
                return False
            continue
        if successful is not None:
            try:
                if float(successful) > 0:
                    return False
                continue
            except (TypeError, ValueError):
                return False

        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_results_table(
    results: OptimizationResult,
    config_space: dict[str, list[Any]],
    objectives: Any,
    *,
    mode_label: str | None = None,
    metric_overrides: dict[str, list[float]] | None = None,
) -> None:
    """Print a rich trial-results table to stdout.

    Args:
        results: The :class:`OptimizationResult` returned by ``func.optimize()``.
        config_space: Configuration space dict (param name → list of values).
        objectives: Objectives list (e.g. ``["accuracy"]``) or ``ObjectiveSchema``.
        mode_label: Optional label rendered in the table title, e.g. ``"MOCK"``.
        metric_overrides: Optional per-metric display values by trial index.
    """
    trials = getattr(results, "trials", [])
    if not trials:
        _safe_table_print("\nNo trials to display.")
        return

    _Colors.refresh()
    C = _Colors

    # Resolve objectives & metrics present in trial data
    objective_info = _get_objective_info(objectives)
    sample_metrics = getattr(trials[0], "metrics", {})
    metric_info = [(n, o) for n, o in objective_info if n in sample_metrics]
    metric_names = [n for n, _ in metric_info]
    param_names = list(config_space.keys())

    # Best-per-objective indices
    best_per_objective = _find_best_per_objective(
        trials,
        metric_info,
        metric_overrides,
    )

    # Overall best trial identity, from optimizer selection metadata when present.
    best_trial_index = _find_best_trial_index(results, trials, metric_info, objectives)

    # When every trial produced zero successful examples, refuse to crown a winner.
    all_failed = _trials_all_failed(trials)
    example_counts = [_example_success_counts(trial) for trial in trials]
    show_examples = any(count is not None for count in example_counts)

    # Column widths
    col_widths: dict[str, int] = {"#": 4}
    for param in param_names:
        max_len = max(
            len(_format_config_value(getattr(t, "config", {}).get(param, "?")))
            for t in trials
        )
        col_widths[param] = max(len(param), max_len) + 1
    for metric in metric_names:
        max_len = max(
            len(
                _format_metric_value(
                    metric,
                    _coerce_float(_get_metric_value(t, metric, i, metric_overrides))
                    or 0.0,
                )
            )
            for i, t in enumerate(trials)
        )
        col_widths[metric] = max(len(metric), max_len) + 1
    if show_examples:
        max_len = max(
            len(f"{count[0]}/{count[1]}") if count is not None else 1
            for count in example_counts
        )
        col_widths["examples"] = max(len("examples"), max_len) + 1

    all_cols = (
        ["#"] + param_names + metric_names + (["examples"] if show_examples else [])
    )
    total_width = sum(col_widths[c] for c in all_cols) + len(all_cols) * 3 - 1

    # Box-drawing characters
    H, V = "─", "│"
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    BT, LT, RT, XT = "┴", "├", "┤", "┼"

    # Title bar
    label_prefix = f"{mode_label.strip()} - " if mode_label else ""
    title = f" Trial Results ({label_prefix}{len(trials)} trials) "
    padding = (total_width - len(title)) // 2

    _safe_table_print("")
    _safe_table_print(f"{C.BOLD}{TL}{H * total_width}{TR}{C.RESET}")
    _safe_table_print(
        f"{C.BOLD}{V}{' ' * padding}{title}"
        f"{' ' * (total_width - padding - len(title))}{V}{C.RESET}"
    )
    _safe_table_print(f"{C.BOLD}{LT}{H * total_width}{RT}{C.RESET}")

    # Column headers
    header_parts = [f"{C.BOLD}{'#':^{col_widths['#']}}{C.RESET}"]
    header_parts.extend(f"{C.CYAN}{p:^{col_widths[p]}}{C.RESET}" for p in param_names)
    header_parts.extend(
        f"{C.YELLOW}{m:^{col_widths[m]}}{C.RESET}" for m in metric_names
    )
    if show_examples:
        header_parts.append(
            f"{C.YELLOW}{'examples':^{col_widths['examples']}}{C.RESET}"
        )
    _safe_table_print(f"{V} " + f" {V} ".join(header_parts) + f" {V}")

    # Separator
    _safe_table_print(
        f"{LT}" + XT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{RT}"
    )

    # Data rows
    for i, trial in enumerate(trials):
        config = getattr(trial, "config", {})
        is_overall_best = i == best_trial_index and not all_failed

        prefix = f"{C.GREEN}★{C.RESET}" if is_overall_best else " "
        row_parts = [f"{prefix}{i + 1:>{col_widths['#'] - 1}}"]

        for param in param_names:
            val = _format_config_value(config.get(param, "?"))
            row_parts.append(f"{val:^{col_widths[param]}}")

        for metric in metric_names:
            raw_metric_val = _get_metric_value(trial, metric, i, metric_overrides)
            metric_val = _coerce_float(raw_metric_val) or 0.0
            formatted = _format_metric_value(metric, metric_val)
            if not all_failed and i in best_per_objective.get(metric, set()):
                cell = f"{C.GREEN}{C.BOLD}{formatted:^{col_widths[metric]}}{C.RESET}"
            else:
                cell = f"{formatted:^{col_widths[metric]}}"
            row_parts.append(cell)

        if show_examples:
            counts = example_counts[i]
            formatted_counts = f"{counts[0]}/{counts[1]}" if counts is not None else "?"
            row_parts.append(f"{formatted_counts:^{col_widths['examples']}}")

        _safe_table_print(f"{V} " + f" {V} ".join(row_parts) + f" {V}")

    # Bottom border
    _safe_table_print(
        f"{BL}" + BT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{BR}"
    )

    # Legend (or "all failed" banner when no trial succeeded)
    if all_failed:
        _safe_table_print(
            f"{C.YELLOW}⚠ All trials failed — no examples succeeded. "
            f"Inspect per-example errors in "
            f".traigent/optimization_logs/experiments/.../runs/{C.RESET}"
        )
    else:
        legend = [f"{C.GREEN}★{C.RESET} Overall Best"]
        legend.extend(f"{C.GREEN}{C.BOLD}{m}{C.RESET} = Best {m}" for m in metric_names)
        _safe_table_print(f"{C.DIM}Legend: {', '.join(legend)}{C.RESET}")
