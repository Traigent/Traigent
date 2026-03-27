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

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.core.result import OptimizationResult


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
    trials: list, metric_info: list[tuple[str, str]]
) -> dict[str, int]:
    """Return {metric_name: best_trial_index}."""
    best_indices: dict[str, int] = {}
    for metric_name, orientation in metric_info:
        best_idx = 0
        best_val = getattr(trials[0], "metrics", {}).get(metric_name, 0)
        is_minimize = orientation == "minimize"
        for i, trial in enumerate(trials[1:], 1):
            val = getattr(trial, "metrics", {}).get(metric_name, 0)
            if (val < best_val) if is_minimize else (val > best_val):
                best_val = val
                best_idx = i
        best_indices[metric_name] = best_idx
    return best_indices


def _find_best_trial(trials: list, metric_names: list[str]) -> Any:
    """Find the best trial by weighted score or accuracy."""
    best_trial = trials[0]
    for trial in trials:
        score = getattr(trial, "weighted_score", None)
        if score is not None:
            best_score = getattr(best_trial, "weighted_score", float("-inf"))
            if best_score < score:
                best_trial = trial
        elif metric_names and "accuracy" in metric_names:
            trial_acc = getattr(trial, "metrics", {}).get("accuracy", 0)
            best_acc = getattr(best_trial, "metrics", {}).get("accuracy", 0)
            if trial_acc > best_acc:
                best_trial = trial
    return best_trial


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_results_table(
    results: OptimizationResult,
    config_space: dict[str, list[Any]],
    objectives: Any,
) -> None:
    """Print a rich trial-results table to stdout.

    Args:
        results: The :class:`OptimizationResult` returned by ``func.optimize()``.
        config_space: Configuration space dict (param name → list of values).
        objectives: Objectives list (e.g. ``["accuracy"]``) or ``ObjectiveSchema``.
    """
    trials = getattr(results, "trials", [])
    if not trials:
        print("\nNo trials to display.")
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
    best_per_objective = _find_best_per_objective(trials, metric_info)

    # Overall best config (weighted scoring when available)
    try:
        weighted = results.calculate_weighted_scores(objective_schema=objectives)
        best_config = weighted.get("best_weighted_config", {})
    except Exception:
        best_config = getattr(_find_best_trial(trials, metric_names), "config", {})

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
            len(_format_metric_value(metric, getattr(t, "metrics", {}).get(metric, 0)))
            for t in trials
        )
        col_widths[metric] = max(len(metric), max_len) + 1

    all_cols = ["#"] + param_names + metric_names
    total_width = sum(col_widths[c] for c in all_cols) + len(all_cols) * 3 - 1

    # Box-drawing characters
    H, V = "─", "│"
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    BT, LT, RT, XT = "┴", "├", "┤", "┼"

    # Title bar
    title = f" Trial Results ({len(trials)} trials) "
    padding = (total_width - len(title)) // 2

    print()
    print(f"{C.BOLD}{TL}{H * total_width}{TR}{C.RESET}")
    print(
        f"{C.BOLD}{V}{' ' * padding}{title}"
        f"{' ' * (total_width - padding - len(title))}{V}{C.RESET}"
    )
    print(f"{C.BOLD}{LT}{H * total_width}{RT}{C.RESET}")

    # Column headers
    header_parts = [f"{C.BOLD}{'#':^{col_widths['#']}}{C.RESET}"]
    header_parts.extend(f"{C.CYAN}{p:^{col_widths[p]}}{C.RESET}" for p in param_names)
    header_parts.extend(
        f"{C.YELLOW}{m:^{col_widths[m]}}{C.RESET}" for m in metric_names
    )
    print(f"{V} " + f" {V} ".join(header_parts) + f" {V}")

    # Separator
    print(f"{LT}" + XT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{RT}")

    # Data rows
    for i, trial in enumerate(trials):
        config = getattr(trial, "config", {})
        metrics = getattr(trial, "metrics", {})
        is_overall_best = config == best_config

        prefix = f"{C.GREEN}★{C.RESET}" if is_overall_best else " "
        row_parts = [f"{prefix}{i + 1:>{col_widths['#'] - 1}}"]

        for param in param_names:
            val = _format_config_value(config.get(param, "?"))
            row_parts.append(f"{val:^{col_widths[param]}}")

        for metric in metric_names:
            metric_val = float(metrics.get(metric, 0))
            formatted = _format_metric_value(metric, metric_val)
            if best_per_objective.get(metric) == i:
                cell = f"{C.GREEN}{C.BOLD}{formatted:^{col_widths[metric]}}{C.RESET}"
            else:
                cell = f"{formatted:^{col_widths[metric]}}"
            row_parts.append(cell)

        print(f"{V} " + f" {V} ".join(row_parts) + f" {V}")

    # Bottom border
    print(f"{BL}" + BT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{BR}")

    # Legend
    legend = [f"{C.GREEN}★{C.RESET} Overall Best"]
    legend.extend(f"{C.GREEN}{C.BOLD}{m}{C.RESET} = Best {m}" for m in metric_names)
    print(f"{C.DIM}Legend: {', '.join(legend)}{C.RESET}")
