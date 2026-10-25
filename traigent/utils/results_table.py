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

from traigent.core import objective_directions
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.result import OptimizationResult

BEST_METRIC_REL_TOL = 1e-9
BEST_METRIC_ABS_TOL = 1e-12

logger = get_logger(__name__)

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


_UNMEASURED_CELL = "unmeasured"


def _format_config_value(val: Any) -> str:
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _format_percent_metric(val: float) -> str:
    """Format a quality metric (accuracy / score / ...) as a percent string.

    Metrics reach the table on either a 0-1 fraction scale or an already-percent
    0-100 scale (e.g. a 0-100 ``accuracy`` of 79.16). Mirror the frontend rule
    (``percentFormatting``): a value ``> 1`` is already a percent and is rendered
    as-is (79.16 -> ``"79.2%"``); a value ``<= 1`` is a fraction and scaled
    (0.79 -> ``"79.0%"``). This never double-scales, so a 0-100 metric can no
    longer render as ">100%" (the prior ``f"{val:.1%}"`` turned 79.16 into
    "7916.2%").
    """
    if val > 1.0:
        return f"{val:.1f}%"
    return f"{val:.1%}"


def _cost_flagged(trial: Any, metric: str) -> bool:
    """True when ``metric`` is a cost metric and the trial flags its cost unmeasured."""
    if "cost" not in metric.lower():
        return False
    from traigent.utils.trial_costs import is_cost_unmeasured  # noqa: PLC0415

    return is_cost_unmeasured(getattr(trial, "metrics", None))


def _render_metric_cell(metric: str, raw_value: Any, trial: Any = None) -> str:
    """Format one table cell; an unmeasured cost reads "unmeasured", not "$0.00000".

    A trial with no captured LLM usage has an UNKNOWN cost (Traigent#2441), and
    printing it as zero makes it look like the cheapest configuration. Other
    metrics keep the historical 0.0 rendering for a missing value.
    """
    value = _coerce_float(raw_value)
    if trial is not None and _cost_flagged(trial, metric):
        # Flagged: the number (if any) is a floor or a stray zero. Show the
        # known spend as a lower bound, else "unmeasured" -- never "$0.00000".
        metrics = getattr(trial, "metrics", None) or {}
        bound = _coerce_float(metrics.get("cost_lower_bound"))
        if bound is not None and bound > 0:
            return f"\u2265 {_format_metric_value(metric, bound)}"
        return _UNMEASURED_CELL
    if value is None and "cost" in metric.lower():
        return _UNMEASURED_CELL
    return _format_metric_value(metric, value or 0.0)


def _format_metric_value(metric: str, val: float) -> str:
    if metric == "cost":
        return f"${val:.5f}"
    if metric == "latency":
        # The "latency" metric holds a mean of per-result ``latency_ms`` values
        # (see ``evaluators/hybrid_api.py`` -> ``aggregated["latency"]``), i.e. it
        # is in MILLISECONDS per the canonical ``_ms`` unit contract. Render it
        # with an ``ms`` label; the prior ``f"{val:.3f}s"`` treated the ms value
        # as seconds, overstating latency 1000x (850 ms -> "850.000s" ~= 14 min).
        return f"{val:.0f}ms"
    return _format_percent_metric(val)


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

    def direction(name: str, explicit: str | None = None) -> str:
        if explicit == "band":
            return "band"
        return objective_directions.resolve_objective_orientation(name, explicit)

    if isinstance(objectives, dict) and isinstance(objectives.get("objectives"), list):
        return [
            (
                str(obj["name"]),
                direction(str(obj["name"]), obj.get("orientation")),
            )
            for obj in objectives["objectives"]
            if isinstance(obj, dict) and "name" in obj
        ]
    if hasattr(objectives, "objectives"):
        return [
            (
                obj.name,
                direction(obj.name, getattr(obj, "orientation", None)),
            )
            for obj in objectives.objectives
        ]
    if isinstance(objectives, (list, tuple)):
        result = []
        for o in objectives:
            name = str(o)
            result.append((name, direction(name)))
        return result
    return []


def _cost_in_play(metric_names: list[str]) -> bool:
    """True when any objective in play is a cost metric (core selection's rule)."""
    from traigent.utils.multi_objective import (  # noqa: PLC0415
        _COST_OBJECTIVE_NAMES,
    )

    return any(name in _COST_OBJECTIVE_NAMES for name in metric_names)


def _trial_rankable(trial: Any, metric_names: list[str]) -> bool:
    """Mirror ``result_selection``: with a cost objective in play, a trial whose
    cost is flagged unmeasured is excluded from any overall-best choice."""
    if not _cost_in_play(metric_names):
        return True
    from traigent.utils.trial_costs import is_cost_unmeasured  # noqa: PLC0415

    return not is_cost_unmeasured(getattr(trial, "metrics", None))


def _all_eligible_cost_unmeasured(trials: list, metric_names: list[str]) -> bool:
    """True when there are eligible trials and each is excluded for unmeasured cost."""
    eligible = [t for t in trials if _trial_is_best_candidate(t)]
    return bool(eligible) and not any(
        _trial_rankable(t, metric_names) for t in eligible
    )


def _find_best_per_objective(
    trials: list,
    metric_info: list[tuple[str, str]],
    metric_overrides: dict[str, list[float]] | None = None,
) -> dict[str, set[int]]:
    """Return {metric_name: {best_trial_indices}} for all tied bests."""
    best_indices: dict[str, set[int]] = {}
    for metric_name, orientation in metric_info:
        if orientation == "band":
            best_indices[metric_name] = set()
            continue
        best_set: set[int] = set()
        best_val: float | None = None
        is_minimize = orientation == "minimize"
        for i, trial in enumerate(trials):
            if not _trial_is_best_candidate(trial):
                continue
            if _cost_flagged(trial, metric_name):
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
        weighted_scores = [
            item for item in weighted_scores if _trial_rankable(item[0], metric_names)
        ]
        if weighted_scores:
            return max(weighted_scores, key=lambda item: item[1])[0]

    eligible_trials = [trial for trial in trials if _trial_is_best_candidate(trial)]
    if not eligible_trials:
        return None

    orientation_by_metric = dict(metric_info or [])
    primary_metric = metric_names[0] if metric_names else None
    if primary_metric is None:
        return eligible_trials[0]

    orientation = orientation_by_metric.get(primary_metric)
    if orientation == "band":
        # The table lacks the target bounds needed to compute band distance.
        # Let the persisted best trial/config fallback identify the winner.
        return None
    is_minimize = orientation == "minimize"
    chooser = min if is_minimize else max
    missing = float("inf") if is_minimize else float("-inf")

    def primary_score(trial: Any) -> float:
        if _cost_flagged(trial, primary_metric):
            return missing
        value = _coerce_float(getattr(trial, "metrics", {}).get(primary_metric))
        return missing if value is None else value

    rankable = [t for t in eligible_trials if _trial_rankable(t, metric_names)]
    if not rankable:
        # Every candidate's cost is unmeasured: no measured winner.
        return None
    return chooser(rankable, key=primary_score)


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
    # The terminal selector found no ranking-eligible trial: the result has no
    # winner, so the table must not star one (Traigent#2664).
    if (
        best_trial_id is None
        and getattr(results, "best_config", None) is None
        and str(getattr(results, "reason_code", None) or "")
        == "NO_RANKING_ELIGIBLE_TRIALS"
    ):
        return None
    best_index = _find_trial_index_by_id(trials, best_trial_id)
    if best_index is not None and _trial_rankable(trials[best_index], metric_names):
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
    except Exception as exc:
        logger.debug(
            "Falling back from weighted to unweighted results-table ranking: %s",
            exc,
            exc_info=True,
        )

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
        if not _trial_rankable(trial, metric_names):
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


def _mock_run_notice(results: Any) -> str | None:
    """Mock-provenance line for a result, from what the run recorded (#2516).

    A recorded ``mock_llm_run`` flag is authoritative in both directions. A
    result with no recorded flag (hand-built, loaded from a legacy save, or an
    empty result) is unknown, never asserted real: it gets a hedged note only
    while mock mode is on in this process, and nothing otherwise.
    """
    metadata = getattr(results, "metadata", None)
    recorded = metadata.get("mock_llm_run") if isinstance(metadata, dict) else None
    if recorded is True and isinstance(metadata, dict):
        if metadata.get("mock_llm_mixed") is True:
            return (
                "Some trials used simulated LLM calls; scores and costs mix "
                "simulated and real results."
            )
        return (
            "MOCK RUN — LLM calls were simulated. Output-based scores "
            "are not meaningful and any cost shown is a synthetic estimate, "
            "not real spend."
        )
    if recorded is False:
        return None
    if _mock_run_active():
        return (
            "Mock mode is on in this process and this result does not record "
            "whether it was produced under it; its scores and costs may be "
            "simulated."
        )
    return None


def _mock_run_active() -> bool:
    """Whether LLM calls are being mocked (never raises; production is False)."""
    try:
        from traigent.utils.env_config import is_mock_llm

        return bool(is_mock_llm())
    except Exception:  # pragma: no cover - display must never break a run
        return False


def _local_fallback_notice(results: OptimizationResult) -> str | None:
    """Return the user-facing local-fallback notice for this result, if any.

    Issue #2024: ``result.metadata`` already carried ``source`` and
    ``fallback_reason``, but nothing printed reflected them, so a run that
    degraded to a local search rendered exactly like managed optimization.
    Printing it under the table makes the caveat travel with the number.

    The remedy is selected from the typed ``fallback_reason_code`` the run
    stamped, never from ``fallback_reason`` — that string embeds the backend's
    own ``error_code``/``message``/``raw_body`` text, so matching against it
    would let a backend 500 body decide what we tell the user about their key.
    """
    from traigent.core.execution_policy_runtime import (
        SOURCE_LOCAL_FALLBACK,
        failure_reason_from_code,
        local_fallback_notice,
    )

    metadata = getattr(results, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return None
    if metadata.get("source") != SOURCE_LOCAL_FALLBACK:
        return None
    code = metadata.get("fallback_reason_code")
    return local_fallback_notice(
        metadata.get("fallback_reason"),
        failure_reason_from_code(code if isinstance(code, str) else None),
    )


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


# Box-drawing characters
_H, _V = "─", "│"
_TL, _TR, _BL, _BR = "┌", "┐", "└", "┘"
_BT, _LT, _RT, _XT = "┴", "├", "┤", "┼"


def _table_column_widths(
    trials: list,
    param_names: list[str],
    metric_names: list[str],
    metric_overrides: dict[str, list[float]] | None,
    example_counts: list[tuple[int, int] | None],
    show_examples: bool,
) -> dict[str, int]:
    """Width of every column: the widest of its header and rendered cells."""
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
                _render_metric_cell(
                    metric, _get_metric_value(t, metric, i, metric_overrides), t
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
    return col_widths


def _print_title_bar(
    colors: Any, trial_count: int, mode_label: str | None, total_width: int
) -> None:
    C = colors
    label_prefix = f"{mode_label.strip()} - " if mode_label else ""
    title = f" Trial Results ({label_prefix}{trial_count} trials) "
    padding = (total_width - len(title)) // 2

    _safe_table_print("")
    _safe_table_print(f"{C.BOLD}{_TL}{_H * total_width}{_TR}{C.RESET}")
    _safe_table_print(
        f"{C.BOLD}{_V}{' ' * padding}{title}"
        f"{' ' * (total_width - padding - len(title))}{_V}{C.RESET}"
    )
    _safe_table_print(f"{C.BOLD}{_LT}{_H * total_width}{_RT}{C.RESET}")


def _print_column_headers(
    colors: Any,
    col_widths: dict[str, int],
    param_names: list[str],
    metric_names: list[str],
    show_examples: bool,
    all_cols: list[str],
) -> None:
    C = colors
    header_parts = [f"{C.BOLD}{'#':^{col_widths['#']}}{C.RESET}"]
    header_parts.extend(f"{C.CYAN}{p:^{col_widths[p]}}{C.RESET}" for p in param_names)
    header_parts.extend(
        f"{C.YELLOW}{m:^{col_widths[m]}}{C.RESET}" for m in metric_names
    )
    if show_examples:
        header_parts.append(
            f"{C.YELLOW}{'examples':^{col_widths['examples']}}{C.RESET}"
        )
    _safe_table_print(f"{_V} " + f" {_V} ".join(header_parts) + f" {_V}")

    # Separator
    _safe_table_print(
        f"{_LT}" + _XT.join(_H * (col_widths[c] + 2) for c in all_cols) + f"{_RT}"
    )


def _table_data_row(
    colors: Any,
    index: int,
    trial: Any,
    col_widths: dict[str, int],
    param_names: list[str],
    metric_names: list[str],
    metric_overrides: dict[str, list[float]] | None,
    *,
    best_trial_index: int | None,
    all_failed: bool,
    best_per_objective: dict[str, set[int]],
    show_examples: bool,
    example_counts: list[tuple[int, int] | None],
) -> str:
    C = colors
    config = getattr(trial, "config", {})
    is_overall_best = index == best_trial_index and not all_failed

    prefix = f"{C.GREEN}★{C.RESET}" if is_overall_best else " "
    row_parts = [f"{prefix}{index + 1:>{col_widths['#'] - 1}}"]

    for param in param_names:
        val = _format_config_value(config.get(param, "?"))
        row_parts.append(f"{val:^{col_widths[param]}}")

    for metric in metric_names:
        raw_metric_val = _get_metric_value(trial, metric, index, metric_overrides)
        formatted = _render_metric_cell(metric, raw_metric_val, trial)
        if not all_failed and index in best_per_objective.get(metric, set()):
            cell = f"{C.GREEN}{C.BOLD}{formatted:^{col_widths[metric]}}{C.RESET}"
        else:
            cell = f"{formatted:^{col_widths[metric]}}"
        row_parts.append(cell)

    if show_examples:
        counts = example_counts[index]
        formatted_counts = f"{counts[0]}/{counts[1]}" if counts is not None else "?"
        row_parts.append(f"{formatted_counts:^{col_widths['examples']}}")

    return f"{_V} " + f" {_V} ".join(row_parts) + f" {_V}"


def _print_table_legend(
    colors: Any,
    trials: list,
    metric_names: list[str],
    *,
    best_trial_index: int | None,
    all_failed: bool,
    show_best: bool,
) -> None:
    """Legend (or "all failed" / "no winner" banner) under the table."""
    C = colors
    if all_failed:
        _safe_table_print(
            f"{C.YELLOW}⚠ All trials failed — no examples succeeded. "
            f"Inspect per-example errors in "
            f".traigent/optimization_logs/experiments/.../runs/{C.RESET}"
        )
    elif not show_best:
        # Plumbing run: no winner framing, so no legend or "no winner" notice.
        pass
    elif best_trial_index is None and _all_eligible_cost_unmeasured(
        trials, metric_names
    ):
        _safe_table_print(
            f"{C.YELLOW}⚠ No measured winner — every candidate's cost is "
            f"unmeasured, so no trial is starred.{C.RESET}"
        )
    elif best_trial_index is None:
        _safe_table_print(
            f"{C.YELLOW}⚠ No overall best could be determined, so no trial "
            f"is starred.{C.RESET}"
        )
    else:
        legend = [f"{C.GREEN}★{C.RESET} Overall Best"]
        legend.extend(f"{C.GREEN}{C.BOLD}{m}{C.RESET} = Best {m}" for m in metric_names)
        _safe_table_print(f"{C.DIM}Legend: {', '.join(legend)}{C.RESET}")


def _print_run_notices(colors: Any, results: Any) -> None:
    """Stop-limit, mock-run and local-fallback caveats printed under the table."""
    C = colors
    # A run cut short by a limit says which one (issue #2518).
    stop_detail = (getattr(results, "metadata", None) or {}).get("stop_detail")
    if isinstance(stop_detail, str) and stop_detail:
        _safe_table_print(f"{C.YELLOW}⚠ {stop_detail}{C.RESET}")

    # A mock run says so next to the numbers it produced (issue #2516): the
    # calls never reached a provider, so scores are canned and any cost is a
    # synthetic estimate, not spend. The notice comes from provenance recorded
    # on the result when the run was built, never from current process state.
    mock_notice = _mock_run_notice(results)
    if mock_notice:
        _safe_table_print(f"{C.YELLOW}⚠ {mock_notice}{C.RESET}")

    # Provenance caveat (issue #2024): a local-fallback run must say so next to
    # the number it produced, not only in a startup log line.
    fallback_notice = _local_fallback_notice(results)
    if fallback_notice:
        _safe_table_print(f"{C.YELLOW}⚠ {fallback_notice}{C.RESET}")


def print_results_table(
    results: OptimizationResult,
    config_space: dict[str, list[Any]],
    objectives: Any,
    *,
    mode_label: str | None = None,
    metric_overrides: dict[str, list[float]] | None = None,
    show_best: bool = True,
) -> None:
    """Print a rich trial-results table to stdout.

    Args:
        results: The :class:`OptimizationResult` returned by ``func.optimize()``.
        config_space: Configuration space dict (param name → list of values).
        objectives: Objectives list (e.g. ``["accuracy"]``) or ``ObjectiveSchema``.
        mode_label: Optional label rendered in the table title, e.g. ``"MOCK"``.
        metric_overrides: Optional per-metric display values by trial index.
        show_best: When False, no trial is starred and no "Overall Best"
            legend or winner notice is printed (plumbing checks, #1690).
    """
    trials = results.trials
    if not trials:
        _safe_table_print("\nNo trials to display.")
        return

    _Colors.refresh()
    C = _Colors

    # Resolve objectives & metrics present in trial data
    objective_info = _get_objective_info(objectives)
    sample_metrics = trials[0].metrics
    # A declared cost objective keeps its column even when no trial measured it:
    # dropping the column would hide the gap, and the cells then read
    # "unmeasured" rather than a fabricated $0 (Traigent#2517).
    metric_info = [
        (n, o) for n, o in objective_info if n in sample_metrics or "cost" in n.lower()
    ]
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
    if not show_best:
        best_trial_index = None

    # When every trial produced zero successful examples, refuse to crown a winner.
    all_failed = _trials_all_failed(trials)
    example_counts = [_example_success_counts(trial) for trial in trials]
    show_examples = any(count is not None for count in example_counts)

    col_widths = _table_column_widths(
        trials,
        param_names,
        metric_names,
        metric_overrides,
        example_counts,
        show_examples,
    )

    all_cols = (
        ["#"] + param_names + metric_names + (["examples"] if show_examples else [])
    )
    total_width = sum(col_widths[c] for c in all_cols) + len(all_cols) * 3 - 1

    _print_title_bar(C, len(trials), mode_label, total_width)
    _print_column_headers(
        C, col_widths, param_names, metric_names, show_examples, all_cols
    )

    # Data rows
    for i, trial in enumerate(trials):
        _safe_table_print(
            _table_data_row(
                C,
                i,
                trial,
                col_widths,
                param_names,
                metric_names,
                metric_overrides,
                best_trial_index=best_trial_index,
                all_failed=all_failed,
                best_per_objective=best_per_objective,
                show_examples=show_examples,
                example_counts=example_counts,
            )
        )

    # Bottom border
    _safe_table_print(
        f"{_BL}" + _BT.join(_H * (col_widths[c] + 2) for c in all_cols) + f"{_BR}"
    )

    _print_table_legend(
        C,
        trials,
        metric_names,
        best_trial_index=best_trial_index,
        all_failed=all_failed,
        show_best=show_best,
    )
    _print_run_notices(C, results)
