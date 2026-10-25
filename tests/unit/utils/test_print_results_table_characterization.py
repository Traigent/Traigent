"""Characterization of ``print_results_table``.

``oracle_print_results_table`` is a frozen copy of the pre-refactor function.
Both versions are run on identical results with every module-level helper they
use replaced by a recording wrapper, and compared on the printed text and on
the full sequence of helper calls (arguments and order).
"""

from __future__ import annotations

import itertools
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from traigent.utils import results_table as rt
from traigent.utils.results_table import print_results_table

if TYPE_CHECKING:
    from traigent.core.result import OptimizationResult


# --- frozen pre-refactor implementation ---
def oracle_print_results_table(
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
        rt._safe_table_print("\nNo trials to display.")
        return

    rt._Colors.refresh()
    C = rt._Colors

    # Resolve objectives & metrics present in trial data
    objective_info = rt._get_objective_info(objectives)
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
    best_per_objective = rt._find_best_per_objective(
        trials,
        metric_info,
        metric_overrides,
    )

    # Overall best trial identity, from optimizer selection metadata when present.
    best_trial_index = rt._find_best_trial_index(
        results, trials, metric_info, objectives
    )
    if not show_best:
        best_trial_index = None

    # When every trial produced zero successful examples, refuse to crown a winner.
    all_failed = rt._trials_all_failed(trials)
    example_counts = [rt._example_success_counts(trial) for trial in trials]
    show_examples = any(count is not None for count in example_counts)

    # Column widths
    col_widths: dict[str, int] = {"#": 4}
    for param in param_names:
        max_len = max(
            len(rt._format_config_value(getattr(t, "config", {}).get(param, "?")))
            for t in trials
        )
        col_widths[param] = max(len(param), max_len) + 1
    for metric in metric_names:
        max_len = max(
            len(
                rt._render_metric_cell(
                    metric, rt._get_metric_value(t, metric, i, metric_overrides), t
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

    rt._safe_table_print("")
    rt._safe_table_print(f"{C.BOLD}{TL}{H * total_width}{TR}{C.RESET}")
    rt._safe_table_print(
        f"{C.BOLD}{V}{' ' * padding}{title}"
        f"{' ' * (total_width - padding - len(title))}{V}{C.RESET}"
    )
    rt._safe_table_print(f"{C.BOLD}{LT}{H * total_width}{RT}{C.RESET}")

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
    rt._safe_table_print(f"{V} " + f" {V} ".join(header_parts) + f" {V}")

    # Separator
    rt._safe_table_print(
        f"{LT}" + XT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{RT}"
    )

    # Data rows
    for i, trial in enumerate(trials):
        config = getattr(trial, "config", {})
        is_overall_best = i == best_trial_index and not all_failed

        prefix = f"{C.GREEN}★{C.RESET}" if is_overall_best else " "
        row_parts = [f"{prefix}{i + 1:>{col_widths['#'] - 1}}"]

        for param in param_names:
            val = rt._format_config_value(config.get(param, "?"))
            row_parts.append(f"{val:^{col_widths[param]}}")

        for metric in metric_names:
            raw_metric_val = rt._get_metric_value(trial, metric, i, metric_overrides)
            formatted = rt._render_metric_cell(metric, raw_metric_val, trial)
            if not all_failed and i in best_per_objective.get(metric, set()):
                cell = f"{C.GREEN}{C.BOLD}{formatted:^{col_widths[metric]}}{C.RESET}"
            else:
                cell = f"{formatted:^{col_widths[metric]}}"
            row_parts.append(cell)

        if show_examples:
            counts = example_counts[i]
            formatted_counts = f"{counts[0]}/{counts[1]}" if counts is not None else "?"
            row_parts.append(f"{formatted_counts:^{col_widths['examples']}}")

        rt._safe_table_print(f"{V} " + f" {V} ".join(row_parts) + f" {V}")

    # Bottom border
    rt._safe_table_print(
        f"{BL}" + BT.join(H * (col_widths[c] + 2) for c in all_cols) + f"{BR}"
    )

    # Legend (or "all failed" banner when no trial succeeded)
    if all_failed:
        rt._safe_table_print(
            f"{C.YELLOW}⚠ All trials failed — no examples succeeded. "
            f"Inspect per-example errors in "
            f".traigent/optimization_logs/experiments/.../runs/{C.RESET}"
        )
    elif not show_best:
        # Plumbing run: no winner framing, so no legend or "no winner" notice.
        pass
    elif best_trial_index is None and rt._all_eligible_cost_unmeasured(
        trials, metric_names
    ):
        rt._safe_table_print(
            f"{C.YELLOW}⚠ No measured winner — every candidate's cost is "
            f"unmeasured, so no trial is starred.{C.RESET}"
        )
    elif best_trial_index is None:
        rt._safe_table_print(
            f"{C.YELLOW}⚠ No overall best could be determined, so no trial "
            f"is starred.{C.RESET}"
        )
    else:
        legend = [f"{C.GREEN}★{C.RESET} Overall Best"]
        legend.extend(f"{C.GREEN}{C.BOLD}{m}{C.RESET} = Best {m}" for m in metric_names)
        rt._safe_table_print(f"{C.DIM}Legend: {', '.join(legend)}{C.RESET}")

    # A run cut short by a limit says which one (issue #2518).
    stop_detail = (getattr(results, "metadata", None) or {}).get("stop_detail")
    if isinstance(stop_detail, str) and stop_detail:
        rt._safe_table_print(f"{C.YELLOW}⚠ {stop_detail}{C.RESET}")

    # A mock run says so next to the numbers it produced (issue #2516): the
    # calls never reached a provider, so scores are canned and any cost is a
    # synthetic estimate, not spend. The notice comes from provenance recorded
    # on the result when the run was built, never from current process state.
    mock_notice = rt._mock_run_notice(results)
    if mock_notice:
        rt._safe_table_print(f"{C.YELLOW}⚠ {mock_notice}{C.RESET}")

    # Provenance caveat (issue #2024): a local-fallback run must say so next to
    # the number it produced, not only in a startup log line.
    fallback_notice = rt._local_fallback_notice(results)
    if fallback_notice:
        rt._safe_table_print(f"{C.YELLOW}⚠ {fallback_notice}{C.RESET}")


_WRAPPED = (
    "_get_objective_info",
    "_find_best_per_objective",
    "_find_best_trial_index",
    "_trials_all_failed",
    "_example_success_counts",
    "_format_config_value",
    "_render_metric_cell",
    "_get_metric_value",
    "_all_eligible_cost_unmeasured",
    "_mock_run_notice",
    "_local_fallback_notice",
)

_ADDRESS = re.compile(r"0x[0-9a-f]+")

_ANSI = ("\033[1m", "\033[92m", "\033[93m", "\033[96m", "\033[2m", "\033[0m")


def _trial(trial_id, config, metrics, ok=True, metadata=None):
    return SimpleNamespace(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        is_successful=ok,
        metadata=metadata,
    )


def _results(
    trials,
    metadata=None,
    best_trial_id=None,
    best_config=None,
    best_score=None,
    reason_code=None,
    weighted="ok",
):
    def calculate_weighted_scores(objective_schema=None):
        if weighted == "boom":
            raise RuntimeError("no weights")
        if weighted == "ok":
            return {
                "weighted_scores": [(t, float(idx)) for idx, t in enumerate(trials)]
            }
        return {"weighted_scores": []}

    return SimpleNamespace(
        trials=trials,
        metadata=metadata,
        best_trial_id=best_trial_id,
        best_config=best_config,
        best_score=best_score,
        reason_code=reason_code,
        calculate_weighted_scores=calculate_weighted_scores,
    )


def _trial_sets():
    good = [
        _trial(
            "a",
            {"model": "x", "temp": 0.0, "flag": True},
            {"accuracy": 0.5, "cost": 0.01, "latency": 120.0},
        ),
        _trial(
            "b",
            {"model": "yy", "temp": 0.5, "flag": False},
            {"accuracy": 0.75, "cost": 0.02, "latency": 80.0},
        ),
        _trial("c", {"model": "z"}, {"accuracy": 0.75, "cost": 0.01, "latency": 80.0}),
    ]
    with_examples = [
        _trial(
            "a",
            {"model": "x"},
            {"accuracy": 0.5, "cost": 0.01},
            metadata={"successful_examples": 3, "examples_attempted": 4},
        ),
        _trial(
            "b",
            {"model": "y"},
            {"accuracy": 90.0, "cost": 0.02},
            metadata={"successful_examples": 0, "examples_attempted": 4},
        ),
        _trial("c", {"model": "z"}, {"accuracy": 0.9, "cost": 0.02}, metadata={}),
    ]
    all_failed = [
        _trial(
            "a",
            {"model": "x"},
            {"accuracy": 0.0, "cost": 0.0},
            metadata={"successful_examples": 0},
        ),
        _trial(
            "b",
            {"model": "y"},
            {"accuracy": 0.0, "cost": 0.0},
            metadata={"successful_examples": 0},
        ),
    ]
    cost_unmeasured = [
        _trial(
            "a", {"model": "x"}, {"accuracy": 0.5, "cost": 0.0, "cost_unmeasured": 1.0}
        ),
        _trial(
            "b",
            {"model": "y"},
            {
                "accuracy": 0.8,
                "cost": 0.0,
                "cost_unmeasured": 1.0,
                "cost_lower_bound": 0.03,
            },
        ),
    ]
    unsuccessful = [
        _trial("a", {"model": "x"}, {"accuracy": 0.5}, ok=False),
        _trial("b", {"model": "y"}, {"accuracy": 0.8}, ok=False),
    ]
    missing_metrics = [
        _trial("a", {"model": "x"}, {}),
        _trial("b", {"model": "y"}, {"accuracy": "n/a", "cost": None}),
    ]
    return {
        "empty": [],
        "good": good,
        "with_examples": with_examples,
        "all_failed": all_failed,
        "cost_unmeasured": cost_unmeasured,
        "unsuccessful": unsuccessful,
        "missing_metrics": missing_metrics,
        "single": good[:1],
        "printer_swap": [_trial("a", {"model": _PrinterSwap()}, {"accuracy": 0.5})],
    }


_OBJECTIVES = {
    "acc": ["accuracy"],
    "acc_cost": ["accuracy", "cost"],
    "cost": ["cost"],
    "acc_latency": ["accuracy", "latency"],
    "unknown": ["accuracy", "nonexistent"],
    "dict_band": {
        "objectives": [
            {"name": "accuracy", "orientation": "band"},
            {"name": "cost", "orientation": "minimize"},
        ]
    },
    "none": [],
}

_RESULT_VARIANTS = {
    "plain": {},
    "stop_detail": {"metadata": {"stop_detail": "Stopped at the trial limit."}},
    "stop_empty": {"metadata": {"stop_detail": ""}},
    "stop_not_str": {"metadata": {"stop_detail": 5}},
    "mock": {"metadata": {"mock_llm_run": True}},
    "mock_mixed": {"metadata": {"mock_llm_run": True, "mock_llm_mixed": True}},
    "not_mock": {"metadata": {"mock_llm_run": False}},
    "fallback": {"metadata": {"source": "local_fallback", "fallback_reason": "boom"}},
    "best_id": {"metadata": {"best_trial_id": "b"}},
    "no_winner": {"reason_code": "NO_RANKING_ELIGIBLE_TRIALS"},
    "best_config": {
        "best_config": {"model": "y"},
        "best_score": 0.8,
        "weighted": "empty",
    },
    "weighted_boom": {"weighted": "boom"},
    "weighted_empty": {"weighted": "empty"},
    "meta_not_dict": {"metadata": "text"},
}


class _ClearingLabel(str):
    """Label whose ``strip()`` empties the trials list, pinning read order."""

    def __new__(cls, text, trials):
        obj = super().__new__(cls, text)
        obj._trials = trials
        return obj

    def strip(self, *args):
        self._trials.clear()
        return str.strip(self, *args)


class _PrinterSwap:
    """Config value that swaps the module printer when rendered the second time."""

    calls = 0
    swap = None

    def __str__(self):
        type(self).calls += 1
        if type(self).calls == 2 and type(self).swap is not None:
            type(self).swap()
        return "swap"


def _install_recorders(monkeypatch, log, colors):
    for name in _WRAPPED:
        real = getattr(rt, name)

        def wrapper(*args, _real=real, _name=name, **kwargs):
            outcome = _real(*args, **kwargs)
            rendered = (repr(args), repr(sorted(kwargs.items())), repr(outcome))
            log.append((_name, *(_ADDRESS.sub("0x", text) for text in rendered)))
            return outcome

        monkeypatch.setattr(rt, name, wrapper)
    monkeypatch.setattr(
        rt,
        "_safe_table_print",
        lambda text, **kw: log.append(("print", text, tuple(sorted(kw)))),
    )
    monkeypatch.setattr(
        rt._Colors, "refresh", classmethod(lambda cls: log.append(("refresh",)))
    )
    for attr, value in zip(
        ("BOLD", "GREEN", "YELLOW", "CYAN", "DIM", "RESET"), colors, strict=True
    ):
        monkeypatch.setattr(rt._Colors, attr, value)


def _run(fn, trials, results_kwargs, config_space, objectives, kwargs, colors):
    """Run ``fn`` with fresh recorders; the patches are undone on exit."""
    log: list = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        _install_recorders(monkeypatch, log, colors)
        _PrinterSwap.calls = 0
        _PrinterSwap.swap = lambda: monkeypatch.setattr(
            rt, "_safe_table_print", lambda text, **kw: log.append(("print2", text))
        )
        results = _results(list(trials), **results_kwargs)
        resolved = {
            key: value(results.trials) if callable(value) else value
            for key, value in kwargs.items()
        }
        try:
            fn(results, config_space, objectives, **resolved)
            outcome = "ok"
        except Exception as exc:  # noqa: BLE001 - characterizing any exception
            outcome = ("err", type(exc).__name__)
        finally:
            _PrinterSwap.swap = None
    return outcome, log


def test_matches_oracle_across_scenarios():
    sets = _trial_sets()
    spaces = [
        {},
        {"model": ["x"]},
        {"model": ["x", "y"], "temp": [0.0], "flag": [True]},
    ]
    kwarg_sets = [
        {},
        {"mode_label": "MOCK"},
        {"mode_label": ""},
        {"mode_label": lambda trials: _ClearingLabel("clear", trials)},
        {"mode_label": "  padded  ", "show_best": False},
        {"metric_overrides": {"accuracy": [0.1, 0.9]}},
        {"show_best": False},
    ]
    checked = 0
    bad = []
    for (
        (t_name, trials),
        (o_name, objectives),
        (v_name, variant),
        space,
        kwargs,
    ) in itertools.product(
        sets.items(), _OBJECTIVES.items(), _RESULT_VARIANTS.items(), spaces, kwarg_sets
    ):
        colors = _ANSI if (checked % 3 == 0) else ("",) * 6
        got = _run(
            print_results_table,
            trials,
            variant,
            space,
            objectives,
            kwargs,
            colors,
        )
        want = _run(
            oracle_print_results_table,
            trials,
            variant,
            space,
            objectives,
            kwargs,
            colors,
        )
        checked += 1
        if got != want:
            bad.append((t_name, o_name, v_name, space, kwargs))
            if len(bad) >= 3:
                break
    assert not bad
    assert checked > 10000


def test_representative_output_is_stable(monkeypatch):
    trials = _trial_sets()["good"]
    log: list = []
    _install_recorders(monkeypatch, log, ("",) * 6)
    results = _results(trials, metadata={"best_trial_id": "b", "mock_llm_run": False})
    print_results_table(results, {"model": ["x"]}, ["accuracy"])
    printed = [entry[1] for entry in log if entry[0] == "print"]
    assert printed[0] == ""
    assert printed[1].startswith("┌") and printed[1].endswith("┐")
    assert " Trial Results (3 trials) " in printed[2]
    assert any(line.startswith(" ") is False and "★" in line for line in printed)
    assert printed[-1] == "Legend: ★ Overall Best, accuracy = Best accuracy"


def test_empty_results_message(monkeypatch):
    log: list = []
    _install_recorders(monkeypatch, log, ("",) * 6)
    print_results_table(_results([]), {}, ["accuracy"])
    assert log == [("print", "\nNo trials to display.", ())]
