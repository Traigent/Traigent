"""Global trace invariants that apply to all tests.

These invariants check structural correctness and data completeness
that should hold for any valid optimization trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

    from .capture import CapturedTrace


# List of all global invariants
GLOBAL_INVARIANTS = [
    # Structural invariants
    "root_span_exists",
    "all_trials_have_parent",
    "no_orphan_spans",
    "valid_timestamps",
    # Data completeness invariants
    "trial_has_trial_id",
    "trial_has_config",
    "trial_has_status",
    "completed_trials_have_metrics",
    # Consistency invariants
    "trial_count_matches_result",
    "config_matches_result",
]


def check_invariant(
    invariant: str,
    trace: CapturedTrace,
    result: OptimizationResult | None = None,
) -> str | None:
    """Check a single invariant.

    Args:
        invariant: Name of invariant to check
        trace: Captured trace to validate
        result: Optional optimization result for consistency checks

    Returns:
        Error message if invariant failed, None if passed
    """
    checkers = {
        "root_span_exists": _check_root_span_exists,
        "all_trials_have_parent": _check_all_trials_have_parent,
        "no_orphan_spans": _check_no_orphan_spans,
        "valid_timestamps": _check_valid_timestamps,
        "trial_has_trial_id": _check_trial_has_trial_id,
        "trial_has_config": _check_trial_has_config,
        "trial_has_status": _check_trial_has_status,
        "completed_trials_have_metrics": _check_completed_trials_have_metrics,
        "trial_count_matches_result": _check_trial_count_matches_result,
        "config_matches_result": _check_config_matches_result,
    }

    if invariant not in checkers:
        return f"Unknown invariant: {invariant}"

    return checkers[invariant](trace, result)


# Structural invariants


def _check_root_span_exists(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that root optimization_session span exists."""
    if not trace.root_span:
        return "Missing root span"

    if trace.root_span.name != "optimization_session":
        return (
            f"Root span should be 'optimization_session', got '{trace.root_span.name}'"
        )

    return None


def _check_all_trials_have_parent(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all trial spans have a parent."""
    root = trace.root_span
    if not root:
        return None  # Will be caught by root_span_exists

    for trial in trace.trial_spans:
        if trial.parent_span_id is None:
            return f"Trial span {trial.attributes.get('trial.id', 'unknown')} has no parent"

        # Parent should be root or another valid span
        parent_found = any(s.span_id == trial.parent_span_id for s in trace.spans)
        if not parent_found:
            return (
                f"Trial span {trial.attributes.get('trial.id', 'unknown')} "
                f"has invalid parent {trial.parent_span_id}"
            )

    return None


def _check_no_orphan_spans(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all spans are connected to the trace."""
    root = trace.root_span
    if not root:
        return None  # Will be caught by root_span_exists

    # Build set of reachable span IDs starting from root
    reachable: set[str] = {root.span_id}
    changed = True

    while changed:
        changed = False
        for span in trace.spans:
            if span.parent_span_id in reachable and span.span_id not in reachable:
                reachable.add(span.span_id)
                changed = True

    # Check all spans are reachable
    for span in trace.spans:
        if span.span_id not in reachable:
            return f"Orphan span detected: {span.name} ({span.span_id})"

    return None


def _check_valid_timestamps(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all spans have valid timestamps (end >= start)."""
    for span in trace.spans:
        if span.end_time_ns < span.start_time_ns:
            return (
                f"Span {span.name} has invalid timestamps: "
                f"end ({span.end_time_ns}) < start ({span.start_time_ns})"
            )

    return None


# Data completeness invariants


def _check_trial_has_trial_id(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all trial spans have trial.id attribute."""
    for trial in trace.trial_spans:
        if "trial.id" not in trial.attributes:
            return f"Trial span missing trial.id attribute (span_id: {trial.span_id})"

    return None


def _check_trial_has_config(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all trial spans have trial.config attribute."""
    for trial in trace.trial_spans:
        if "trial.config" not in trial.attributes:
            trial_id = trial.attributes.get("trial.id", "unknown")
            return f"Trial span {trial_id} missing trial.config attribute"

    return None


def _check_trial_has_status(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that all trial spans have trial.status attribute."""
    for trial in trace.trial_spans:
        if "trial.status" not in trial.attributes:
            trial_id = trial.attributes.get("trial.id", "unknown")
            return f"Trial span {trial_id} missing trial.status attribute"

    return None


def _check_completed_trials_have_metrics(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that completed trial spans have metrics."""
    for trial in trace.trial_spans:
        status = trial.attributes.get("trial.status")
        if status == "completed":
            if "trial.metrics" not in trial.attributes:
                trial_id = trial.attributes.get("trial.id", "unknown")
                return f"Completed trial span {trial_id} missing trial.metrics"

    return None


# Consistency invariants


def _check_trial_count_matches_result(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that trace trial count matches result trial count."""
    if result is None:
        return None  # Skip if no result provided

    trace_count = len(trace.trial_spans)
    result_count = len(result.trials)

    if trace_count != result_count:
        return (
            f"Trial count mismatch: trace has {trace_count} spans, "
            f"result has {result_count} trials"
        )

    return None


def _check_config_matches_result(
    trace: CapturedTrace,
    result: OptimizationResult | None,
) -> str | None:
    """Check that trial span configs match result trial configs."""
    if result is None:
        return None  # Skip if no result provided

    import json

    for trial_result in result.trials:
        trial_span = trace.get_trial_span(trial_result.trial_id)
        if trial_span:
            span_config_str = trial_span.attributes.get("trial.config", "{}")
            try:
                span_config = json.loads(span_config_str)
                if span_config != trial_result.config:
                    return (
                        f"Config mismatch for trial {trial_result.trial_id}: "
                        f"span has {span_config}, result has {trial_result.config}"
                    )
            except json.JSONDecodeError:
                return (
                    f"Invalid JSON in trial.config for {trial_result.trial_id}: "
                    f"{span_config_str}"
                )

    return None
