"""Ratchet on content leaking over SDK optimization traffic (Phase 1: measure).

The number of distinct (route, canary kind) leak pairs may never go up, and
may not go down without lowering the pin in ``pins.py``, so progress is
recorded in the same change that makes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .canaries import distinct_pairs
from .pins import BASELINE_PAIRS, PINNED_LEAK_PAIRS
from .scenario import ScenarioResult, run_scenario

# The run is mocked at every transport; the marker only selects connected mode.
pytestmark = pytest.mark.backend_online

REQUIRED_ROUTES = (
    "POST /api/v1/sessions",
    "POST /api/v1/sessions/{id}/next-trial",
    "POST /api/v1/sessions/{id}/results",
    "POST /api/v1/sessions/{id}/finalize",
)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> ScenarioResult:
    return run_scenario(Path(tmp_path_factory.mktemp("egress-canary")))


def _fmt(pairs: set[tuple[str, str]]) -> str:
    return "\n".join(f"  {route}  [{kind}]" for route, kind in sorted(pairs))


def test_canary_run_exercised_the_optimization_routes(scenario: ScenarioResult) -> None:
    """A ratchet over an empty capture would pass vacuously; prove coverage."""

    assert scenario.optimization_error is None, scenario.optimization_error
    assert scenario.result_source == "cloud_brain"
    missing = [r for r in REQUIRED_ROUTES if r not in scenario.routes]
    assert not missing, f"routes not captured: {missing}; got {scenario.routes}"
    assert scenario.guard_violations == []
    failed = {k: v for k, v in scenario.exerciser_outcomes.items() if v is not None}
    assert not failed, f"direct-path exercisers failed (coverage shrank): {failed}"
    assert len(scenario.exerciser_outcomes) >= 7
    assert scenario.unhandled_routes == [], scenario.unhandled_routes


def test_leak_pair_count_does_not_increase(scenario: ScenarioResult) -> None:
    pairs = distinct_pairs(scenario.leaks)
    assert len(pairs) <= PINNED_LEAK_PAIRS, (
        f"leak pairs went UP: {len(pairs)} > pinned {PINNED_LEAK_PAIRS}. "
        f"New pairs not in the baseline:\n{_fmt(pairs - set(BASELINE_PAIRS))}"
    )


def test_leak_pair_count_does_not_decrease_without_lowering_pin(
    scenario: ScenarioResult,
) -> None:
    pairs = distinct_pairs(scenario.leaks)
    assert len(pairs) >= PINNED_LEAK_PAIRS, (
        f"leak pairs went DOWN: {len(pairs)} < pinned {PINNED_LEAK_PAIRS}. "
        "Lower PINNED_LEAK_PAIRS and drop the fixed pairs from BASELINE_PAIRS "
        f"in tests/egress_canary/pins.py. Gone:\n{_fmt(set(BASELINE_PAIRS) - pairs)}"
    )


def test_no_pair_outside_the_baseline(scenario: ScenarioResult) -> None:
    """A fix offset by a new leak keeps the count equal; catch it by identity."""

    assert len(BASELINE_PAIRS) == PINNED_LEAK_PAIRS
    new = distinct_pairs(scenario.leaks) - set(BASELINE_PAIRS)
    assert not new, f"new leak pairs outside the baseline:\n{_fmt(new)}"
