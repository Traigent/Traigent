"""CTD-style matrix tests for CostEnforcer state transitions.

This module adds constrained pairwise coverage across key cost-enforcement factors:
- mock mode
- cost value shape
- strict tracking mode
- budget state
- permit state
- unknown-cost mode pre-state
- concurrency mode

The matrix is intentionally constrained to meaningful operational combinations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations, product

import pytest

from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
    Permit,
)

# Fixed reservation amount for deterministic budget-state behavior.
ESTIMATED_COST_PER_TRIAL = 0.05
BUDGET_LIMITS = {
    "under_limit": 0.20,
    "at_limit": 0.05,
    "over_limit": 0.01,
}


@dataclass(frozen=True)
class CTDCase:
    mock_mode: bool
    cost_value: str
    strict_tracking: bool
    budget_state: str
    permit_state: str
    unknown_cost_mode: str
    concurrency: str


# 11 cases are a full constrained pairwise cover; we add 10 extra high-risk cases
# to keep this matrix explicit and easier to review in PRs.
PAIRWISE_CASES: tuple[CTDCase, ...] = (
    CTDCase(True, "positive", False, "under_limit", "active", "not_entered", "single"),
    CTDCase(
        False,
        "positive",
        True,
        "at_limit",
        "already_released",
        "already_entered",
        "parallel",
    ),
    CTDCase(False, "zero", True, "under_limit", "foreign", "not_entered", "single"),
    CTDCase(False, "none", False, "under_limit", "active", "already_entered", "parallel"),
    CTDCase(
        False,
        "none",
        False,
        "at_limit",
        "already_released",
        "not_entered",
        "single",
    ),
    CTDCase(False, "zero", False, "at_limit", "foreign", "already_entered", "parallel"),
    CTDCase(False, "positive", False, "over_limit", "active", "not_entered", "single"),
    CTDCase(False, "zero", True, "at_limit", "active", "not_entered", "parallel"),
    CTDCase(
        False,
        "zero",
        True,
        "under_limit",
        "already_released",
        "already_entered",
        "single",
    ),
    CTDCase(False, "none", True, "under_limit", "foreign", "not_entered", "single"),
    CTDCase(False, "positive", True, "under_limit", "foreign", "not_entered", "single"),
    CTDCase(False, "none", False, "under_limit", "active", "not_entered", "single"),
    CTDCase(False, "none", True, "under_limit", "active", "not_entered", "parallel"),
    CTDCase(
        False,
        "positive",
        False,
        "under_limit",
        "already_released",
        "not_entered",
        "single",
    ),
    CTDCase(
        False,
        "positive",
        False,
        "under_limit",
        "foreign",
        "already_entered",
        "parallel",
    ),
    CTDCase(False, "zero", False, "under_limit", "active", "not_entered", "parallel"),
    CTDCase(False, "positive", False, "at_limit", "active", "not_entered", "single"),
    CTDCase(False, "zero", False, "at_limit", "active", "already_entered", "single"),
    CTDCase(
        False,
        "none",
        False,
        "under_limit",
        "foreign",
        "already_entered",
        "parallel",
    ),
    CTDCase(False, "positive", True, "under_limit", "active", "not_entered", "parallel"),
    CTDCase(False, "zero", True, "under_limit", "foreign", "already_entered", "parallel"),
)

FACTOR_NAMES = (
    "mock_mode",
    "cost_value",
    "strict_tracking",
    "budget_state",
    "permit_state",
    "unknown_cost_mode",
    "concurrency",
)

FACTOR_VALUES = {
    "mock_mode": (True, False),
    "cost_value": ("positive", "zero", "none"),
    "strict_tracking": (True, False),
    "budget_state": ("under_limit", "at_limit", "over_limit"),
    "permit_state": ("active", "already_released", "foreign"),
    "unknown_cost_mode": ("not_entered", "already_entered"),
    "concurrency": ("single", "parallel"),
}


def _case_id(case: CTDCase) -> str:
    return (
        f"mock-{'on' if case.mock_mode else 'off'}_"
        f"cost-{case.cost_value}_"
        f"strict-{'on' if case.strict_tracking else 'off'}_"
        f"budget-{case.budget_state}_"
        f"permit-{case.permit_state}_"
        f"unknown-{case.unknown_cost_mode}_"
        f"{case.concurrency}"
    )


def _is_valid_case(case: CTDCase) -> bool:
    # Mock mode bypasses all cost-accounting behavior; keep one canonical case.
    if case.mock_mode:
        return case == CTDCase(
            True,
            "positive",
            False,
            "under_limit",
            "active",
            "not_entered",
            "single",
        )

    # Over-limit combinations are represented by a single deny-on-acquire scenario.
    if case.budget_state == "over_limit":
        return (
            case.strict_tracking is False
            and case.cost_value == "positive"
            and case.permit_state == "active"
            and case.unknown_cost_mode == "not_entered"
            and case.concurrency == "single"
        )

    # strict + unknown cost fails before budget differences matter.
    if (
        case.cost_value == "none"
        and case.strict_tracking
        and case.budget_state != "under_limit"
    ):
        return False

    return True


def _setup_budget_state(budget_state: str) -> CostEnforcerConfig:
    return CostEnforcerConfig(
        limit=BUDGET_LIMITS[budget_state],
        estimated_cost_per_trial=ESTIMATED_COST_PER_TRIAL,
        fallback_trial_limit=6,
    )


def _drive_into_unknown_mode(enforcer: CostEnforcer) -> None:
    permit = enforcer.acquire_permit()
    assert permit.is_granted
    enforcer.assert_invariants()
    enforcer.track_cost(None, permit=permit)
    enforcer.assert_invariants()
    assert enforcer.get_status().unknown_cost_mode is True


def _setup_permit_state(
    enforcer: CostEnforcer,
    permit_state: str,
    budget_state: str,
) -> tuple[Permit, bool]:
    if permit_state == "foreign":
        # Foreign permit not present in the enforcer registry.
        return Permit(id=999_999, amount=ESTIMATED_COST_PER_TRIAL, active=True), True

    permit = enforcer.acquire_permit()
    enforcer.assert_invariants()

    if budget_state == "over_limit":
        assert permit.is_granted is False
        return permit, False

    assert permit.is_granted is True
    if permit_state == "already_released":
        assert enforcer.release_permit(permit) is True
        enforcer.assert_invariants()

    return permit, True


def _seed_unknown_mode(
    enforcer: CostEnforcer,
    mode: str,
    strict_tracking: bool,
) -> None:
    if mode == "not_entered":
        return

    # With strict tracking latched ON, unknown mode cannot be entered organically.
    # Seed it directly to validate behavior in that pre-existing state.
    if strict_tracking:
        with enforcer._lock:
            enforcer._unknown_cost_mode = True
        enforcer.assert_invariants()
        return

    _drive_into_unknown_mode(enforcer)


def _cost_value(label: str) -> float | None:
    mapping: dict[str, float | None] = {
        "positive": 0.05,
        "zero": 0.0,
        "none": None,
    }
    return mapping[label]


def _run_single_transition(
    enforcer: CostEnforcer,
    permit: Permit,
    cost_value: float | None,
    should_raise: bool,
) -> None:
    if should_raise:
        with pytest.raises(CostTrackingRequiredError):
            enforcer.track_cost(cost_value, permit=permit)
    else:
        enforcer.track_cost(cost_value, permit=permit)
    enforcer.assert_invariants()


def _run_parallel_transitions(
    enforcer: CostEnforcer,
    permits: list[Permit],
    cost_value: float | None,
    should_raise: bool,
) -> None:
    def _worker(permit: Permit) -> Exception | None:
        try:
            enforcer.track_cost(cost_value, permit=permit)
            enforcer.assert_invariants()
            return None
        except Exception as exc:
            enforcer.assert_invariants()
            return exc

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_worker, permits))

    if should_raise:
        assert all(isinstance(r, CostTrackingRequiredError) for r in results)
    else:
        assert all(r is None for r in results)
    enforcer.assert_invariants()


def _build_parallel_permits(
    enforcer: CostEnforcer,
    base_permit: Permit,
    permit_state: str,
) -> list[Permit]:
    if permit_state in {"already_released", "foreign"}:
        # Intentional: reuse one stale/foreign permit across threads to exercise
        # lock contention around mark_released() and repeated already-released paths.
        return [base_permit, base_permit, base_permit]

    permits = [base_permit]
    for _ in range(2):
        permit = enforcer.acquire_permit()
        enforcer.assert_invariants()
        if permit.is_granted:
            permits.append(permit)
        else:
            # Keep 3 operations by reusing a known valid permit.
            permits.append(base_permit)
    return permits


def _pairs_for_case(case: CTDCase) -> set[tuple[str, object, str, object]]:
    pairs: set[tuple[str, object, str, object]] = set()
    case_dict = case.__dict__
    for left, right in combinations(FACTOR_NAMES, 2):
        pairs.add((left, case_dict[left], right, case_dict[right]))
    return pairs


def _all_valid_cases() -> list[CTDCase]:
    all_cases: list[CTDCase] = []
    for values in product(*(FACTOR_VALUES[name] for name in FACTOR_NAMES)):
        case = CTDCase(*values)
        if _is_valid_case(case):
            all_cases.append(case)
    return all_cases


class TestCostEnforcerCTDPairwise:
    """Constrained pairwise matrix coverage tests."""

    @pytest.mark.parametrize(
        "case",
        [pytest.param(case, id=_case_id(case)) for case in PAIRWISE_CASES],
    )
    def test_cost_enforcer_transition_matrix(self, case: CTDCase, monkeypatch) -> None:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true" if case.mock_mode else "false")
        monkeypatch.setenv(
            "TRAIGENT_REQUIRE_COST_TRACKING",
            "true" if case.strict_tracking else "false",
        )
        monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

        enforcer = CostEnforcer(config=_setup_budget_state(case.budget_state))
        enforcer.assert_invariants()

        _seed_unknown_mode(enforcer, case.unknown_cost_mode, case.strict_tracking)

        permit, should_track = _setup_permit_state(
            enforcer,
            case.permit_state,
            case.budget_state,
        )
        if not should_track:
            # Over-limit path: permit denied before track_cost transition.
            assert permit.is_granted is False
            enforcer.assert_invariants()
            return

        cost_value = _cost_value(case.cost_value)
        should_raise = (
            not case.mock_mode
            and case.cost_value == "none"
            and case.strict_tracking is True
        )

        if case.concurrency == "single":
            _run_single_transition(enforcer, permit, cost_value, should_raise)
        else:
            permits = _build_parallel_permits(enforcer, permit, case.permit_state)
            _run_parallel_transitions(enforcer, permits, cost_value, should_raise)

        status = enforcer.get_status()
        if case.mock_mode:
            assert status.trial_count == 0
            assert status.accumulated_cost_usd == pytest.approx(0.0)
            return

        if case.cost_value == "none" and not case.strict_tracking:
            assert status.unknown_cost_mode is True

    def test_pairwise_matrix_covers_all_valid_pairs(self) -> None:
        valid_cases = _all_valid_cases()
        # Guard against accidental over-pruning in _is_valid_case(), which could
        # otherwise make coverage checks pass vacuously with a shrunk domain.
        assert len(valid_cases) == 134
        required_pairs = set().union(*(_pairs_for_case(case) for case in valid_cases))
        assert len(required_pairs) == 107
        covered_pairs = set().union(*(_pairs_for_case(case) for case in PAIRWISE_CASES))
        assert covered_pairs >= required_pairs, (
            f"Pair coverage {len(covered_pairs & required_pairs)}/{len(required_pairs)} "
            "is incomplete"
        )
        assert 20 <= len(PAIRWISE_CASES) <= 25


class TestCostEnforcerCTDCriticalTriples:
    """Critical triple scenarios that must stay green."""

    def test_critical_unknown_none_strict_on_parallel(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
        monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true")
        monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

        enforcer = CostEnforcer(config=_setup_budget_state("under_limit"))
        permits = [enforcer.acquire_permit() for _ in range(3)]
        assert all(p.is_granted for p in permits)
        enforcer.assert_invariants()

        _run_parallel_transitions(
            enforcer,
            permits,
            cost_value=None,
            should_raise=True,
        )

    def test_critical_at_limit_already_released_permit_parallel(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
        monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "false")
        monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

        enforcer = CostEnforcer(config=_setup_budget_state("at_limit"))
        permit = enforcer.acquire_permit()
        assert permit.is_granted
        assert enforcer.release_permit(permit) is True
        enforcer.assert_invariants()

        _run_parallel_transitions(
            enforcer,
            [permit, permit, permit],
            cost_value=0.05,
            should_raise=False,
        )

    def test_critical_foreign_permit_parallel_exception_path(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
        monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true")
        monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

        enforcer = CostEnforcer(config=_setup_budget_state("under_limit"))
        foreign_permit = Permit(id=424242, amount=ESTIMATED_COST_PER_TRIAL, active=True)
        enforcer.assert_invariants()

        _run_parallel_transitions(
            enforcer,
            [foreign_permit, foreign_permit, foreign_permit],
            cost_value=None,
            should_raise=True,
        )
