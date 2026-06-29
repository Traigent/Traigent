"""Promotion gate for epsilon-Pareto dominance evaluation.

This module implements the promotion policy logic from TVL 0.9,
including epsilon-Pareto dominance testing, chance constraint
evaluation, and multi-objective comparison.

The decision rule mirrors the canonical ``epsilon_pareto_gate`` shipped in
``tvl/python/tvl/promotion.py`` (Intersection-Union test): a candidate is
promoted iff every objective is non-inferior AND at least one objective is
superior; it is rejected the moment any objective fails non-inferiority, any
chance constraint fails, or any banded objective falls outside its band.

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import BandTarget, PromotionPolicy, validate_adjust_method
from .objectives import tost_equivalence_test
from .statistics import (
    benjamini_hochberg_adjust,
    bonferroni_adjust,
    holm_bonferroni_adjust,
    paired_comparison_test,
)

try:  # scipy ships only in the optional ``[bayesian]`` extra, mirroring the
    # canonical gate's own guard (tvl/python/tvl/promotion.py:18-23).
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised only on a core (no-extra) install
    _scipy_stats = None  # type: ignore[assignment]

_VALID_OBJECTIVE_DIRECTIONS = frozenset({"maximize", "minimize", "band"})


def clopper_pearson_upper_bound(
    violations: int,
    trials: int,
    confidence: float,
) -> float:
    """Compute the exact one-sided Clopper-Pearson upper bound on a proportion.

    This bounds the *violation* rate from above and matches the canonical
    chance-constraint test bit-for-bit (tvl/python/tvl/promotion.py:664-678),
    which uses the EXACT beta Clopper-Pearson upper quantile for ALL sample
    sizes::

        ci_upper = beta.ppf(confidence, violations + 1, trials - violations)

    There is deliberately NO normal/Wilson large-sample approximation. A Wilson
    upper bound is systematically *smaller* than the exact beta quantile and can
    flip a promotion verdict: e.g. ``violations=0, trials=100, confidence=0.95``
    gives a Wilson upper of ``0.0264`` but an exact beta upper of ``0.0295``,
    which straddle a threshold of ``0.028`` (the exact bound correctly fails).

    When scipy is installed (the ``[bayesian]`` extra) the exact beta quantile
    comes straight from :func:`scipy.stats.beta.ppf`, matching canonical
    exactly. When scipy is NOT installed this FAILS CLOSED for any case that
    needs the general beta quantile: it raises :class:`ImportError` rather than
    falling back to the pure-python ``_beta_quantile_approx``. That
    approximation is not reliable for high-violation / high-confidence inputs
    (its damped Newton step can clamp to ``0.001``), which would silently
    *under-estimate* the violation-rate upper bound and flip a chance-constraint
    verdict to "promote" when it must reject. A chance constraint is policy-like
    and must never fail open, so we refuse to produce a verdict instead of
    trusting the approximation -- mirroring the canonical gate, which
    hard-requires scipy for chance-constraint evaluation
    (tvl/python/tvl/promotion.py:285-302, ``require_scipy()``). The trivial
    edges (``violations == trials`` -> ``1.0``) need no quantile and still work
    without scipy.

    Args:
        violations: Number of observed violations (non-negative).
        trials: Total number of trials (positive).
        confidence: One-sided confidence level (0 < confidence < 1).

    Returns:
        Upper bound of the confidence interval for the true violation rate.

    Raises:
        ValueError: If inputs are invalid.
        ImportError: If the general beta quantile is required but scipy is not
            installed (fail-closed; install ``traigent[bayesian]``).
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if violations < 0:
        raise ValueError(f"violations must be non-negative, got {violations}")
    if violations > trials:
        raise ValueError(f"violations ({violations}) cannot exceed trials ({trials})")
    if confidence <= 0 or confidence >= 1:  # NOSONAR - defensive validation
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # All trials violated: the upper bound is trivially 1.0 (canonical
    # promotion.py:674-675). The general beta.ppf would need shape
    # ``trials - violations == 0`` here, which is undefined; guard it exactly as
    # canonical does.
    if violations == trials:
        return 1.0

    # Exact beta Clopper-Pearson upper bound for ALL n (canonical
    # promotion.py:677-678): ci_upper = beta.ppf(confidence, k + 1, n - k).
    # No special case for violations == 0 -- the general formula already yields
    # the exact value beta.ppf(confidence, 1, trials).
    k = violations
    n = trials
    if _scipy_stats is None:
        # FAIL CLOSED. The only reliable exact-beta quantile available is
        # scipy.stats.beta.ppf. statistics.py's pure-python
        # _beta_quantile_approx clamps to 0.001 on high-violation /
        # high-confidence inputs (e.g. clopper_pearson_upper_bound(95, 100,
        # 0.99) returns 0.001 instead of the canonical 0.987031), which would
        # silently flip a chance-constraint verdict to "promote" when it must
        # reject. A chance constraint is policy-like and must never fail open,
        # so refuse to produce a verdict instead of trusting the approximation
        # (canonical require_scipy(), promotion.py:285-302).
        raise ImportError(
            "Chance-constraint evaluation requires scipy for the exact beta "
            "Clopper-Pearson upper bound; the pure-python approximation is not "
            "reliable enough for a fail-closed policy gate. Install the optional "
            "extra to enable chance constraints: pip install 'traigent[bayesian]'."
        )
    return float(_scipy_stats.beta.ppf(confidence, k + 1, n - k))


@dataclass(slots=True)
class ObjectiveResult:
    """Result of comparing a single objective.

    Attributes:
        name: Name of the objective.
        direction: Optimization direction or "band".
        candidate_better: Whether the candidate is *superior* (strictly better
            by more than epsilon, after multiplicity adjustment).
        p_value: Primary p-value (superiority p-value for standard objectives,
            TOST p-value for banded objectives). Retained for compatibility.
        effect_size: Estimated effect size.
        candidate_mean: Mean value for candidate.
        incumbent_mean: Mean value for incumbent.
        epsilon: Epsilon tolerance applied.
        p_value_noninf: Non-inferiority p-value (unadjusted). Small values
            prove the candidate is not worse by more than epsilon.
        p_value_super: Superiority p-value (raw, before multiplicity
            adjustment). Small values prove the candidate is strictly better.
        verdict: Per-objective verdict ("superior", "noninferior", "inferior"
            for standard objectives; "in_band" or "out_of_band" for banded).
    """

    name: str
    direction: Literal["maximize", "minimize", "band"]
    candidate_better: bool
    p_value: float
    effect_size: float
    candidate_mean: float
    incumbent_mean: float
    epsilon: float
    p_value_noninf: float = 1.0
    p_value_super: float = 1.0
    verdict: Literal[
        "superior", "noninferior", "inferior", "in_band", "out_of_band"
    ] = "noninferior"


@dataclass(slots=True)
class ChanceConstraintResult:
    """Result of evaluating a chance constraint.

    Chance constraints are upper bounds on a *violation* rate: the constraint
    is satisfied iff the upper confidence bound on the violation rate is at or
    below the threshold (matching the canonical gate).

    Attributes:
        name: Name of the constrained metric.
        satisfied: Whether the constraint is satisfied.
        observed_rate: Observed violation rate.
        upper_bound: Upper confidence bound for the violation rate.
        threshold: Maximum allowed violation rate.
        confidence: Required confidence level.
    """

    name: str
    satisfied: bool
    observed_rate: float
    upper_bound: float
    threshold: float
    confidence: float


@dataclass(slots=True)
class PromotionDecision:
    """Result of the promotion gate evaluation.

    Attributes:
        decision: The promotion decision ("promote", "reject", "no_decision").
        reason: Human-readable explanation of the decision.
        objective_results: Per-objective comparison results.
        chance_results: Per-constraint evaluation results.
        adjusted_p_values: Superiority p-values after multiple-testing
            adjustment (standard objectives only).
        dominance_satisfied: Whether the candidate was promoted (all objectives
            non-inferior and at least one superior).
    """

    decision: Literal["promote", "reject", "no_decision"]
    reason: str
    objective_results: list[ObjectiveResult] = field(default_factory=list)
    chance_results: list[ChanceConstraintResult] = field(default_factory=list)
    adjusted_p_values: dict[str, float] = field(default_factory=dict)
    dominance_satisfied: bool = False


@dataclass
class ObjectiveSpec:
    """Specification for an objective in the promotion gate.

    Attributes:
        name: Name of the objective.
        direction: Optimization direction ("maximize", "minimize", or "band").
        band: Band target for banded objectives (None for standard).
        band_alpha: Significance level for TOST (banded objectives only).
    """

    name: str
    direction: Literal["maximize", "minimize", "band"]
    band: BandTarget | None = None
    band_alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate the objective direction.

        Mirrors the spec-load guard (spec_loader.py:1980-1983) so that a direct
        ``ObjectiveSpec`` caller cannot smuggle an unsupported direction past
        the (compile-time-only) ``Literal`` annotation and silently fall into
        the minimize branch.
        """
        if self.direction not in _VALID_OBJECTIVE_DIRECTIONS:
            allowed = ", ".join(sorted(_VALID_OBJECTIVE_DIRECTIONS))
            raise ValueError(
                f"Objective '{self.name}' direction must be one of: {allowed}; "
                f"got {self.direction!r}"
            )


class PromotionGate:
    """Promotion gate implementing the canonical epsilon-Pareto decision rule.

    The promotion gate evaluates whether a candidate configuration should
    be promoted over the incumbent using an Intersection-Union test: every
    objective must be non-inferior, at least one must be superior, and all
    chance constraints and banded objectives must pass.

    Example:
        ```python
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01, "latency": 5.0},
            adjust="holm",
        )
        objectives = [
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("latency", "minimize"),
        ]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.85, 0.87], "latency": [100, 95]},
            candidate_metrics={"accuracy": [0.90, 0.88], "latency": [90, 92]},
        )
        print(f"Decision: {decision.decision} - {decision.reason}")
        ```
    """

    def __init__(
        self,
        policy: PromotionPolicy,
        objectives: list[ObjectiveSpec],
    ) -> None:
        """Initialize the promotion gate.

        Args:
            policy: Promotion policy configuration.
            objectives: List of objective specifications.
        """
        self.policy = policy
        self.objectives = {obj.name: obj for obj in objectives}
        self._objective_list = objectives

    def _evaluate_objectives(
        self,
        incumbent_metrics: dict[str, Sequence[float]],
        candidate_metrics: dict[str, Sequence[float]],
    ) -> tuple[list[ObjectiveResult], dict[str, float]]:
        """Evaluate each objective and collect results and superiority p-values.

        Returns:
            Tuple of (objective_results, raw_superiority_p_values). The
            superiority p-values include standard objectives only (banded
            objectives never contribute to superiority), matching the canonical
            gate's union component (promotion.py:231).
        """
        objective_results: list[ObjectiveResult] = []
        super_p_values: dict[str, float] = {}

        for obj in self._objective_list:
            if obj.name not in incumbent_metrics or obj.name not in candidate_metrics:
                continue
            result = self._compare_objective(
                obj, incumbent_metrics[obj.name], candidate_metrics[obj.name]
            )
            objective_results.append(result)
            if result.direction != "band":
                super_p_values[obj.name] = result.p_value_super

        return objective_results, super_p_values

    def _apply_p_value_adjustment(
        self, raw_p_values: dict[str, float]
    ) -> dict[str, float]:
        """Apply multiple-testing adjustment to superiority p-values.

        Only the superiority (union) component is adjusted; the non-inferiority
        (intersection) component stays unadjusted, matching the canonical gate
        (promotion.py:233-244).
        """
        adjust = validate_adjust_method(self.policy.adjust)
        if adjust == "none" or len(raw_p_values) <= 1:
            return raw_p_values.copy()

        p_list = list(raw_p_values.values())
        if adjust == "BH":
            adjusted_list = benjamini_hochberg_adjust(p_list)
        elif adjust == "holm":
            adjusted_list = holm_bonferroni_adjust(p_list)
        elif adjust == "bonferroni":
            adjusted_list = bonferroni_adjust(p_list)
        else:
            # validate_adjust_method is the primary fail-closed path; keep this
            # branch as a defensive guard against future enum drift.
            raise ValueError(f"Unsupported promotion_policy.adjust {adjust!r}")

        return dict(zip(raw_p_values.keys(), adjusted_list, strict=True))

    def _classify_objectives(
        self,
        objective_results: list[ObjectiveResult],
        adjusted_super: dict[str, float],
    ) -> tuple[bool, bool, bool]:
        """Assign per-objective verdicts and compute the IUT aggregates.

        Mirrors the canonical verdict loop (promotion.py:244-265) and banded
        gate (promotion.py:221-222).

        Returns:
            Tuple of (all_noninferior, any_superior, all_bands_pass).
        """
        alpha = self.policy.alpha
        all_noninferior = True
        any_superior = False
        all_bands_pass = True

        for result in objective_results:
            if result.direction == "band":
                if result.verdict != "in_band":
                    all_bands_pass = False
                continue

            # Non-inferiority is the unadjusted intersection component.
            if result.p_value_noninf >= alpha:
                result.verdict = "inferior"
                all_noninferior = False
                continue

            # Superiority uses the multiplicity-adjusted p-value.
            p_super = adjusted_super.get(result.name, result.p_value_super)
            if p_super < alpha:
                result.verdict = "superior"
                result.candidate_better = True
                any_superior = True
            else:
                result.verdict = "noninferior"

        return all_noninferior, any_superior, all_bands_pass

    def evaluate(
        self,
        incumbent_metrics: dict[str, Sequence[float]],
        candidate_metrics: dict[str, Sequence[float]],
        constraint_data: dict[str, tuple[int, int]] | None = None,
    ) -> PromotionDecision:
        """Evaluate candidate for promotion against incumbent.

        Args:
            incumbent_metrics: Per-objective samples for the incumbent.
            candidate_metrics: Per-objective samples for the candidate.
            constraint_data: Dict mapping each chance-constraint name to
                ``(violations, trials)`` (NOT successes).
        """
        # Evaluate objectives first (the canonical gate rejects on
        # non-inferiority before anything else; promotion.py:768).
        objective_results, raw_super_p = self._evaluate_objectives(
            incumbent_metrics, candidate_metrics
        )
        if not objective_results:
            return PromotionDecision(
                decision="no_decision",
                reason="No objectives to compare",
                objective_results=[],
                chance_results=self._evaluate_chance_constraints(constraint_data or {}),
                dominance_satisfied=False,
            )

        chance_results = self._evaluate_chance_constraints(constraint_data or {})

        adjusted_super = self._apply_p_value_adjustment(raw_super_p)
        all_noninferior, any_superior, all_bands_pass = self._classify_objectives(
            objective_results, adjusted_super
        )
        all_chance_pass = all(r.satisfied for r in chance_results)

        decision, reason = self._decide(
            objective_results,
            chance_results,
            all_noninferior=all_noninferior,
            any_superior=any_superior,
            all_bands_pass=all_bands_pass,
            all_chance_pass=all_chance_pass,
        )

        return PromotionDecision(
            decision=decision,
            reason=reason,
            objective_results=objective_results,
            chance_results=chance_results,
            adjusted_p_values=adjusted_super,
            dominance_satisfied=decision == "promote",
        )

    def _decide(
        self,
        objective_results: list[ObjectiveResult],
        chance_results: list[ChanceConstraintResult],
        *,
        all_noninferior: bool,
        any_superior: bool,
        all_bands_pass: bool,
        all_chance_pass: bool,
    ) -> tuple[Literal["promote", "reject", "no_decision"], str]:
        """Make the final decision, mirroring canonical ``_make_decision``.

        Decision precedence (promotion.py:765-798):
        Reject on non-inferiority failure, then chance-constraint failure, then
        banded-objective failure; Promote if any objective is superior;
        otherwise NoDecision.
        """
        if not all_noninferior:
            failing = [r.name for r in objective_results if r.verdict == "inferior"]
            return "reject", f"Non-inferiority failed on: {', '.join(failing)}"

        if not all_chance_pass:
            failing = [r.name for r in chance_results if not r.satisfied]
            return "reject", f"Chance constraints not satisfied: {', '.join(failing)}"

        if not all_bands_pass:
            failing = [r.name for r in objective_results if r.verdict == "out_of_band"]
            return "reject", f"Banded objectives failed TOST: {', '.join(failing)}"

        if any_superior:
            superior = [r.name for r in objective_results if r.verdict == "superior"]
            return (
                "promote",
                f"All objectives non-inferior; superior on: {', '.join(superior)}",
            )

        return (
            "no_decision",
            "All objectives non-inferior but no superiority demonstrated",
        )

    def _compare_objective(
        self,
        obj: ObjectiveSpec,
        incumbent_samples: Sequence[float],
        candidate_samples: Sequence[float],
    ) -> ObjectiveResult:
        """Compare a single objective between incumbent and candidate.

        For standard objectives this computes both a non-inferiority p-value
        and a superiority p-value, mirroring the canonical two-test design
        (promotion.py:401-408): the non-inferiority test uses margin ``-epsilon``
        and the superiority test uses margin ``+epsilon``.

        Args:
            obj: Objective specification.
            incumbent_samples: Samples for incumbent.
            candidate_samples: Samples for candidate.

        Returns:
            ObjectiveResult with comparison outcome.
        """
        epsilon = self.policy.get_epsilon(obj.name, 0.0)

        inc_mean = sum(incumbent_samples) / len(incumbent_samples)
        cand_mean = sum(candidate_samples) / len(candidate_samples)

        if obj.direction == "band" and obj.band is not None:
            return self._compare_banded(
                obj, incumbent_samples, candidate_samples, epsilon
            )

        # Standard objective: direction-string for the one-sided paired test.
        test_direction: Literal["greater", "less"] = (
            "greater" if obj.direction == "maximize" else "less"
        )

        # Superiority test: candidate better by more than +epsilon.
        super_cmp = paired_comparison_test(
            list(candidate_samples),
            list(incumbent_samples),
            epsilon,
            test_direction,
        )
        # Non-inferiority test: candidate not worse by more than epsilon
        # (equivalently, "superior" with margin -epsilon).
        noninf_cmp = paired_comparison_test(
            list(candidate_samples),
            list(incumbent_samples),
            -epsilon,
            test_direction,
        )

        if obj.direction == "maximize":
            effect_size = cand_mean - inc_mean
        else:
            effect_size = inc_mean - cand_mean  # Positive if candidate is better

        return ObjectiveResult(
            name=obj.name,
            direction=obj.direction,
            candidate_better=super_cmp.p_value < self.policy.alpha,
            p_value=super_cmp.p_value,
            effect_size=effect_size,
            candidate_mean=cand_mean,
            incumbent_mean=inc_mean,
            epsilon=epsilon,
            p_value_noninf=noninf_cmp.p_value,
            p_value_super=super_cmp.p_value,
        )

    def _compare_banded(
        self,
        obj: ObjectiveSpec,
        incumbent_samples: Sequence[float],
        candidate_samples: Sequence[float],
        epsilon: float,
    ) -> ObjectiveResult:
        """Compare a banded objective using TOST on the candidate.

        Matching the canonical gate, a banded objective acts purely as a
        pass/fail band constraint: it passes iff the candidate is statistically
        within the band (in_band) and never contributes to superiority
        (promotion.py:217-222, :595-601).

        Args:
            obj: Objective specification (must have band).
            incumbent_samples: Samples for incumbent (reported only).
            candidate_samples: Samples for candidate.
            epsilon: Epsilon tolerance (reported only).

        Returns:
            ObjectiveResult for the banded comparison.
        """
        if obj.band is None:
            raise ValueError("Banded comparison requires obj.band to be set")

        inc_tost = tost_equivalence_test(incumbent_samples, obj.band, obj.band_alpha)
        cand_tost = tost_equivalence_test(candidate_samples, obj.band, obj.band_alpha)

        inc_mean = inc_tost.sample_mean
        cand_mean = cand_tost.sample_mean

        # Band gate: the candidate must itself be within the band.
        verdict: Literal["in_band", "out_of_band"] = (
            "in_band" if cand_tost.is_equivalent else "out_of_band"
        )
        # TOST p-value (in_band iff this is below band_alpha).
        p_value = max(cand_tost.p_lower, cand_tost.p_upper)

        if obj.band.low is not None and obj.band.high is not None:
            center = (obj.band.low + obj.band.high) / 2
            effect_size = abs(inc_mean - center) - abs(cand_mean - center)
        else:
            effect_size = 0.0

        return ObjectiveResult(
            name=obj.name,
            direction="band",
            candidate_better=False,  # Banded objectives never grant superiority.
            p_value=p_value,
            effect_size=effect_size,
            candidate_mean=cand_mean,
            incumbent_mean=inc_mean,
            epsilon=epsilon,
            p_value_noninf=p_value,
            p_value_super=1.0,
            verdict=verdict,
        )

    def _evaluate_chance_constraints(
        self,
        constraint_data: dict[str, tuple[int, int]],
    ) -> list[ChanceConstraintResult]:
        """Evaluate chance constraints.

        Each constraint is an upper bound on a violation rate: it is satisfied
        iff the one-sided upper confidence bound on the violation rate is at or
        below the threshold (canonical: promotion.py:627-628, :677-678).

        Args:
            constraint_data: Dict mapping constraint name to (violations, trials).

        Returns:
            List of ChanceConstraintResult.
        """
        results = []

        for constraint in self.policy.chance_constraints:
            if constraint.name not in constraint_data:
                # Missing data - treat as not satisfied (fail-closed).
                results.append(
                    ChanceConstraintResult(
                        name=constraint.name,
                        satisfied=False,
                        observed_rate=0.0,
                        upper_bound=1.0,
                        threshold=constraint.threshold,
                        confidence=constraint.confidence,
                    )
                )
                continue

            violations, trials = constraint_data[constraint.name]

            if trials == 0:
                results.append(
                    ChanceConstraintResult(
                        name=constraint.name,
                        satisfied=False,
                        observed_rate=0.0,
                        upper_bound=1.0,
                        threshold=constraint.threshold,
                        confidence=constraint.confidence,
                    )
                )
                continue

            observed_rate = violations / trials
            upper_bound = clopper_pearson_upper_bound(
                violations, trials, constraint.confidence
            )

            # Constraint is satisfied iff the upper bound on the violation rate
            # is at or below the allowed threshold.
            satisfied = upper_bound <= constraint.threshold

            results.append(
                ChanceConstraintResult(
                    name=constraint.name,
                    satisfied=satisfied,
                    observed_rate=observed_rate,
                    upper_bound=upper_bound,
                    threshold=constraint.threshold,
                    confidence=constraint.confidence,
                )
            )

        return results

    @classmethod
    def from_spec_artifact(
        cls,
        artifact: Any,  # TVLSpecArtifact - avoid circular import
    ) -> PromotionGate | None:
        """Create a PromotionGate from a TVLSpecArtifact.

        Args:
            artifact: The loaded TVL spec artifact.

        Returns:
            PromotionGate if promotion policy is defined, None otherwise.
        """
        return cls.from_policy(
            promotion_policy=getattr(artifact, "promotion_policy", None),
            objective_schema=getattr(artifact, "objective_schema", None),
        )

    @classmethod
    def from_policy(
        cls,
        promotion_policy: PromotionPolicy | None,
        objective_schema: Any | None,
    ) -> PromotionGate | None:
        """Create a PromotionGate from policy + objective schema."""
        if promotion_policy is None:
            return None

        # Build objective specs from the objective schema.
        objectives: list[ObjectiveSpec] = []

        if objective_schema is not None:
            for obj_def in objective_schema.objectives:
                # obj_def is an ObjectiveDefinition, not a dict
                band = obj_def.band  # Already a BandTarget or None
                band_alpha = obj_def.band_alpha if obj_def.band_alpha else 0.05

                objectives.append(
                    ObjectiveSpec(
                        name=obj_def.name,
                        direction=obj_def.orientation,  # orientation, not direction
                        band=band,
                        band_alpha=band_alpha,
                    )
                )

        return cls(promotion_policy, objectives)
