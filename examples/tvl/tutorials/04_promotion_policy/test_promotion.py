#!/usr/bin/env python3
"""
TVL 0.9 Tutorial: Testing Promotion Policy

This script demonstrates how to use the PromotionGate to make
statistically rigorous promotion decisions.

Key concepts:
- PromotionGate: Evaluates if a candidate should replace incumbent
- Epsilon-Pareto dominance: Requires meaningful improvements
- Chance constraints: Probabilistic safety guarantees
- BH correction: Controls false discovery rate

Run with (from repo root): .venv/bin/python examples/tvl/tutorials/04_promotion_policy/test_promotion.py
"""

import random
from pathlib import Path

from traigent.tvl import (
    load_tvl_spec,
)
from traigent.tvl.promotion_gate import (
    ObjectiveSpec,
    PromotionGate,
)
from traigent.tvl.statistics import (
    benjamini_hochberg_adjust,
    clopper_pearson_lower_bound,
)

SPEC_PATH = Path(__file__).parent / "safety_critical.tvl.yml"


def generate_samples(mean: float, std: float, n: int = 20) -> list[float]:
    """Generate n samples from a normal distribution."""
    return [mean + random.gauss(0, std) for _ in range(n)]


def demonstrate_bh_correction():
    """Show Benjamini-Hochberg correction in action."""
    print("\n" + "-" * 50)
    print("Benjamini-Hochberg Correction Demo")
    print("-" * 50)

    # Simulate p-values from testing multiple objectives
    # Some are significant (< 0.05), some are not
    raw_p_values = [0.001, 0.015, 0.030, 0.049, 0.080, 0.120]

    print(f"\n  Raw p-values:      {[f'{p:.3f}' for p in raw_p_values]}")

    adjusted = benjamini_hochberg_adjust(raw_p_values)
    print(f"  BH-adjusted:       {[f'{p:.3f}' for p in adjusted]}")

    print("\n  Interpretation (alpha=0.05):")
    for i, (raw, adj) in enumerate(zip(raw_p_values, adjusted, strict=True)):
        sig_raw = "sig" if raw < 0.05 else "not sig"
        sig_adj = "sig" if adj < 0.05 else "not sig"
        change = "CHANGED" if sig_raw != sig_adj else ""
        print(
            f"    Objective {i+1}: raw={raw:.3f} ({sig_raw}), adj={adj:.3f} ({sig_adj}) {change}"
        )

    print("\n  BH correction reduces false discoveries when testing")
    print("  multiple objectives simultaneously.")


def demonstrate_chance_constraints():
    """Show chance constraint evaluation."""
    print("\n" + "-" * 50)
    print("Chance Constraints Demo")
    print("-" * 50)

    # Simulate safety score samples
    good_samples = [
        0.96,
        0.97,
        0.95,
        0.98,
        0.96,
        0.97,
        0.95,
        0.96,
        0.98,
        0.97,
        0.96,
        0.95,
        0.97,
        0.96,
        0.98,
        0.95,
        0.97,
        0.96,
        0.95,
        0.96,
    ]
    bad_samples = [
        0.92,
        0.93,
        0.91,
        0.94,
        0.90,
        0.93,
        0.91,
        0.92,
        0.94,
        0.90,
        0.93,
        0.91,
        0.92,
        0.90,
        0.93,
        0.91,
        0.92,
        0.94,
        0.91,
        0.93,
    ]

    threshold = 0.95
    confidence = 0.99

    print(f"\n  Chance constraint: P(safety_score >= {threshold}) >= {confidence}")

    # Check good samples
    successes_good = sum(1 for s in good_samples if s >= threshold)
    lb_good = clopper_pearson_lower_bound(successes_good, len(good_samples), confidence)
    passes_good = lb_good >= threshold

    print(f"\n  Good configuration (mean={sum(good_samples)/len(good_samples):.3f}):")
    print(f"    Samples >= threshold: {successes_good}/{len(good_samples)}")
    print(f"    Lower bound ({confidence:.0%} CI): {lb_good:.3f}")
    print(f"    Constraint satisfied: {passes_good}")

    # Check bad samples
    successes_bad = sum(1 for s in bad_samples if s >= threshold)
    lb_bad = clopper_pearson_lower_bound(successes_bad, len(bad_samples), confidence)
    passes_bad = lb_bad >= threshold

    print(f"\n  Bad configuration (mean={sum(bad_samples)/len(bad_samples):.3f}):")
    print(f"    Samples >= threshold: {successes_bad}/{len(bad_samples)}")
    print(f"    Lower bound ({confidence:.0%} CI): {lb_bad:.3f}")
    print(f"    Constraint satisfied: {passes_bad}")


def demonstrate_promotion_gate():
    """Show full promotion gate evaluation."""
    print("\n" + "-" * 50)
    print("Promotion Gate Demo")
    print("-" * 50)

    # Create promotion policy from spec
    spec = load_tvl_spec(spec_path=SPEC_PATH)
    policy = spec.promotion_policy

    if not policy:
        print("  No promotion policy in spec!")
        return

    print("\n  Promotion Policy:")
    print(f"    Dominance: {policy.dominance}")
    print(f"    Alpha: {policy.alpha}")
    print(f"    Min effects: {policy.min_effect}")
    print(f"    Chance constraints: {len(policy.chance_constraints)}")

    # Create PromotionGate
    objectives = [
        ObjectiveSpec("task_accuracy", "maximize"),
        ObjectiveSpec("response_latency", "minimize"),
        ObjectiveSpec("safety_score", "maximize"),
        ObjectiveSpec("cost_per_request", "minimize"),
    ]

    gate = PromotionGate(policy, objectives)

    # Simulate incumbent and candidate metrics
    random.seed(42)

    print("\n  Scenario 1: Clear improvement")
    incumbent_1 = {
        "task_accuracy": generate_samples(0.82, 0.02),
        "response_latency": generate_samples(500, 30),
        "safety_score": generate_samples(0.96, 0.01),
        "cost_per_request": generate_samples(0.001, 0.0001),
    }
    candidate_1 = {
        "task_accuracy": generate_samples(0.88, 0.02),  # Better
        "response_latency": generate_samples(450, 30),  # Better
        "safety_score": generate_samples(0.97, 0.01),  # Better
        "cost_per_request": generate_samples(0.0009, 0.0001),  # Better
    }
    # Provide constraint_data for chance constraints: (successes, trials)
    # Candidate passes safety check 19/20 times
    constraint_data_1 = {"safety_score": (19, 20)}

    decision_1 = gate.evaluate(incumbent_1, candidate_1, constraint_data_1)
    print(f"    Decision: {decision_1.decision}")
    print(f"    Reason: {decision_1.reason}")

    print("\n  Scenario 2: Trade-off (better accuracy, worse latency)")
    incumbent_2 = {
        "task_accuracy": generate_samples(0.85, 0.02),
        "response_latency": generate_samples(400, 30),
        "safety_score": generate_samples(0.96, 0.01),
        "cost_per_request": generate_samples(0.001, 0.0001),
    }
    candidate_2 = {
        "task_accuracy": generate_samples(0.90, 0.02),  # Better
        "response_latency": generate_samples(600, 30),  # Worse!
        "safety_score": generate_samples(0.96, 0.01),  # Same
        "cost_per_request": generate_samples(0.0012, 0.0001),  # Worse!
    }
    constraint_data_2 = {"safety_score": (18, 20)}

    decision_2 = gate.evaluate(incumbent_2, candidate_2, constraint_data_2)
    print(f"    Decision: {decision_2.decision}")
    print(f"    Reason: {decision_2.reason}")

    print("\n  Scenario 3: Marginal improvement (within epsilon)")
    incumbent_3 = {
        "task_accuracy": generate_samples(0.85, 0.02),
        "response_latency": generate_samples(500, 30),
        "safety_score": generate_samples(0.96, 0.01),
        "cost_per_request": generate_samples(0.001, 0.0001),
    }
    candidate_3 = {
        "task_accuracy": generate_samples(0.86, 0.02),  # Only 1% better (epsilon=2%)
        "response_latency": generate_samples(490, 30),  # Only 10ms better (epsilon=25)
        "safety_score": generate_samples(0.962, 0.01),  # Marginally better
        "cost_per_request": generate_samples(0.00099, 0.0001),  # Marginally better
    }
    # Still need constraint_data for chance constraints
    constraint_data_3 = {"safety_score": (19, 20)}

    decision_3 = gate.evaluate(incumbent_3, candidate_3, constraint_data_3)
    print(f"    Decision: {decision_3.decision}")
    print(f"    Reason: {decision_3.reason}")


def main():
    """Main tutorial demonstration."""
    print("=" * 60)
    print("TVL 0.9 Tutorial: Promotion Policy")
    print("=" * 60)

    # Load and display spec info
    print("\n1. Loading TVL Spec...")
    spec = load_tvl_spec(spec_path=SPEC_PATH)

    policy = spec.promotion_policy
    if policy:
        print(f"   Dominance: {policy.dominance}")
        print(f"   Alpha: {policy.alpha}")
        print(f"   Adjustment: {policy.adjust}")
        print(f"   Min effects: {len(policy.min_effect)} objectives")
        print(f"   Chance constraints: {len(policy.chance_constraints)}")

    # Demonstrate components
    print("\n2. Component Demonstrations...")
    demonstrate_bh_correction()
    demonstrate_chance_constraints()
    demonstrate_promotion_gate()

    print("\n" + "=" * 60)
    print("Tutorial complete! Next: 05_statistical_testing")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
