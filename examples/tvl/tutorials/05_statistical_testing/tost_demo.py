#!/usr/bin/env python3
"""
TVL 0.9 Tutorial: TOST and Banded Objectives

This script demonstrates Two One-Sided Tests (TOST) for equivalence testing,
which is used to determine if a metric is statistically within a target band.

Key concepts:
- TOST: Tests if mean is within [lower, upper] bounds
- Banded objectives: Metrics that should stay within a range
- BandTarget: Defines the target band (low/high or center/tolerance)
- BandedObjectiveSpec: Evaluates samples against a band

Run with (from repo root): .venv/bin/python examples/tvl/tutorials/05_statistical_testing/tost_demo.py
"""

import random
from pathlib import Path

from traigent.tvl import (
    BandTarget,
    load_tvl_spec,
)
from traigent.tvl.objectives import (
    BandedObjectiveSpec,
    compare_banded_objectives,
    compare_banded_with_tost,
    tost_equivalence_test,
)

SPEC_PATH = Path(__file__).parent / "banded_objectives.tvl.yml"


def demonstrate_band_target():
    """Show how to create and use BandTarget."""
    print("\n" + "-" * 50)
    print("BandTarget Demo")
    print("-" * 50)

    # Create band using low/high
    band1 = BandTarget(low=0.85, high=0.95)
    print(f"\n  Band from low/high: [{band1.low}, {band1.high}]")

    # Create band using center/tolerance
    band2 = BandTarget(center=0.90, tol=0.05)
    print(f"  Band from center/tol: [{band2.low}, {band2.high}]")

    # Test if values are in band
    test_values = [0.80, 0.85, 0.90, 0.95, 1.00]
    print(f"\n  Testing values against band [{band1.low}, {band1.high}]:")
    for val in test_values:
        in_band = band1.contains(val)
        deviation = band1.deviation(val)
        print(f"    {val:.2f}: in_band={in_band}, deviation={deviation:.3f}")


def demonstrate_tost():
    """Show TOST equivalence testing."""
    print("\n" + "-" * 50)
    print("TOST Equivalence Testing Demo")
    print("-" * 50)

    band = BandTarget(low=0.85, high=0.95)
    print(f"\n  Target band: [{band.low}, {band.high}]")

    # Samples clearly inside the band
    samples_inside = [
        0.88,
        0.91,
        0.89,
        0.92,
        0.90,
        0.87,
        0.91,
        0.89,
        0.90,
        0.88,
        0.89,
        0.91,
        0.88,
        0.90,
        0.92,
        0.89,
        0.87,
        0.90,
        0.91,
        0.89,
    ]
    result_inside = tost_equivalence_test(samples_inside, band, alpha=0.05)
    print(f"\n  Samples inside band (mean={result_inside.sample_mean:.3f}):")
    print(f"    Is equivalent: {result_inside.is_equivalent}")
    print(f"    P-value (lower): {result_inside.p_lower:.4f}")
    print(f"    P-value (upper): {result_inside.p_upper:.4f}")
    print(f"    Sample std: {result_inside.sample_std:.4f}")
    ci_low, ci_high = result_inside.confidence_interval
    print(
        f"    90% CI: [{ci_low:.3f}, {ci_high:.3f}]"
    )  # 90% CI corresponds to alpha=0.05 TOST

    # Samples outside the band (too low)
    samples_low = [
        0.78,
        0.80,
        0.79,
        0.82,
        0.81,
        0.77,
        0.80,
        0.79,
        0.81,
        0.78,
        0.79,
        0.80,
        0.78,
        0.81,
        0.79,
        0.80,
        0.77,
        0.79,
        0.80,
        0.78,
    ]
    result_low = tost_equivalence_test(samples_low, band, alpha=0.05)
    print(f"\n  Samples below band (mean={result_low.sample_mean:.3f}):")
    print(f"    Is equivalent: {result_low.is_equivalent}")
    print(f"    P-value (lower): {result_low.p_lower:.4f}")
    print(f"    P-value (upper): {result_low.p_upper:.4f}")

    # Samples outside the band (too high)
    samples_high = [
        0.98,
        0.99,
        0.97,
        1.00,
        0.98,
        0.99,
        0.97,
        0.98,
        1.00,
        0.99,
        0.98,
        0.97,
        0.99,
        0.98,
        1.00,
        0.99,
        0.97,
        0.98,
        0.99,
        0.98,
    ]
    result_high = tost_equivalence_test(samples_high, band, alpha=0.05)
    print(f"\n  Samples above band (mean={result_high.sample_mean:.3f}):")
    print(f"    Is equivalent: {result_high.is_equivalent}")
    print(f"    P-value (lower): {result_high.p_lower:.4f}")
    print(f"    P-value (upper): {result_high.p_upper:.4f}")

    # Edge case: single sample
    print("\n  Edge case: n=1 sample")
    result_n1_inside = tost_equivalence_test([0.90], band)
    print(f"    Sample=0.90: is_equivalent={result_n1_inside.is_equivalent}")
    result_n1_outside = tost_equivalence_test([0.80], band)
    print(f"    Sample=0.80: is_equivalent={result_n1_outside.is_equivalent}")


def demonstrate_banded_comparison():
    """Show how to compare configurations on banded objectives."""
    print("\n" + "-" * 50)
    print("Banded Objective Comparison Demo")
    print("-" * 50)

    band = BandTarget(low=0.85, high=0.95)
    print(f"\n  Target band: [{band.low}, {band.high}]")

    # Compare two point values
    print("\n  Point value comparison:")
    scenarios = [
        (0.90, 0.80, "A in band, B below"),
        (0.80, 0.90, "A below, B in band"),
        (0.90, 0.92, "Both in band"),
        (0.80, 0.75, "Both below band"),
    ]

    for val_a, val_b, desc in scenarios:
        result = compare_banded_objectives(val_a, val_b, band)
        print(f"    {desc}: A={val_a}, B={val_b}")
        print(f"      Winner: {result.winner}")
        print(f"      A in band: {result.a_in_band}, B in band: {result.b_in_band}")
        print(
            f"      A deviation: {result.a_deviation:.3f}, B deviation: {result.b_deviation:.3f}"
        )
        print()


def demonstrate_banded_with_tost():
    """Show comparison using TOST for statistical rigor."""
    print("\n" + "-" * 50)
    print("Banded Comparison with TOST Demo")
    print("-" * 50)

    band = BandTarget(low=0.85, high=0.95)
    print(f"\n  Target band: [{band.low}, {band.high}]")

    random.seed(42)

    # Config A: clearly in band
    samples_a = [0.89 + random.gauss(0, 0.02) for _ in range(20)]
    # Config B: below band
    samples_b = [0.80 + random.gauss(0, 0.02) for _ in range(20)]

    print(f"\n  Config A mean: {sum(samples_a)/len(samples_a):.3f}")
    print(f"  Config B mean: {sum(samples_b)/len(samples_b):.3f}")

    winner, tost_a, tost_b = compare_banded_with_tost(samples_a, samples_b, band)

    print("\n  TOST Results:")
    print(f"    A equivalent to band: {tost_a.is_equivalent}")
    print(f"    B equivalent to band: {tost_b.is_equivalent}")
    print(f"    Winner: {winner}")


def demonstrate_banded_objective_spec():
    """Show programmatic use of BandedObjectiveSpec."""
    print("\n" + "-" * 50)
    print("BandedObjectiveSpec Demo")
    print("-" * 50)

    # Create a banded objective specification
    band = BandTarget(low=0.85, high=0.95)
    spec = BandedObjectiveSpec(
        name="response_consistency",
        target=band,
        test="TOST",
        alpha=0.05,
    )

    print(f"\n  Banded Objective: {spec.name}")
    print(f"  Target: [{spec.target.low}, {spec.target.high}]")
    print(f"  Test: {spec.test}")
    print(f"  Alpha: {spec.alpha}")

    # Evaluate samples
    random.seed(42)
    good_samples = [0.90 + random.gauss(0, 0.02) for _ in range(20)]
    bad_samples = [0.78 + random.gauss(0, 0.02) for _ in range(20)]

    print(f"\n  Good config (mean={sum(good_samples)/len(good_samples):.3f}):")
    good_result = spec.evaluate(good_samples)
    print(f"    Satisfied: {spec.is_satisfied(good_samples)}")
    print(f"    TOST result: is_equivalent={good_result.is_equivalent}")

    print(f"\n  Bad config (mean={sum(bad_samples)/len(bad_samples):.3f}):")
    bad_result = spec.evaluate(bad_samples)
    print(f"    Satisfied: {spec.is_satisfied(bad_samples)}")
    print(f"    TOST result: is_equivalent={bad_result.is_equivalent}")


def main():
    """Main tutorial demonstration."""
    print("=" * 60)
    print("TVL 0.9 Tutorial: Banded Objectives & TOST")
    print("=" * 60)

    # Load spec to show context
    print("\n1. Loading TVL Spec...")
    spec = load_tvl_spec(spec_path=SPEC_PATH)
    print(f"   Loaded: {spec.path}")
    print(f"   Objectives: {len(spec.objective_schema.objectives)}")

    # Run demonstrations
    print("\n2. Demonstrations...")
    demonstrate_band_target()
    demonstrate_tost()
    demonstrate_banded_comparison()
    demonstrate_banded_with_tost()
    demonstrate_banded_objective_spec()

    print("\n" + "=" * 60)
    print("Tutorial complete!")
    print("=" * 60)
    print("\n  You've learned:")
    print("  - BandTarget: Define target bands for metrics")
    print("  - TOST: Test if samples are statistically within a band")
    print("  - Banded comparisons: Choose configs that best fit bands")
    print("  - BandedObjectiveSpec: Programmatic banded objective evaluation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
