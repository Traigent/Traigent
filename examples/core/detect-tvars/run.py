#!/usr/bin/env python3
"""Detect tunable variables in a sample agent using static analysis.

Demonstrates the Traigent tuned variable detection engine:
1. Programmatic detection via TunedVariableDetector
2. Multi-strategy results (AST name-matching + data-flow slicing)
3. Configuration space generation from detected candidates

Usage:
    # Run this script directly
    python examples/core/detect-tvars/run.py

    # Or use the CLI for the same result
    traigent detect-tvars examples/core/detect-tvars/sample_agent.py
    traigent detect-tvars examples/core/detect-tvars/sample_agent.py --json
"""

from __future__ import annotations

from pathlib import Path

from traigent.tuned_variables import TunedVariableDetector


def main() -> None:
    sample_file = Path(__file__).parent / "sample_agent.py"
    detector = TunedVariableDetector()

    print("=" * 60)
    print("Traigent Tuned Variable Detection Demo")
    print("=" * 60)
    print(f"\nScanning: {sample_file.name}\n")

    # Detect from file — analyzes all top-level functions
    results = detector.detect_from_file(sample_file)

    if not results:
        print("No tunable variable candidates detected.")
        return

    total = sum(r.count for r in results)
    print(f"Found {total} candidate(s) across {len(results)} function(s)\n")

    for result in results:
        print(f"--- {result.function_name} ---")
        print(f"  Strategies used: {', '.join(result.detection_strategies_used)}")

        for c in result.candidates:
            source_tag = f"[{c.detection_source}]"
            value_str = repr(c.current_value) if c.current_value is not None else "?"
            print(
                f"  {c.name:20s} {c.confidence.value:6s}  {source_tag:12s}  = {value_str}"
            )
            if c.suggested_range:
                print(f"  {'':20s} -> {c.suggested_range.to_parameter_range_code()}")
            if c.reasoning:
                print(f"  {'':20s}    {c.reasoning}")
        print()

    # Generate a ready-to-use configuration space
    print("=" * 60)
    print("Suggested configuration_space (HIGH + MEDIUM candidates):")
    print("=" * 60)
    for result in results:
        config_space = result.to_configuration_space()
        if config_space:
            print(f"\n# {result.function_name}")
            print("configuration_space = {")
            for name, range_obj in config_space.items():
                print(f"    {name!r}: {range_obj!r},")
            print("}")


if __name__ == "__main__":
    main()
