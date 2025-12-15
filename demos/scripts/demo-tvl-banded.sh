#!/bin/bash
# Demo: TVL 0.9 Banded Objectives - TOST Equivalence Testing
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# TVL 0.9: Banded Objectives & TOST"
echo "# When you want 'good enough', not maximum"
echo ""
sleep 1.5

echo "# Not all objectives should be maximized/minimized!"
echo ""
echo "# Example scenarios:"
echo "  - Latency: 'Under 500ms is fine, don't over-optimize'"
echo "  - Response length: 'Between 100-300 words is ideal'"
echo "  - Confidence score: '0.85-0.95 is the sweet spot'"
echo ""
sleep 2

echo "# Solution: BANDED OBJECTIVES"
echo ""
sleep 0.5

cat << 'YAML'
# banded_objectives.tvl.yml
spec:
  id: banded-optimization
  version: "1.0"

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o", "gpt-4o-mini"]
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]

objectives:
  # Standard objectives
  - name: accuracy
    direction: maximize

  # BANDED OBJECTIVE: Just needs to be within range
  - name: latency_ms
    direction: band
    target:
      low: 200
      high: 500

  # BANDED with center/tolerance notation
  - name: response_length
    direction: band
    target:
      center: 200          # Target 200 words
      tol: 50              # +/- 50 words acceptable
YAML
sleep 3

echo ""
echo "# What is TOST (Two One-Sided Tests)?"
echo ""
echo "Traditional test: 'Is X different from Y?'"
echo "TOST test:        'Is X equivalent to target band?'"
echo ""
sleep 2

echo "# TOST checks BOTH bounds simultaneously:"
echo "  H0: value < low  OR  value > high  (outside band)"
echo "  H1: low <= value <= high           (inside band)"
echo ""
echo "  If BOTH one-sided tests pass -> value is IN the band"
echo ""
sleep 2

echo "# Using Banded Objectives in Python"
echo ""
sleep 0.5

cat << 'PYTHON'
from traigent.tvl import load_tvl_spec, BandTarget
from traigent.tvl.objectives import BandedObjectiveSpec, tost_equivalence_test

spec = load_tvl_spec("banded_objectives.tvl.yml")

# Create a banded objective evaluator
# Get band from the objective definition
latency_obj = spec.objective_schema.objectives[1]  # latency_ms
banded = BandedObjectiveSpec(
    name="latency_ms",
    target=latency_obj.band,  # BandTarget(low=200, high=500)
    alpha=0.05
)

# Test if samples are within the band
latency_samples = [245, 312, 289, 267, 301, 256, 278, 295]

result = banded.evaluate(latency_samples)  # Returns TOSTResult
print(f"Mean latency: {result.sample_mean:.1f}ms")
print(f"TOST p-values: lower={result.p_lower:.4f}, upper={result.p_upper:.4f}")
print(f"Equivalent at alpha=0.05: {result.is_equivalent}")
PYTHON
sleep 3

echo ""
echo "# Demo: TOST Equivalence Testing"
echo ""
sleep 0.5

echo '$ python tost_demo.py'
sleep 0.5
echo ""
echo "Testing latency samples against band [200, 500]ms"
echo ""
echo "Sample statistics:"
echo "  n = 8 samples"
echo "  Mean = 280.4ms"
echo "  Std = 23.1ms"
echo ""
sleep 1

echo "TOST Results:"
echo "  Lower bound test (>= 200): p = 0.0001"
echo "  Upper bound test (<= 500): p = 0.0000"
echo "  Combined p-value: 0.0001"
echo ""
sleep 1

echo "========================================"
echo "RESULT: EQUIVALENT"
echo "Latency is within [200-500]ms band"
echo "at alpha=0.05 significance level"
echo "========================================"
sleep 2

echo ""
echo "# Banded vs Standard Objectives"
echo ""
echo "Standard (maximize/minimize):"
echo "  - Always push for more/less"
echo "  - Can over-optimize at cost of other objectives"
echo ""
echo "Banded (target range):"
echo "  - 'Good enough' is good enough"
echo "  - Frees up optimization budget for other objectives"
echo "  - Better for SLO compliance checking"
echo ""
sleep 2

echo "# Real-World Use Cases:"
echo "  - API latency SLOs (must be 200-500ms)"
echo "  - Token usage budgets (stay within allocation)"
echo "  - Quality metrics with diminishing returns"
echo "  - Regulatory compliance ranges"
sleep 2
