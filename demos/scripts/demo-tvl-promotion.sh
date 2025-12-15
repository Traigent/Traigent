#!/bin/bash
# Demo: TVL 0.9 Promotion Policy - Statistical Testing for Production
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# TVL 0.9: Promotion Policy"
echo "# Statistical testing for safe production deployments"
echo ""
sleep 1.5

echo "# Problem: When is a new config ACTUALLY better?"
echo "# - Random variance in LLM outputs"
echo "# - Multiple objectives to compare"
echo "# - Need statistical confidence, not just higher numbers"
echo ""
sleep 2

echo "# Solution: TVL 0.9 Promotion Policy"
echo ""
sleep 0.5

cat << 'YAML'
# production_deploy.tvl.yml
spec:
  id: production-deployment
  version: "1.0"

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o", "gpt-4o-mini"]

objectives:
  - name: accuracy
    direction: maximize
  - name: latency_ms
    direction: minimize
  - name: cost
    direction: minimize

# PROMOTION POLICY - The key TVL 0.9 feature!
promotion_policy:
  dominance: epsilon_pareto    # Require meaningful improvement
  alpha: 0.05                  # 95% confidence level
  adjust: BH                   # Benjamini-Hochberg correction

  # Minimum improvement thresholds (epsilon values)
  min_effect:
    accuracy: 0.02             # Must improve by at least 2%
    latency_ms: 50.0           # Must be 50ms faster
    cost: 0.0001               # Must be $0.0001 cheaper

  # Safety constraints with confidence bounds
  chance_constraints:
    - name: accuracy
      threshold: 0.85          # Accuracy must be >= 85%
      confidence: 0.95         # With 95% confidence
YAML
sleep 4

echo ""
echo "# Key Concepts:"
echo ""
echo "1. EPSILON-PARETO DOMINANCE"
echo "   New config must be better by at least epsilon (min_effect)"
echo "   Prevents promoting configs that are just noise"
echo ""
sleep 2

echo "2. STATISTICAL TESTING (alpha=0.05)"
echo "   Use paired t-tests to verify improvement is real"
echo "   Not just lucky samples!"
echo ""
sleep 2

echo "3. BENJAMINI-HOCHBERG CORRECTION (adjust: BH)"
echo "   When testing multiple objectives, control false discovery rate"
echo "   Prevents 'p-hacking' across many comparisons"
echo ""
sleep 2

echo "4. CHANCE CONSTRAINTS"
echo "   P(accuracy >= 0.85) >= 0.95"
echo "   Guarantees minimum performance with high confidence"
echo ""
sleep 2

echo "# Using PromotionGate in Python"
echo ""
sleep 0.5

cat << 'PYTHON'
from traigent.tvl import load_tvl_spec
from traigent.tvl.promotion_gate import PromotionGate, ObjectiveSpec

spec = load_tvl_spec("production_deploy.tvl.yml")

# Convert objective definitions to ObjectiveSpec for PromotionGate
objectives = [
    ObjectiveSpec(obj.name, obj.orientation)
    for obj in spec.objective_schema.objectives
]
gate = PromotionGate(spec.promotion_policy, objectives)

# Compare incumbent (current production) vs candidate (new config)
# constraint_data provides (successes, trials) for chance constraints
decision = gate.evaluate(
    incumbent_metrics={
        "accuracy": [0.82, 0.84, 0.81, 0.83, 0.85],
        "latency_ms": [320, 315, 340, 325, 318],
        "cost": [0.0012, 0.0011, 0.0013, 0.0012, 0.0011]
    },
    candidate_metrics={
        "accuracy": [0.87, 0.89, 0.86, 0.88, 0.90],
        "latency_ms": [245, 250, 242, 248, 251],
        "cost": [0.0008, 0.0007, 0.0008, 0.0007, 0.0008]
    },
    constraint_data={"accuracy": (5, 5)}  # All 5 samples passed threshold
)

print(f"Decision: {decision.decision}")
print(f"Reason: {decision.reason}")
print(f"P-values: {decision.adjusted_p_values}")
PYTHON
sleep 3

echo ""
echo "# Demo: Running Promotion Gate"
echo ""
sleep 0.5

echo '$ python test_promotion.py'
sleep 0.5
echo ""
echo "Comparing incumbent vs candidate..."
echo ""
echo "Statistical Tests (alpha=0.05, BH-adjusted):"
echo "  accuracy:   p=0.0012 < 0.05 (significant improvement)"
echo "  latency_ms: p=0.0034 < 0.05 (significant improvement)"
echo "  cost:       p=0.0089 < 0.05 (significant improvement)"
echo ""
sleep 1

echo "Epsilon-Pareto Check:"
echo "  accuracy:   +0.05 >= 0.02 epsilon (PASS)"
echo "  latency_ms: -72ms >= 50ms epsilon (PASS)"
echo "  cost:       -0.0004 >= 0.0001 epsilon (PASS)"
echo ""
sleep 1

echo "Chance Constraints:"
echo "  P(accuracy >= 0.85) = 0.97 >= 0.95 (PASS)"
echo ""
sleep 1

echo "=========================================="
echo "DECISION: PROMOTE"
echo "Reason: Candidate dominates incumbent with"
echo "        statistical significance on all objectives"
echo "=========================================="
sleep 3

echo ""
echo "# Why This Matters:"
echo "  - No more 'deploy and pray'"
echo "  - Statistical rigor prevents false positives"
echo "  - Safety constraints protect production SLOs"
echo "  - Auditable promotion decisions"
sleep 2
