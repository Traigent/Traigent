#!/bin/bash
# Demo: TVL 0.9 Basics - Typed Tuned Variables
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# TVL 0.9: Typed Tuned Variables Language"
echo ""
sleep 1

echo "# TVL is a declarative language for LLM optimization"
echo "# Define WHAT to optimize, let TraiGent figure out HOW"
echo ""
sleep 2

echo "# Step 1: Create a TVL spec file (chatbot.tvl.yml)"
echo ""
sleep 0.5

cat << 'YAML'
# chatbot.tvl.yml - TVL 0.9 Specification
spec:
  id: chatbot-optimization
  version: "1.0"

metadata:
  owner: ml-team@company.com
  description: Optimize customer support chatbot

# TYPED TUNED VARIABLES (TVARs)
tvars:
  - name: model
    type: enum[str]                    # Strongly typed!
    domain: ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet"]
    default: "gpt-4o-mini"

  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
      resolution: 0.1                  # Discretized search
    default: 0.7

  - name: max_tokens
    type: int
    domain:
      range: [256, 2048]
    default: 512

# What to optimize
objectives:
  - name: response_quality
    direction: maximize

  - name: latency_ms
    direction: minimize

  - name: cost_per_query
    direction: minimize
YAML
sleep 4

echo ""
echo "# Key TVL 0.9 Features:"
echo "  1. Typed TVARs: bool, int, float, enum[T], tuple[...], callable[Proto]"
echo "  2. Domain specs: ranges, enums, resolution for discretization"
echo "  3. Multi-objective optimization support"
echo ""
sleep 2

echo "# Step 2: Load and use the spec in Python"
echo ""
sleep 0.5

cat << 'PYTHON'
import traigent
from traigent.tvl import load_tvl_spec

# Load and inspect the spec
spec = load_tvl_spec("chatbot.tvl.yml")

print(f"Spec ID: {spec.metadata['spec_id']}")
print(f"TVARs: {[t.name for t in spec.tvars]}")
print(f"Objectives: {[o.name for o in spec.objective_schema.objectives]}")

# Use with decorator
@traigent.optimize(tvl_spec="chatbot.tvl.yml")
def chatbot_respond(query: str, *, model: str, temperature: float, max_tokens: int):
    """TraiGent injects optimized parameters automatically."""
    return call_llm(model, temperature, max_tokens, query)
PYTHON
sleep 3

echo ""
echo "# Step 3: Run optimization"
echo ""
sleep 0.5

echo '$ TRAIGENT_MOCK_MODE=true python optimize.py'
sleep 0.5
echo ""
echo "Loading TVL spec: chatbot.tvl.yml"
echo "  3 TVARs defined (model, temperature, max_tokens)"
echo "  3 objectives (response_quality, latency_ms, cost_per_query)"
echo ""
sleep 1

echo "Running optimization..."
echo "[1/24] model=gpt-4o-mini, temp=0.3, tokens=512"
echo "       quality=0.82, latency=245ms, cost=$0.0004"
sleep 0.3
echo "[2/24] model=gpt-4o-mini, temp=0.5, tokens=512"
echo "       quality=0.79, latency=251ms, cost=$0.0004"
sleep 0.3
echo "[3/24] model=gpt-4o, temp=0.3, tokens=512"
echo "       quality=0.91, latency=892ms, cost=$0.0032"
sleep 0.3
echo "..."
sleep 0.5
echo "[24/24] model=claude-3-sonnet, temp=0.7, tokens=1024"
echo "        quality=0.88, latency=634ms, cost=$0.0018"
echo ""
sleep 1

echo "Pareto-optimal configurations found: 4"
echo ""
echo "Best trade-offs:"
echo "  1. gpt-4o, temp=0.3 -> Best quality (0.91), highest cost"
echo "  2. gpt-4o-mini, temp=0.3 -> Good quality (0.82), lowest cost"
echo "  3. claude-3-sonnet, temp=0.5 -> Balanced (0.88 quality, mid cost)"
sleep 3

echo ""
echo "# TVL 0.9 Benefits:"
echo "  - Type-safe parameter definitions"
echo "  - Declarative, version-controlled configs"
echo "  - Multi-objective Pareto optimization"
echo "  - Separation of WHAT vs HOW"
sleep 2
