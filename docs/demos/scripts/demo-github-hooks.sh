#!/bin/bash
# Demo: Traigent Agent Configuration Hooks
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# Traigent Hooks: Agent Configuration Validation"
echo ""
sleep 1

echo "# Traigent provides Git hooks to validate agent configurations"
echo "# before they reach production."
echo ""
sleep 0.5

echo "Available Traigent hooks:"
echo "  traigent-validate    - Validates agent config against constraints"
echo "  traigent-performance - Ensures config meets performance baseline"
echo "  traigent-cost        - Enforces cost budget limits"
echo ""
sleep 2

echo "# Step 1: Install Traigent hooks"
echo ""
sleep 0.5

echo "$ traigent hooks install"
sleep 0.5
echo "Installing Traigent Git hooks..."
echo "  Installed: .git/hooks/pre-push (traigent-validate)"
echo "  Installed: .git/hooks/pre-commit (traigent-config-check)"
echo ""
echo "Hooks installed successfully!"
sleep 2

echo ""
echo "# Step 2: Define validation constraints in traigent.yml"
echo ""
sleep 0.5

cat << 'YAML'
# traigent.yml - Agent configuration constraints
validation:
  hooks:
    pre-push:
      - traigent-validate
      - traigent-performance

constraints:
  # Cost constraints
  max_cost_per_query: 0.05      # $0.05 max per query
  max_monthly_budget: 1000      # $1000/month limit

  # Performance constraints
  min_accuracy: 0.85            # 85% minimum accuracy
  max_latency_ms: 500           # 500ms max response time

  # Model constraints
  allowed_models:
    - gpt-4o-mini
    - gpt-4o
    - claude-3-haiku
  blocked_models:
    - gpt-4-32k                 # Too expensive for production
YAML
sleep 3

echo ""
echo "# Step 3: Commit an agent with valid configuration"
echo ""
sleep 0.5

cat << 'PYTHON'
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini"],      # Allowed model
        "temperature": [0.1, 0.3]
    }
)
def support_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return llm.invoke(query).content
PYTHON
sleep 2

echo ""
echo '$ git commit -m "Add support agent with gpt-4o-mini"'
echo '$ git push origin main'
echo ""
sleep 0.5

echo "Running Traigent validation hooks..."
echo ""
sleep 0.3

echo "[traigent-validate] Checking agent configurations..."
echo "  Found 1 decorated function: support_agent"
echo "  Model: gpt-4o-mini .......... ALLOWED"
echo "  Est. cost: \$0.008/query ..... WITHIN BUDGET"
sleep 0.5
echo ""
echo "[traigent-performance] Checking performance baseline..."
echo "  Accuracy: 0.89 .............. PASSES (min: 0.85)"
echo "  Latency: 120ms .............. PASSES (max: 500ms)"
sleep 0.5
echo ""
echo "All Traigent hooks passed!"
echo "Pushing to origin/main..."
sleep 2

echo ""
echo "# Step 4: Try to push an agent that VIOLATES constraints"
echo ""
sleep 0.5

cat << 'PYTHON'
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4-32k"],        # BLOCKED model!
        "temperature": [0.9]
    }
)
def expensive_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4-32k", temperature=0.9)
    return llm.invoke(query).content
PYTHON
sleep 2

echo ""
echo '$ git commit -m "Add expensive agent with gpt-4-32k"'
echo '$ git push origin main'
echo ""
sleep 0.5

echo "Running Traigent validation hooks..."
echo ""
sleep 0.3

echo "[traigent-validate] Checking agent configurations..."
echo "  Found 1 decorated function: expensive_agent"
echo ""
sleep 0.3
echo "  ERROR: Model 'gpt-4-32k' is in blocked_models list"
echo "         Reason: Too expensive for production"
echo ""
echo "  ERROR: Estimated cost \$0.12/query exceeds max_cost_per_query (\$0.05)"
echo ""
sleep 0.5
echo "PUSH REJECTED: Agent configuration violates constraints"
echo ""
echo "To bypass (not recommended):"
echo "  git push --no-verify"
echo ""
echo "To fix:"
echo "  1. Use an allowed model: gpt-4o-mini, gpt-4o, claude-3-haiku"
echo "  2. Or request budget increase in traigent.yml"
sleep 3

echo ""
echo "# Step 5: Fix the configuration and push successfully"
echo ""
sleep 0.5

cat << 'PYTHON'
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],  # Allowed models
        "temperature": [0.1, 0.5]
    }
)
def fixed_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return llm.invoke(query).content
PYTHON
sleep 2

echo ""
echo '$ git push origin main'
echo ""
sleep 0.5

echo "[traigent-validate] Checking agent configurations..."
echo "  Model: gpt-4o-mini .......... ALLOWED"
echo "  Model: gpt-4o ............... ALLOWED"
echo "  Est. cost: \$0.008/query ..... WITHIN BUDGET"
echo ""
echo "All Traigent hooks passed!"
echo "Pushing to origin/main..."
sleep 2

echo ""
echo "# Summary: Traigent Agent Hooks"
echo ""
echo "  Validate:    Block pushes with invalid agent configs"
echo "  Performance: Ensure accuracy/latency meet baseline"
echo "  Cost:        Enforce budget limits before production"
echo ""
echo "  Prevent expensive mistakes BEFORE they reach production!"
sleep 2
