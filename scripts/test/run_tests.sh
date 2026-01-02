#!/bin/bash
# Run tests with different configurations

set -e

echo "🧪 Running Traigent SDK Tests"
echo "============================"

# Function to run tests and capture results
run_test_suite() {
    local name=$1
    local cmd=$2

    echo -e "\n📋 Running $name..."

    if eval "$cmd"; then
        echo "✅ $name passed"
        return 0
    else
        echo "❌ $name failed"
        return 1
    fi
}

# Create test results directory
mkdir -p test_results

# 1. Quick smoke tests
echo -e "\n🔥 Smoke Tests"
python -m pytest tests/unit/api/test_types.py -v -x

# 2. Unit tests by module
echo -e "\n🧩 Unit Tests by Module"

# Test each module separately to identify issues
modules=(
    "api"
    "core"
    "config"
    "utils"
    "optimizers"
    "evaluators"
    "analytics"
)

for module in "${modules[@]}"; do
    if [ -d "tests/unit/$module" ]; then
        echo -e "\n  Testing $module..."
        python -m pytest "tests/unit/$module" -v --tb=short -x || echo "  ⚠️  $module has failing tests"
    fi
done

# 3. Integration tests (if they don't take too long)
echo -e "\n🔗 Integration Tests"
python -m pytest tests/integration -v --tb=short -x --maxfail=3 || echo "⚠️  Integration tests have issues"

# 4. Generate coverage report
echo -e "\n📊 Coverage Report"
python -m pytest tests/unit -q --cov=traigent --cov-report=term-missing --cov-report=html --cov-report=xml

# 5. Summary
echo -e "\n📈 Test Summary"
echo "==============="
echo "✅ Test run complete"
echo "📁 Coverage report: htmlcov/index.html"
echo "📄 Coverage XML: coverage.xml"

# Check if running in CI
if [ -n "$CI" ]; then
    # In CI, fail if coverage is below threshold
    coverage report --fail-under=60
fi
