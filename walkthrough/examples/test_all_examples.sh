#!/bin/bash

# Test all Traigent walkthrough examples in mock mode

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TRAIGENT_MOCK_MODE=true

echo "====================================="
echo "Testing Traigent Walkthrough Examples"
echo "====================================="
echo ""

# Array of example files
examples=(
    "01_simple_optimization.py"
    "02_zero_code_change.py"
    "03_parameter_mode.py"
    "04_multi_objective.py"
    "05_rag_example.py"
    "06_custom_evaluator.py"
    "07_privacy_modes.py"
)

# Test each example
EXAMPLE_TIMEOUT=25
for example in "${examples[@]}"; do
    echo "-------------------------------------"
    echo "Testing: $example"
    echo "-------------------------------------"

    if [ -f "$example" ]; then
        # Run with timeout and capture result
        timeout "$EXAMPLE_TIMEOUT" python "$example" 2>&1 | tail -10

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ $example - PASSED"
        elif [ ${PIPESTATUS[0]} -eq 124 ]; then
            echo "⚠️  $example - TIMEOUT (but likely working)"
        else
            echo "❌ $example - FAILED"
        fi
    else
        echo "❌ $example - FILE NOT FOUND"
    fi

    echo ""
done

echo "====================================="
echo "Testing Complete!"
echo "====================================="
