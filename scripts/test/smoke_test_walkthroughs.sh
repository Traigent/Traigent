#!/bin/bash

# Lightweight smoke test that runs every walkthrough example in mock mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export TRAIGENT_MOCK_LLM="${TRAIGENT_MOCK_LLM:-true}"
EXAMPLE_TIMEOUT="${WALKTHROUGH_TIMEOUT:-30}"

examples=(walkthrough/examples/0*.py)

echo "🚬 Running walkthrough smoke test (timeout: ${EXAMPLE_TIMEOUT}s)"
for example in "${examples[@]}"; do
    if [[ ! -f "$example" ]]; then
        continue
    fi

    echo "-------------------------------------"
    echo "Testing: $example"
    echo "-------------------------------------"

    if timeout "${EXAMPLE_TIMEOUT}" python "$example"; then
        echo "✅ $example - PASSED"
    else
        rc=$?
        echo "❌ $example - FAILED (exit code $rc)"
        exit $rc
    fi
done

echo "✅ All walkthroughs passed"
