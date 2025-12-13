#!/bin/bash
# Verify Tests Script for Release Review Protocol
#
# Captain uses this to spot-check agent test claims.
# Re-runs tests and compares with agent's claimed results.
#
# Usage: ./verify_tests.sh <test_path> <claimed_passed> <claimed_total>
# Example: ./verify_tests.sh tests/unit/core 20 20

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arguments
TEST_PATH="${1:-tests/}"
CLAIMED_PASSED="${2:-0}"
CLAIMED_TOTAL="${3:-0}"

echo "=========================================="
echo "Release Review: Test Verification"
echo "=========================================="
echo "Test path: $TEST_PATH"
echo "Agent claimed: $CLAIMED_PASSED/$CLAIMED_TOTAL passed"
echo ""

# Set mock mode for safety
export TRAIGENT_MOCK_MODE=true

# Run tests and capture output
echo "Running tests..."
TEST_OUTPUT=$(pytest "$TEST_PATH" -q --tb=no 2>&1) || true

# Parse results
# pytest output format: "X passed, Y failed, Z skipped in Xs"
# or just "X passed in Xs"
PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1 || echo "0")
FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1 || echo "0")
SKIPPED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= skipped)' | tail -1 || echo "0")
ERRORS=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= error)' | tail -1 || echo "0")

# Default to 0 if not found
PASSED=${PASSED:-0}
FAILED=${FAILED:-0}
SKIPPED=${SKIPPED:-0}
ERRORS=${ERRORS:-0}

ACTUAL_TOTAL=$((PASSED + FAILED + SKIPPED + ERRORS))

echo ""
echo "Actual results:"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  Errors:  $ERRORS"
echo "  Total:   $ACTUAL_TOTAL"
echo ""

# Compare with claims
VERIFICATION_STATUS="UNKNOWN"

if [ "$PASSED" -eq "$CLAIMED_PASSED" ] && [ "$ACTUAL_TOTAL" -eq "$CLAIMED_TOTAL" ]; then
    VERIFICATION_STATUS="VERIFIED"
    echo -e "${GREEN}✅ VERIFIED: Test results match agent claims${NC}"
elif [ "$PASSED" -ge "$CLAIMED_PASSED" ] && [ "$ACTUAL_TOTAL" -ge "$CLAIMED_TOTAL" ]; then
    VERIFICATION_STATUS="VERIFIED_BETTER"
    echo -e "${GREEN}✅ VERIFIED (better): Actual results exceed claims${NC}"
    echo "   (Agent may have tested a subset)"
elif [ "$PASSED" -eq "$CLAIMED_PASSED" ] && [ "$ACTUAL_TOTAL" -ne "$CLAIMED_TOTAL" ]; then
    VERIFICATION_STATUS="PARTIAL_MATCH"
    echo -e "${YELLOW}⚠️  PARTIAL MATCH: Pass count matches but total differs${NC}"
    echo "   Claimed total: $CLAIMED_TOTAL, Actual total: $ACTUAL_TOTAL"
else
    VERIFICATION_STATUS="MISMATCH"
    echo -e "${RED}❌ MISMATCH: Test results do not match agent claims${NC}"
    echo "   Claimed: $CLAIMED_PASSED/$CLAIMED_TOTAL"
    echo "   Actual:  $PASSED/$ACTUAL_TOTAL"
fi

# Output JSON for programmatic use
echo ""
echo "JSON output:"
cat << EOF
{
  "status": "$VERIFICATION_STATUS",
  "claimed": {
    "passed": $CLAIMED_PASSED,
    "total": $CLAIMED_TOTAL
  },
  "actual": {
    "passed": $PASSED,
    "failed": $FAILED,
    "skipped": $SKIPPED,
    "errors": $ERRORS,
    "total": $ACTUAL_TOTAL
  },
  "test_path": "$TEST_PATH",
  "timestamp": "$(date -Iseconds)"
}
EOF

# Exit code based on verification
case "$VERIFICATION_STATUS" in
    "VERIFIED"|"VERIFIED_BETTER")
        exit 0
        ;;
    "PARTIAL_MATCH")
        exit 1
        ;;
    "MISMATCH")
        exit 2
        ;;
    *)
        exit 3
        ;;
esac
