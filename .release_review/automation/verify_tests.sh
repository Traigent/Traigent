#!/usr/bin/env bash
# Verify test claims using pytest JSON report output.
# Exit codes:
#   0 = verified
#   1 = mismatch
#   2 = infra/tooling error

set -euo pipefail

TEST_PATH="${1:-tests/}"
CLAIMED_PASSED="${2:-0}"
CLAIMED_TOTAL="${3:-0}"

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest not found"
  exit 2
fi

REPORT_FILE="$(mktemp -t release_review_pytest_report.XXXXXX.json)"
cleanup() {
  rm -f "$REPORT_FILE"
}
trap cleanup EXIT

set +e
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true PYTHONPATH=. \
pytest "$TEST_PATH" -q --tb=no --json-report --json-report-file "$REPORT_FILE" >/tmp/release_review_verify_tests.out 2>/tmp/release_review_verify_tests.err
PYTEST_EXIT=$?
set -e

if [ ! -s "$REPORT_FILE" ]; then
  echo "pytest JSON report missing or empty"
  exit 2
fi

readarray -t PARSED < <(python3 - "$REPORT_FILE" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
summary = report.get("summary", {})
passed = int(summary.get("passed", 0))
failed = int(summary.get("failed", 0))
errors = int(summary.get("error", 0)) + int(summary.get("errors", 0))
skipped = int(summary.get("skipped", 0))
total = int(summary.get("total", passed + failed + errors + skipped))
print(passed)
print(failed)
print(errors)
print(skipped)
print(total)
PY
)

PASSED="${PARSED[0]:-0}"
FAILED="${PARSED[1]:-0}"
ERRORS="${PARSED[2]:-0}"
SKIPPED="${PARSED[3]:-0}"
ACTUAL_TOTAL="${PARSED[4]:-0}"

STATUS="MISMATCH"
if [ "$PASSED" -eq "$CLAIMED_PASSED" ] && [ "$ACTUAL_TOTAL" -eq "$CLAIMED_TOTAL" ] && [ "$PYTEST_EXIT" -eq 0 ]; then
  STATUS="VERIFIED"
elif [ "$PASSED" -ge "$CLAIMED_PASSED" ] && [ "$ACTUAL_TOTAL" -ge "$CLAIMED_TOTAL" ] && [ "$PYTEST_EXIT" -eq 0 ]; then
  STATUS="VERIFIED_BETTER"
fi

cat <<JSON
{
  "status": "$STATUS",
  "claimed": {
    "passed": $CLAIMED_PASSED,
    "total": $CLAIMED_TOTAL
  },
  "actual": {
    "passed": $PASSED,
    "failed": $FAILED,
    "errors": $ERRORS,
    "skipped": $SKIPPED,
    "total": $ACTUAL_TOTAL
  },
  "pytest_exit_code": $PYTEST_EXIT,
  "test_path": "$TEST_PATH",
  "report_file": "$REPORT_FILE"
}
JSON

if [ "$STATUS" = "VERIFIED" ] || [ "$STATUS" = "VERIFIED_BETTER" ]; then
  exit 0
fi
exit 1
