#!/usr/bin/env bash
# Docker-free shell-level self-test for live_contract_fill_optuna_pool.sh.
#
# Stubs `docker` on PATH so this exercises argument construction and failure
# handling without a running compose stack -- it can run in ordinary unit CI,
# not just the sdk-backend-live-contract lane. It does not exercise the real
# backend, Postgres, or provision_optuna_project_storage.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_UNDER_TEST="$SCRIPT_DIR/live_contract_fill_optuna_pool.sh"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

FAKE_BIN="$WORKDIR/bin"
mkdir -p "$FAKE_BIN"
ARGV_LOG="$WORKDIR/docker_argv.log"
FREE_COUNT_FILE="$WORKDIR/free_count"
EXIT_CODE_FILE="$WORKDIR/exit_code"
: > "$ARGV_LOG"
echo 1 > "$FREE_COUNT_FILE"
echo 0 > "$EXIT_CODE_FILE"

# A fake `docker` that records every invocation and simulates the two
# backend-side commands the script under test issues: the pool-filler CLI
# (exit code from $EXIT_CODE_FILE) and the free_slot_count probe (prints
# $FREE_COUNT_FILE). State lives in files, not env vars, so it survives the
# command-substitution subshells below.
cat > "$FAKE_BIN/docker" <<'FAKE_DOCKER_EOF'
#!/usr/bin/env bash
state_dir="${FAKE_DOCKER_STATE_DIR:?FAKE_DOCKER_STATE_DIR must be set}"
printf '%s\n' "$*" >> "$state_dir/docker_argv.log"

case "$*" in
  *free_slot_count*)
    cat "$state_dir/free_count"
    exit 0
    ;;
  *provision_optuna_project_storage.py*)
    exit "$(cat "$state_dir/exit_code")"
    ;;
esac
exit 0
FAKE_DOCKER_EOF
chmod +x "$FAKE_BIN/docker"

export FAKE_DOCKER_STATE_DIR="$WORKDIR"
export PATH="$FAKE_BIN:$PATH"

pass=0
fail=0

assert_eq() {
  local description="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "FAIL: $description -- expected [$expected], got [$actual]" >&2
  fi
}

assert_contains() {
  local description="$1" haystack="$2" needle="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "FAIL: $description -- expected to find [$needle] in:" >&2
    echo "$haystack" >&2
  fi
}

assert_not_contains() {
  local description="$1" haystack="$2" needle="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "FAIL: $description -- must not contain [$needle]" >&2
  fi
}

# --- F4: pin the one formula a slot's login-role name is derived from. This
# is the coupling between compose.optuna-pool.override.yml's
# TRAIGENT_OPTUNA_CI_SLOT_0000_USERNAME and what
# provision_optuna_project_storage.py's _slot_names() derives from
# (--name-prefix, index) -- a drift here fails the slot at fill time with
# "resolved username does not match the slot's login role".
actual="$("$SCRIPT_UNDER_TEST" slot-username 0)"
assert_eq "slot-username(0) with the default CI prefix" "ci_optuna_login_0000" "$actual"

actual="$(POOL_NAME_PREFIX=other_prefix "$SCRIPT_UNDER_TEST" slot-username 3)"
assert_eq "slot-username(3) with a custom prefix" "other_prefix_login_0003" "$actual"

actual="$("$SCRIPT_UNDER_TEST" slot-credential-ref 0)"
assert_eq "slot-credential-ref(0)" "env:TRAIGENT_OPTUNA_CI_SLOT_0000" "$actual"

# --- Argument construction: a successful fill issues the pool-filler CLI
# TWICE (F5's real login-role re-probe) plus exactly one free_slot_count
# read, all pinned to the same compose project and the same two -f files.
: > "$ARGV_LOG"
echo 1 > "$FREE_COUNT_FILE"
echo 0 > "$EXIT_CODE_FILE"

output="$(
  COMPOSE_PROJECT_NAME=test-project \
  BACKEND_DIR=/workspace/TraigentBackend \
  SDK_DIR=/workspace/Traigent \
  "$SCRIPT_UNDER_TEST" fill
)"

fill_invocations="$(grep -c 'provision_optuna_project_storage.py' "$ARGV_LOG" || true)"
assert_eq "a successful fill issues exactly 2 filler invocations" "2" "$fill_invocations"

# free_slot_count's python -c payload has embedded newlines, so each
# invocation spans several log lines and a plain `grep -c free_slot_count`
# over-counts; os.environ["DATABASE_URL"] appears exactly once per
# invocation regardless of line-wrapping.
probe_invocations="$(grep -c 'os.environ\["DATABASE_URL"\]' "$ARGV_LOG" || true)"
assert_eq "a successful fill reads free_slot_count exactly once" "1" "$probe_invocations"

argv="$(cat "$ARGV_LOG")"
assert_contains "fill targets the given compose project" "$argv" "-p test-project"
assert_contains "fill layers the backend dev compose file" "$argv" "/workspace/TraigentBackend/docker/docker-compose.dev.yml"
assert_contains "fill layers the Optuna pool override file" "$argv" "/workspace/Traigent/.github/live-contract/compose.optuna-pool.override.yml"
assert_contains "fill names the CI maintenance credential ref" "$argv" "--maintenance-credential-ref env:TRAIGENT_OPTUNA_CI_MAINTENANCE"
assert_contains "fill names the per-slot credential ref template" "$argv" "--credential-ref-template env:TRAIGENT_OPTUNA_CI_SLOT_{index:04d}"
assert_contains "fill names the CI slot name-prefix" "$argv" "--name-prefix ci_optuna"
assert_contains "fill reads the free-slot count via the container's DATABASE_URL" "$argv" 'os.environ["DATABASE_URL"]'
assert_contains "success output reports the free-slot count" "$output" "1 free slot(s)"
assert_not_contains "fill argv never carries a password-looking value" "$argv" "PASSWORD"
assert_not_contains "fill argv never carries the maintenance username literal as a value" "$argv" "credential-ref-template=env:TRAIGENT_OPTUNA_CI_MAINTENANCE_PASSWORD"

# --- Failure handling: the filler CLI failing on the FIRST pass must fail
# the script loudly, before ever attempting the second pass or the
# free-slot-count read (a fabricated count would hide the failure).
: > "$ARGV_LOG"
echo 1 > "$EXIT_CODE_FILE"
set +e
failure_output="$(
  COMPOSE_PROJECT_NAME=test-project \
  BACKEND_DIR=/workspace/TraigentBackend \
  SDK_DIR=/workspace/Traigent \
  "$SCRIPT_UNDER_TEST" fill 2>&1
)"
failure_status=$?
set -e

assert_eq "a failing first-pass fill exits non-zero" "1" "$failure_status"
assert_contains "a failing fill reports an ::error:: line" "$failure_output" "::error::"
assert_not_contains "a failing fill never echoes credential material" "$failure_output" "PASSWORD"
fill_invocations_on_failure="$(grep -c 'provision_optuna_project_storage.py' "$ARGV_LOG" || true)"
assert_eq "a failing first-pass fill never attempts the second pass" "1" "$fill_invocations_on_failure"
probe_invocations_on_failure="$(grep -c 'os.environ\["DATABASE_URL"\]' "$ARGV_LOG" || true)"
assert_eq "a failing fill never reads free_slot_count" "0" "$probe_invocations_on_failure"

# --- F5: exit 0 from the CLI is not capacity proof by itself -- a pool
# report that free_slot_count came back short of what was requested must
# still fail the script, even though both filler invocations "succeeded".
: > "$ARGV_LOG"
echo 0 > "$EXIT_CODE_FILE"
echo 0 > "$FREE_COUNT_FILE"
set +e
short_output="$(
  COMPOSE_PROJECT_NAME=test-project \
  BACKEND_DIR=/workspace/TraigentBackend \
  SDK_DIR=/workspace/Traigent \
  "$SCRIPT_UNDER_TEST" fill 2>&1
)"
short_status=$?
set -e

assert_eq "a short free-slot count fails the script despite exit-0 fills" "1" "$short_status"
assert_contains "a short free-slot count is reported by name" "$short_output" "0 free slot(s), expected exactly 1"

echo ""
echo "test_live_contract_fill_optuna_pool: ${pass} passed, ${fail} failed"
if [ "$fail" -ne 0 ]; then
  exit 1
fi
