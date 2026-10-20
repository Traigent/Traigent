#!/usr/bin/env bash
# PY-1 live-dev acceptance (design sdk-reactive-policy.md section 7.5). PREPARED, NOT RUN BY THE WORKER.
#
# Needs a dev tenant with PRODUCTION_REQUIREMENTS_ENABLED=true and a test agent that has a baseline
# run and validity V1. Evidence = this transcript + the printed ids; copy the run dir under runs/.
#
#   TRAIGENT_BACKEND_URL=https://<dev backend> TRAIGENT_API_KEY=<key via your own secret store> \
#   PROJECT_ID=<pid> AGENT_ID=<server Agent.id> PY312=<python3.12 with the runtime-tvl extra> \
#   scripts/validation/runtime_requirements/live_dev_acceptance.sh
#
# Optional: WAIT_OPEN_S (default 6000), WAIT_RESOLVE_S (default 14400), POLL_S (default 60),
#           MIN_SAMPLES_TRACES (default 120; must be >= the requirement's min_samples of 100), RUN_DIR.
# Every blocking call below is bounded by `timeout`. The script never deploys anything.
set -uo pipefail

: "${TRAIGENT_BACKEND_URL:?set the dev backend origin}"
: "${TRAIGENT_API_KEY:?set an API key (from your secret store; never paste it into a file)}"
: "${PROJECT_ID:?set the project id}"
: "${AGENT_ID:?set the server Agent.id (never the trace agent key)}"
: "${PY312:?set a Python 3.12+ with the runtime-tvl extra for step 2}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
RUN_DIR="${RUN_DIR:-runs/py1-live-dev-$(date -u +%Y%m%dT%H%M%SZ)}"
WAIT_OPEN_S="${WAIT_OPEN_S:-6000}"
WAIT_RESOLVE_S="${WAIT_RESOLVE_S:-14400}"
POLL_S="${POLL_S:-60}"
N_TRACES="${MIN_SAMPLES_TRACES:-120}"
mkdir -p "$RUN_DIR"
export TRAIGENT_RUNTIME_STATE_DIR="$RUN_DIR/state"
LOG="$RUN_DIR/transcript.txt"
FAILED=0

log()  { printf '\n== %s\n' "$*" | tee -a "$LOG"; }
step() { # step <label> <timeout-s> <expected-grep|-> -- cmd...
  local label="$1" secs="$2" expect="$3"; shift 4
  log "$label"
  local out; out="$(timeout "$secs" "$@" 2>&1)"; local rc=$?
  printf '%s\nexit=%s\n' "$out" "$rc" | tee -a "$LOG"
  if [[ "$rc" -ne 0 ]]; then echo "FAIL: $label (exit $rc)" | tee -a "$LOG"; FAILED=1; return 1; fi
  if [[ "$expect" != "-" ]] && ! grep -Eq "$expect" <<<"$out"; then
    echo "FAIL: $label (expected /$expect/)" | tee -a "$LOG"; FAILED=1; return 1
  fi
}
CLI=("$PY" -m traigent.cli.main requirements)
CLI312=("$PY312" -m traigent.cli.main requirements)
COMMON=(--project-id "$PROJECT_ID" --agent-id "$AGENT_ID")
cd "$HERE"

# 1. kwargs apply: created, then idempotent no-op.
step "1a apply (kwargs) -> created" 120 'created via native' -- "${CLI[@]}" apply live_agent.py -f answer "${COMMON[@]}"
step "1b apply again -> noop"       120 'noop via native'    -- "${CLI[@]}" apply live_agent.py -f answer "${COMMON[@]}"

# 2. TVL fragment on Python 3.12 + runtime-tvl: created (key 2500 ms), then identical import = 200 noop.
step "2a apply (TVL fragment) -> created" 180 'created via tvl_import' -- "${CLI312[@]}" apply live_agent_tvl.py -f answer "${COMMON[@]}"
step "2b apply again -> identical import" 180 'noop via tvl_import'    -- "${CLI312[@]}" apply live_agent_tvl.py -f answer "${COMMON[@]}"

# 3. slow traces -> episode_opened within one window + grace + ~15 min.
step "3a emit slow traces (3500 ms > 2000 ms)" 600 'emitted' -- \
  "$PY" emit_traces.py --agent-id "$AGENT_ID" --count "$N_TRACES" --latency-ms 3500
log "3b watch until episode_opened (<= ${WAIT_OPEN_S}s)"
deadline=$(( $(date +%s) + WAIT_OPEN_S )); opened=0
while (( $(date +%s) < deadline )); do
  out="$(timeout 120 "${CLI[@]}" watch live_agent.py -f answer "${COMMON[@]}" 2>&1)"; echo "$out" | tee -a "$LOG"
  if grep -q 'episode_opened' <<<"$out" && grep -q 'coverage: complete' <<<"$out"; then opened=1; break; fi
  sleep "$POLL_S"
done
(( opened )) || { echo "FAIL: no episode_opened with coverage complete" | tee -a "$LOG"; FAILED=1; }

# 4. respond: shows RESTORE, asks, runs optimize after y, prints the run id; second run reports the used occurrence.
#    A pty is required (non-interactive respond is refused by design); `script` supplies one.
log "4a respond (answer y)"
printf 'y\n' | timeout 3600 script -qec "$(printf '%q ' "${CLI[@]}" respond live_agent.py -f answer "${COMMON[@]}")" /dev/null 2>&1 | tee -a "$LOG"
grep -q 'run id:' "$LOG" || { echo "FAIL: no run id printed" | tee -a "$LOG"; FAILED=1; }
log "4b respond again -> occurrence already used"
printf 'y\n' | timeout 300 script -qec "$(printf '%q ' "${CLI[@]}" respond live_agent.py -f answer "${COMMON[@]}")" /dev/null 2>&1 | tee -a "$LOG"
grep -q 'occurrence_already_used' "$LOG" || { echo "FAIL: second respond was not refused" | tee -a "$LOG"; FAILED=1; }

# 5. healthy traces -> after K windows watch prints episode_resolved.
step "5a emit healthy traces (200 ms)" 600 'emitted' -- \
  "$PY" emit_traces.py --agent-id "$AGENT_ID" --count "$N_TRACES" --latency-ms 200
log "5b watch until episode_resolved (<= ${WAIT_RESOLVE_S}s)"
deadline=$(( $(date +%s) + WAIT_RESOLVE_S )); resolved=0
while (( $(date +%s) < deadline )); do
  out="$(timeout 120 "${CLI[@]}" watch live_agent.py -f answer "${COMMON[@]}" 2>&1)"; echo "$out" | tee -a "$LOG"
  if grep -q 'episode_resolved' <<<"$out"; then resolved=1; break; fi
  sleep "$POLL_S"
done
(( resolved )) || { echo "FAIL: no episode_resolved" | tee -a "$LOG"; FAILED=1; }

step "status (ids for the evidence record)" 120 - -- "${CLI[@]}" status live_agent.py -f answer "${COMMON[@]}"
log "done; FAILED=$FAILED; transcript: $LOG"
exit "$FAILED"
