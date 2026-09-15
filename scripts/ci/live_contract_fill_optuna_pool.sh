#!/usr/bin/env bash
# Fills the Optuna runtime-slot pool inside the dockerized live-contract
# backend so a typed session create does not 503
# (typed_storage_mapping_unavailable, TraigentBackend#3131/#3152,
# Traigent#2243). Run this after the compose stack reports healthy and
# BEFORE any API key is minted or session created -- filling ahead of demand
# means the guard never fires during a normal lane run.
#
# Uses only the backend's own provisioning entry point
# (scripts/maintenance/provision_optuna_project_storage.py --pool); this
# script never touches the 503 guard or Optuna DDL directly.
#
# Required environment (already job-level env in live-contract.yml):
#   COMPOSE_PROJECT_NAME  docker compose -p project name for this run
#   BACKEND_DIR           absolute path to the TraigentBackend checkout
#   SDK_DIR               absolute path to this SDK checkout (locates
#                         compose.optuna-pool.override.yml)
#
# Optional overrides (scripts/ci/test_live_contract_fill_optuna_pool.sh
# exercises these without Docker):
#   POOL_NAME_PREFIX      identifier prefix for derived schema/role/login
#                         names (default: ci_optuna)
#   POOL_COUNT            number of slots to fill (default: 1)
#   POOL_START_INDEX      first slot index (default: 0)
#   POOL_STORAGE_LOCATOR  credential-free postgres URL the runtime planes
#                         live in, reachable from inside the backend
#                         container (default: postgresql://postgres:5432/traigent)
#
# Usage:
#   live_contract_fill_optuna_pool.sh                 fill the pool (default)
#   live_contract_fill_optuna_pool.sh fill
#   live_contract_fill_optuna_pool.sh slot-username <index>
#   live_contract_fill_optuna_pool.sh slot-credential-ref <index>
set -euo pipefail

POOL_NAME_PREFIX="${POOL_NAME_PREFIX:-ci_optuna}"
POOL_COUNT="${POOL_COUNT:-1}"
POOL_START_INDEX="${POOL_START_INDEX:-0}"
POOL_STORAGE_LOCATOR="${POOL_STORAGE_LOCATOR:-postgresql://postgres:5432/traigent}"

# env: prefixes are validated by credential_ref_adapter.py against
# ^TRAIGENT_OPTUNA_[A-Z][A-Z0-9_]{0,63}$ -- both names below satisfy it.
MAINTENANCE_CREDENTIAL_REF="env:TRAIGENT_OPTUNA_CI_MAINTENANCE"
CREDENTIAL_REF_TEMPLATE="env:TRAIGENT_OPTUNA_CI_SLOT_{index:04d}"

# F4: the ONE formula for a slot's login-role name. The live-contract
# workflow's "Provision Optuna pool-fill credentials" step calls this same
# subcommand to set TRAIGENT_OPTUNA_CI_SLOT_<NNNN>_USERNAME -- so the value
# baked into the compose override and the name
# provision_optuna_project_storage.py derives from (--name-prefix, index)
# can never drift apart. Must match _slot_names() in
# scripts/maintenance/provision_optuna_project_storage.py:
#   runtime_login_role = f"{name_prefix}_login_{slot_index:04d}"
slot_username() {
  local index="$1"
  printf '%s_login_%04d' "$POOL_NAME_PREFIX" "$index"
}

slot_credential_ref() {
  local index="$1"
  printf 'env:TRAIGENT_OPTUNA_CI_SLOT_%04d' "$index"
}

_compose() {
  docker compose \
    -p "$COMPOSE_PROJECT_NAME" \
    -f "$BACKEND_DIR/docker/docker-compose.dev.yml" \
    -f "$SDK_DIR/.github/live-contract/compose.optuna-pool.override.yml" \
    "$@"
}

_run_fill() {
  _compose exec -T backend python scripts/maintenance/provision_optuna_project_storage.py \
    --pool \
    --count "$POOL_COUNT" \
    --start-index "$POOL_START_INDEX" \
    --provider postgres \
    --name-prefix "$POOL_NAME_PREFIX" \
    --storage-locator "$POOL_STORAGE_LOCATOR" \
    --credential-ref-template "$CREDENTIAL_REF_TEMPLATE" \
    --maintenance-credential-ref "$MAINTENANCE_CREDENTIAL_REF"
}

# free_slot_count() (src/services/optuna/slot_claim.py) is the same query the
# request-path low-water alert reads; running it through the container's own
# DATABASE_URL (the app role project creation actually claims with) proves
# capacity from the claimer's point of view, not the maintenance principal's.
_free_slot_count() {
  _compose exec -T backend python -c '
import os
from sqlalchemy import create_engine
from src.services.optuna.slot_claim import free_slot_count

engine = create_engine(os.environ["DATABASE_URL"])
try:
    with engine.connect() as connection:
        print(free_slot_count(connection))
finally:
    engine.dispose()
'
}

fill() {
  : "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME must be set}"
  : "${BACKEND_DIR:?BACKEND_DIR must be set}"
  : "${SDK_DIR:?SDK_DIR must be set}"

  echo "Filling Optuna runtime-slot pool: prefix=${POOL_NAME_PREFIX} count=${POOL_COUNT} start=${POOL_START_INDEX}"
  if ! _run_fill; then
    echo "::error::Optuna pool fill failed (first pass) -- see the backend container logs for the affected schema/roles; no credential material is ever printed here." >&2
    exit 1
  fi

  # F5: exit 0 alone does not prove capacity -- a slot the ledger already
  # shows claimed or quarantined is reported "skipped" and still exits 0.
  # Re-running the identical fill forces provision_runtime_slot's "existing"
  # branch, which opens a real authenticated connection with the resolved
  # password (_verify_login_credential) -- that is the actual capacity proof,
  # not the exit code.
  if ! _run_fill; then
    echo "::error::Optuna pool fill failed (second pass -- the login-credential probe). See the backend container logs." >&2
    exit 1
  fi

  local free
  free="$(_free_slot_count)"
  if [ "$free" != "$POOL_COUNT" ]; then
    echo "::error::Optuna runtime slot pool has ${free} free slot(s), expected exactly ${POOL_COUNT}." >&2
    exit 1
  fi
  echo "Optuna runtime slot pool has ${free} free slot(s) (expected ${POOL_COUNT})."
}

case "${1:-fill}" in
  slot-username)
    slot_username "${2:?slot index required}"
    ;;
  slot-credential-ref)
    slot_credential_ref "${2:?slot index required}"
    ;;
  fill)
    fill
    ;;
  *)
    echo "usage: $0 [fill|slot-username <index>|slot-credential-ref <index>]" >&2
    exit 2
    ;;
esac
