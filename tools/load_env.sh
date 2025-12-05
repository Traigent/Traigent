#!/usr/bin/env bash
# shellcheck disable=SC2034

# Helper to export variables from the repository-level .env file into the
# current shell session. Must be sourced:
#   source tools/load_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: load_env.sh must be sourced, not executed. Use: source tools/load_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "warning: no .env file found at ${ENV_FILE}. Nothing to load." >&2
  return 0
fi

# Export every variable defined in .env for the caller's shell.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

echo "Loaded environment variables from ${ENV_FILE}"
