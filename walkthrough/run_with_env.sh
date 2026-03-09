#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${1:-}"
SCRIPT_PATH="${2:-}"

if [[ -z "$ENV_FILE" || -z "$SCRIPT_PATH" ]]; then
  echo "Usage: walkthrough/run_with_env.sh <env-file> <script>"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

npm run build:sdk >/dev/null
node "$SCRIPT_PATH"
