#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---mock}"
BASE="walkthrough/mock"

if [[ "$MODE" == "--real" ]]; then
  BASE="walkthrough/real"
fi

npm run build:sdk >/dev/null
node scripts/examples/run_examples.mjs --base "$BASE" --pattern .mjs
