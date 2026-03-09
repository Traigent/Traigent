#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

npm run build:sdk >/dev/null
node scripts/examples/run_examples.mjs --base examples/quickstart --pattern .mjs
node scripts/examples/run_examples.mjs --base examples/core --pattern run.mjs
