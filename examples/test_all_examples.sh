#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

node "$ROOT/examples/quickstart/01_simple_qa.mjs"
node "$ROOT/examples/quickstart/02_customer_support_rag.mjs"
node "$ROOT/examples/quickstart/03_custom_objectives.mjs"
node "$ROOT/examples/core/native-grid-search/run.mjs"
node "$ROOT/examples/core/seamless-autowrap/run.mjs"
node "$ROOT/examples/core/tvl-loading/run.mjs"
node "$ROOT/examples/adoption/seamless/runner.mjs"
