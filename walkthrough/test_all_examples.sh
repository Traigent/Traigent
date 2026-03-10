#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

node "$ROOT/walkthrough/mock/01_tuning_qa.mjs"
node "$ROOT/walkthrough/mock/02_zero_code_change.mjs"
node "$ROOT/walkthrough/mock/03_parameter_mode.mjs"
node "$ROOT/walkthrough/mock/04_multi_objective.mjs"
node "$ROOT/walkthrough/mock/05_rag_parallel.mjs"
