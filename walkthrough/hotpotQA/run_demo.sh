#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optuna-backed optimizers are enabled by default (no env var needed).
export TRAIGENT_MOCK_LLM="${TRAIGENT_MOCK_LLM:-true}"

python "$SCRIPT_DIR/run_demo.py" "$@"
