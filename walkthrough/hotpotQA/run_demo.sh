#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIGENT_OPTUNA_ENABLED="${TRAIGENT_OPTUNA_ENABLED:-1}"
export TRAIGENT_MOCK_MODE="${TRAIGENT_MOCK_MODE:-true}"

python "$SCRIPT_DIR/run_demo.py" "$@"
