#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true \
  "$PYTHON_BIN" -m pytest -m smoke tests/smoke/test_observability_phase4_smoke.py -q
