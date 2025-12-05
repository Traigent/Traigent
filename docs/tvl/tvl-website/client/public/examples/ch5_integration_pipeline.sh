#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="${ROOT_DIR}/ch1_motivation_experiment.tvl.yml"
MANIFEST="${ROOT_DIR}/ch5_integration_manifest.yaml"

echo "[validate] Checking spec..."
tvl validate "${SPEC}"

echo "[export] Producing deterministic artifacts..."
tvl export --lock "${SPEC}"

echo "[triagent] Dry-running pipeline..."
triagent deploy --dry-run RAGOrientation

echo "[dvl] Running validation suites..."
dvl validate suites/faq-orientation.json

echo "[manifest] Capturing manifest..."
cat "${MANIFEST}"
