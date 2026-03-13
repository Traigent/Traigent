#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: walkthrough/run_with_env.sh <script>"
  exit 1
fi

node "$ROOT/$1"
