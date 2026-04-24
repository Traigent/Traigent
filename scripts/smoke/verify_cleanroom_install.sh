#!/usr/bin/env bash
# Clean-room install smoke for the Python SDK.
# Builds a wheel (if dist/ missing), installs it into a throwaway venv,
# imports traigent, and runs `traigent --help`. Exits non-zero on any failure.
#
# Usage: scripts/smoke/verify_cleanroom_install.sh
# Assumes: invoked from the Traigent/ repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if ! ls dist/*.whl >/dev/null 2>&1; then
  echo "[cleanroom] no wheel in dist/; building..."
  if command -v uv >/dev/null 2>&1; then
    uv build
  else
    python3 -m pip install --quiet build
    python3 -m build
  fi
fi

WHL="$(ls -1 dist/*.whl | head -n1)"
if [ -z "$WHL" ]; then
  echo "[cleanroom] FAIL: no wheel produced" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

if command -v uv >/dev/null 2>&1; then
  uv venv --clear --python 3.12 .venv
  . .venv/bin/activate
  uv pip install --quiet "$REPO_ROOT/$WHL"
else
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet "$REPO_ROOT/$WHL"
fi

python3 -c "import traigent; print('import_ok version=', getattr(traigent, '__version__', 'unknown'))"

if command -v traigent >/dev/null 2>&1; then
  traigent --help | head -n 20
  echo "[cleanroom] cli_ok"
else
  echo "[cleanroom] WARN: traigent CLI not on PATH after install" >&2
fi

echo "[cleanroom] wheel=$WHL"
echo "[cleanroom] OK"
