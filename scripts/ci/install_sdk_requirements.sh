#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# traigent-schema 5.0.0 is not on PyPI (newest published is 4.8.0), so this must
# be the exact Git pin, not a version specifier. The pin lives in
# scripts/ci/schema-pin.txt — the single source of truth shared with pr-gate.yml
# and publish.yml — and deliberately NOT in pyproject.toml, because PyPI rejects
# direct URL references in package metadata even inside an optional extra.
#
# There is no `internal_schema` extra any more, but the token is still accepted
# here so existing invocations keep working for a developer running this by hand.
schema_pin_file="$(dirname "$0")/schema-pin.txt"
schema_requirement="${TRAIGENT_SCHEMA_REQUIREMENT:-}"
install_schema="${TRAIGENT_INSTALL_SCHEMA:-}"

for arg in "$@"; do
  case "$arg" in
    *internal_schema*) install_schema="1" ;;
  esac
done

if [ "$install_schema" = "1" ]; then
  if [ -n "$schema_requirement" ]; then
    python -m pip install "$schema_requirement"
  elif [ -f "$schema_pin_file" ]; then
    python -m pip install -r "$schema_pin_file"
  else
    echo "error: $schema_pin_file not found and TRAIGENT_SCHEMA_REQUIREMENT is unset" >&2
    exit 1
  fi
fi

if [ "$#" -gt 0 ]; then
  python -m pip install "$@"
fi
