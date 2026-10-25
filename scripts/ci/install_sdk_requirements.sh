#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# The pinned traigent-schema build is not on PyPI, so this must
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

# Opt-in (TRAIGENT_INSTALL_LOCKED=1): constrain every package to the versions in
# uv.lock, so CI tests the same dependency set `uv sync` developers get instead
# of floating to the newest release each range allows (issue #2475). Left off by
# default so jobs that must resolve fresh (e.g. install-smoke) keep doing so.
constraint_args=()
if [ "${TRAIGENT_INSTALL_LOCKED:-}" = "1" ]; then
  python -m pip install uv
  constraints_file="$(mktemp)"
  uv export --frozen --no-hashes --no-emit-project --all-extras -o "$constraints_file" >/dev/null
  constraint_args=(-c "$constraints_file")
fi

if [ "$#" -gt 0 ]; then
  python -m pip install ${constraint_args[@]+"${constraint_args[@]}"} "$@"
fi
