#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# traigent-schema 5.0.0 is not on PyPI yet (see pr-gate.yml), so the fallback
# must be the exact Git pin, not a version specifier. This branch is currently
# unreached in CI: no workflow passes an "internal_schema" extras token or sets
# TRAIGENT_INSTALL_SCHEMA (verified 2026-07-24), so it only fires for a
# developer running this script by hand.
schema_requirement="${TRAIGENT_SCHEMA_REQUIREMENT:-traigent-schema @ git+https://github.com/Traigent/TraigentSchema.git@01f3e2a2bbc1ca7d1b1cc8dde94f82d73dbe822a}"
install_schema="${TRAIGENT_INSTALL_SCHEMA:-}"

for arg in "$@"; do
  case "$arg" in
    *internal_schema*) install_schema="1" ;;
  esac
done

if [ "$install_schema" = "1" ]; then
  python -m pip install "$schema_requirement"
fi

if [ "$#" -gt 0 ]; then
  python -m pip install "$@"
fi
