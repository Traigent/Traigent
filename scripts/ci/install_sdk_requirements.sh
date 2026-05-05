#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

private_deps_token="${TRAIGENT_PRIVATE_DEPS_TOKEN:-${TRAIGENT_SCHEMA_TOKEN:-${TRAIGENT_SCHEMAS_PAT:-}}}"
schema_ref="${TRAIGENT_SCHEMA_REF:-7d8539f45e978c5148e4335e8250843f097e3b67}"
install_schema="${TRAIGENT_INSTALL_SCHEMA:-}"

for arg in "$@"; do
  case "$arg" in
    *internal_schema*) install_schema="1" ;;
  esac
done

if [ -n "$private_deps_token" ]; then
  git config --global url."https://x-access-token:${private_deps_token}@github.com/Traigent/".insteadOf "https://github.com/Traigent/"
fi

if [ "$install_schema" = "1" ]; then
  if [ -n "$private_deps_token" ]; then
    python -m pip install "traigent-schema @ git+https://x-access-token:${private_deps_token}@github.com/Traigent/TraigentSchema.git@${schema_ref}"
  else
    python -m pip install "traigent-schema>=3.2.0"
  fi
fi

if [ "$#" -gt 0 ]; then
  python -m pip install "$@"
fi
