#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

private_deps_token="${TRAIGENT_PRIVATE_DEPS_TOKEN:-${TRAIGENT_SCHEMA_TOKEN:-${TRAIGENT_SCHEMAS_PAT:-}}}"
schema_ref="${TRAIGENT_SCHEMA_REF:-7d8539f45e978c5148e4335e8250843f097e3b67}"

if [ -n "$private_deps_token" ]; then
  python -m pip install "traigent-schema @ git+https://x-access-token:${private_deps_token}@github.com/Traigent/TraigentSchema.git@${schema_ref}"
else
  python -m pip install "traigent-schema>=3.2.0"
fi

if [ "$#" -gt 0 ]; then
  python -m pip install "$@"
fi
