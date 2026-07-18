"""Shared fixtures for economics telemetry emitter tests (WI-B).

The contract fixture validates emitted payloads against the INSTALLED exact
TraigentSchema (the git-pinned economics commit), the same package the emitter
fails closed on at runtime. It does NOT skip when the schema is missing: the
economics surface requires the exact schema, so its absence is a failure, not a
reason to pass silently.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest


class _InstalledContract:
    """Validates payloads and exposes raw schema/endpoint definitions.

    Uses the installed ``traigent_schema`` package: :class:`SchemaValidator` for
    validation (reference resolution included) and the package's on-disk schema
    files for raw definition/endpoint reads (the endpoints file is intentionally
    not loaded by the validator).
    """

    def __init__(self) -> None:
        import traigent_schema
        from traigent_schema.validator import SchemaValidator

        self._validator = SchemaValidator()
        self.version = traigent_schema.__version__
        self.economics_dir = os.path.join(
            os.path.dirname(traigent_schema.__file__), "schemas", "economics"
        )

    def has_request_schema(self) -> bool:
        return (
            "economics_telemetry_ingest_request_schema"
            in self._validator.available_schemas
        )

    def schema(self, schema_name: str) -> dict[str, Any]:
        return self._read(f"{schema_name}.json")

    def endpoints(self) -> dict[str, Any]:
        return self._read("economics_endpoints.json")

    def _read(self, filename: str) -> dict[str, Any]:
        with open(
            os.path.join(self.economics_dir, filename), encoding="utf-8"
        ) as handle:
            return json.load(handle)

    def validate(self, payload: Any, schema_name: str) -> list[str]:
        return self._validator.validate_json(payload, schema_name)


@pytest.fixture(scope="session")
def local_contract() -> _InstalledContract:
    try:
        contract = _InstalledContract()
    except Exception as exc:  # noqa: BLE001 - required dependency, surface as failure
        pytest.fail(
            "the exact economics TraigentSchema must be installed for contract "
            f"drift tests (import/build failed: {exc})"
        )
    if not contract.has_request_schema():
        pytest.fail(
            "installed traigent-schema predates the economics contract "
            f"(version {contract.version}); pin the exact economics commit"
        )
    return contract
