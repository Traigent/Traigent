"""Legend test support: schema validation against the Schema repo's contract files.

The contract JSON files are snapshotted in ``fixtures/`` (Schema commit 864aefe,
design note ``docs/identity/pillar-legend-v0.md``). Point
``TRAIGENT_LEGEND_SCHEMA_DIR`` at a live ``traigent_schema/schemas/legend`` to
validate against it instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft7Validator
from referencing import Registry, Resource

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.identity.keys import (
    ContentIdentityKeys,
    clear_content_identity_keys,
    set_content_identity_keys,
)

FIXTURES = Path(__file__).parent / "fixtures"


def schema_dir() -> Path:
    return Path(os.environ.get("TRAIGENT_LEGEND_SCHEMA_DIR") or FIXTURES)


class LegendSchemas:
    def __init__(self) -> None:
        self.docs = {
            p.name: json.loads(p.read_text()) for p in schema_dir().glob("*.json")
        }
        self.registry: Registry = Registry()
        for doc in self.docs.values():
            self.registry = self.registry.with_resource(
                doc["$id"], Resource.from_contents(doc)
            )

    def validate(self, file: str, definition: str | None, instance: Any) -> None:
        base = self.docs[file]
        schema = (
            {"$ref": f"{base['$id']}#/definitions/{definition}"} if definition else base
        )
        Draft7Validator(schema, registry=self.registry).validate(instance)

    def is_valid(self, file: str, definition: str | None, instance: Any) -> bool:
        try:
            self.validate(file, definition, instance)
        except Exception:
            return False
        return True


@pytest.fixture(scope="session")
def schemas() -> LegendSchemas:
    return LegendSchemas()


@pytest.fixture
def keys():
    grant = ContentIdentityKeys(
        tenant_id="tenant-a",
        kid="k" + "ab" * 8,
        example_id_key=b"\x01" * 32,
        example_version_key=b"\x02" * 32,
    )
    set_content_identity_keys(grant)
    yield grant
    clear_content_identity_keys()


def make_dataset(rows: list[tuple[Any, Any]]) -> Dataset:
    return Dataset(
        examples=[EvaluationExample(input_data=i, expected_output=o) for i, o in rows]
    )
