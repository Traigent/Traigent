"""Shared fixtures for the label-service conformance tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "labels"


@pytest.fixture(scope="session")
def vectors() -> dict:
    return json.loads((FIXTURES / "label_vectors_v0.json").read_text("utf-8"))


@pytest.fixture(scope="session")
def rfc() -> dict:
    return json.loads(
        (FIXTURES / "rfc9180_x25519_base_vectors.json").read_text("utf-8")
    )


@pytest.fixture()
def v(vectors: dict) -> dict:
    return copy.deepcopy(vectors)
