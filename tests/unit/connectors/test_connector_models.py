from dataclasses import FrozenInstanceError

import pytest

from traigent.connectors.models import (
    ConnectionRef,
    DatasetItem,
    ExternalRef,
    Observation,
    Score,
)


def test_content_models_are_frozen():
    values = [
        Observation("q", "a"),
        Score("quality", 1.0),
        DatasetItem("q", "a"),
        ExternalRef("vendor", "x", "tk_0123456789abcdefghjkmnpqrs", "opaque-id"),
        ConnectionRef("langfuse"),
    ]
    for value in values:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            value.__dict__[next(iter(value.__dict__))] = "canary"


def test_locator_serialization_rejects_non_code_strings():
    from traigent.connectors.models import VerifiedCodeFact, serialize_locator

    with pytest.raises(ValueError):
        serialize_locator("https://example.com/user/alice")

    forged = object.__new__(VerifiedCodeFact)
    object.__setattr__(forged, "function_ref", "agent:run")
    object.__setattr__(forged, "file_path", "agent.py")
    with pytest.raises(ValueError):
        serialize_locator(forged)


def test_verified_code_fact_rejects_missing_callable_source():
    from traigent.connectors.models import VerifiedCodeFact

    namespace: dict[str, object] = {}
    exec(
        compile("def agent(): pass", "datasets/missing_canary.json", "exec"), namespace
    )

    with pytest.raises(ValueError, match="callable source must exist"):
        VerifiedCodeFact.from_callable(namespace["agent"], repository_root=".")


def test_content_model_reprs_do_not_expose_canaries():
    canary = "CUSTOMER_CONTENT_CANARY"
    values = [
        Observation(canary, canary, {canary: canary}),
        Score(canary, canary, {canary: canary}),
        DatasetItem(canary, canary, (Score(canary, canary),)),
    ]

    assert all(canary not in repr(value) for value in values)


def test_external_ref_requires_a_validated_connection_token():
    with pytest.raises(ValueError, match="validated connection token"):
        ExternalRef("vendor", "x", "CUSTOMER_CONTENT_CANARY", "opaque-id")
