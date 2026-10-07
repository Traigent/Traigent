from dataclasses import FrozenInstanceError

import pytest

from traigent.connectors.models import ConnectionRef, DatasetItem, ExternalRef, Observation, Score


def test_content_models_are_frozen():
    values = [Observation("q", "a"), Score("quality", 1.0), DatasetItem("q", "a"), ExternalRef("vendor", "x"), ConnectionRef("langfuse")]
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
