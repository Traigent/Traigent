"""OpenAI models retired 2026-10-23 are rejected; still-live ids stay valid."""

from __future__ import annotations

import pytest

from traigent.config.retired_models import is_retired_model
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery

RETIRED_1023 = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "o1",
    "o1-pro",
    "o3-mini",
    "o4-mini",
    "gpt-4.1-nano",
]

STILL_LIVE = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]


@pytest.mark.parametrize("model_id", RETIRED_1023)
def test_retired_1023_is_retired(model_id: str) -> None:
    assert is_retired_model(model_id)


@pytest.mark.parametrize("model_id", RETIRED_1023)
def test_retired_1023_rejected_by_validation(model_id: str) -> None:
    assert OpenAIDiscovery().is_valid_model(model_id) is False


@pytest.mark.parametrize("model_id", STILL_LIVE)
def test_still_live_not_retired_and_valid(model_id: str) -> None:
    assert not is_retired_model(model_id)
    assert OpenAIDiscovery().is_valid_model(model_id) is True


@pytest.mark.parametrize("model_id", ["openai/gpt-4", "openai/o4-mini"])
def test_provider_prefixed_retired(model_id: str) -> None:
    assert is_retired_model(model_id)


def test_gpt4_entry_does_not_catch_gpt4o_or_gpt41_via_prefix() -> None:
    for model_id in ("openai/gpt-4o", "openai/gpt-4.1", "gpt-4o-mini-2024-07-18"):
        assert not is_retired_model(model_id)


def test_runtime_fallback_list_has_no_retired_ids() -> None:
    assert not [m for m in OpenAIDiscovery().list_models() if is_retired_model(m)]
