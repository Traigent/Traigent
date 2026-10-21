"""Guard: shipped examples, defaults and probes must not name retired models.

Regression for #2651 (Anthropic key-validation probe).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from traigent.config.retired_models import (
    is_retired_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_anthropic_key_probe_uses_non_retired_model(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from traigent.providers import ProviderValidator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
    sent: dict[str, object] = {}

    def fake_count_tokens(self, **kwargs):  # noqa: ANN001, ANN003
        sent.update(kwargs)
        return mock.Mock()

    with mock.patch(
        "anthropic.resources.messages.Messages.count_tokens", fake_count_tokens
    ):
        status = ProviderValidator()._validate_anthropic()
    assert status.valid, status
    assert not is_retired_model(str(sent["model"])), sent["model"]


def test_anthropic_probe_model_is_known_to_validator() -> None:
    from traigent.providers.validation import _KNOWN_MODELS

    assert "gemini-3.8-flash" in _KNOWN_MODELS["google"]
    assert "claude-haiku-4-5-20251001" in _KNOWN_MODELS["anthropic"]
