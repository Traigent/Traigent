"""APIKeyError built from a provider failure: readable fix line, mock-aware
hint (#2417)."""

from traigent.utils.error_handler import APIKeyError


def test_message_is_not_treated_as_key_name(monkeypatch):
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    err = APIKeyError.from_provider_error(RuntimeError("Missing credentials"))
    assert err.message.startswith("API key error detected")
    assert err.fix is not None
    assert "API key error detected" not in err.fix
    assert "to your .env file" not in err.fix
    assert "TRAIGENT_MOCK_LLM=true" in err.fix


def test_hint_omits_mock_advice_when_mock_already_on(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    err = APIKeyError.from_provider_error(RuntimeError("Missing credentials"))
    assert "TRAIGENT_MOCK_LLM=true" not in err.fix
    assert "already on" in err.fix
    assert "placeholder key" in err.fix


def test_key_name_form_unchanged():
    err = APIKeyError("OPENAI_API_KEY")
    assert err.message == "Missing or invalid API key: OPENAI_API_KEY"
    assert (
        err.fix == "Add OPENAI_API_KEY to your .env file or set as environment variable"
    )
