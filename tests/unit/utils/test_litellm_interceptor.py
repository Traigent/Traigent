"""Tests for the LiteLLM response interceptor."""

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.langchain_interceptor import (
    clear_captured_responses,
    get_all_captured_responses,
)


def _make_mock_response(model="gpt-4o"):
    """Create a mock LiteLLM ModelResponse with OpenAI-compatible usage."""
    response = MagicMock()
    response.model = model
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello world"
    return response


def _make_mock_litellm_module():
    """Create a fake litellm module with completion/acompletion."""
    mock_module = types.ModuleType("litellm")
    mock_module.completion = MagicMock(
        side_effect=lambda *a, **kw: _make_mock_response(kw.get("model", "gpt-4o"))
    )

    async def _async_completion(*args, **kwargs):
        return _make_mock_response(kwargs.get("model", "gpt-4o"))

    mock_module.acompletion = _async_completion
    mock_module._traigent_patched_completion = False
    mock_module._traigent_patched_acompletion = False
    return mock_module


@pytest.fixture(autouse=True)
def _clear_responses():
    """Clear captured responses before and after each test."""
    clear_captured_responses()
    yield
    clear_captured_responses()


@pytest.fixture()
def litellm_module():
    """Provide a fresh mock litellm module, isolated from sys.modules."""
    mock_mod = _make_mock_litellm_module()
    with patch.dict("sys.modules", {"litellm": mock_mod}):
        yield mock_mod


@pytest.fixture(autouse=True)
def _disable_mock_adapter():
    """Disable MockAdapter so the real (mock) completion runs."""
    with patch(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        return_value=False,
    ):
        yield


class TestPatchLiteLLM:
    """Patching litellm.completion and litellm.acompletion."""

    def test_patch_returns_true(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        result = patch_litellm_for_metadata_capture()
        assert result is True

    def test_patch_returns_false_when_litellm_missing(self):
        with patch.dict("sys.modules", {"litellm": None}):
            from traigent.utils.litellm_interceptor import (
                patch_litellm_for_metadata_capture,
            )

            result = patch_litellm_for_metadata_capture()
        assert result is False

    def test_patch_is_idempotent(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        # Second call should skip (already patched)
        result = patch_litellm_for_metadata_capture()
        assert result is False  # no new patching done

        litellm_module.completion(model="gpt-4o", messages=[])
        captured = get_all_captured_responses()
        assert len(captured) == 1  # not double-captured


class TestSyncCapture:
    """Sync litellm.completion response capture."""

    def test_captures_response(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert captured[0] is response

    def test_injects_timing(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert hasattr(response, "response_time_ms")
        assert response.response_time_ms > 0

    def test_preserves_usage(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 150

    def test_preserves_return_value(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert response.model == "gpt-4o"
        assert response.choices[0].message.content == "Hello world"

    def test_multiple_calls_captured_in_order(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        litellm_module.completion(model="gpt-4o", messages=[])
        litellm_module.completion(model="claude-3-sonnet", messages=[])
        litellm_module.completion(model="gemini-pro", messages=[])

        captured = get_all_captured_responses()
        assert len(captured) == 3


class TestAsyncCapture:
    """Async litellm.acompletion response capture."""

    def test_captures_response(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = asyncio.run(litellm_module.acompletion(model="gpt-4o", messages=[]))

        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert captured[0] is response

    def test_injects_timing(self, litellm_module):
        from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

        patch_litellm_for_metadata_capture()
        response = asyncio.run(litellm_module.acompletion(model="gpt-4o", messages=[]))

        assert hasattr(response, "response_time_ms")
        assert response.response_time_ms > 0


class TestLiteLLMPlugin:
    """LiteLLM plugin parameter mappings and overrides."""

    def setup_method(self):
        from traigent.integrations.llms.litellm_plugin import LiteLLMPlugin

        self.plugin = LiteLLMPlugin()

    def test_metadata_name(self):
        metadata = self.plugin._get_metadata()
        assert metadata.name == "litellm"
        assert "litellm" in metadata.supported_packages

    def test_framework_is_litellm(self):
        from traigent.integrations.utils import Framework

        assert self.plugin.FRAMEWORK == Framework.LITELLM

    def test_extra_mappings_include_litellm_params(self):
        mappings = self.plugin._get_extra_mappings()
        assert "api_base" in mappings
        assert "custom_llm_provider" in mappings
        assert "num_retries" in mappings

    def test_model_validation_accepts_provider_prefixed(self):
        errors = self.plugin._validate_model("model", "openrouter/openai/gpt-4o")
        assert errors == []

    def test_model_validation_accepts_direct(self):
        errors = self.plugin._validate_model("model", "gpt-4o")
        assert errors == []

    def test_model_validation_rejects_empty(self):
        errors = self.plugin._validate_model("model", "")
        assert len(errors) > 0

    def test_model_validation_rejects_non_string(self):
        errors = self.plugin._validate_model("model", 42)
        assert len(errors) > 0

    def test_apply_overrides_passes_through(self):
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        config = {"model": "gpt-4o", "temperature": 0.5}
        result = self.plugin.apply_overrides(kwargs, config)
        assert result["temperature"] == 0.5

    def test_apply_overrides_with_none_config(self):
        kwargs = {"model": "gpt-4o", "messages": []}
        result = self.plugin.apply_overrides(kwargs, None)
        assert result == kwargs
        assert result is not kwargs

    def test_target_methods_include_completion(self):
        methods = self.plugin.get_target_methods()
        assert "litellm" in methods
        assert "completion" in methods["litellm"]
        assert "acompletion" in methods["litellm"]
