"""Tests for the LiteLLM response interceptor."""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

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
    mock_module.pre_call_rules = []
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
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

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
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

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
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert captured[0] is response

    def test_injects_timing(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert hasattr(response, "response_time_ms")
        assert response.response_time_ms > 0

    def test_preserves_usage(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 150

    def test_preserves_return_value(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = litellm_module.completion(model="gpt-4o", messages=[])

        assert response.model == "gpt-4o"
        assert response.choices[0].message.content == "Hello world"

    def test_multiple_calls_captured_in_order(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        litellm_module.completion(model="gpt-4o", messages=[])
        litellm_module.completion(model="claude-3-sonnet", messages=[])
        litellm_module.completion(model="gemini-pro", messages=[])

        captured = get_all_captured_responses()
        assert len(captured) == 3


class TestAsyncCapture:
    """Async litellm.acompletion response capture."""

    def test_captures_response(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = asyncio.run(litellm_module.acompletion(model="gpt-4o", messages=[]))

        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert captured[0] is response

    def test_injects_timing(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        patch_litellm_for_metadata_capture()
        response = asyncio.run(litellm_module.acompletion(model="gpt-4o", messages=[]))

        assert hasattr(response, "response_time_ms")
        assert response.response_time_ms > 0


class TestMockModeLiteLLMShape:
    """LiteLLM mock mode must preserve ModelResponse access patterns."""

    @pytest.mark.asyncio
    async def test_async_mock_mode_supports_choices_message_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm
        from litellm import ModelResponse

        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        underlying = AsyncMock(side_effect=AssertionError("real LiteLLM called"))
        monkeypatch.setattr(litellm, "acompletion", underlying)
        monkeypatch.setattr(
            litellm, "_traigent_patched_acompletion", False, raising=False
        )

        with patch(
            "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
            return_value=True,
        ):
            patch_litellm_for_metadata_capture()

            async def user_code() -> str:
                resp = await litellm.acompletion(
                    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                    messages=[{"role": "user", "content": "hello"}],
                )
                assert isinstance(resp, ModelResponse)
                assert resp.usage.prompt_tokens == 10
                assert resp.usage.completion_tokens == 20
                assert resp.usage.total_tokens == 30
                return resp.choices[0].message.content

            content = await user_code()

        assert content == "This is a mock response for testing."
        underlying.assert_not_called()
        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert isinstance(captured[0], ModelResponse)

    def test_sync_mock_mode_supports_choices_message_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm
        from litellm import ModelResponse

        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        underlying = MagicMock(side_effect=AssertionError("real LiteLLM called"))
        monkeypatch.setattr(litellm, "completion", underlying)
        monkeypatch.setattr(
            litellm, "_traigent_patched_completion", False, raising=False
        )

        with patch(
            "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
            return_value=True,
        ):
            patch_litellm_for_metadata_capture()
            resp = litellm.completion(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert isinstance(resp, ModelResponse)
        assert resp.choices[0].message.content == "This is a mock response for testing."
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 20
        assert resp.usage.total_tokens == 30
        underlying.assert_not_called()
        captured = get_all_captured_responses()
        assert len(captured) == 1
        assert captured[0] is resp


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


class TestFromImportForm:
    """`from litellm import completion` bound before the patch (#2515)."""

    def test_name_imported_before_patch_is_intercepted(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        user_mod = types.ModuleType("user_agent_2515")
        user_mod.completion = (
            litellm_module.completion
        )  # from litellm import completion
        user_mod.acompletion = litellm_module.acompletion
        with patch.dict("sys.modules", {"user_agent_2515": user_mod}):
            patch_litellm_for_metadata_capture()
            assert user_mod.completion is litellm_module.completion
            assert user_mod.acompletion is litellm_module.acompletion
            user_mod.completion(model="gpt-4o", messages=[])
            asyncio.run(user_mod.acompletion(model="gpt-4o", messages=[]))

        assert len(get_all_captured_responses()) == 2

    def test_late_bound_name_is_rebound_on_repeat_patch_call(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        original = litellm_module.completion
        patch_litellm_for_metadata_capture()
        late_mod = types.ModuleType("late_agent_2515")
        late_mod.completion = original
        with patch.dict("sys.modules", {"late_agent_2515": late_mod}):
            patch_litellm_for_metadata_capture()
            assert late_mod.completion is litellm_module.completion

    def test_from_import_form_is_mocked_without_provider_call(self, litellm_module):
        from traigent.utils.litellm_interceptor import (
            patch_litellm_for_metadata_capture,
        )

        provider = litellm_module.completion
        user_mod = types.ModuleType("user_agent_2515_mock")
        user_mod.completion = provider
        with (
            patch.dict("sys.modules", {"user_agent_2515_mock": user_mod}),
            patch(
                "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
                return_value=True,
            ),
        ):
            patch_litellm_for_metadata_capture()
            user_mod.completion(model="gpt-4o", messages=[])

        provider.assert_not_called()


class TestMockModeBoundaryGuard:
    """Sol r4 finding 1 (#2515): stale references must not reach the provider."""

    @pytest.fixture()
    def real_litellm(self):
        litellm = pytest.importorskip("litellm")
        from traigent.utils import litellm_interceptor as li

        saved = (
            litellm.completion,
            litellm.acompletion,
            list(litellm.pre_call_rules),
            getattr(litellm, "_traigent_patched_completion", False),
            getattr(litellm, "_traigent_patched_acompletion", False),
        )
        # Capture the unwrapped references BEFORE patching, as an agent
        # factory that imported them earlier would.
        orig_completion = getattr(litellm.completion, "_traigent_original", None) or (
            litellm.completion
        )
        orig_acompletion = getattr(litellm.acompletion, "_traigent_original", None) or (
            litellm.acompletion
        )
        litellm.completion, litellm.acompletion = orig_completion, orig_acompletion
        litellm._traigent_patched_completion = False
        litellm._traigent_patched_acompletion = False
        litellm.pre_call_rules[:] = []
        try:
            yield litellm, li, orig_completion, orig_acompletion
        finally:
            litellm.completion, litellm.acompletion = saved[0], saved[1]
            litellm.pre_call_rules[:] = saved[2]
            litellm._traigent_patched_completion = saved[3]
            litellm._traigent_patched_acompletion = saved[4]

    @pytest.fixture()
    def mock_enabled(self):
        with patch(
            "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
            return_value=True,
        ):
            yield

    @pytest.fixture()
    def transport_spy(self, monkeypatch):
        import socket

        attempts: list[object] = []

        def _spy(self, address, *a, **kw):
            attempts.append(address)
            raise OSError("transport spy")

        monkeypatch.setattr(socket.socket, "connect", _spy)
        return attempts

    def test_closure_and_default_arg_captured_completion_blocked(
        self, real_litellm, transport_spy, mock_enabled
    ):
        litellm, li, orig_completion, orig_acompletion = real_litellm
        captured = orig_completion  # imported earlier, closed over below

        def make_agent():
            def agent(prompt):
                return captured(
                    model="gpt-4o-mini",
                    api_key="sk-test",
                    messages=[{"role": "user", "content": prompt}],
                )

            return agent

        def agent_default(prompt, fn=orig_completion):
            return fn(
                model="gpt-4o-mini",
                api_key="sk-test",
                messages=[{"role": "user", "content": prompt}],
            )

        async def async_agent(prompt):
            return await orig_acompletion(
                model="gpt-4o-mini",
                api_key="sk-test",
                messages=[{"role": "user", "content": prompt}],
            )

        li.patch_litellm_for_metadata_capture()

        with pytest.raises(li.MockModeBypassError):
            make_agent()("hello")
        with pytest.raises(li.MockModeBypassError):
            agent_default("hello")
        with pytest.raises(li.MockModeBypassError):
            asyncio.run(async_agent("hello"))
        assert transport_spy == []

    def test_guard_is_inert_when_mock_disabled_and_idempotent(
        self, real_litellm, monkeypatch
    ):
        litellm, li, *_ = real_litellm
        li.patch_litellm_for_metadata_capture()
        li.patch_litellm_for_metadata_capture()
        assert litellm.pre_call_rules.count(li._mock_mode_pre_call_guard) == 1
        assert li._mock_mode_pre_call_guard("text") is True
