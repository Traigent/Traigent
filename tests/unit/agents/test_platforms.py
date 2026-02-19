"""Unit tests for platform-specific agent executors."""

from unittest.mock import Mock, patch

import pytest

from traigent.agents.executor import AgentExecutor
from traigent.agents.platforms import (
    LangChainAgentExecutor,
    OpenAIAgentExecutor,
    PlatformRegistry,
    get_executor_for_platform,
)
from traigent.cloud.models import AgentSpecification


@pytest.mark.asyncio
async def test_langchain_modern_ainvoke(monkeypatch):
    """LangChain executor should use modern LCEL .ainvoke when available."""
    import sys
    import types

    # Stub langchain_openai.ChatOpenAI with ainvoke
    class _StubLLM:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, messages):
            class _Msg:
                def __init__(self, content):
                    self.content = content

            last = messages[-1].content if messages else ""
            return _Msg(f"Echo: {last}")

    mod_lco = types.ModuleType("langchain_openai")
    mod_lco.ChatOpenAI = _StubLLM

    # Stub langchain_core.messages.HumanMessage
    mod_lcc = types.ModuleType("langchain_core")
    mod_lcc_messages = types.ModuleType("langchain_core.messages")

    class _HumanMessage:
        def __init__(self, content):
            self.content = content

    mod_lcc_messages.HumanMessage = _HumanMessage

    monkeypatch.setitem(sys.modules, "langchain_openai", mod_lco)
    monkeypatch.setitem(sys.modules, "langchain_core", mod_lcc)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", mod_lcc_messages)

    # Ensure initialize marks langchain as available
    def _find_spec_side_effect(name):
        if name in ("langchain", "langchain_openai"):
            return object()
        return None

    monkeypatch.setattr("importlib.util.find_spec", _find_spec_side_effect)

    executor = LangChainAgentExecutor()
    await executor.initialize()

    spec = AgentSpecification(
        id="lc-modern",
        name="LC Modern",
        agent_type="conversational",
        agent_platform="langchain",
        prompt_template="Answer the question: {question}",
        model_parameters={"model": "gpt-4o-mini"},
    )

    result = await executor.execute(spec, {"question": "What is AI?"})

    assert result.output.startswith("Echo: Answer the question:")
    assert result.tokens_used is None


def test_langchain_import_runtime_error_propagates(monkeypatch):
    """Non-ImportError failures during LangChain imports should bubble up."""
    import sys
    import types

    executor = LangChainAgentExecutor()

    # Simulate langchain_openai module raising a runtime error when accessed
    mod_lco = types.ModuleType("langchain_openai")

    def _raise(name):
        raise RuntimeError("unexpected import failure")

    mod_lco.__getattr__ = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", mod_lco)

    with pytest.raises(RuntimeError, match="unexpected import failure"):
        executor._import_langchain_components()


def test_safe_prompt_formatting_unknown_placeholder():
    """Unknown placeholders should remain intact with safe formatting."""
    executor = LangChainAgentExecutor()
    template = "Hello {name}, unknown={unknown}"
    out = executor._format_prompt(template, {"name": "Alice"})
    assert out == "Hello Alice, unknown={unknown}"


@pytest.mark.asyncio
async def test_openai_v1_async_client_execution(monkeypatch):
    """OpenAI executor uses AsyncOpenAI v1 client when available."""
    import sys
    import types

    # Stub openai.AsyncOpenAI client
    class _StubResponse:
        def __init__(self):
            class _Choice:
                def __init__(self):
                    class _Msg:
                        def __init__(self):
                            self.content = "stub v1 response"

                    self.message = _Msg()

            class _Usage:
                def __init__(self):
                    self.prompt_tokens = 5
                    self.completion_tokens = 7
                    self.total_tokens = 12

            self.choices = [_Choice()]
            self.usage = _Usage()

    class _StubCompletions:
        async def create(self, **kwargs):
            return _StubResponse()

    class _StubChat:
        def __init__(self):
            self.completions = _StubCompletions()

    class AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _StubChat()

    mod_openai = types.ModuleType("openai")
    mod_openai.AsyncOpenAI = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod_openai)

    # Stub unified auth manager
    class _AuthMgr:
        async def get_auth_headers(self, target="cloud"):
            return {}

        async def authenticate(self, creds):
            class _Res:
                success = True

            return _Res()

    monkeypatch.setattr(
        "traigent.agents.platforms.get_auth_manager", lambda: _AuthMgr()
    )

    executor = OpenAIAgentExecutor(platform_config={"api_key": "testkey"})
    await executor.initialize()

    spec = AgentSpecification(
        id="openai-v1",
        name="OpenAI v1",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Query: {q}",
        model_parameters={"model": "gpt-4o-mini"},
    )

    result = await executor.execute(spec, {"q": "hi"})
    assert result.output == "stub v1 response"
    assert result.tokens_used == 12


@pytest.mark.asyncio
async def test_openai_legacy_chat_completion_execution(monkeypatch):
    """OpenAI executor falls back to legacy ChatCompletion when v1 not available."""
    import sys
    import types

    # Stub legacy openai.ChatCompletion.acreate
    class _LegacyResponse:
        def __init__(self):
            class _Choice:
                def __init__(self):
                    class _Msg:
                        def __init__(self):
                            self.content = "stub legacy response"

                    self.message = _Msg()

            class _Usage:
                def __init__(self):
                    self.prompt_tokens = 3
                    self.completion_tokens = 4
                    self.total_tokens = 7

            self.choices = [_Choice()]
            self.usage = _Usage()

    class _ChatCompletion:
        @classmethod
        async def acreate(cls, **kwargs):
            return _LegacyResponse()

    mod_openai = types.ModuleType("openai")
    mod_openai.ChatCompletion = _ChatCompletion
    # Ensure AsyncOpenAI import fails by not providing it
    monkeypatch.setitem(sys.modules, "openai", mod_openai)

    # Stub unified auth manager
    class _AuthMgr:
        async def get_auth_headers(self, target="cloud"):
            return {}

        async def authenticate(self, creds):
            class _Res:
                success = True

            return _Res()

    monkeypatch.setattr(
        "traigent.agents.platforms.get_auth_manager", lambda: _AuthMgr()
    )

    executor = OpenAIAgentExecutor(platform_config={"api_key": "testkey"})
    await executor.initialize()

    spec = AgentSpecification(
        id="openai-legacy",
        name="OpenAI Legacy",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Query: {q}",
        model_parameters={"model": "gpt-4o-mini"},
    )

    result = await executor.execute(spec, {"q": "hi"})
    assert result.output == "stub legacy response"
    assert result.tokens_used == 7


@pytest.fixture
def langchain_agent_spec():
    """Create LangChain agent specification."""
    return AgentSpecification(
        id="langchain-agent",
        name="LangChain Test Agent",
        agent_type="conversational",
        agent_platform="langchain",
        prompt_template="Answer the question: {question}",
        model_parameters={"model": "o4-mini", "temperature": 0.7, "max_tokens": 100},
        persona="helpful assistant",
        guidelines=["Be concise", "Be accurate"],
    )


@pytest.fixture
def openai_agent_spec():
    """Create OpenAI agent specification."""
    return AgentSpecification(
        id="openai-agent",
        name="OpenAI Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Query: {query}\nResponse:",
        model_parameters={
            "model": "o4-mini",
            "temperature": 0.5,
            "max_tokens": 150,
            "top_p": 0.9,
        },
    )


class TestLangChainAgentExecutor:
    """Test LangChain agent executor."""

    @pytest.mark.asyncio
    async def test_platform_initialize_with_langchain(self):
        """Test initialization when LangChain is available."""
        with patch("traigent.agents.platforms.logger") as mock_logger:
            # Mock importlib.util.find_spec to return a spec (package found)
            with patch("importlib.util.find_spec", return_value=Mock()):
                executor = LangChainAgentExecutor()
                await executor.initialize()

                assert executor._langchain_available is True
                mock_logger.info.assert_called_with(
                    "LangChain initialized successfully"
                )

    @pytest.mark.asyncio
    async def test_platform_initialize_without_langchain(self):
        """Test initialization when LangChain is not available."""
        with patch("traigent.agents.platforms.logger") as mock_logger:
            # Mock importlib.util.find_spec to return None (package not found)
            with patch("importlib.util.find_spec", return_value=None):
                executor = LangChainAgentExecutor()
                await executor.initialize()

                assert executor._langchain_available is False
                mock_logger.warning.assert_called()

    def test_validate_platform_spec(self):
        """Test platform specification validation."""
        executor = LangChainAgentExecutor()

        # Valid spec
        valid_spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="conversational",
            agent_platform="langchain",
            prompt_template="Test",
            model_parameters={"model": "o4-mini"},
        )

        executor._validate_platform_spec(valid_spec)  # Should not raise

        # Wrong platform
        with pytest.raises(ValueError, match="Invalid platform"):
            invalid_spec = AgentSpecification(
                id="test",
                name="Test",
                agent_type="conversational",
                agent_platform="openai",
                prompt_template="Test",
                model_parameters={"model": "o4-mini"},
            )
            executor._validate_platform_spec(invalid_spec)

        # Missing model parameter
        with pytest.raises(ValueError, match="require 'model' parameter"):
            invalid_spec = AgentSpecification(
                id="test",
                name="Test",
                agent_type="conversational",
                agent_platform="langchain",
                prompt_template="Test",
                model_parameters={},
            )
            executor._validate_platform_spec(invalid_spec)

    @pytest.mark.asyncio
    async def test_validate_platform_config(self):
        """Test configuration validation."""
        executor = LangChainAgentExecutor()

        # Valid config
        result = await executor._validate_platform_config(
            {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 100}
        )

        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0

        # Invalid temperature
        result = await executor._validate_platform_config({"temperature": 3.0})

        assert len(result["errors"]) == 1
        assert "exceeds maximum 2.0" in result["errors"][0]

        # Unknown model warning
        result = await executor._validate_platform_config({"model": "unknown-model"})

        assert len(result["warnings"]) == 1
        assert "unknown-model" in result["warnings"][0]

    def test_format_prompt(self):
        """Test prompt formatting."""
        executor = LangChainAgentExecutor()

        template = "Hello {name}, your question is: {question}"
        input_data = {"name": "Alice", "question": "What is AI?"}

        formatted = executor._format_prompt(template, input_data)

        assert formatted == "Hello Alice, your question is: What is AI?"

    def test_get_platform_capabilities(self):
        """Test getting platform capabilities."""
        executor = LangChainAgentExecutor()
        capabilities = executor._get_platform_capabilities()

        assert "conversational" in capabilities
        assert "task" in capabilities
        assert "tools" in capabilities
        assert "async" in capabilities

    @pytest.mark.asyncio
    async def test_execute_without_langchain(self, langchain_agent_spec):
        """Test execution when LangChain is not available."""
        executor = LangChainAgentExecutor()
        # Initialize first to prevent re-initialization
        await executor.initialize()
        # Then set the flag to simulate langchain not being available
        executor._langchain_available = False

        result = await executor.execute(
            langchain_agent_spec, {"question": "What is AI?"}
        )

        assert result.output is None
        assert result.error is not None
        # Check for expected error indicating LangChain is not available
        assert "langchain not available" in result.error.lower()


class TestOpenAIAgentExecutor:
    """Test OpenAI agent executor."""

    @pytest.mark.asyncio
    async def test_platform_initialize_with_openai(self):
        """Test initialization with OpenAI SDK."""
        with patch("traigent.agents.platforms.logger") as mock_logger:
            # Mock openai import
            mock_openai = Mock()
            with patch("builtins.__import__", return_value=mock_openai):
                executor = OpenAIAgentExecutor(platform_config={"api_key": "test-key"})
                await executor.initialize()

                assert executor._openai_available is True
                mock_logger.info.assert_called_with(
                    "OpenAI SDK initialized successfully"
                )

    def test_prepare_messages(self):
        """Test message preparation for OpenAI API."""
        executor = OpenAIAgentExecutor()

        # With persona and guidelines
        agent_spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Test prompt",
            model_parameters={},
            persona="a helpful assistant",
            guidelines=["Be concise", "Be accurate"],
        )

        messages = executor._prepare_messages(agent_spec, "What is AI?", {})

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert "Be concise" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is AI?"

        # Without persona/guidelines
        simple_spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Test",
            model_parameters={},
        )

        messages = executor._prepare_messages(simple_spec, "Query", {})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_calculate_cost(self):
        """Test _calculate_cost delegates to CostCalculator correctly."""
        from traigent.utils.cost_calculator import get_cost_calculator

        executor = OpenAIAgentExecutor()

        # Mock usage object
        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        calc = get_cost_calculator()

        # Test gpt-4o — verify result matches CostCalculator
        cost = executor._calculate_cost("gpt-4o", MockUsage())
        assert isinstance(cost, float)
        assert cost > 0, "Known model should return positive cost"
        input_cost, output_cost = calc._calculate_from_tokens(100, 50, "gpt-4o")
        expected = float(input_cost + output_cost)
        assert (
            abs(cost - expected) < 1e-10
        ), f"_calculate_cost should match CostCalculator: {cost} vs {expected}"

        # Test with model alias (GPT-4o → gpt-4o via MODEL_ALIASES)
        cost_alias = executor._calculate_cost("GPT-4o", MockUsage())
        assert (
            abs(cost_alias - expected) < 1e-10
        ), "Aliased model should produce same cost as canonical name"

    @pytest.mark.asyncio
    async def test_estimate_cost(self, openai_agent_spec):
        """Test cost estimation."""
        executor = OpenAIAgentExecutor()

        estimate = await executor.estimate_cost(
            openai_agent_spec, {"query": "Test query"}
        )

        assert estimate["estimated_cost"] > 0
        assert "estimated_input_cost" in estimate
        assert "estimated_output_cost" in estimate
        assert estimate["confidence"] > 0

    @pytest.mark.asyncio
    async def test_validate_platform_config(self):
        """Test configuration validation."""
        executor = OpenAIAgentExecutor()

        # Valid config
        result = await executor._validate_platform_config(
            {"model": "o4-mini", "temperature": 0.7, "top_p": 0.9}
        )

        assert result["valid"] is True

        # Invalid temperature
        result = await executor._validate_platform_config({"temperature": -1})

        assert len(result["errors"]) == 1
        assert "Temperature" in result["errors"][0]

        # Invalid top_p
        result = await executor._validate_platform_config({"top_p": 1.5})

        assert len(result["errors"]) == 1
        assert "top_p" in result["errors"][0]

    def test_extract_api_kwargs(self):
        """Test extracting API kwargs."""
        executor = OpenAIAgentExecutor()

        config = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "unknown_param": "ignored",
        }

        kwargs = executor._extract_api_kwargs(config)

        assert "top_p" in kwargs
        assert "presence_penalty" in kwargs
        assert "frequency_penalty" in kwargs
        assert "temperature" not in kwargs  # Already handled
        assert "unknown_param" not in kwargs


class TestPlatformRegistry:
    """Test platform registry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the platform registry before each test."""
        # Save original state
        original_executors = PlatformRegistry._executors.copy()
        yield
        # Restore original state
        PlatformRegistry._executors = original_executors

    def test_register_executor(self):
        """Test registering a new executor."""

        # Create a mock executor class
        class CustomExecutor(AgentExecutor):
            async def _platform_initialize(self):
                pass

            async def _execute_agent(self, spec, data, config):
                return {}

            def _validate_platform_spec(self, spec):
                pass

            async def _validate_platform_config(self, config):
                return {"errors": [], "warnings": []}

            def _get_platform_capabilities(self):
                return ["custom"]

        # Register it
        PlatformRegistry.register_executor("custom", CustomExecutor)

        # Get it back
        executor_class = PlatformRegistry.get_executor("custom")
        assert executor_class == CustomExecutor

        # Case insensitive
        executor_class = PlatformRegistry.get_executor("CUSTOM")
        assert executor_class == CustomExecutor

    def test_get_unknown_executor(self):
        """Test getting unknown executor."""
        with pytest.raises(ValueError, match="Unknown platform"):
            PlatformRegistry.get_executor("unknown")

    def test_list_platforms(self):
        """Test listing available platforms."""
        platforms = PlatformRegistry.list_platforms()

        assert "langchain" in platforms
        assert "openai" in platforms

    def test_get_executor_for_platform(self):
        """Test getting executor instance."""
        # LangChain executor
        executor = get_executor_for_platform("langchain")
        assert isinstance(executor, LangChainAgentExecutor)

        # OpenAI executor with config
        executor = get_executor_for_platform(
            "openai", {"api_key": "test-key"}  # pragma: allowlist secret
        )
        assert isinstance(executor, OpenAIAgentExecutor)
        assert (
            executor.platform_config["api_key"] == "test-key"  # pragma: allowlist secret
        )
