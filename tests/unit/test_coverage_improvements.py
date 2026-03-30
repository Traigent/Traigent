"""Tests to improve coverage for SonarQube approval.

This module adds tests for specific uncovered code paths in:
- traigent/_version.py
- traigent/telemetry/optuna_metrics.py
- traigent/agents/executor.py
- traigent/storage/local_storage.py
- traigent/utils/persistence.py
- traigent/adapters/execution_adapter.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# =============================================================================
# _version.py coverage tests
# =============================================================================


def test_version_force_override():
    """Test TRAIGENT_FORCE_VERSION environment variable override."""
    with patch.dict(os.environ, {"TRAIGENT_FORCE_VERSION": "99.99.99"}):
        # Re-import to trigger the override path
        import importlib

        import traigent._version as version_module

        importlib.reload(version_module)
        assert version_module.get_version() == "99.99.99"

    # Clean up by reloading without the override
    with patch.dict(os.environ, {}, clear=False):
        if "TRAIGENT_FORCE_VERSION" in os.environ:
            del os.environ["TRAIGENT_FORCE_VERSION"]
        import importlib

        import traigent._version as version_module

        importlib.reload(version_module)


def test_version_package_metadata_path():
    """Test TRAIGENT_USE_PACKAGE_METADATA path when pyproject.toml doesn't exist."""
    import importlib

    import traigent._version as version_module

    # Test the package metadata path by mocking Path.exists to return False
    # and setting the env var
    with patch.dict(os.environ, {"TRAIGENT_USE_PACKAGE_METADATA": "1"}):
        with patch.object(Path, "exists", return_value=False):
            # Also mock importlib.metadata.version to return a version
            with patch("importlib.metadata.version", return_value="1.2.3"):
                importlib.reload(version_module)
                result = version_module.get_version()
                assert result == "1.2.3"

    # Clean up
    if "TRAIGENT_USE_PACKAGE_METADATA" in os.environ:
        del os.environ["TRAIGENT_USE_PACKAGE_METADATA"]
    importlib.reload(version_module)


def test_version_package_metadata_not_found():
    """Test fallback when package metadata is not found."""
    import importlib
    import importlib.metadata

    import traigent._version as version_module

    with patch.dict(os.environ, {"TRAIGENT_USE_PACKAGE_METADATA": "1"}):
        with patch.object(Path, "exists", return_value=False):
            # Mock importlib.metadata.version to raise PackageNotFoundError
            with patch(
                "importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError("traigent"),
            ):
                importlib.reload(version_module)
                result = version_module.get_version()
                # Should fall back to hardcoded version
                assert result == "0.11.0"

    # Clean up
    if "TRAIGENT_USE_PACKAGE_METADATA" in os.environ:
        del os.environ["TRAIGENT_USE_PACKAGE_METADATA"]
    importlib.reload(version_module)


def test_get_version_info():
    """Test get_version_info returns proper structure."""
    from traigent._version import get_version_info

    info = get_version_info()
    assert "version" in info
    assert "major" in info
    assert "minor" in info
    assert "patch" in info
    # Version should be 0.9.0 format
    assert info["major"] == "0"
    assert info["minor"] == "10"
    assert info["patch"] == "0"


# =============================================================================
# optuna_metrics.py coverage tests
# =============================================================================


def test_optuna_emitter_disabled_by_env():
    """Test OptunaMetricsEmitter respects TRAIGENT_DISABLE_TELEMETRY."""
    from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter

    with patch.dict(os.environ, {"TRAIGENT_DISABLE_TELEMETRY": "true"}):
        emitter = OptunaMetricsEmitter()
        # When disabled, emit_trial_update should return empty dict
        result = emitter.emit_trial_update(
            event="trial_started",
            trial_id=1,
            study_name="test_study",
            payload={"test": "data"},
        )
        assert result == {}


def test_optuna_emitter_subscribe_unsubscribe():
    """Test subscribe/unsubscribe functionality."""
    from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter

    emitter = OptunaMetricsEmitter()
    events = []

    def listener(event):
        events.append(event)

    emitter.subscribe(listener)
    emitter.emit_trial_update(event="test", trial_id=1, study_name="s")
    assert len(events) == 1

    emitter.unsubscribe(listener)
    emitter.emit_trial_update(event="test2", trial_id=2, study_name="s")
    # Should still be 1 after unsubscribe
    assert len(events) == 1


def test_optuna_emitter_unsubscribe_nonexistent():
    """Test unsubscribe with listener not in list."""
    from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter

    emitter = OptunaMetricsEmitter()

    def listener(event):
        pass

    # Unsubscribe without subscribing should not raise
    result = emitter.unsubscribe(listener)
    assert result is None  # Method returns None


# =============================================================================
# executor.py coverage tests
# =============================================================================


class CancellingExecutor:
    """Mock executor that triggers CancelledError."""

    def __init__(self):
        self._initialized = True
        self._cleanup_done = False
        self.platform_config = {}

    async def _platform_initialize(self):
        pass

    async def _execute_agent(self, spec, data, config):
        raise asyncio.CancelledError("Task cancelled")

    def _validate_agent_spec(self, spec):
        pass

    def _validate_platform_spec(self, spec):
        pass

    def _merge_configurations(self, base, overrides):
        return base if not overrides else {**base, **(overrides or {})}


@pytest.mark.asyncio
async def test_executor_cancelled_error_propagates():
    """Test that asyncio.CancelledError is re-raised properly."""
    from traigent.agents.executor import AgentExecutor
    from traigent.cloud.models import AgentSpecification

    class CancelTestExecutor(AgentExecutor):
        async def _platform_initialize(self):
            pass

        async def _execute_agent(self, agent_spec, input_data, config):
            raise asyncio.CancelledError()

        def _validate_platform_spec(self, agent_spec):
            pass

        async def _validate_platform_config(self, config):
            return {"errors": [], "warnings": []}

        def _get_platform_capabilities(self):
            return []

    executor = CancelTestExecutor()
    await executor.initialize()

    spec = AgentSpecification(
        id="test",
        name="Test",
        agent_type="task",
        agent_platform="test",
        prompt_template="Test prompt",
        model_parameters={},
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(spec, {"query": "test"})


@pytest.mark.asyncio
async def test_executor_cleanup_error_handling():
    """Test cleanup handles errors during platform cleanup."""
    from traigent.agents.executor import AgentExecutor

    class FailingCleanupExecutor(AgentExecutor):
        async def _platform_initialize(self):
            pass

        async def _execute_agent(self, agent_spec, input_data, config):
            return {"output": "test"}

        def _validate_platform_spec(self, agent_spec):
            pass

        async def _validate_platform_config(self, config):
            return {"errors": [], "warnings": []}

        def _get_platform_capabilities(self):
            return []

        async def _platform_cleanup(self):
            raise RuntimeError("Cleanup failed")

    executor = FailingCleanupExecutor()
    await executor.initialize()

    # Cleanup should raise but still mark as done
    with pytest.raises(RuntimeError, match="Cleanup failed"):
        await executor.cleanup()

    # Should still be marked as cleanup done
    assert executor._cleanup_done is True


@pytest.mark.asyncio
async def test_executor_batch_execute_invalid_concurrency_type():
    """Test batch_execute with non-integer concurrency raises ValueError."""
    from traigent.agents.executor import AgentExecutor
    from traigent.cloud.models import AgentSpecification

    class SimpleExecutor(AgentExecutor):
        async def _platform_initialize(self):
            pass

        async def _execute_agent(self, agent_spec, input_data, config):
            return {"output": "test"}

        def _validate_platform_spec(self, agent_spec):
            pass

        async def _validate_platform_config(self, config):
            return {"errors": [], "warnings": []}

        def _get_platform_capabilities(self):
            return []

    executor = SimpleExecutor()
    spec = AgentSpecification(
        id="test",
        name="Test",
        agent_type="task",
        agent_platform="test",
        prompt_template="Test",
        model_parameters={},
    )

    with pytest.raises(ValueError, match="positive integer"):
        await executor.batch_execute(spec, [{}], max_concurrent="invalid")


# =============================================================================
# local_storage.py coverage tests
# =============================================================================


def test_local_storage_lock_timeout_raises():
    """Test acquire_lock raises TimeoutError when lock is held."""
    from traigent.storage.local_storage import LocalStorageManager
    from traigent.utils.function_identity import sanitize_identifier

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = LocalStorageManager(tmp_dir)
        lock_dir = storage.storage_path / ".locks"
        lock_dir.mkdir(exist_ok=True, parents=True)

        # Create lock file using the sanitized name (matches what acquire_lock creates)
        safe_name = sanitize_identifier("test_lock")
        lock_path = lock_dir / f"{safe_name}.lock"
        lock_path.touch()

        # Try to acquire with very short timeout - should raise TimeoutError
        with pytest.raises(TimeoutError, match="Could not acquire lock"):
            with storage.acquire_lock("test_lock", timeout=0.1):
                pass  # Should never reach here

        # Clean up
        if lock_path.exists():
            lock_path.unlink()


def test_local_storage_delete_session_validates_path():
    """Test delete_session uses path validation."""
    from traigent.storage.local_storage import LocalStorageManager

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = LocalStorageManager(tmp_dir)
        # Non-existent session should return False (not crash)
        assert storage.delete_session("nonexistent_session_id") is False


def test_local_storage_export_session_unsupported_format():
    """Test export_session returns False for unsupported format."""

    from traigent.storage.local_storage import LocalStorageManager

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = LocalStorageManager(tmp_dir)
        session_id = storage.create_session("test_func")

        # Try to export with unsupported format
        result = storage.export_session(session_id, "/tmp/test.csv", format="csv")
        assert result is False


# =============================================================================
# execution_adapter.py coverage tests
# =============================================================================


@pytest.mark.asyncio
async def test_local_adapter_unknown_evaluation_type():
    """Test LocalExecutionAdapter handles unknown evaluation type."""
    from traigent.adapters.execution_adapter import LocalExecutionAdapter

    class SimpleAgent:
        async def execute(self, _input):
            return "result"

    class AgentBuilder:
        def build_agent(self, spec):
            return SimpleAgent()

    adapter = LocalExecutionAdapter(AgentBuilder())

    dataset = {
        "examples": [
            {
                "input": {"text": "test"},
                "expected_output": "result",
                "metadata": {"evaluation_type": "unknown_custom_type"},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="unknown-type-1"
    )

    # Unknown evaluation type should still complete
    assert result["metrics"]["total_examples"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_no_expected_output():
    """Test LocalExecutionAdapter handles examples without expected_output."""
    from traigent.adapters.execution_adapter import LocalExecutionAdapter

    class SimpleAgent:
        async def execute(self, _input):
            return "result"

    class AgentBuilder:
        def build_agent(self, spec):
            return SimpleAgent()

    adapter = LocalExecutionAdapter(AgentBuilder())

    dataset = {
        "examples": [
            {
                "input": {"text": "test"},
                # No expected_output - should still work
                "metadata": {},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="no-expected-1"
    )

    assert result["metrics"]["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_hybrid_adapter_get_execution_mode():
    """Test HybridPlatformAdapter returns hybrid execution mode."""
    from traigent.adapters.execution_adapter import HybridPlatformAdapter

    class AgentBuilder:
        def build_platform_agent(self, spec, client):
            return Mock()

    adapter = HybridPlatformAdapter(
        platform_client=Mock(), agent_builder=AgentBuilder()
    )

    mode = await adapter.get_execution_mode()
    assert mode == "hybrid"


@pytest.mark.asyncio
async def test_local_adapter_get_execution_mode():
    """Test LocalExecutionAdapter returns edge_analytics execution mode."""
    from traigent.adapters.execution_adapter import LocalExecutionAdapter

    adapter = LocalExecutionAdapter(Mock())
    mode = await adapter.get_execution_mode()
    assert mode == "edge_analytics"


# =============================================================================
# persistence.py coverage tests
# =============================================================================


def test_safe_json_value_numpy_types():
    """Test _safe_json_value handles numpy-like types."""
    from traigent.utils.persistence import _safe_json_value

    # Test with a mock numpy-like object
    class MockNumpyScalar:
        def item(self):
            return 42

    result = _safe_json_value(MockNumpyScalar())
    assert result == 42


def test_safe_json_value_set():
    """Test _safe_json_value handles set types."""
    from traigent.utils.persistence import _safe_json_value

    result = _safe_json_value({1, 2, 3})
    assert isinstance(result, list)
    assert set(result) == {1, 2, 3}


def test_safe_json_value_tuple():
    """Test _safe_json_value handles tuple types."""
    from traigent.utils.persistence import _safe_json_value

    result = _safe_json_value((1, 2, 3))
    assert result == [1, 2, 3]


def test_safe_json_value_fallback():
    """Test _safe_json_value fallback to string representation."""
    from traigent.utils.persistence import _safe_json_value

    class CustomObject:
        def __repr__(self):
            return "CustomObject()"

    result = _safe_json_value(CustomObject())
    assert result == "CustomObject()"


def test_persistence_delete_result_not_found(tmp_path):
    """Test delete_result returns False when result doesn't exist."""
    from traigent.utils.persistence import PersistenceManager

    persistence = PersistenceManager(base_dir=tmp_path)
    result = persistence.delete_result("nonexistent_result")
    assert result is False


def test_persistence_load_result_not_found(tmp_path):
    """Test load_result raises FileNotFoundError for missing result."""
    from traigent.utils.persistence import PersistenceManager

    persistence = PersistenceManager(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        persistence.load_result("nonexistent_result")


def test_persistence_load_result_missing_metadata(tmp_path):
    """Test load_result raises ValueError when metadata is missing."""
    from traigent.utils.persistence import PersistenceManager

    persistence = PersistenceManager(base_dir=tmp_path)

    # Create result directory without metadata file
    result_dir = tmp_path / "incomplete_result"
    result_dir.mkdir()

    with pytest.raises(ValueError, match="Metadata file missing"):
        persistence.load_result("incomplete_result")


# =============================================================================
# platforms.py coverage tests
# =============================================================================


@pytest.mark.asyncio
async def test_langchain_task_agent_type():
    """Test LangChain executor with task agent type."""
    import sys
    import types

    from traigent.agents.platforms import LangChainAgentExecutor
    from traigent.cloud.models import AgentSpecification

    # Stub langchain modules
    class StubLLM:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, messages):
            class Msg:
                content = "Task result"

            return Msg()

    mod_lco = types.ModuleType("langchain_openai")
    mod_lco.ChatOpenAI = StubLLM

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    mod_lcc_messages = types.ModuleType("langchain_core.messages")
    mod_lcc_messages.HumanMessage = HumanMessage

    with patch.dict(sys.modules, {"langchain_openai": mod_lco}):
        with patch.dict(sys.modules, {"langchain_core.messages": mod_lcc_messages}):
            with patch("importlib.util.find_spec", return_value=object()):
                executor = LangChainAgentExecutor()
                await executor.initialize()

                spec = AgentSpecification(
                    id="task-test",
                    name="Task Test",
                    agent_type="task",  # Use task type
                    agent_platform="langchain",
                    prompt_template="Do this: {task}",
                    model_parameters={"model": "gpt-4o-mini"},
                )

                result = await executor.execute(spec, {"task": "test task"})
                assert result.output == "Task result"


@pytest.mark.asyncio
async def test_openai_executor_prepare_messages_with_aux():
    """Test OpenAI executor includes aux_user_message when present."""
    from traigent.agents.platforms import OpenAIAgentExecutor
    from traigent.cloud.models import AgentSpecification

    executor = OpenAIAgentExecutor()

    spec = AgentSpecification(
        id="test",
        name="Test",
        agent_type="task",
        agent_platform="openai",
        prompt_template="Test",
        model_parameters={},
    )

    messages = executor._prepare_messages(
        spec, "Main prompt", {"aux_user_message": "Additional context"}
    )

    # Should have user message + aux message
    assert len(messages) == 2
    assert messages[0]["content"] == "Main prompt"
    assert messages[1]["content"] == "Additional context"


@pytest.mark.asyncio
async def test_openai_executor_format_prompt_fallback():
    """Test OpenAI executor prompt formatting fallback path."""
    from traigent.agents.platforms import OpenAIAgentExecutor

    executor = OpenAIAgentExecutor()

    # Template with complex formatting that might fail
    template = "Hello {name}, {greeting}"
    input_data = {"name": "World"}  # Missing 'greeting'

    # Should handle missing keys gracefully
    result = executor._format_prompt(template, input_data)
    assert "World" in result


def test_langchain_extract_llm_kwargs_validation():
    """Test LangChain executor validates LLM kwargs."""
    from traigent.agents.platforms import LangChainAgentExecutor

    executor = LangChainAgentExecutor()

    config = {
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stop_sequences": ["END"],
    }

    kwargs = executor._extract_llm_kwargs(config)

    assert kwargs["top_p"] == 0.9
    assert kwargs["frequency_penalty"] == 0.5
    assert kwargs["presence_penalty"] == 0.3
    assert kwargs["stop_sequences"] == ["END"]


def test_langchain_extract_llm_kwargs_invalid_values():
    """Test LangChain executor handles invalid LLM kwargs by logging warning."""
    from traigent.agents.platforms import LangChainAgentExecutor

    executor = LangChainAgentExecutor()

    # Test with params that are not in the validation list (e.g. stop_sequences as list)
    config = {
        "stop_sequences": ["END", "STOP"],  # Valid list param
    }

    kwargs = executor._extract_llm_kwargs(config)
    assert kwargs["stop_sequences"] == ["END", "STOP"]

    # Test with out of range value - should raise ValidationError
    from traigent.utils.exceptions import ValidationError

    with pytest.raises(ValidationError):
        executor._extract_llm_kwargs({"top_p": 5.0})


@pytest.mark.asyncio
async def test_openai_cleanup():
    """Test OpenAI executor cleanup releases resources."""
    from traigent.agents.platforms import OpenAIAgentExecutor

    executor = OpenAIAgentExecutor()

    # Mock the client
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    executor._openai_client = mock_client
    executor._openai_available = True
    executor._initialized = True

    await executor._platform_cleanup()

    # Client should be closed and cleared
    mock_client.close.assert_called_once()
    assert executor._openai_client is None
    assert executor._openai_available is False
