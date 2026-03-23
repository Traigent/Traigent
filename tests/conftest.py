"""Shared test fixtures and configuration for Traigent test suite.

This module provides:
- Common test datasets and examples
- Mock LLM responses and utilities
- Shared evaluators and optimizers
- Test environment setup and configuration
- Performance testing utilities
"""

# CRITICAL: Set LITELLM_LOCAL_MODEL_COST_MAP BEFORE any imports that might
# trigger litellm loading. This prevents network calls to api.litellm.ai
# during tests and uses bundled pricing data instead.
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import asyncio
import functools
import json
import os
import sys
import tempfile
import weakref
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from traigent import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.logging import get_logger

# Import rate limit fixtures for all tests
pytest_plugins = ["tests.fixtures.rate_limit_fixtures"]

# Increase recursion limit to handle complex test scenarios
sys.setrecursionlimit(2000)


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    """Expose common Traigent symbols to package doctests."""
    import traigent
    from traigent import (
        Choices,
        ConfigSpace,
        IntRange,
        LogRange,
        Range,
        TraigentConfig,
        configure,
        get_config,
        get_current_config,
        get_optimization_insights,
        get_trial_config,
        get_version_info,
        initialize,
        optimize,
        override_config,
        set_strategy,
    )
    from traigent.api.constraints import (
        AndCondition,
        BoolExpr,
        Condition,
        Constraint,
        NotCondition,
        OrCondition,
        WhenBuilder,
        implies,
        require,
        when,
    )

    doctest_namespace["traigent"] = traigent
    doctest_namespace["Range"] = Range
    doctest_namespace["IntRange"] = IntRange
    doctest_namespace["LogRange"] = LogRange
    doctest_namespace["Choices"] = Choices
    doctest_namespace["ConfigSpace"] = ConfigSpace
    doctest_namespace["TraigentConfig"] = TraigentConfig

    doctest_namespace["Constraint"] = Constraint
    doctest_namespace["Condition"] = Condition
    doctest_namespace["AndCondition"] = AndCondition
    doctest_namespace["OrCondition"] = OrCondition
    doctest_namespace["NotCondition"] = NotCondition
    doctest_namespace["BoolExpr"] = BoolExpr
    doctest_namespace["WhenBuilder"] = WhenBuilder
    doctest_namespace["when"] = when
    doctest_namespace["require"] = require
    doctest_namespace["implies"] = implies

    doctest_namespace["optimize"] = optimize
    doctest_namespace["configure"] = configure
    doctest_namespace["initialize"] = initialize
    doctest_namespace["get_config"] = get_config
    doctest_namespace["get_current_config"] = get_current_config
    doctest_namespace["get_trial_config"] = get_trial_config
    doctest_namespace["get_version_info"] = get_version_info
    doctest_namespace["get_optimization_insights"] = get_optimization_insights
    doctest_namespace["override_config"] = override_config
    doctest_namespace["set_strategy"] = set_strategy


def pytest_addoption(parser):
    """Add shared test suite options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless they are explicitly requested."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Set JWT validation to development mode for all tests
@pytest.fixture(autouse=True)
def jwt_development_mode(monkeypatch):
    """Set JWT validator to development mode for all tests."""
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "development")
    # Ensure mock LLM mode to avoid real LLM API calls during tests
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    # Ensure offline mode to avoid real backend calls during tests
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("MOCK_MODE", "true")


# Global State Reset Fixture - Critical for test isolation
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset all global state before and after each test to prevent state pollution.

    This fixture addresses the root cause of test flakiness where global singletons
    like _API_KEY_MANAGER and _GLOBAL_CONFIG maintain state across tests, causing
    tests to pass in isolation but fail when run together.

    Also resets the context-based configuration (ContextVars) to prevent leakage
    between tests.
    """
    # Import here to ensure modules are loaded
    from traigent.api.functions import _GLOBAL_CONFIG
    from traigent.config.api_keys import _API_KEY_MANAGER
    from traigent.config.context import (
        config_context,
        config_space_context,
        trial_context,
    )
    from traigent.config.types import TraigentConfig

    # Store original state for restoration (good practice)
    _API_KEY_MANAGER._keys.copy()
    _GLOBAL_CONFIG.copy()

    # Reset to clean state BEFORE test
    _API_KEY_MANAGER._keys.clear()
    _API_KEY_MANAGER._warned = False
    _GLOBAL_CONFIG.clear()
    _GLOBAL_CONFIG.update(
        {
            "default_storage_backend": "edge_analytics",
            "parallel_workers": 1,
            "cache_policy": "memory",
            "logging_level": "INFO",
            "api_keys": {},
        }
    )

    # Reset context-based configuration (ContextVars)
    # Use a fresh TraigentConfig to ensure clean state
    config_token = config_context.set(TraigentConfig())
    space_token = config_space_context.set(None)
    trial_token = trial_context.set(None)

    # Clear any cached warning filters
    import warnings

    warnings.resetwarnings()

    yield  # Run the test

    # Clean up AFTER test to ensure next test gets clean state
    _API_KEY_MANAGER._keys.clear()
    _API_KEY_MANAGER._warned = False
    _GLOBAL_CONFIG.clear()
    _GLOBAL_CONFIG.update(
        {
            "default_storage_backend": "edge_analytics",
            "parallel_workers": 1,
            "cache_policy": "memory",
            "logging_level": "INFO",
            "api_keys": {},
        }
    )

    # Reset context vars after test
    config_context.reset(config_token)
    config_space_context.reset(space_token)
    trial_context.reset(trial_token)


# Cloud session cleanup for integration tests that hit the real backend
@pytest.fixture(autouse=True)
def cloud_session_cleanup(monkeypatch):
    """Automatically delete cloud sessions created during tests when enabled.

    When `TRAIGENT_RUN_CLOUD_TESTS` is truthy, this fixture tracks every session
    created through `BackendIntegratedClient` or `TraigentCloudClient` and
    deletes it via the new backend cleanup endpoint after the test completes.
    """

    flag = os.getenv("TRAIGENT_RUN_CLOUD_TESTS", "").lower()
    if flag not in {"1", "true", "yes", "on"}:
        yield
        return

    tracker: dict[str, list[weakref.ReferenceType]] = defaultdict(list)
    logger = get_logger("tests.cloud_cleanup")

    def register_session(client, session_id):
        if not session_id:
            return
        tracker[session_id].append(weakref.ref(client))

    try:
        from traigent.cloud.backend_client import BackendIntegratedClient
    except ImportError:
        BackendIntegratedClient = None  # type: ignore

    if BackendIntegratedClient is not None:
        if hasattr(BackendIntegratedClient, "create_session"):
            original_create_session = BackendIntegratedClient.create_session

            @functools.wraps(original_create_session)
            def wrapped_create_session(self, *args, **kwargs):
                session_id = original_create_session(self, *args, **kwargs)
                register_session(self, session_id)
                return session_id

            monkeypatch.setattr(
                BackendIntegratedClient, "create_session", wrapped_create_session
            )

        if hasattr(BackendIntegratedClient, "create_hybrid_session"):
            original_create_hybrid = BackendIntegratedClient.create_hybrid_session

            @functools.wraps(original_create_hybrid)
            async def wrapped_create_hybrid(self, *args, **kwargs):
                result = await original_create_hybrid(self, *args, **kwargs)
                if isinstance(result, tuple) and result:
                    register_session(self, result[0])
                return result

            monkeypatch.setattr(
                BackendIntegratedClient,
                "create_hybrid_session",
                wrapped_create_hybrid,
            )

        if hasattr(BackendIntegratedClient, "_create_traigent_session_via_api"):
            original_create_via_api = (
                BackendIntegratedClient._create_traigent_session_via_api
            )

            @functools.wraps(original_create_via_api)
            async def wrapped_create_via_api(self, *args, **kwargs):
                response = await original_create_via_api(self, *args, **kwargs)
                session_id = None
                if isinstance(response, dict):
                    session_id = response.get("session_id")
                elif isinstance(response, tuple) and response:
                    session_id = response[0]
                register_session(self, session_id)
                return response

            monkeypatch.setattr(
                BackendIntegratedClient,
                "_create_traigent_session_via_api",
                wrapped_create_via_api,
            )

    try:
        from traigent.cloud.client import TraigentCloudClient
    except ImportError:
        TraigentCloudClient = None  # type: ignore

    if TraigentCloudClient is not None and hasattr(
        TraigentCloudClient, "create_optimization_session"
    ):
        original_create_opt_session = TraigentCloudClient.create_optimization_session

        @functools.wraps(original_create_opt_session)
        async def wrapped_create_opt_session(self, *args, **kwargs):
            response = await original_create_opt_session(self, *args, **kwargs)
            session_id = getattr(response, "session_id", None)
            if session_id is None and isinstance(response, dict):
                session_id = response.get("session_id")
            register_session(self, session_id)
            return response

        monkeypatch.setattr(
            TraigentCloudClient,
            "create_optimization_session",
            wrapped_create_opt_session,
        )

    yield

    async def _cleanup_sessions():
        for session_id, refs in tracker.items():
            deleted = False
            for ref in refs:
                client = ref()
                if client is None:
                    continue

                delete_method = getattr(client, "delete_session", None)
                if delete_method is None:
                    continue

                try:
                    result = await delete_method(session_id, cascade=True)
                    deleted = bool(result) or result is None
                    if deleted:
                        break
                except TypeError:
                    # Bound method may not accept cascade keyword (legacy fallback)
                    try:
                        result = await delete_method(session_id)
                        deleted = bool(result) or result is None
                        if deleted:
                            break
                    except Exception as exc:
                        logger.warning(
                            "Cleanup call failed for session %s: %s", session_id, exc
                        )
                except Exception as exc:
                    logger.warning(
                        "Cleanup call failed for session %s: %s", session_id, exc
                    )

            if not deleted:
                logger.warning(
                    "Unable to confirm deletion for session %s; manual cleanup may be required",
                    session_id,
                )

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_cleanup_sessions())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# Mock Response Classes


class MockLLMResponse:
    """Standard mock LLM response with comprehensive metadata."""

    def __init__(
        self,
        text: str,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cost: float = 0.003,
        response_time: float = 1500,
        provider: str = "openai",
    ):
        self.text = text
        self.provider = provider

        # OpenAI-style usage
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                # Also support Anthropic naming
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )()

        # Response timing
        self.response_time_ms = response_time

        # Cost metadata
        self.cost_metadata = {
            "input_cost": cost * 0.4,
            "output_cost": cost * 0.6,
            "total_cost": cost,
        }

        # Additional metadata for testing
        self.metadata = {
            "tokens": {"input": input_tokens, "output": output_tokens},
            "cost": self.cost_metadata,
            "response_time_ms": response_time,
            "provider": provider,
        }

    def __str__(self):
        return self.text


class MockCachedResponse(MockLLMResponse):
    """Mock LLM response that simulates cached behavior."""

    def __init__(self, text: str, cached: bool = True, **kwargs):
        # Cached responses should be faster
        if cached and "response_time" not in kwargs:
            kwargs["response_time"] = 50  # Fast cache response

        super().__init__(text, **kwargs)
        self.cached = cached

        # Update metadata to indicate cache status
        self.metadata["cached"] = cached


class MockErrorResponse:
    """Mock response that simulates LLM errors."""

    def __init__(
        self, error_message: str = "Mock LLM error", error_type: str = "APIError"
    ):
        self.error_message = error_message
        self.error_type = error_type
        self.success = False

    def raise_error(self):
        """Raise the appropriate error type."""
        if self.error_type == "TimeoutError":
            raise TimeoutError(self.error_message)
        elif self.error_type == "ValueError":
            raise ValueError(self.error_message)
        else:
            raise RuntimeError(self.error_message)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Dataset Fixtures


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    examples = [
        EvaluationExample({"query": "Hello"}, "Hi there!"),
        EvaluationExample({"query": "Goodbye"}, "See you later!"),
        EvaluationExample({"query": "How are you?"}, "I'm doing well!"),
    ]
    return Dataset(examples, name="test_dataset", description="Test dataset")


@pytest.fixture
def simple_dataset():
    """Simple 3-example dataset for quick tests."""
    examples = [
        EvaluationExample(
            input_data={"text": "This is a positive example"},
            expected_output="positive",
        ),
        EvaluationExample(
            input_data={"text": "This is a negative example"},
            expected_output="negative",
        ),
        EvaluationExample(
            input_data={"text": "This is a neutral example"}, expected_output="neutral"
        ),
    ]
    return Dataset(examples=examples, name="simple_test")


@pytest.fixture
def sentiment_dataset():
    """Comprehensive sentiment analysis dataset."""
    examples = [
        # Strong positive
        EvaluationExample(
            input_data={
                "text": "This product is absolutely fantastic and exceeded all expectations!"
            },
            expected_output="positive",
        ),
        EvaluationExample(
            input_data={
                "text": "Amazing quality and excellent customer service experience!"
            },
            expected_output="positive",
        ),
        EvaluationExample(
            input_data={
                "text": "Outstanding performance and remarkable build quality!"
            },
            expected_output="positive",
        ),
        # Strong negative
        EvaluationExample(
            input_data={
                "text": "Terrible quality, completely broken and unusable product."
            },
            expected_output="negative",
        ),
        EvaluationExample(
            input_data={
                "text": "Poor design and very disappointing overall experience."
            },
            expected_output="negative",
        ),
        EvaluationExample(
            input_data={"text": "Awful experience, would not recommend to anyone."},
            expected_output="negative",
        ),
        # Neutral
        EvaluationExample(
            input_data={
                "text": "It's an okay product, nothing special but works as expected."
            },
            expected_output="neutral",
        ),
        EvaluationExample(
            input_data={
                "text": "Average product with standard features and decent quality."
            },
            expected_output="neutral",
        ),
    ]
    return Dataset(examples=examples, name="sentiment_analysis_test")


@pytest.fixture
def sample_jsonl_file():
    """Create a temporary JSONL file for testing."""
    data = [
        {"input": {"text": "Hello world"}, "output": "HELLO WORLD"},
        {"input": {"text": "Good morning"}, "output": "GOOD MORNING"},
        {"input": {"text": "Python rocks"}, "output": "PYTHON ROCKS"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


# Function Fixtures


@pytest.fixture
def simple_sentiment_function():
    """Simple synchronous sentiment analysis function."""

    def sentiment_analysis(text: str, **kwargs) -> str:
        text_lower = text.lower()
        if any(
            word in text_lower
            for word in ["excellent", "fantastic", "amazing", "outstanding"]
        ):
            return "positive"
        elif any(
            word in text_lower
            for word in ["terrible", "awful", "horrible", "disappointing"]
        ):
            return "negative"
        else:
            return "neutral"

    return sentiment_analysis


@pytest.fixture
def async_sentiment_function():
    """Asynchronous sentiment analysis function."""

    async def async_sentiment_analysis(text: str, **kwargs) -> str:
        # Simulate some async processing time
        await asyncio.sleep(0.01)

        text_lower = text.lower()
        if any(
            word in text_lower
            for word in ["excellent", "fantastic", "amazing", "outstanding"]
        ):
            return "positive"
        elif any(
            word in text_lower
            for word in ["terrible", "awful", "horrible", "disappointing"]
        ):
            return "negative"
        else:
            return "neutral"

    return async_sentiment_analysis


@pytest.fixture
def llm_response_function():
    """Function that returns MockLLMResponse objects."""

    async def llm_sentiment_analysis(
        text: str, model: str = "gpt-3.5-turbo", temperature: float = 0.5, **kwargs
    ) -> MockLLMResponse:
        # Simulate model-specific behavior
        input_tokens = max(10, len(text) // 4)
        output_tokens = max(5, 10 + int(temperature * 20))

        # Cost varies by model
        if "gpt-4" in model:
            base_cost = 0.006
        elif "gpt-3.5" in model:
            base_cost = 0.002
        else:
            base_cost = 0.001

        cost = base_cost * (input_tokens + output_tokens) / 150
        response_time = 800 + int(temperature * 1200)

        # Simple sentiment logic
        text_lower = text.lower()
        if any(word in text_lower for word in ["excellent", "fantastic", "amazing"]):
            sentiment = "positive"
        elif any(word in text_lower for word in ["terrible", "awful", "horrible"]):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return MockLLMResponse(
            text=sentiment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            response_time=response_time,
            provider="openai" if "gpt" in model else "anthropic",
        )

    return llm_sentiment_analysis


# Evaluator and Optimizer Fixtures


@pytest.fixture
def basic_evaluator():
    """Basic LocalEvaluator for standard testing."""
    return LocalEvaluator(
        metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
    )


@pytest.fixture
def comprehensive_evaluator():
    """LocalEvaluator with comprehensive metrics."""
    return LocalEvaluator(
        metrics=["accuracy", "precision", "recall", "f1"],
        detailed=True,
        execution_mode="edge_analytics",
    )


@pytest.fixture
def privacy_evaluator():
    """LocalEvaluator configured for privacy mode."""
    return LocalEvaluator(metrics=["accuracy"], detailed=True, execution_mode="privacy")


@pytest.fixture
def grid_optimizer():
    """Basic GridSearchOptimizer for testing."""
    config_space = {"temperature": [0.3, 0.5, 0.7], "model": ["gpt-3.5-turbo", "gpt-4"]}
    return GridSearchOptimizer(config_space=config_space, objectives=["accuracy"])


@pytest.fixture
def weighted_grid_optimizer():
    """GridSearchOptimizer with weighted objectives."""
    config_space = {
        "temperature": [0.1, 0.5, 0.9],
        "approach": ["conservative", "balanced", "aggressive"],
    }
    return GridSearchOptimizer(
        config_space=config_space,
        objectives=["accuracy", "cost"],
        objective_weights={"accuracy": 0.8, "cost": 0.2},
    )


# Environment and Configuration Fixtures


@pytest.fixture
def mock_environment():
    """Set up mock environment variables."""
    original_vars = {}
    mock_vars = {
        "MOCK_MODE": "true",
        "TRAIGENT_MOCK_LLM": "true",
        "OPENAI_API_KEY": "mock-key-for-testing",  # pragma: allowlist secret
        "ANTHROPIC_API_KEY": "mock-key-for-testing",  # pragma: allowlist secret
    }

    # Store original values
    for key in mock_vars:
        original_vars[key] = os.environ.get(key)
        os.environ[key] = mock_vars[key]

    yield

    # Restore original values
    for key, original_value in original_vars.items():
        if original_value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_cache"
        cache_dir.mkdir()
        yield cache_dir


@pytest.fixture
def traigent_config():
    """Basic Traigent configuration for testing."""
    return TraigentConfig(execution_mode="edge_analytics", detailed_metrics=True)


@pytest.fixture
def privacy_config():
    """Traigent configuration for privacy mode testing."""
    return TraigentConfig(execution_mode="privacy", detailed_metrics=False)


# Original fixtures for backward compatibility


@pytest.fixture
def simple_config_space():
    """Simple configuration space for testing."""
    return {"model": ["o4-mini", "GPT-4o"], "temperature": [0.0, 0.5, 1.0]}


@pytest.fixture
def continuous_config_space():
    """Configuration space with continuous parameters."""
    return {"temperature": (0.0, 1.0), "max_tokens": (100, 1000)}


@pytest.fixture
def mixed_config_space():
    """Configuration space with mixed parameter types."""
    return {
        "model": ["o4-mini", "GPT-4o"],
        "temperature": (0.0, 1.0),
        "strategy": ["conservative", "balanced", "aggressive"],
    }


# Validation Utilities


def validate_example_metrics(example_result) -> bool:
    """Validate that an example result has all required metrics."""
    required_metrics = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "accuracy",
    ]

    for metric in required_metrics:
        if metric not in example_result.metrics:
            return False

        value = example_result.metrics[metric]
        if metric in ["input_tokens", "output_tokens", "total_tokens"]:
            if not isinstance(value, (int, float)) or value < 0:
                return False
        elif metric in ["input_cost", "output_cost", "total_cost"]:
            if not isinstance(value, (int, float)) or value < 0:
                return False
        elif metric == "accuracy":
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                return False

    # Verify token math
    total_tokens = example_result.metrics["total_tokens"]
    input_tokens = example_result.metrics["input_tokens"]
    output_tokens = example_result.metrics["output_tokens"]

    if total_tokens != input_tokens + output_tokens:
        return False

    return True


def validate_measures_array(measures: list[dict[str, Any]]) -> bool:
    """Validate that measures array has proper format."""
    if not isinstance(measures, list):
        return False

    for measure in measures:
        if not isinstance(measure, dict):
            return False

        required_fields = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
            "accuracy",
            "score",
        ]

        for field in required_fields:
            if field not in measure:
                return False

            value = measure[field]
            if field in ["input_tokens", "output_tokens", "total_tokens"]:
                if not isinstance(value, (int, float)) or value < 0:
                    return False
            elif field in ["accuracy", "score"]:
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                    return False

    return True


def validate_summary_stats(summary_stats: dict[str, Any]) -> bool:
    """Validate summary_stats follow pandas.describe() format."""
    if not isinstance(summary_stats, dict):
        return False

    required_top_level = ["metrics", "execution_time", "total_examples", "metadata"]
    for field in required_top_level:
        if field not in summary_stats:
            return False

    # Check metadata
    metadata = summary_stats["metadata"]
    if metadata.get("aggregation_method") != "pandas.describe":
        return False

    # Check metrics format
    metrics = summary_stats["metrics"]
    if not isinstance(metrics, dict):
        return False

    for _metric_name, stats in metrics.items():
        if not isinstance(stats, dict):
            return False

        required_stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for stat_field in required_stats:
            if stat_field not in stats:
                return False

            if not isinstance(stats[stat_field], (int, float)):
                return False

    return True


# Export validation functions for use in tests
__all__ = [
    "MockLLMResponse",
    "MockCachedResponse",
    "MockErrorResponse",
    "validate_example_metrics",
    "validate_measures_array",
    "validate_summary_stats",
]
