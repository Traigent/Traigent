"""Shared mock classes for Traigent SDK test suite.

This module provides reusable mock implementations to eliminate duplication
across test files and ensure consistent behavior in tests.
"""

from .cloud_services import (
    MockHybridCloudClient,
    MockPrivacyCloudClient,
    MockTraigentCloudClient,
    MockTraigentCloudClientWithAuth,
)
from .data_generators import (
    create_config_space,
    create_evaluation_examples,
    create_mock_trial_results,
    create_test_dataset,
)
from .evaluators import (
    MockAsyncEvaluator,
    MockDetailedEvaluator,
    MockEvaluator,
    create_mock_evaluator,
)
from .llm_providers import (
    ConfigurationLogger,
    MockAnthropic,
    MockAsyncAnthropic,
    MockAsyncOpenAI,
    MockCohere,
    MockHuggingFacePipeline,
    MockLangChainLLM,
    MockOpenAI,
)
from .optimizers import (
    MockAsyncOptimizer,
    MockBayesianOptimizer,
    MockGridOptimizer,
    MockOptimizer,
    create_mock_optimizer,
)

__all__ = [
    # LLM Providers
    "MockOpenAI",
    "MockAsyncOpenAI",
    "MockAnthropic",
    "MockAsyncAnthropic",
    "MockLangChainLLM",
    "MockHuggingFacePipeline",
    "MockCohere",
    "ConfigurationLogger",
    # Cloud Services
    "MockTraigentCloudClient",
    "MockTraigentCloudClientWithAuth",
    "MockPrivacyCloudClient",
    "MockHybridCloudClient",
    # Data Generators
    "create_test_dataset",
    "create_config_space",
    "create_evaluation_examples",
    "create_mock_trial_results",
    # Optimizers
    "MockOptimizer",
    "MockAsyncOptimizer",
    "MockBayesianOptimizer",
    "MockGridOptimizer",
    "create_mock_optimizer",
    # Evaluators
    "MockEvaluator",
    "MockAsyncEvaluator",
    "MockDetailedEvaluator",
    "create_mock_evaluator",
]
