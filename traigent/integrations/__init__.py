"""TraiGent integrations with ML/AI frameworks and experiment tracking.

TraiGent provides two types of integrations:
# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

1. **Zero-Code-Change Framework Integration**:
   - Automatically optimizes LangChain, OpenAI SDK, Anthropic, Cohere, HuggingFace
   - No code changes required - just add @traigent.optimize decorator
   - Parameters are automatically overridden during optimization
   - Streaming and tool/function calling support

2. **Experiment Tracking Integrations**:
   - MLflow: Track optimization experiments and results
   - Weights & Biases: Advanced experiment tracking and visualization

Example of zero-code-change optimization:

    @traigent.optimize(
        auto_override_frameworks=True,  # Enable automatic parameter override
        configuration_space={
            "model": ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"],
            "temperature": [0.1, 0.5, 0.9],
            "stream": [True, False]
        }
    )
    def my_function():
        # Your existing LangChain/OpenAI code stays unchanged
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        return llm.invoke("Hello!")
"""

from __future__ import annotations

from typing import Any

# Activation state management (thread-safe state)
from .activation import ActivationState, create_activation_state

# Base manager (for subclassing and advanced use)
from .base import BaseOverrideManager

# Core framework override (primary API)
from .framework_override import (
    FrameworkOverrideManager,
    disable_framework_overrides,
    enable_framework_overrides,
    override_all_platforms,
    override_anthropic,
    override_cohere,
    override_context,
    override_huggingface,
    override_langchain,
    override_openai_sdk,
    register_framework_mapping,
)

# Static mappings (fallback mappings when no plugin registered)
from .mappings import (
    METHOD_MAPPINGS,
    PARAMETER_MAPPINGS,
    get_method_mapping,
    get_parameter_mapping,
    get_supported_frameworks,
)

# Wrapper utilities (Protocol and helper functions)
from .wrappers import (
    OverrideContext,
    apply_parameter_overrides,
    create_method_wrapper,
    create_resilient_wrapper,
    create_wrapper,
)

# MLflow integration
try:
    from .observability.mlflow import (
        TraigentMLflowTracker,
        compare_traigent_runs,
        create_mlflow_tracker,
        enable_mlflow_autolog,
        get_best_traigent_run,
        log_traigent_optimization,
    )

    MLFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    TraigentMLflowTracker = None  # type: ignore[assignment, misc]

    def _mlflow_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "MLflow integration is unavailable. Install the optional `mlflow` dependency."
        )

    compare_traigent_runs = _mlflow_unavailable
    create_mlflow_tracker = _mlflow_unavailable
    enable_mlflow_autolog = _mlflow_unavailable
    get_best_traigent_run = _mlflow_unavailable
    log_traigent_optimization = _mlflow_unavailable
    MLFLOW_INTEGRATION_AVAILABLE = False

# Weights & Biases integration
try:
    from .observability.wandb import (  # type: ignore[assignment]
        TraigentWandBTracker,
        WandBIntegration,
        create_wandb_sweep_config,
        create_wandb_tracker,
        enable_wandb_autolog,
        init_wandb_run,
        log_final_results_to_wandb,
        log_traigent_optimization,
        log_trial_to_wandb,
    )

    WANDB_INTEGRATION_AVAILABLE = True
except ImportError:
    TraigentWandBTracker = None  # type: ignore[assignment, misc]

    def _wandb_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "W&B integration is unavailable. Install the optional `wandb` dependency."
        )

    create_wandb_sweep_config = _wandb_unavailable
    create_wandb_tracker = _wandb_unavailable
    enable_wandb_autolog = _wandb_unavailable
    init_wandb_run = _wandb_unavailable
    log_final_results_to_wandb = _wandb_unavailable
    log_traigent_optimization = _wandb_unavailable
    log_trial_to_wandb = _wandb_unavailable
    WandBIntegration = _wandb_unavailable  # type: ignore[assignment, misc]
    WANDB_INTEGRATION_AVAILABLE = False

# LangChain integration
try:
    from .llms.langchain.base import (
        add_langchain_llm_mapping,
        auto_detect_langchain_llms,
        enable_chatgpt_optimization,
        enable_claude_optimization,
        enable_langchain_optimization,
        get_supported_langchain_llms,
    )

    LANGCHAIN_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGCHAIN_INTEGRATION_AVAILABLE = False

# OpenAI SDK integration
try:
    from .llms.openai import (
        auto_detect_openai,
        enable_async_openai,
        enable_openai_optimization,
        enable_streaming_optimization,
        enable_sync_openai,
        enable_tools_optimization,
        get_supported_openai_clients,
        openai_context,
    )

    OPENAI_SDK_INTEGRATION_AVAILABLE = True
except ImportError:
    OPENAI_SDK_INTEGRATION_AVAILABLE = False

__all__ = [
    # Core framework override (primary API)
    "FrameworkOverrideManager",
    "enable_framework_overrides",
    "disable_framework_overrides",
    "override_context",
    "register_framework_mapping",
    "override_openai_sdk",
    "override_langchain",
    "override_anthropic",
    "override_cohere",
    "override_huggingface",
    "override_all_platforms",
    # Base manager (for subclassing)
    "BaseOverrideManager",
    # Activation state management
    "ActivationState",
    "create_activation_state",
    # Static mappings
    "PARAMETER_MAPPINGS",
    "METHOD_MAPPINGS",
    "get_parameter_mapping",
    "get_method_mapping",
    "get_supported_frameworks",
    # Wrapper utilities
    "OverrideContext",
    "apply_parameter_overrides",
    "create_wrapper",
    "create_method_wrapper",
    "create_resilient_wrapper",
]

# Add MLflow exports if available
if MLFLOW_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "TraigentMLflowTracker",
            "create_mlflow_tracker",
            "enable_mlflow_autolog",
            "log_traigent_optimization",
            "compare_traigent_runs",
            "get_best_traigent_run",
        ]
    )

# Add W&B exports if available
if WANDB_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "TraigentWandBTracker",
            "init_wandb_run",
            "log_traigent_optimization",
            "log_trial_to_wandb",
            "log_final_results_to_wandb",
            "create_wandb_tracker",
            "enable_wandb_autolog",
            "create_wandb_sweep_config",
            "WandBIntegration",
        ]
    )

# Add LangChain exports if available
if LANGCHAIN_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "enable_langchain_optimization",
            "get_supported_langchain_llms",
            "add_langchain_llm_mapping",
            "enable_chatgpt_optimization",
            "enable_claude_optimization",
            "auto_detect_langchain_llms",
        ]
    )

# Add OpenAI SDK exports if available
if OPENAI_SDK_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "enable_openai_optimization",
            "get_supported_openai_clients",
            "enable_sync_openai",
            "enable_async_openai",
            "openai_context",
            "auto_detect_openai",
            "enable_streaming_optimization",
            "enable_tools_optimization",
        ]
    )
