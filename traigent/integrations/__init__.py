"""Traigent integrations with ML/AI frameworks and experiment tracking.

Traigent provides two types of integrations:
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

# DSPy integration
try:
    from .dspy_adapter import (
        DSPY_AVAILABLE,
        DSPyPromptOptimizer,
        PromptOptimizationResult,
        create_dspy_integration,
    )

    DSPY_INTEGRATION_AVAILABLE = DSPY_AVAILABLE
except (ImportError, KeyError, RuntimeError):
    # KeyError/RuntimeError can occur due to pydantic/litellm version conflicts
    DSPY_INTEGRATION_AVAILABLE = False
    DSPY_AVAILABLE = False

    def _dspy_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "DSPy integration is unavailable. Install the optional `dspy-ai` dependency."
        )

    DSPyPromptOptimizer = _dspy_unavailable  # type: ignore[assignment, misc]
    PromptOptimizationResult = _dspy_unavailable  # type: ignore[assignment, misc]
    create_dspy_integration = _dspy_unavailable

# PydanticAI integration
try:
    from .pydantic_ai import (
        PYDANTICAI_AVAILABLE,
        PydanticAIHandler,
        PydanticAIPlugin,
        create_pydantic_ai_handler,
    )

    PYDANTICAI_INTEGRATION_AVAILABLE = PYDANTICAI_AVAILABLE
except (ImportError, KeyError, RuntimeError):
    PYDANTICAI_INTEGRATION_AVAILABLE = False
    PYDANTICAI_AVAILABLE = False

    def _pydanticai_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "PydanticAI integration is unavailable. "
            "Install with: pip install 'pydantic-ai>=1,<2'"
        )

    PydanticAIHandler = _pydanticai_unavailable  # type: ignore[assignment, misc]
    PydanticAIPlugin = _pydanticai_unavailable  # type: ignore[assignment, misc]
    create_pydantic_ai_handler = _pydanticai_unavailable

# Provider-based model selection
from .providers import (
    get_all_tiers,
    get_models_for_tier,
    list_available_providers,
    register_provider_tiers,
)

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

# Add DSPy exports if available
if DSPY_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "DSPyPromptOptimizer",
            "PromptOptimizationResult",
            "create_dspy_integration",
            "DSPY_AVAILABLE",
        ]
    )

# Add PydanticAI exports if available
if PYDANTICAI_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "PydanticAIHandler",
            "PydanticAIPlugin",
            "create_pydantic_ai_handler",
            "PYDANTICAI_AVAILABLE",
        ]
    )

# Langfuse integration (always available - graceful degradation for langfuse SDK)
try:
    from .langfuse import (
        LANGFUSE_AVAILABLE,
        LangfuseClient,
        LangfuseObservation,
        LangfuseOptimizationCallback,
        LangfuseTraceMetrics,
        LangfuseTracker,
        TraceIdResolver,
        create_langfuse_tracker,
    )

    LANGFUSE_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    LANGFUSE_INTEGRATION_AVAILABLE = False

    def _langfuse_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Langfuse integration is unavailable. "
            "Install the optional dependencies: pip install aiohttp requests"
        )

    LangfuseClient = _langfuse_unavailable  # type: ignore[assignment, misc]
    LangfuseObservation = _langfuse_unavailable  # type: ignore[assignment, misc]
    LangfuseOptimizationCallback = _langfuse_unavailable  # type: ignore[assignment, misc]
    LangfuseTraceMetrics = _langfuse_unavailable  # type: ignore[assignment, misc]
    LangfuseTracker = _langfuse_unavailable  # type: ignore[assignment, misc]
    TraceIdResolver = _langfuse_unavailable  # type: ignore[assignment, misc]
    create_langfuse_tracker = _langfuse_unavailable

# Workflow traces integration (always available - graceful degradation for OTEL)
from .observability.workflow_traces import (
    OTEL_AVAILABLE,
    SpanPayload,
    SpanStatus,
    SpanType,
    TraceIngestionRequest,
    TraceIngestionResponse,
    TraigentSpanExporter,
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowLoop,
    WorkflowNode,
    WorkflowTracesClient,
    WorkflowTracesTracker,
    create_workflow_tracker,
    detect_loops_in_graph,
    extract_edges_from_langgraph,
    extract_nodes_from_langgraph,
    setup_workflow_tracing,
)

WORKFLOW_TRACES_INTEGRATION_AVAILABLE = True

# Add provider exports (always available)
__all__.extend(
    [
        "get_models_for_tier",
        "list_available_providers",
        "get_all_tiers",
        "register_provider_tiers",
    ]
)

# Add Workflow Traces exports (always available)
__all__.extend(
    [
        "OTEL_AVAILABLE",
        "WORKFLOW_TRACES_INTEGRATION_AVAILABLE",
        "SpanStatus",
        "SpanType",
        "SpanPayload",
        "WorkflowNode",
        "WorkflowEdge",
        "WorkflowLoop",
        "WorkflowGraphPayload",
        "TraceIngestionRequest",
        "TraceIngestionResponse",
        "WorkflowTracesClient",
        "TraigentSpanExporter",
        "WorkflowTracesTracker",
        "create_workflow_tracker",
        "setup_workflow_tracing",
        "extract_nodes_from_langgraph",
        "extract_edges_from_langgraph",
        "detect_loops_in_graph",
    ]
)

# Add Langfuse exports if available
if LANGFUSE_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "LANGFUSE_AVAILABLE",
            "LANGFUSE_INTEGRATION_AVAILABLE",
            "LangfuseClient",
            "LangfuseObservation",
            "LangfuseOptimizationCallback",
            "LangfuseTraceMetrics",
            "LangfuseTracker",
            "TraceIdResolver",
            "create_langfuse_tracker",
        ]
    )

# LangChain/LangGraph handler integration (graceful degradation for langchain-core)
try:
    from .langchain import (
        LANGCHAIN_AVAILABLE,
        LLMCallMetrics,
        ToolCallMetrics,
        TraigentHandler,
        TraigentHandlerMetrics,
        create_traigent_handler,
        get_current_node_name,
        get_current_trial_config,
        node_context,
        trial_context,
    )

    LANGCHAIN_HANDLER_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_HANDLER_AVAILABLE = False

    def _langchain_handler_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "LangChain handler integration is unavailable. "
            "Install langchain-core: pip install langchain-core"
        )

    TraigentHandler = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    TraigentHandlerMetrics = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    LLMCallMetrics = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    ToolCallMetrics = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    create_traigent_handler = _langchain_handler_unavailable
    trial_context = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    node_context = _langchain_handler_unavailable  # type: ignore[assignment, misc]
    get_current_trial_config = _langchain_handler_unavailable
    get_current_node_name = _langchain_handler_unavailable

# Add LangChain handler exports if available
if LANGCHAIN_HANDLER_AVAILABLE:
    __all__.extend(
        [
            "LANGCHAIN_AVAILABLE",
            "LANGCHAIN_HANDLER_AVAILABLE",
            "TraigentHandler",
            "TraigentHandlerMetrics",
            "LLMCallMetrics",
            "ToolCallMetrics",
            "create_traigent_handler",
            "trial_context",
            "node_context",
            "get_current_trial_config",
            "get_current_node_name",
        ]
    )
