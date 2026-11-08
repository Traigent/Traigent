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
            "model": ["gpt-5.6-terra", "gpt-5.6-sol", "claude-3-opus"],
            "temperature": [0.1, 0.5, 0.9],
            "stream": [True, False]
        }
    )
    def my_function():
        # Your existing LangChain/OpenAI code stays unchanged
        llm = ChatOpenAI(model="gpt-5.6-terra", temperature=0.7)
        return llm.invoke("Hello!")
"""

from __future__ import annotations

import importlib
import threading
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

# Provider-based model selection
from .providers import (
    get_all_tiers,
    get_models_for_tier,
    list_available_providers,
    register_provider_tiers,
)

# Wrapper utilities (Protocol and helper functions)
from .wrappers import (
    OverrideContext,
    apply_parameter_overrides,
    create_method_wrapper,
    create_resilient_wrapper,
    create_wrapper,
)

WORKFLOW_TRACES_INTEGRATION_AVAILABLE = True

# ---------------------------------------------------------------------------
# Optional integrations are loaded lazily (PEP 562) on first attribute access,
# so ``import traigent`` / ``traigent.optimize`` never imports third-party
# frameworks (dspy, litellm, langchain, mlflow, wandb, ...) as a side effect.
# Public names, availability flags and unavailable-stub behaviour are unchanged.
# ---------------------------------------------------------------------------


def _unavailable_stub(message: str) -> Any:
    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(message)

    return _stub


# group -> module, exported names, flag assignments, caught errors, stub message,
# names that fall back to ``None`` instead of a raising stub.
_LAZY_GROUPS: dict[str, dict[str, Any]] = {
    "mlflow": {
        "module": ".observability.mlflow",
        "names": (
            "TraigentMLflowTracker",
            "compare_traigent_runs",
            "create_mlflow_tracker",
            "enable_mlflow_autolog",
            "get_best_traigent_run",
        ),
        "flags": {"MLFLOW_INTEGRATION_AVAILABLE": None},
        "catch": (ImportError,),
        "message": (
            "MLflow integration is unavailable. Install `traigent[integrations]` "
            "or install `mlflow-skinny` and `pandas`."
        ),
        "none_names": ("TraigentMLflowTracker",),
    },
    "wandb": {
        "module": ".observability.wandb",
        "names": (
            "TraigentWandBTracker",
            "WandBIntegration",
            "create_wandb_sweep_config",
            "create_wandb_tracker",
            "enable_wandb_autolog",
            "init_wandb_run",
            "log_final_results_to_wandb",
            "log_traigent_optimization",
            "log_trial_to_wandb",
        ),
        "flags": {"WANDB_INTEGRATION_AVAILABLE": None},
        "catch": (ImportError,),
        "message": (
            "W&B integration is unavailable. Install the optional `wandb` dependency."
        ),
        "none_names": ("TraigentWandBTracker",),
    },
    "langchain_llm": {
        "module": ".llms.langchain.base",
        "names": (
            "add_langchain_llm_mapping",
            "auto_detect_langchain_llms",
            "enable_chatgpt_optimization",
            "enable_claude_optimization",
            "enable_langchain_optimization",
            "get_supported_langchain_llms",
        ),
        "flags": {"LANGCHAIN_INTEGRATION_AVAILABLE": None},
        "catch": (ImportError,),
        "message": "LangChain integration is unavailable. Install `langchain`.",
        "none_names": (),
    },
    "openai_sdk": {
        "module": ".llms.openai",
        "names": (
            "auto_detect_openai",
            "enable_async_openai",
            "enable_openai_optimization",
            "enable_streaming_optimization",
            "enable_sync_openai",
            "enable_tools_optimization",
            "get_supported_openai_clients",
            "openai_context",
        ),
        "flags": {"OPENAI_SDK_INTEGRATION_AVAILABLE": None},
        "catch": (ImportError,),
        "message": "OpenAI SDK integration is unavailable. Install `openai`.",
        "none_names": (),
    },
    "dspy": {
        "module": ".dspy_adapter",
        "names": (
            "DSPyPromptOptimizer",
            "PromptOptimizationResult",
            "create_dspy_integration",
        ),
        # DSPY_AVAILABLE is exported by the adapter; mirrored into the flag below.
        "flags": {
            "DSPY_AVAILABLE": "DSPY_AVAILABLE",
            "DSPY_INTEGRATION_AVAILABLE": "DSPY_AVAILABLE",
        },
        # KeyError/RuntimeError can occur due to pydantic/litellm version conflicts
        "catch": (ImportError, KeyError, RuntimeError),
        "message": (
            "DSPy integration is unavailable. Traigent does not bundle DSPy; bring "
            "your own compatible installation only after reviewing its dependency "
            "security."
        ),
        "none_names": (),
    },
    "pydantic_ai": {
        "module": ".pydantic_ai",
        "names": (
            "PydanticAIHandler",
            "PydanticAIPlugin",
            "create_pydantic_ai_handler",
        ),
        "flags": {
            "PYDANTICAI_AVAILABLE": "PYDANTICAI_AVAILABLE",
            "PYDANTICAI_INTEGRATION_AVAILABLE": "PYDANTICAI_AVAILABLE",
        },
        "catch": (ImportError, KeyError, RuntimeError),
        "message": (
            "PydanticAI integration is unavailable. "
            "Install with: pip install 'pydantic-ai>=1,<2'"
        ),
        "none_names": (),
    },
    "langfuse": {
        "module": ".langfuse",
        "names": (
            "LangfuseClient",
            "LangfuseObservation",
            "LangfuseOptimizationCallback",
            "LangfuseTraceMetrics",
            "LangfuseTracker",
            "TraceIdResolver",
            "create_langfuse_tracker",
        ),
        "flags": {
            "LANGFUSE_AVAILABLE": "LANGFUSE_AVAILABLE",
            "LANGFUSE_INTEGRATION_AVAILABLE": None,
        },
        "catch": (ImportError,),
        "message": (
            "Langfuse integration is unavailable. "
            "Install the optional dependencies: pip install aiohttp requests"
        ),
        "none_names": (),
    },
    "langchain_handler": {
        "module": ".langchain",
        "names": (
            "LLMCallMetrics",
            "ToolCallMetrics",
            "TraigentHandler",
            "TraigentHandlerMetrics",
            "create_traigent_handler",
            "get_current_node_name",
            "get_current_trial_config",
            "node_context",
            "trial_context",
        ),
        "flags": {
            "LANGCHAIN_AVAILABLE": "LANGCHAIN_AVAILABLE",
            "LANGCHAIN_HANDLER_AVAILABLE": None,
        },
        "catch": (ImportError,),
        "message": (
            "LangChain handler integration is unavailable. "
            "Install langchain-core: pip install langchain-core"
        ),
        "none_names": (),
    },
}

# Public attribute -> owning group. Flag names resolve to their group as well.
# (``log_traigent_optimization`` is exported by both MLflow and W&B; W&B won
# historically because it was imported last, so it keeps winning.)
_LAZY_ATTRS: dict[str, str] = {}
for _group_name, _spec in _LAZY_GROUPS.items():
    for _attr in (*_spec["names"], *_spec["flags"]):
        _LAZY_ATTRS[_attr] = _group_name
_LAZY_ATTRS["log_traigent_optimization"] = "wandb"
del _group_name, _spec, _attr

_loaded_groups: set[str] = set()
_lazy_lock = threading.RLock()


def _load_group(group: str) -> None:
    """Import one optional integration group and publish its names."""
    with _lazy_lock:
        if group in _loaded_groups:
            return
        spec = _LAZY_GROUPS[group]
        namespace = globals()
        try:
            module = importlib.import_module(spec["module"], __name__)
            exported = {name: getattr(module, name) for name in spec["names"]}
            flags = {
                flag: (True if source is None else getattr(module, source))
                for flag, source in spec["flags"].items()
            }
        except spec["catch"]:
            stub = _unavailable_stub(spec["message"])
            exported = {
                name: (None if name in spec["none_names"] else stub)
                for name in spec["names"]
            }
            flags = dict.fromkeys(spec["flags"], False)
        for name, value in exported.items():
            # Do not let a group clobber a name another group owns.
            if _LAZY_ATTRS.get(name) == group:
                namespace[name] = value
        namespace.update(flags)
        _loaded_groups.add(group)


def __getattr__(name: str) -> Any:
    group = _LAZY_ATTRS.get(name)
    if group is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_group(group)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


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

# Exports of lazily-loaded optional integrations, workflow traces and providers.
__all__.extend(sorted(_LAZY_ATTRS))
__all__.extend(
    [
        "get_models_for_tier",
        "list_available_providers",
        "get_all_tiers",
        "register_provider_tiers",
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
