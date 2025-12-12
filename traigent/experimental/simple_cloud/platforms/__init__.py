"""Platform-specific agent executors.

This module provides implementations for various AI platforms including
Anthropic, Cohere, HuggingFace, OpenAI, and LangChain.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

from .base_platform import BasePlatformExecutor
from .parameter_mapping import (
    PLATFORM_MAPPINGS,
    UNIFIED_PARAMS,
    ParameterMapper,
    get_mapper,
)

# Import platform executors
try:
    from .anthropic_executor import AnthropicAgentExecutor

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from .cohere_executor import CohereAgentExecutor

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from .huggingface_executor import HuggingFaceAgentExecutor

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Re-export existing platforms from parent module
try:
    from traigent.agents.platforms import (
        LangChainAgentExecutor,
        OpenAIAgentExecutor,
    )

    LANGCHAIN_AVAILABLE = True
    OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    OPENAI_AVAILABLE = False

__all__ = [
    "BasePlatformExecutor",
    "ParameterMapper",
    "get_mapper",
    "UNIFIED_PARAMS",
    "PLATFORM_MAPPINGS",
]

# Add available executors to exports
if ANTHROPIC_AVAILABLE:
    __all__.append("AnthropicAgentExecutor")

if COHERE_AVAILABLE:
    __all__.append("CohereAgentExecutor")

if HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceAgentExecutor")

if LANGCHAIN_AVAILABLE:
    __all__.append("LangChainAgentExecutor")

if OPENAI_AVAILABLE:
    __all__.append("OpenAIAgentExecutor")
