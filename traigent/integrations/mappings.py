"""Static parameter and method mappings for framework overrides.

This module contains the static mappings used as FALLBACK when no plugin is registered.
Plugin mappings (via LLMPlugin._get_default_mappings) take precedence over these static
mappings. These are primarily used for:
1. Legacy override path support
2. Frameworks without dedicated plugins
3. Method-level parameter injection (specifying which params each method accepts)

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

# Parameter mappings from Traigent canonical names to framework-specific parameter names.
# Keys are fully-qualified class names, values are dicts mapping traigent_param -> framework_param
PARAMETER_MAPPINGS: dict[str, dict[str, str]] = {
    # OpenAI SDK mappings
    "openai.OpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stream": "stream",
        "tools": "tools",
    },
    "openai.AsyncOpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stream": "stream",
        "tools": "tools",
    },
    # LangChain mappings
    "langchain.llms.OpenAI": {
        "model": "model_name",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
    },
    "langchain_openai.OpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "streaming": "streaming",
    },
    "langchain_openai.ChatOpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "streaming": "streaming",
    },
    # Anthropic mappings
    "anthropic.Anthropic": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens_to_sample",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop_sequences",
        "stream": "stream",
        "tools": "tools",
        "system_prompt": "system",
    },
    "anthropic.AsyncAnthropic": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens_to_sample",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop_sequences",
        "stream": "stream",
        "tools": "tools",
        "system_prompt": "system",
    },
    # LangChain Anthropic mappings
    "langchain_anthropic.ChatAnthropic": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop",
        "streaming": "streaming",
    },
    # Cohere mappings
    "cohere.Client": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "p",
        "top_k": "k",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop_sequences": "stop_sequences",
        "stream": "stream",
        "tools": "tools",
    },
    "cohere.AsyncClient": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "p",
        "top_k": "k",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop_sequences": "stop_sequences",
        "stream": "stream",
        "tools": "tools",
    },
    # HuggingFace mappings
    "transformers.pipeline": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_new_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop",
        "stream": "streamer",
    },
    "transformers.AutoModelForCausalLM": {
        "model": "pretrained_model_name_or_path",
        "temperature": "temperature",
        "max_tokens": "max_new_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
    },
    # Mock classes for testing
    "MockOpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
    },
    "MockLangChainOpenAI": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
    },
}

# Method mappings: Maps class names to methods and which parameters each method accepts.
# Used for method-level parameter injection (e.g., chat.completions.create).
METHOD_MAPPINGS: dict[str, dict[str, list[str]]] = {
    # OpenAI SDK methods that accept parameters
    "openai.OpenAI": {
        "completions.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ],
        "chat.completions.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ],
    },
    "openai.AsyncOpenAI": {
        "completions.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ],
        "chat.completions.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "tools",
        ],
    },
    # Anthropic methods
    "anthropic.Anthropic": {
        "messages.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stream",
            "tools",
            "system",
        ],
        "messages.stream": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "tools",
            "system",
        ],
    },
    "anthropic.AsyncAnthropic": {
        "messages.create": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stream",
            "tools",
            "system",
        ],
        "messages.stream": [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "tools",
            "system",
        ],
    },
    # LangChain methods
    "langchain_openai.ChatOpenAI": {
        "invoke": ["model", "temperature", "max_tokens", "streaming"],
        "stream": ["model", "temperature", "max_tokens"],
        "astream": ["model", "temperature", "max_tokens"],
    },
    "langchain_anthropic.ChatAnthropic": {
        "invoke": ["model", "temperature", "max_tokens", "streaming"],
        "stream": ["model", "temperature", "max_tokens"],
        "astream": ["model", "temperature", "max_tokens"],
    },
    # Cohere methods
    "cohere.Client": {
        "generate": ["model", "temperature", "max_tokens", "p", "k", "stream"],
        "chat": [
            "model",
            "temperature",
            "max_tokens",
            "p",
            "k",
            "stream",
            "tools",
        ],
    },
    "cohere.AsyncClient": {
        "generate": ["model", "temperature", "max_tokens", "p", "k", "stream"],
        "chat": [
            "model",
            "temperature",
            "max_tokens",
            "p",
            "k",
            "stream",
            "tools",
        ],
    },
}


def get_parameter_mapping(class_name: str) -> dict[str, str]:
    """Get parameter mapping for a framework class.

    Args:
        class_name: Fully-qualified class name (e.g., "openai.OpenAI")

    Returns:
        Parameter mapping dict, or empty dict if not found
    """
    return PARAMETER_MAPPINGS.get(class_name, {})


def get_method_mapping(class_name: str) -> dict[str, list[str]]:
    """Get method mapping for a framework class.

    Args:
        class_name: Fully-qualified class name (e.g., "openai.OpenAI")

    Returns:
        Method mapping dict (method_name -> list of supported params), or empty dict
    """
    return METHOD_MAPPINGS.get(class_name, {})


def get_method_params(class_name: str, method_name: str) -> list[str]:
    """Get supported parameters for a specific method.

    Args:
        class_name: Fully-qualified class name
        method_name: Method name (e.g., "chat.completions.create")

    Returns:
        List of parameter names the method accepts, or empty list
    """
    return METHOD_MAPPINGS.get(class_name, {}).get(method_name, [])


def register_parameter_mapping(class_name: str, mapping: dict[str, str]) -> None:
    """Register a custom parameter mapping for a class.

    Note: For new integrations, prefer creating an LLMPlugin subclass instead.
    This function is provided for dynamic registration at runtime.

    Args:
        class_name: Fully-qualified class name
        mapping: Parameter mapping dict (traigent_param -> framework_param)
    """
    PARAMETER_MAPPINGS[class_name] = mapping


def register_method_mapping(class_name: str, mapping: dict[str, list[str]]) -> None:
    """Register a custom method mapping for a class.

    Args:
        class_name: Fully-qualified class name
        mapping: Method mapping dict (method_name -> list of params)
    """
    METHOD_MAPPINGS[class_name] = mapping


def get_all_supported_classes() -> list[str]:
    """Get list of all classes with parameter mappings.

    Returns:
        List of fully-qualified class names
    """
    return list(PARAMETER_MAPPINGS.keys())


def get_supported_frameworks() -> list[str]:
    """Get list of supported framework class names.

    This is an alias for get_all_supported_classes() with a more intuitive name
    for external API use.

    Returns:
        List of fully-qualified class names that have parameter mappings
    """
    return get_all_supported_classes()


def has_mapping(class_name: str) -> bool:
    """Check if a class has a parameter mapping defined.

    Args:
        class_name: Fully-qualified class name

    Returns:
        True if mapping exists, False otherwise
    """
    return class_name in PARAMETER_MAPPINGS
