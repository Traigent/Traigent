"""Static parameter and method mappings for framework overrides.

This module is the single active source of truth for framework parameter mappings.
FrameworkOverrideManager loads these directly — there is no separate plugin-precedence
layer on top of them. They are used for:
1. Class-level (constructor) parameter injection — ONLY constructor-valid kwargs
2. Method-level parameter injection — specifying which params each method accepts
3. Method-specific parameter name translations (see METHOD_PARAMETER_TRANSLATIONS)

IMPORTANT — constructor vs method params:
  Class-level PARAMETER_MAPPINGS must contain ONLY parameters that are valid kwargs
  for the class __init__. Generation params (temperature, max_tokens, top_p, etc.)
  that are NOT accepted by __init__ must NOT appear in PARAMETER_MAPPINGS — doing so
  causes TypeError when the constructor override injects them. Generation params for
  such classes belong exclusively in METHOD_MAPPINGS and METHOD_PARAMETER_TRANSLATIONS.

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
    # HuggingFace mappings (transformers — legacy/direct-model path)
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
    # HuggingFace Hub InferenceClient mappings (primary auto-override surface)
    #
    # CONSTRUCTOR-ONLY params: InferenceClient.__init__ accepts model, token, timeout,
    # base_url, headers, cookies, api_key — it does NOT accept generation kwargs such as
    # temperature, max_tokens/max_new_tokens, top_p, top_k, stop, or stream.
    # Injecting those into __init__ raises TypeError (issue #1570).
    #
    # Generation params live ONLY in METHOD_MAPPINGS (to control which methods receive them)
    # and METHOD_PARAMETER_TRANSLATIONS (for max_tokens → max_new_tokens in text_generation).
    "huggingface_hub.InferenceClient": {
        "model": "model",
    },
    "huggingface_hub.AsyncInferenceClient": {
        "model": "model",
    },
    # PydanticAI mappings — discovery/documentation only.
    # IMPORTANT: PydanticAI requires these params inside the `model_settings` dict,
    # NOT as top-level kwargs. Actual injection must go through
    # PydanticAIPlugin.apply_overrides() which handles the nesting.
    "pydantic_ai.Agent": {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
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
    # HuggingFace Hub InferenceClient methods (primary auto-override surface)
    # max_tokens is listed for text_generation only: maps to max_new_tokens via PARAMETER_MAPPINGS.
    # chat_completion omits max_tokens because its framework param name is max_tokens (not
    # max_new_tokens), so injecting max_new_tokens there would silently be ignored/error.
    "huggingface_hub.InferenceClient": {
        "text_generation": [
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
        ],
        "chat_completion": [
            "model",
            "temperature",
            "top_p",
            "stop",
            "stream",
        ],
    },
    "huggingface_hub.AsyncInferenceClient": {
        "text_generation": [
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
        ],
        "chat_completion": [
            "model",
            "temperature",
            "top_p",
            "stop",
            "stream",
        ],
    },
    # PydanticAI methods — params are injected via model_settings dict
    # (see PydanticAIPlugin.apply_overrides for actual nesting logic)
    "pydantic_ai.Agent": {
        "run": ["temperature", "max_tokens", "top_p"],
        "run_sync": ["temperature", "max_tokens", "top_p"],
        "run_stream": ["temperature", "max_tokens", "top_p"],
        "run_stream_sync": ["temperature", "max_tokens", "top_p"],
    },
}

# Method-level parameter name translations.
# Used when a method's framework param name differs from the Traigent canonical name
# AND the class-level PARAMETER_MAPPINGS does not carry that translation (because the
# param is not a constructor arg and therefore was intentionally excluded from
# PARAMETER_MAPPINGS to prevent TypeError on __init__).
#
# Structure: class_name -> method_name -> {traigent_param: framework_param}
#
# These translations are merged by FrameworkOverrideManager._create_override_method
# with the class-level PARAMETER_MAPPINGS and an identity fallback for remaining
# supported_params, so only entries that differ from the canonical name are needed.
METHOD_PARAMETER_TRANSLATIONS: dict[str, dict[str, dict[str, str]]] = {
    # HuggingFace text_generation: the framework param is max_new_tokens, not max_tokens.
    # All other generation params use the canonical Traigent name (temperature, top_p, etc.).
    "huggingface_hub.InferenceClient": {
        "text_generation": {
            "max_tokens": "max_new_tokens",
        },
        # chat_completion: max_tokens is intentionally excluded from METHOD_MAPPINGS
        # (it is not a valid chat_completion kwarg); no special translation needed.
        "chat_completion": {},
    },
    "huggingface_hub.AsyncInferenceClient": {
        "text_generation": {
            "max_tokens": "max_new_tokens",
        },
        "chat_completion": {},
    },
}


def get_method_parameter_translation(
    class_name: str, method_name: str
) -> dict[str, str]:
    """Get method-level parameter translations for a specific class and method.

    These supplement the class-level PARAMETER_MAPPINGS for cases where a method's
    framework param name differs from the Traigent canonical name but the param is
    NOT a valid constructor arg (and therefore absent from PARAMETER_MAPPINGS).

    Args:
        class_name: Fully-qualified class name (e.g., "huggingface_hub.InferenceClient")
        method_name: Method name (e.g., "text_generation")

    Returns:
        Dict mapping traigent_param -> framework_param for this method, or empty dict
    """
    return METHOD_PARAMETER_TRANSLATIONS.get(class_name, {}).get(method_name, {})


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
