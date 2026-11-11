"""Response wrapper utilities for LLM integrations.

Provides utilities to extract metadata from SDK responses WITHOUT
modifying the return type. Plugins intercept SDK methods and must
return SDK-compatible response objects for downstream compatibility.

CRITICAL: These wrappers are for internal Traigent analytics/logging only.
They do NOT replace SDK return types - the actual SDK response object
must always be returned from intercepted methods.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Internal Traigent response wrapper for analytics/logging.

    This dataclass captures response metadata for Traigent's internal
    tracking and analytics. It does NOT replace SDK return types.

    When intercepting SDK methods like openai.chat.completions.create,
    the actual ChatCompletion object must be returned. This wrapper
    is used alongside to capture metrics.

    Attributes:
        text: Extracted text content from the response.
        raw: Original SDK response object (ChatCompletion, etc.).
        usage: Token usage information if available.
        model: Model name used for the request.
        provider: Provider name (openai, anthropic, gemini, etc.).
        latency_ms: Request latency in milliseconds if tracked.
        metadata: Additional provider-specific metadata.
    """

    text: str
    raw: Any
    usage: dict[str, int] | None = None
    model: str | None = None
    provider: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] | None = None


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute, returning default if not present."""
    return getattr(obj, attr, default) if hasattr(obj, attr) else default


def _token_count(value: Any) -> int | None:
    """Return a vendor token field as an int, or None when it is not measured.

    A missing or null count must stay unknown: 0 would make an unmeasured
    call look free to a cost objective.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return None
    return int(value)


def _build_usage(
    first_key: str,
    second_key: str,
    first: Any,
    second: Any,
    total: Any = None,
) -> dict[str, Any]:
    """Build a usage dict that keeps unknown counts as None.

    When any count is unknown the dict carries ``cost_unmeasured: True`` (the
    flag ``traigent.utils.trial_costs.is_cost_unmeasured`` understands), so
    cost paths omit the cost instead of treating it as $0.
    """
    n1, n2, nt = _token_count(first), _token_count(second), _token_count(total)
    if nt is None and n1 is not None and n2 is not None:
        nt = n1 + n2
    usage: dict[str, Any] = {first_key: n1, second_key: n2, "total_tokens": nt}
    if n1 is None or n2 is None or nt is None:
        usage["cost_unmeasured"] = True
    return usage


def _extract_openai_text(response: Any) -> str:
    """Extract text content from OpenAI response choices."""
    if not hasattr(response, "choices") or not response.choices:
        return ""
    choice = response.choices[0]
    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        return choice.message.content or ""
    if hasattr(choice, "text"):
        return choice.text or ""
    return ""


def _extract_openai_finish_reason(response: Any) -> str | None:
    """Extract finish reason from OpenAI response."""
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "finish_reason"):
            finish_reason = choice.finish_reason
            return str(finish_reason) if finish_reason is not None else None
    return None


def _extract_anthropic_text(response: Any) -> str:
    """Extract text content from Anthropic content blocks."""
    if not hasattr(response, "content") or not response.content:
        return ""
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            texts.append(block["text"])
    return "".join(texts)


def _extract_gemini_text(response: Any) -> str:
    """Extract text content from Gemini response."""
    # Simple property access
    if hasattr(response, "text"):
        text_val = response.text
        return str(text_val) if text_val is not None else ""

    # Extract from candidates
    if not hasattr(response, "candidates") or not response.candidates:
        return ""
    candidate = response.candidates[0]
    if not hasattr(candidate, "content") or not hasattr(candidate.content, "parts"):
        return ""
    texts = []
    for part in candidate.content.parts:
        if hasattr(part, "text"):
            texts.append(part.text)
    return "".join(texts)


def extract_response_metadata(
    sdk_response: Any,
    provider: str,
) -> LLMResponse:
    """Extract metadata from SDK response without modifying return type.

    This function extracts useful information from various SDK response
    types for internal Traigent tracking. The original sdk_response
    should still be returned to the caller.

    Args:
        sdk_response: The raw SDK response object.
        provider: Provider name (openai, anthropic, gemini, azure_openai).

    Returns:
        LLMResponse with extracted metadata.
    """
    extractors = {
        "openai": _extract_openai_metadata,
        "azure_openai": _extract_openai_metadata,  # Same format as OpenAI
        "anthropic": _extract_anthropic_metadata,
        "gemini": _extract_gemini_metadata,
        "cohere": _extract_cohere_metadata,
    }

    extractor = extractors.get(provider, _extract_generic_metadata)
    return extractor(sdk_response, provider)


def _extract_openai_metadata(
    response: Any,
    provider: str,
) -> LLMResponse:
    """Extract metadata from OpenAI ChatCompletion response."""
    metadata: dict[str, Any] = {}

    try:
        text = _extract_openai_text(response)
        finish_reason = _extract_openai_finish_reason(response)
        if finish_reason:
            metadata["finish_reason"] = finish_reason

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = _build_usage(
                "prompt_tokens",
                "completion_tokens",
                getattr(response.usage, "prompt_tokens", None),
                getattr(response.usage, "completion_tokens", None),
                getattr(response.usage, "total_tokens", None),
            )

        model = _safe_getattr(response, "model")
        response_id = _safe_getattr(response, "id")
        if response_id:
            metadata["response_id"] = response_id

    except Exception as e:
        logger.debug(f"Error extracting OpenAI metadata: {e}")
        text = ""
        usage = None
        model = None

    return LLMResponse(
        text=text,
        raw=response,
        usage=usage,
        model=model,
        provider=provider,
        metadata=metadata if metadata else None,
    )


def _extract_anthropic_metadata(
    response: Any,
    provider: str,
) -> LLMResponse:
    """Extract metadata from Anthropic Message response.

    Text, usage and model are extracted independently so that a failure (or a
    null token count) in one never wipes the others.
    """
    metadata: dict[str, Any] = {}
    text = ""
    usage = None
    model = None

    try:
        text = _extract_anthropic_text(response)
    except Exception as e:
        logger.debug(f"Error extracting Anthropic text: {e}")

    try:
        if hasattr(response, "usage") and response.usage:
            usage = _build_usage(
                "input_tokens",
                "output_tokens",
                getattr(response.usage, "input_tokens", None),
                getattr(response.usage, "output_tokens", None),
            )
    except Exception as e:
        logger.debug(f"Error extracting Anthropic usage: {e}")

    try:
        model = _safe_getattr(response, "model")
        stop_reason = _safe_getattr(response, "stop_reason")
        if stop_reason:
            metadata["stop_reason"] = stop_reason
        response_id = _safe_getattr(response, "id")
        if response_id:
            metadata["response_id"] = response_id
    except Exception as e:
        logger.debug(f"Error extracting Anthropic metadata: {e}")

    return LLMResponse(
        text=text,
        raw=response,
        usage=usage,
        model=model,
        provider=provider,
        metadata=metadata if metadata else None,
    )


def _extract_gemini_finish_reason(response: Any) -> str | None:
    """Extract finish reason from Gemini response candidate."""
    if not hasattr(response, "candidates") or not response.candidates:
        return None
    candidate = response.candidates[0]
    if hasattr(candidate, "finish_reason"):
        return str(candidate.finish_reason)
    return None


def _extract_gemini_metadata(
    response: Any,
    provider: str,
) -> LLMResponse:
    """Extract metadata from Gemini GenerateContentResponse."""
    metadata: dict[str, Any] = {}
    model: str | None = None

    try:
        text = _extract_gemini_text(response)
        finish_reason = _extract_gemini_finish_reason(response)
        if finish_reason:
            metadata["finish_reason"] = finish_reason

        # Extract usage metadata
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = _build_usage(
                "prompt_tokens",
                "completion_tokens",
                getattr(um, "prompt_token_count", None),
                getattr(um, "candidates_token_count", None),
                getattr(um, "total_token_count", None),
            )

        model_version = _safe_getattr(response, "model_version")
        if isinstance(model_version, str) and model_version:
            model = model_version

    except Exception as e:
        logger.debug(f"Error extracting Gemini metadata: {e}")
        text = ""
        usage = None

    return LLMResponse(
        text=text,
        raw=response,
        usage=usage,
        model=model,
        provider=provider,
        metadata=metadata if metadata else None,
    )


def _extract_cohere_v2_text(response: Any) -> str:
    """Extract text from a Cohere v2 chat response (message.content[].text)."""
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, (list, tuple)):
        return ""
    texts = []
    for block in content:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str):
            texts.append(block_text)
    return "".join(texts)


def _extract_cohere_metadata(
    response: Any,
    provider: str,
) -> LLMResponse:
    """Extract metadata from Cohere response."""
    text = ""
    usage = None
    model = None
    metadata = {}

    try:
        # Extract text: v1 (text / generations), then v2 (message.content[].text)
        if isinstance(getattr(response, "text", None), str):
            text = response.text
        elif hasattr(response, "generations") and response.generations:
            text = response.generations[0].text
        else:
            text = _extract_cohere_v2_text(response)

        # Extract usage: v1 meta.billed_units, v2 usage.billed_units
        billed = None
        for holder in ("meta", "usage"):
            container = getattr(response, holder, None)
            if container and hasattr(container, "billed_units"):
                billed = container.billed_units
                break
        if billed is not None:
            usage = _build_usage(
                "input_tokens",
                "output_tokens",
                getattr(billed, "input_tokens", None),
                getattr(billed, "output_tokens", None),
            )

        # Extract response ID
        if hasattr(response, "id"):
            metadata["response_id"] = response.id

    except Exception as e:
        logger.debug(f"Error extracting Cohere metadata: {e}")

    return LLMResponse(
        text=text,
        raw=response,
        usage=usage,
        model=model,
        provider=provider,
        metadata=metadata if metadata else None,
    )


def _extract_generic_metadata(
    response: Any,
    provider: str,
) -> LLMResponse:
    """Generic metadata extraction for unknown providers."""
    text = ""

    try:
        # Try common text attributes
        for attr in ["text", "content", "output", "result", "response"]:
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, str):
                    text = val
                    break

        # If response is a string, use it directly
        if isinstance(response, str):
            text = response

    except Exception as e:
        logger.debug(f"Error extracting generic metadata: {e}")

    return LLMResponse(
        text=text,
        raw=response,
        provider=provider,
    )
