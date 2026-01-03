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
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

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
    """Extract metadata from Anthropic Message response."""
    metadata: dict[str, Any] = {}

    try:
        text = _extract_anthropic_text(response)

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        model = _safe_getattr(response, "model")
        stop_reason = _safe_getattr(response, "stop_reason")
        if stop_reason:
            metadata["stop_reason"] = stop_reason
        response_id = _safe_getattr(response, "id")
        if response_id:
            metadata["response_id"] = response_id

    except Exception as e:
        logger.debug(f"Error extracting Anthropic metadata: {e}")
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

    try:
        text = _extract_gemini_text(response)
        finish_reason = _extract_gemini_finish_reason(response)
        if finish_reason:
            metadata["finish_reason"] = finish_reason

        # Extract usage metadata
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

    except Exception as e:
        logger.debug(f"Error extracting Gemini metadata: {e}")
        text = ""
        usage = None

    return LLMResponse(
        text=text,
        raw=response,
        usage=usage,
        model=None,
        provider=provider,
        metadata=metadata if metadata else None,
    )


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
        # Extract text
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "generations") and response.generations:
            text = response.generations[0].text

        # Extract usage from meta
        if hasattr(response, "meta") and response.meta:
            meta = response.meta
            if hasattr(meta, "billed_units"):
                billed = meta.billed_units
                usage = {
                    "input_tokens": getattr(billed, "input_tokens", 0),
                    "output_tokens": getattr(billed, "output_tokens", 0),
                }
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

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
