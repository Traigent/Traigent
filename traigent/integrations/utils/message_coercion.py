"""Message coercion utilities for LLM integrations.

Provides functions to normalize various message input formats
into the standard role/content dict format used by most LLM APIs.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from collections.abc import Mapping, Sequence
from typing import Any


def coerce_messages(
    messages: str | list[str] | list[dict[str, Any]] | Any,
    default_role: str = "user",
) -> list[dict[str, Any]]:
    """Normalize message input to list of role/content dicts.

    Accepts various input formats and converts them to the standard
    message format expected by LLM APIs:
    [{"role": "user", "content": "message"}]

    Args:
        messages: Input messages in various formats:
            - str: Single message string
            - list[str]: List of message strings (all assigned default_role)
            - list[dict]: Already formatted messages
            - dict: Single message dict
        default_role: Role to assign when not specified (default: "user")

    Returns:
        List of message dicts with role and content keys.

    Examples:
        >>> coerce_messages("Hello")
        [{"role": "user", "content": "Hello"}]

        >>> coerce_messages(["Hello", "World"])
        [{"role": "user", "content": "Hello"}, {"role": "user", "content": "World"}]

        >>> coerce_messages([{"role": "assistant", "content": "Hi"}])
        [{"role": "assistant", "content": "Hi"}]
    """
    if messages is None:
        return []

    # Single string -> single user message
    if isinstance(messages, str):
        return [{"role": default_role, "content": messages}]

    # Single dict -> wrap in list
    if isinstance(messages, Mapping):
        return [_normalize_message_dict(messages, default_role)]

    # Sequence of messages
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        result = []
        for msg in messages:
            if isinstance(msg, str):
                result.append({"role": default_role, "content": msg})
            elif isinstance(msg, Mapping):
                result.append(_normalize_message_dict(msg, default_role))
            else:
                # Try to convert to string
                result.append({"role": default_role, "content": str(msg)})
        return result

    # Fallback: convert to string
    return [{"role": default_role, "content": str(messages)}]


def _normalize_message_dict(
    msg: Mapping[str, Any],
    default_role: str = "user",
) -> dict[str, Any]:
    """Normalize a single message dict.

    Handles various message dict formats:
    - {"role": "user", "content": "text"}
    - {"content": "text"} (role inferred)
    - {"text": "text"} (content aliased)
    - {"parts": [...]} (Gemini format)

    Args:
        msg: Message dict to normalize.
        default_role: Default role if not specified.

    Returns:
        Normalized message dict with role and content.
    """
    result = dict(msg)

    # Ensure role exists
    if "role" not in result:
        result["role"] = default_role

    # Handle content aliases
    if "content" not in result:
        # Check for 'text' alias (common in some APIs)
        if "text" in result:
            result["content"] = result.pop("text")
        # Check for Gemini 'parts' format
        elif "parts" in result:
            parts = result.get("parts", [])
            result["content"] = _extract_text_from_parts(parts)
        else:
            result["content"] = ""

    return result


def _extract_text_from_parts(parts: Any) -> str:
    """Extract text content from Gemini-style parts.

    Args:
        parts: Parts list from Gemini message format.

    Returns:
        Concatenated text content.
    """
    if isinstance(parts, str):
        return parts

    if not isinstance(parts, Sequence):
        return str(parts) if parts else ""

    texts = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, Mapping):
            # {"text": "..."} format
            if "text" in part:
                texts.append(str(part["text"]))
            elif "content" in part:
                texts.append(str(part["content"]))

    return " ".join(texts)


def coerce_to_openai_format(
    messages: str | list[str] | list[dict[str, Any]] | Any,
) -> list[dict[str, str]]:
    """Coerce messages to OpenAI chat format.

    OpenAI expects: [{"role": "user", "content": "message"}]

    Args:
        messages: Input messages in any format.

    Returns:
        Messages in OpenAI format.
    """
    return coerce_messages(messages, default_role="user")


def coerce_to_anthropic_format(
    messages: str | list[str] | list[dict[str, Any]] | Any,
) -> tuple[str | None, list[dict[str, str]]]:
    """Coerce messages to Anthropic format.

    Anthropic expects system message as separate parameter.

    Args:
        messages: Input messages in any format.

    Returns:
        Tuple of (system_message, messages_list).
        system_message is None if no system message found.
    """
    normalized = coerce_messages(messages)

    system_message: str | None = None
    conversation: list[dict[str, str]] = []

    for msg in normalized:
        if msg.get("role") == "system":
            # Extract system message
            if system_message is None:
                system_message = str(msg.get("content", ""))
            else:
                # Append additional system messages
                system_message += "\n" + str(msg.get("content", ""))
        else:
            conversation.append(msg)

    return system_message, conversation


def coerce_to_gemini_format(
    messages: str | list[str] | list[dict[str, Any]] | Any,
) -> str | list[dict[str, Any]]:
    """Coerce messages to Gemini format.

    Gemini's generate_content can accept:
    - Simple string
    - List of content dicts with role and parts

    For simple single-turn, returns string.
    For multi-turn, returns content list.

    Args:
        messages: Input messages in any format.

    Returns:
        String for single message, or content list for multi-turn.
    """
    normalized = coerce_messages(messages)

    if len(normalized) == 1 and normalized[0].get("role") == "user":
        # Single user message -> return just the content string
        return str(normalized[0].get("content", ""))

    # Multi-turn -> convert to Gemini content format
    gemini_contents = []
    for msg in normalized:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Gemini uses "model" instead of "assistant"
        if role == "assistant":
            role = "model"

        gemini_contents.append(
            {
                "role": role,
                "parts": [{"text": str(content)}],
            }
        )

    return gemini_contents
