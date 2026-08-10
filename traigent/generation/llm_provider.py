"""LLM provider abstraction for client-side guided generation.

In privacy mode the user's OWN LLM does all generation, so the provider is a
thin seam the caller supplies. The privacy-clean default is ``CallbackRewriteLLM``
wrapping a user closure that holds the user's credentials — Traigent never sees
the prompt text or example content, and this module NEVER auto-instantiates a
network client from ambient environment keys or routes through Traigent's cloud
credential resolver. A missing provider fails closed (raises), never fabricating
candidates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


class GenerationProviderError(RuntimeError):
    """Raised when no usable client-side generation LLM is available."""


@runtime_checkable
class RewriteLLM(Protocol):
    """Minimal text-completion interface the generation engines call.

    Implementations run on the USER's infrastructure/credentials. The engines
    pass a fully-formed meta-prompt and expect raw text back.
    """

    def complete(self, prompt: str) -> str: ...


class CallbackRewriteLLM:
    """Wrap a user closure ``fn(prompt) -> str`` as a RewriteLLM.

    The closure captures the user's own LLM client and credentials. This is the
    privacy-clean default: no Traigent credential path is touched.
    """

    def __init__(self, fn: Callable[[str], str]) -> None:
        if not callable(fn):
            raise GenerationProviderError(
                "CallbackRewriteLLM requires a callable fn(prompt) -> str"
            )
        self._fn = fn

    def complete(self, prompt: str) -> str:
        result = self._fn(prompt)
        if not isinstance(result, str):
            raise GenerationProviderError(
                f"generation LLM returned {type(result).__name__}, expected str"
            )
        return result


class _DuckTypedClientAdapter:
    """Adapt an already-constructed user client that exposes a known method.

    Tries common method names on an object the USER built (so their creds are
    already bound). We do not construct the client ourselves.
    """

    _CANDIDATE_METHODS = ("complete", "generate", "__call__")

    def __init__(self, client: Any) -> None:
        method = next(
            (
                getattr(client, name)
                for name in self._CANDIDATE_METHODS
                if callable(getattr(client, name, None))
            ),
            None,
        )
        if method is None:
            raise GenerationProviderError(
                "rewrite_llm object exposes none of complete/generate/__call__; "
                "pass a callable fn(prompt) -> str instead."
            )
        self._method = method

    def complete(self, prompt: str) -> str:
        result = self._method(prompt)
        if not isinstance(result, str):
            raise GenerationProviderError(
                f"generation LLM returned {type(result).__name__}, expected str"
            )
        return result


def resolve_rewrite_llm(spec: Any) -> RewriteLLM:
    """Resolve a user-provided spec into a RewriteLLM.

    Accepts a RewriteLLM, a callable ``fn(prompt) -> str``, or an already
    constructed client exposing complete/generate/__call__. Never builds a
    network client from ambient env keys. ``None`` fails closed.
    """
    if spec is None:
        raise GenerationProviderError(
            "client-side generation requires an explicit rewrite_llm "
            "(a callable fn(prompt) -> str or a constructed LLM client); "
            "Traigent will not instantiate one from environment credentials."
        )
    if isinstance(spec, RewriteLLM) and not callable(spec):
        return spec
    if callable(spec):
        # A bare function/lambda is the privacy-clean default.
        return CallbackRewriteLLM(spec)
    return _DuckTypedClientAdapter(spec)


__all__ = [
    "RewriteLLM",
    "CallbackRewriteLLM",
    "GenerationProviderError",
    "resolve_rewrite_llm",
]
