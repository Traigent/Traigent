"""Capture-only nonstream OpenAI calls within existing evaluation scopes."""

import functools
import inspect
import threading
from typing import Any, cast

_lock = threading.RLock()
_users = 0
_methods: list[tuple[Any, Any, Any]] = []


def _capture(response: Any, kwargs: dict[str, Any]) -> None:
    from traigent.integrations.framework_override import _capture_openai_usage
    from traigent.utils.langchain_interceptor import has_active_capture_scope

    if has_active_capture_scope():
        _capture_openai_usage(response, kwargs)


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def create(instance: Any, *args: Any, **kwargs: Any) -> Any:
        from traigent.utils.langchain_interceptor import (
            has_active_capture_scope,
            instrumented_provider_call,
        )

        if not has_active_capture_scope():
            return original(instance, *args, **kwargs)
        with instrumented_provider_call():
            response = original(instance, *args, **kwargs)
        if inspect.isawaitable(response):

            async def finish() -> Any:
                with instrumented_provider_call():
                    completed = await response
                _capture(completed, kwargs)
                return completed

            return finish()
        _capture(response, kwargs)
        return response

    cast(Any, create)._traigent_capture_original = original
    return create


def acquire_openai_capture() -> None:
    """Share temporary resource wrappers across nested/concurrent scopes."""
    global _users
    with _lock:
        if _users:
            _users += 1
            return
        try:
            from openai.resources.chat.completions import AsyncCompletions, Completions
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                _users += 1
                return
            raise
        try:
            for resource in (Completions, AsyncCompletions):
                original = resource.create
                wrapper = _wrap(original)
                resource.create = wrapper
                _methods.append((resource, original, wrapper))
        except BaseException:
            for resource, original, wrapper in _methods:
                if resource.create is wrapper:
                    resource.create = original
            _methods.clear()
            raise
        _users = 1


def release_openai_capture() -> None:
    """Restore owned methods without overwriting an active framework override."""
    global _users
    with _lock:
        _users -= 1
        if _users:
            return
        for resource, original, wrapper in _methods:
            if resource.create is wrapper:
                resource.create = original
        _methods.clear()


def restore_openai_capture_original(
    resource: Any, method_name: str, method: Any
) -> None:
    """Restore framework ownership atomically with capture acquire/release."""
    with _lock:
        restored = method
        if not _users:
            restored = getattr(method, "_traigent_capture_original", method)
        else:
            for index, (target, _original, wrapper) in enumerate(_methods):
                if target is resource and method_name == "create":
                    # The framework owns this restoration; activation may
                    # have overwritten capture while acquiring the scope.
                    # Preserve active capture after removing that override.
                    if method is wrapper:
                        # Framework activation followed this capture scope.
                        # Reuse its owned wrapper and original restoration pin.
                        restored = wrapper
                    else:
                        restored = _wrap(method)
                        _methods[index] = (target, method, restored)
                    break
        setattr(resource, method_name, restored)
