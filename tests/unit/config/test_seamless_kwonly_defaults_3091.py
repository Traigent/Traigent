"""Seamless injection keeps keyword-only defaults of the user's function (#3091).

The AST-rewritten function is rebuilt with types.FunctionType, which takes
positional defaults only. Keyword-only defaults (__kwdefaults__) were dropped,
so calling the rewritten function without the keyword-only argument raised
a TypeError on every call.
"""

from __future__ import annotations

import types

from traigent.config.providers import SeamlessParameterProvider, _rebuild_function


def test_keyword_only_default_survives_seamless_rewrite() -> None:
    def agent(question: str, *, retries: int = 3) -> tuple[str, int]:
        model = "gpt-4o-mini"
        return (model, retries)

    wrapped = SeamlessParameterProvider().inject_config(agent, {"model": "gpt-4o"})

    assert wrapped("q") == ("gpt-4o", 3)


def test_keyword_only_default_can_still_be_overridden() -> None:
    def agent(question: str, *, retries: int = 3) -> tuple[str, int]:
        model = "gpt-4o-mini"
        return (model, retries)

    wrapped = SeamlessParameterProvider().inject_config(agent, {"model": "gpt-4o"})

    assert wrapped("q", retries=5) == ("gpt-4o", 5)


def test_keyword_only_default_survives_closure_stripped_rebuild() -> None:
    """The closure-stripped fallback rebuild keeps keyword-only defaults too (#3091).

    A closure-capturing function rebuilt from code with no free variables takes
    the fallback branch in ``_rebuild_function``, which must copy ``__kwdefaults__``
    just as the primary rebuild does.
    """

    def make():
        prefix = "hello"

        def agent(x: str, *, retries: int = 2) -> str:
            return prefix + str(retries) + x

        return agent

    original = make()
    module = compile(
        "def agent(x, *, retries=2):\n    return (x, retries)\n", "<test>", "exec"
    )
    body = next(c for c in module.co_consts if isinstance(c, types.CodeType))

    rebuilt, closure_stripped = _rebuild_function(original, body)

    assert closure_stripped is True
    assert rebuilt.__kwdefaults__ == {"retries": 2}
    assert rebuilt("q") == ("q", 2)
