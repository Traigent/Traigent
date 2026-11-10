"""Seamless injection keeps keyword-only defaults of the user's function (#3091).

The AST-rewritten function is rebuilt with types.FunctionType, which takes
positional defaults only. Keyword-only defaults (__kwdefaults__) were dropped,
so calling the rewritten function without the keyword-only argument raised
a TypeError on every call.
"""

from __future__ import annotations

from traigent.config.providers import SeamlessParameterProvider


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
