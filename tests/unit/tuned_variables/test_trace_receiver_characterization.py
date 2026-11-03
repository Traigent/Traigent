"""Golden characterization for ``_SinkFinder._trace_receiver`` via the finder."""

from __future__ import annotations

import ast
import textwrap

from traigent.tuned_variables.dataflow_strategy import (
    DEFAULT_SINK_PATTERNS,
    _build_def_use_map,
    _SinkFinder,
)


def _find(src: str, hints=("ChatOpenAI",)):
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    sinks = {p.method_name: p for p in DEFAULT_SINK_PATTERNS}
    finder = _SinkFinder(sinks, frozenset(hints), _build_def_use_map(fn))
    finder.visit(fn)
    return [
        (a.var_name, a.param_name, a.sink_method, a.sink_category, a.line)
        for a in finder.sink_args
    ]


def test_constructor_kwargs_and_literals_traced():
    out = _find(
        """
        def f(t, m):
            llm = ChatOpenAI(temperature=t, model="gpt", n=m + t)
            return llm.invoke("hi")
        """
    )
    ctor = [r for r in out if r[2] == "ChatOpenAI.__init__"]
    # kwargs keep their order; names inside one expression come from a set
    assert [r[1] for r in ctor] == ["temperature", "model", "n", "n"]
    assert ctor[0] == ("t", "temperature", "ChatOpenAI.__init__", "llm", 3)
    assert ctor[1] == ("__literal__model", "model", "ChatOpenAI.__init__", "llm", 3)
    assert sorted(r[0] for r in ctor[2:]) == ["m", "t"]
    assert all(r[3:] == ("llm", 3) for r in ctor)


def test_attribute_class_name_and_nested_receiver():
    out = _find(
        """
        def f(t):
            llm = lc.ChatOpenAI(temperature=t)
            return llm.client.chat.create()
        """
    )
    assert ("t", "temperature", "ChatOpenAI.__init__", "llm", 3) in out


def test_unknown_class_not_traced():
    out = _find(
        """
        def f(t):
            llm = Other(temperature=t)
            return llm.invoke("x")
        """
    )
    assert out == []


def test_non_call_definition_or_unknown_receiver_not_traced():
    assert (
        _find(
            """
        def f(t):
            llm = t
            return llm.invoke("x")
        """
        )
        == []
    )
    assert _find(
        """
        def f(t):
            return make()[0].invoke(t)
        """
    ) == [("t", None, "invoke", "llm", 3)]


def test_bare_function_sink_has_no_receiver_trace():
    out = _find(
        """
        def f(t):
            llm = ChatOpenAI(temperature=t)
            return completion(model=t)
        """
    )
    assert all(r[2] != "ChatOpenAI.__init__" for r in out)


def test_multiple_definitions_each_traced_and_double_star_skipped():
    out = _find(
        """
        def f(a, b, kw):
            llm = ChatOpenAI(temperature=a)
            llm = ChatOpenAI(top_p=b, **kw)
            return llm.invoke("x")
        """
    )
    ctor = [r for r in out if r[2] == "ChatOpenAI.__init__"]
    assert [(r[0], r[1]) for r in ctor] == [
        ("a", "temperature"),
        ("b", "top_p"),
        ("kw", None),
    ]
