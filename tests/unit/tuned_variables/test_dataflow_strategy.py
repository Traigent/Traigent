"""Unit tests for dataflow_strategy.py.

Tests cover:
- _names_in_expr: name extraction from AST expressions
- _DefUseBuilder: definition tracking for all assignment forms
- _SinkFinder: statistical call site identification and argument extraction
- DataFlowDetectionStrategy: end-to-end backward slicing detection
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from traigent.tuned_variables.dataflow_strategy import (
    CONSTRUCTOR_CLASS_HINTS,
    DEFAULT_SINK_PATTERNS,
    DataFlowDetectionStrategy,
    SinkPattern,
    _build_def_use_map,
    _DefUseBuilder,
    _DefUseMap,
    _hop_to_confidence,
    _names_in_expr,
    _SinkFinder,
)
from traigent.tuned_variables.detection_types import CandidateType, DetectionConfidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


def _names(candidates) -> list[str]:
    return [c.name for c in candidates]


def _by_name(candidates, name: str):
    for c in candidates:
        if c.name == name:
            return c
    return None


def _parse_func(src: str, func_name: str = "fn"):
    """Parse source and return the target FunctionDef node."""
    tree = ast.parse(_dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                return node
    raise ValueError(f"Function {func_name!r} not found in source")


# ---------------------------------------------------------------------------
# _names_in_expr
# ---------------------------------------------------------------------------


class TestNamesInExpr:
    def test_name_node(self) -> None:
        node = ast.parse("x", mode="eval").body
        assert _names_in_expr(node) == {"x"}

    def test_attribute_extracts_base(self) -> None:
        node = ast.parse("obj.attr", mode="eval").body
        assert _names_in_expr(node) == {"obj"}

    def test_binop_extracts_both_sides(self) -> None:
        node = ast.parse("a + b", mode="eval").body
        assert _names_in_expr(node) == {"a", "b"}

    def test_call_extracts_func_and_args(self) -> None:
        node = ast.parse("f(x, y=z)", mode="eval").body
        assert _names_in_expr(node) == {"f", "x", "z"}

    def test_joinedstr_extracts_interpolated(self) -> None:
        node = ast.parse('f"Hello {name} at {place}"', mode="eval").body
        assert _names_in_expr(node) == {"name", "place"}

    def test_subscript_extracts_base_and_key(self) -> None:
        node = ast.parse("d[key]", mode="eval").body
        assert _names_in_expr(node) == {"d", "key"}

    def test_nested_expression(self) -> None:
        node = ast.parse("a + f(b[c])", mode="eval").body
        assert _names_in_expr(node) == {"a", "f", "b", "c"}

    def test_none_returns_empty(self) -> None:
        assert _names_in_expr(None) == set()

    def test_constant_returns_empty(self) -> None:
        node = ast.parse("42", mode="eval").body
        assert _names_in_expr(node) == set()


# ---------------------------------------------------------------------------
# _DefUseBuilder
# ---------------------------------------------------------------------------


class TestDefUseBuilder:
    def test_simple_assignment(self) -> None:
        func = _parse_func(
            """
            def fn():
                x = 1
        """
        )
        du = _build_def_use_map(func)
        assert "x" in du.definitions
        assert du.definitions["x"][0].var_name == "x"
        assert du.definitions["x"][0].line > 0

    def test_annotated_assignment(self) -> None:
        func = _parse_func(
            """
            def fn():
                x: int = 1
        """
        )
        du = _build_def_use_map(func)
        assert "x" in du.definitions

    def test_augmented_assignment(self) -> None:
        func = _parse_func(
            """
            def fn():
                x = 0
                x += 1
        """
        )
        du = _build_def_use_map(func)
        assert len(du.definitions["x"]) == 2

    def test_tuple_unpacking(self) -> None:
        func = _parse_func(
            """
            def fn():
                a, b = (1, 2)
        """
        )
        du = _build_def_use_map(func)
        assert "a" in du.definitions
        assert "b" in du.definitions
        # Tuple unpack gives value_node=None for each element
        assert du.definitions["a"][0].value_node is None

    def test_walrus_operator(self) -> None:
        func = _parse_func(
            """
            def fn():
                if (x := compute()):
                    pass
        """
        )
        du = _build_def_use_map(func)
        assert "x" in du.definitions

    def test_for_target(self) -> None:
        func = _parse_func(
            """
            def fn():
                for i in items:
                    pass
        """
        )
        du = _build_def_use_map(func)
        assert "i" in du.definitions

    def test_with_as_variable(self) -> None:
        func = _parse_func(
            """
            def fn():
                with open("f") as handle:
                    pass
        """
        )
        du = _build_def_use_map(func)
        assert "handle" in du.definitions

    def test_nested_function_skipped(self) -> None:
        func = _parse_func(
            """
            def fn():
                x = 1
                def inner():
                    y = 2
                return x
        """
        )
        du = _build_def_use_map(func)
        assert "x" in du.definitions
        assert "y" not in du.definitions

    def test_function_parameters_recorded(self) -> None:
        func = _parse_func(
            """
            def fn(a, b, *args, c=1, **kwargs):
                pass
        """
        )
        du = _build_def_use_map(func)
        assert du.func_params == {"a", "b", "args", "c", "kwargs"}


# ---------------------------------------------------------------------------
# _SinkFinder
# ---------------------------------------------------------------------------


class TestSinkFinder:
    def _find_sinks(self, src: str, func_name: str = "fn", **kwargs):
        func = _parse_func(src, func_name)
        du = _build_def_use_map(func)
        sink_methods = {sp.method_name: sp for sp in DEFAULT_SINK_PATTERNS}
        finder = _SinkFinder(
            sink_methods=sink_methods,
            constructor_hints=kwargs.get("constructor_hints", CONSTRUCTOR_CLASS_HINTS),
            def_use=du,
        )
        finder.visit(func)
        return finder.sink_args

    def test_llm_invoke_detected(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                llm.invoke(prompt)
        """
        )
        names = [s.var_name for s in sinks]
        assert "prompt" in names

    def test_chained_create_detected(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                client.chat.completions.create(model="gpt-4", temperature=0.7)
        """
        )
        # Should detect literal kwargs
        params = [s.param_name for s in sinks if s.param_name]
        assert "model" in params
        assert "temperature" in params

    def test_module_level_completion_detected(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                result = completion(model=m, temperature=t)
        """
        )
        names = [s.var_name for s in sinks]
        assert "m" in names
        assert "t" in names

    def test_retrieval_search_detected(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                docs = db.similarity_search(query, k=5)
        """
        )
        names = [s.var_name for s in sinks]
        assert "query" in names

    def test_embedding_detected(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                vec = embedder.embed_query(text)
        """
        )
        names = [s.var_name for s in sinks]
        assert "text" in names

    def test_non_sink_ignored(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                result = obj.process(data)
        """
        )
        assert sinks == [], f"process() should not be a sink; got {sinks}"

    def test_constructor_kwargs_traced(self) -> None:
        sinks = self._find_sinks(
            """
            def fn():
                llm = ChatOpenAI(temperature=0.7, model="gpt-4")
                result = llm.invoke(prompt)
        """
        )
        # The constructor kwargs should appear as sink arguments
        params = [s.param_name for s in sinks]
        assert "temperature" in params
        assert "model" in params


# ---------------------------------------------------------------------------
# Constructor tracing
# ---------------------------------------------------------------------------


class TestConstructorTracing:
    @pytest.fixture
    def strategy(self) -> DataFlowDetectionStrategy:
        return DataFlowDetectionStrategy()

    def test_constructor_literal_kwargs(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                llm = ChatOpenAI(temperature=0.7, model="gpt-4")
                result = llm.invoke("hello")
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        names = _names(candidates)
        assert "temperature" in names, f"temperature not found; got {names}"
        assert "model" in names, f"model not found; got {names}"

    def test_constructor_variable_arg_traced(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                t = 0.7
                llm = ChatOpenAI(temperature=t)
                result = llm.invoke("hello")
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "t")
        assert c is not None, f"t not detected; got {_names(candidates)}"
        assert c.current_value == pytest.approx(0.7)

    def test_unknown_constructor_ignored(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                obj = MyUnknownClass(temperature=0.7)
                result = obj.invoke("hello")
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        # "temperature" may still be detected via the invoke() call
        # but NOT via constructor tracing (MyUnknownClass is not in hints)
        ctor_reasoned = [
            c for c in candidates if c.reasoning and "__init__" in c.reasoning
        ]
        assert (
            ctor_reasoned == []
        ), "Unknown class should not trigger constructor tracing"


# ---------------------------------------------------------------------------
# Backward slicing
# ---------------------------------------------------------------------------


class TestBackwardSlicing:
    @pytest.fixture
    def strategy(self) -> DataFlowDetectionStrategy:
        return DataFlowDetectionStrategy()

    def test_direct_literal_kwarg_high_confidence(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                result = llm.invoke(temperature=0.7)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "temperature")
        assert c is not None, f"temperature not detected; got {_names(candidates)}"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.7)

    def test_one_hop_variable_high_confidence(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                t = 0.7
                result = llm.invoke(temperature=t)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "t")
        assert c is not None, f"t not detected; got {_names(candidates)}"
        assert c.confidence == DetectionConfidence.HIGH
        assert c.current_value == pytest.approx(0.7)

    def test_two_hop_transitive_medium_confidence(self, strategy) -> None:
        # t at hop 0, y at hop 1, x at hop 2 → MEDIUM
        src = _dedent(
            """
            def fn():
                x = 0.7
                y = x
                t = y
                result = llm.invoke(temperature=t)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "x")
        assert c is not None, f"x not detected; got {_names(candidates)}"
        assert c.confidence == DetectionConfidence.MEDIUM

    def test_function_param_medium_confidence(self, strategy) -> None:
        src = _dedent(
            """
            def fn(temp):
                result = llm.invoke(temperature=temp)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "temp")
        assert c is not None, f"temp not detected; got {_names(candidates)}"
        assert c.confidence == DetectionConfidence.MEDIUM

    def test_subscript_traces_base(self, strategy) -> None:
        # config is assigned from an external variable, so backward walk
        # traces through the subscript base to find the literal origin
        src = _dedent(
            """
            def fn():
                temp_val = 0.7
                config = {"temperature": temp_val}
                t = config["temperature"]
                result = llm.invoke(temperature=t)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        # temp_val should be traced via config dict → subscript → invoke
        names = _names(candidates)
        assert "temp_val" in names, f"temp_val not traced; got {names}"

    def test_call_traces_args(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                x = 0.7
                t = float(x)
                result = llm.invoke(temperature=t)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "x")
        assert c is not None, f"x not traced through call; got {_names(candidates)}"

    def test_fstring_detects_variables(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                topic = "science"
                prompt = f"Tell me about {topic}"
                result = llm.invoke(prompt)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "topic")
        assert (
            c is not None
        ), f"topic not detected in f-string; got {_names(candidates)}"

    def test_string_concat_detects_both(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                prefix = "Answer: "
                query = "what is AI?"
                prompt = prefix + query
                result = llm.invoke(prompt)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        names = _names(candidates)
        assert "prefix" in names, f"prefix not detected; got {names}"
        assert "query" in names, f"query not detected; got {names}"

    def test_max_hops_respected(self) -> None:
        strategy = DataFlowDetectionStrategy(max_hops=1)
        src = _dedent(
            """
            def fn():
                a = 0.7
                b = a
                c = b
                result = llm.invoke(temperature=c)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        names = _names(candidates)
        # With max_hops=1, only c (hop=0) and b (hop=1) should be visited.
        # a is at hop=2, beyond max_hops=1.
        assert "a" not in names, f"a should not be reached with max_hops=1; got {names}"

    def test_no_infinite_loop_on_self_reference(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                x = 1
                x = x + 1
                result = llm.invoke(temperature=x)
                return result
        """
        )
        # Should terminate without hanging
        candidates = strategy.detect(src, "fn")
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# DataFlowDetectionStrategy — Integration
# ---------------------------------------------------------------------------


class TestDataFlowDetectionStrategy:
    @pytest.fixture
    def strategy(self) -> DataFlowDetectionStrategy:
        return DataFlowDetectionStrategy()

    def test_langchain_pattern(self, strategy) -> None:
        src = _dedent(
            """
            def answer(question):
                llm = ChatOpenAI(temperature=0.7, model="gpt-4")
                prompt = f"Answer: {question}"
                response = llm.invoke(prompt)
                return response.content
        """
        )
        candidates = strategy.detect(src, "answer")
        names = _names(candidates)
        assert "temperature" in names
        assert "model" in names
        assert "question" in names or "prompt" in names

    def test_openai_sdk_pattern(self, strategy) -> None:
        src = _dedent(
            """
            def chat(msg):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": msg}],
                    temperature=0.5,
                )
                return response
        """
        )
        candidates = strategy.detect(src, "chat")
        names = _names(candidates)
        assert "temperature" in names or "model" in names

    def test_rag_pattern(self, strategy) -> None:
        src = _dedent(
            """
            def answer(question):
                docs = retriever.similarity_search(question, k=5)
                context = "\\n".join(d.page_content for d in docs)
                prompt = f"Context: {context}\\nQuestion: {question}"
                response = llm.invoke(prompt)
                return response
        """
        )
        candidates = strategy.detect(src, "answer")
        names = _names(candidates)
        # question flows to both similarity_search AND llm.invoke
        assert "question" in names, f"question not detected; got {names}"

    def test_empty_function_returns_empty(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                pass
        """
        )
        candidates = strategy.detect(src, "fn")
        assert candidates == []

    def test_no_sinks_returns_empty(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                x = 42
                y = x + 1
                return y
        """
        )
        candidates = strategy.detect(src, "fn")
        assert candidates == []

    def test_existing_tvars_filtered(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                t = 0.7
                m = "gpt-4"
                result = llm.invoke(temperature=t, model=m)
                return result
        """
        )
        candidates = strategy.detect(src, "fn", context={"existing_tvars": {"t"}})
        names = _names(candidates)
        assert "t" not in names, "existing tvar should be filtered"
        assert "m" in names, "non-existing tvar should still be detected"

    def test_dedup_keeps_highest_confidence(self, strategy) -> None:
        # A variable detected via both direct kwarg and transitive path
        # should keep the higher confidence.
        src = _dedent(
            """
            def fn():
                t = 0.7
                result = llm.invoke(temperature=t)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        t_candidates = [c for c in candidates if c.name == "t"]
        assert len(t_candidates) == 1, "Should be deduplicated to 1"

    def test_syntax_error_returns_empty(self, strategy) -> None:
        candidates = strategy.detect("def broken(: pass", "broken")
        assert candidates == []

    def test_function_not_found_returns_empty(self, strategy) -> None:
        src = _dedent(
            """
            def actual():
                llm.invoke(temperature=0.7)
        """
        )
        candidates = strategy.detect(src, "nonexistent")
        assert candidates == []

    def test_async_function_supported(self, strategy) -> None:
        src = _dedent(
            """
            async def fn():
                result = await llm.ainvoke(temperature=0.7)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        names = _names(candidates)
        assert "temperature" in names

    def test_detection_source_is_dataflow(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                t = 0.7
                llm.invoke(temperature=t)
        """
        )
        candidates = strategy.detect(src, "fn")
        for c in candidates:
            assert c.detection_source == "dataflow"

    def test_reasoning_includes_sink_info(self, strategy) -> None:
        src = _dedent(
            """
            def fn():
                t = 0.7
                llm.invoke(temperature=t)
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "t")
        assert c is not None
        assert (
            "invoke" in c.reasoning
        ), f"Reasoning should mention sink; got {c.reasoning!r}"


# ---------------------------------------------------------------------------
# _hop_to_confidence
# ---------------------------------------------------------------------------


class TestHopToConfidence:
    def test_hop_0_is_high(self) -> None:
        assert _hop_to_confidence(0) == DetectionConfidence.HIGH

    def test_hop_1_is_high(self) -> None:
        assert _hop_to_confidence(1) == DetectionConfidence.HIGH

    def test_hop_2_is_medium(self) -> None:
        assert _hop_to_confidence(2) == DetectionConfidence.MEDIUM

    def test_hop_3_is_medium(self) -> None:
        assert _hop_to_confidence(3) == DetectionConfidence.MEDIUM

    def test_hop_4_is_low(self) -> None:
        assert _hop_to_confidence(4) == DetectionConfidence.LOW


# ---------------------------------------------------------------------------
# Regression tests for Codex review fixes
# ---------------------------------------------------------------------------


class TestCodexReviewFixes:
    """Tests for bugs identified in the Codex code review."""

    @pytest.fixture()
    def strategy(self) -> DataFlowDetectionStrategy:
        return DataFlowDetectionStrategy()

    def test_generic_bare_name_not_matched(self, strategy) -> None:
        """P1: Bare `run(cmd)` should NOT be a sink (too generic)."""
        src = _dedent(
            """
            def fn():
                cmd = "ls -la"
                result = run(cmd)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        assert (
            candidates == []
        ), f"Bare run() should not be a sink; got {_names(candidates)}"

    def test_generic_attribute_still_matched(self, strategy) -> None:
        """P1: `obj.run(cmd)` should still be a sink (attribute context)."""
        src = _dedent(
            """
            def fn():
                prompt = "hello"
                result = chain.run(prompt)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        names = _names(candidates)
        assert "prompt" in names, f"chain.run() should be a sink; got {names}"

    def test_generic_query_bare_not_matched(self, strategy) -> None:
        """P1: Bare `query(sql)` should NOT be a sink."""
        src = _dedent(
            """
            def fn():
                sql = "SELECT 1"
                result = query(sql)
                return result
        """
        )
        candidates = strategy.detect(src, "fn")
        assert (
            candidates == []
        ), f"Bare query() should not be a sink; got {_names(candidates)}"

    def test_bfs_shortest_hop_wins(self, strategy) -> None:
        """P2: BFS ensures shortest hop is processed first.

        x is reachable at hop 0 via model=x and at hop 2 via temperature=y -> z -> x.
        With BFS, x should get HIGH confidence (hop 0), not MEDIUM (hop 2).
        """
        src = _dedent(
            """
            def fn():
                x = 0.7
                z = x
                y = z
                llm.invoke(model=x, temperature=y)
        """
        )
        candidates = strategy.detect(src, "fn")
        c = _by_name(candidates, "x")
        assert c is not None, f"x not detected; got {_names(candidates)}"
        assert (
            c.confidence == DetectionConfidence.HIGH
        ), f"x should be HIGH (hop 0 via model=x), got {c.confidence}"

    def test_existing_tvars_filters_literal_kwargs(self, strategy) -> None:
        """P3: existing_tvars should filter literal kwargs too."""
        src = _dedent(
            """
            def fn():
                result = llm.invoke(temperature=0.7, model="gpt-4")
                return result
        """
        )
        candidates = strategy.detect(
            src, "fn", context={"existing_tvars": {"temperature"}}
        )
        names = _names(candidates)
        assert (
            "temperature" not in names
        ), f"Literal kwarg 'temperature' should be filtered; got {names}"
        assert "model" in names, f"model should still be detected; got {names}"
