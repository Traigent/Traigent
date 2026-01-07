"""Tests for TunedCallable and built-in callables."""

from __future__ import annotations

import pytest


class TestTunedCallable:
    """Tests for TunedCallable composition pattern."""

    def test_basic_creation(self):
        """Test creating a basic TunedCallable."""
        from traigent_tuned_variables import TunedCallable

        def func_a():
            return "a"

        def func_b():
            return "b"

        tc = TunedCallable(
            name="my_callable",
            callables={"a": func_a, "b": func_b},
        )

        assert tc.name == "my_callable"
        assert len(tc) == 2
        assert "a" in tc
        assert "b" in tc

    def test_as_choices(self):
        """Test converting to Choices for config space."""
        from traigent_tuned_variables import TunedCallable

        tc = TunedCallable(
            name="strategy",
            callables={"fast": lambda: 1, "slow": lambda: 2},
        )

        choices = tc.as_choices()

        assert choices.name == "strategy"
        assert set(choices.values) == {"fast", "slow"}

    def test_get_callable(self):
        """Test getting callable by name."""
        from traigent_tuned_variables import TunedCallable

        def my_func():
            return 42

        tc = TunedCallable(
            name="test",
            callables={"my_func": my_func},
        )

        retrieved = tc.get_callable("my_func")
        assert retrieved is my_func
        assert retrieved() == 42

    def test_get_callable_not_found(self):
        """Test getting non-existent callable raises KeyError."""
        from traigent_tuned_variables import TunedCallable

        tc = TunedCallable(name="test", callables={"a": lambda: 1})

        with pytest.raises(KeyError, match="not found"):
            tc.get_callable("nonexistent")

    def test_invoke(self):
        """Test invoking callable by name."""
        from traigent_tuned_variables import TunedCallable

        def add(x, y):
            return x + y

        tc = TunedCallable(name="math", callables={"add": add})

        result = tc.invoke("add", 2, 3)
        assert result == 5

    def test_get_parameters(self):
        """Test getting per-callable parameters."""
        from traigent_tuned_variables import TunedCallable

        from traigent.api.parameter_ranges import Range

        tc = TunedCallable(
            name="retriever",
            callables={"mmr": lambda: None},
            parameters={
                "mmr": {"lambda_mult": Range(0.0, 1.0)},
            },
        )

        params = tc.get_parameters("mmr")
        assert "lambda_mult" in params
        assert params["lambda_mult"].low == 0.0
        assert params["lambda_mult"].high == 1.0

    def test_get_parameters_empty(self):
        """Test getting parameters for callable without any."""
        from traigent_tuned_variables import TunedCallable

        tc = TunedCallable(
            name="test",
            callables={"simple": lambda: 1},
        )

        params = tc.get_parameters("simple")
        assert params == {}

    def test_get_full_space(self):
        """Test getting full configuration space including dependent params."""
        from traigent_tuned_variables import TunedCallable

        from traigent.api.parameter_ranges import Range

        tc = TunedCallable(
            name="retriever",
            callables={"similarity": lambda: None, "mmr": lambda: None},
            parameters={
                "mmr": {"lambda_mult": Range(0.0, 1.0, name="lambda_mult")},
            },
        )

        space = tc.get_full_space()

        # Should have main choice and conditional param
        assert "retriever" in space
        assert "retriever.mmr.lambda_mult" in space
        assert set(space["retriever"].values) == {"similarity", "mmr"}

    def test_extract_callable_params(self):
        """Test extracting params for selected callable from config."""
        from traigent_tuned_variables import TunedCallable

        tc = TunedCallable(
            name="retriever",
            callables={"similarity": lambda: None, "mmr": lambda: None},
        )

        config = {
            "retriever": "mmr",
            "retriever.mmr.lambda_mult": 0.7,
            "retriever.similarity.threshold": 0.5,  # Should be ignored
        }

        params = tc.extract_callable_params(config)

        assert params == {"lambda_mult": 0.7}

    def test_register(self):
        """Test registering new callable."""
        from traigent_tuned_variables import TunedCallable

        from traigent.api.parameter_ranges import IntRange

        tc = TunedCallable(name="funcs", callables={})

        tc.register("new_func", lambda: 1, parameters={"k": IntRange(1, 10)})

        assert "new_func" in tc
        assert "k" in tc.get_parameters("new_func")

    def test_iteration(self):
        """Test iterating over callable names."""
        from traigent_tuned_variables import TunedCallable

        tc = TunedCallable(
            name="test",
            callables={"a": lambda: 1, "b": lambda: 2, "c": lambda: 3},
        )

        names = list(tc)
        assert set(names) == {"a", "b", "c"}


class TestContextFormatters:
    """Tests for built-in context formatters."""

    def test_bullet_format(self):
        """Test bullet formatting."""
        from traigent_tuned_variables import ContextFormatters

        docs = [
            {"content": "First doc"},
            {"content": "Second doc"},
        ]

        result = ContextFormatters.bullet(docs)

        assert "• First doc" in result
        assert "• Second doc" in result

    def test_numbered_format(self):
        """Test numbered formatting."""
        from traigent_tuned_variables import ContextFormatters

        docs = [
            {"content": "First"},
            {"content": "Second"},
        ]

        result = ContextFormatters.numbered(docs)

        assert "1. First" in result
        assert "2. Second" in result

    def test_xml_format(self):
        """Test XML formatting."""
        from traigent_tuned_variables import ContextFormatters

        docs = [{"content": "Content here"}]

        result = ContextFormatters.xml(docs)

        assert '<document id="1">' in result
        assert "Content here" in result
        assert "</document>" in result

    def test_as_choices(self):
        """Test getting formatters as Choices."""
        from traigent_tuned_variables import ContextFormatters

        choices = ContextFormatters.as_choices()

        assert choices.name == "context_format"
        assert "bullet" in choices.values
        assert "xml" in choices.values
        assert "json" in choices.values

    def test_invoke(self):
        """Test invoking formatter by name."""
        from traigent_tuned_variables import ContextFormatters

        docs = [{"content": "Test"}]

        result = ContextFormatters.invoke("bullet", docs)
        assert "• Test" in result


class TestRetrievers:
    """Tests for built-in retrievers."""

    def test_as_choices(self):
        """Test getting retrievers as Choices."""
        from traigent_tuned_variables import Retrievers

        choices = Retrievers.as_choices()

        assert choices.name == "retriever"
        assert "similarity" in choices.values
        assert "mmr" in choices.values

    def test_as_tuned_callable(self):
        """Test getting retrievers as TunedCallable."""
        from traigent_tuned_variables import Retrievers

        tc = Retrievers.as_tuned_callable()

        assert tc.name == "retriever"
        assert "similarity" in tc
        assert "mmr" in tc

        # MMR should have lambda_mult parameter
        mmr_params = tc.get_parameters("mmr")
        assert "lambda_mult" in mmr_params
