"""Unit tests for the DeepEval metric integration adapter."""

from __future__ import annotations

import copy
import inspect
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock DeepEval so tests don't require the real package
# ---------------------------------------------------------------------------


def _make_mock_metric(score: float = 0.85, name: str | None = None) -> MagicMock:
    """Create a mock DeepEval metric instance."""
    metric = MagicMock()
    metric.score = score
    if name is not None:
        metric.name = name
    else:
        # Remove the .name attribute so getattr falls back
        del metric.name
    metric.measure = MagicMock()
    return metric


def _make_mock_deepeval_modules() -> dict[str, ModuleType]:
    """Build mock modules for deepeval.metrics and deepeval.test_case."""
    # deepeval.test_case
    test_case_mod = ModuleType("deepeval.test_case")
    test_case_mod.LLMTestCase = MagicMock(name="LLMTestCase")  # type: ignore[attr-defined]

    # deepeval.metrics — create mock classes for each metric type
    metrics_mod = ModuleType("deepeval.metrics")
    for class_name in [
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "HallucinationMetric",
        "ToxicityMetric",
        "BiasMetric",
        "ContextualRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "SummarizationMetric",
        "GEval",
    ]:
        mock_cls = MagicMock(name=class_name)
        # Instances returned by the class should have .measure() and .score
        mock_instance = MagicMock()
        mock_instance.score = 0.85
        mock_instance.measure = MagicMock()
        mock_cls.return_value = mock_instance
        mock_cls.__name__ = class_name
        setattr(metrics_mod, class_name, mock_cls)

    # deepeval package itself
    deepeval_mod = ModuleType("deepeval")

    return {
        "deepeval": deepeval_mod,
        "deepeval.metrics": metrics_mod,
        "deepeval.test_case": test_case_mod,
    }


@pytest.fixture()
def mock_deepeval():
    """Patch sys.modules so deepeval_metrics imports our mocks."""
    mocks = _make_mock_deepeval_modules()

    with patch.dict(sys.modules, mocks):
        # Force reimport so the module picks up our mocks
        import importlib

        # Clear cached shortcut classes
        import traigent.metrics.deepeval_metrics as dm

        dm._SHORTCUT_TO_CLASS = None
        importlib.reload(dm)
        yield dm, mocks

    # Reset after test
    dm._SHORTCUT_TO_CLASS = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Test behaviour when deepeval is not installed."""

    def test_not_installed_raises_import_error(self):
        """DeepEvalScorer() should raise ImportError with install hint."""
        with patch.dict(
            sys.modules,
            {"deepeval": None, "deepeval.metrics": None, "deepeval.test_case": None},
        ):
            import importlib

            import traigent.metrics.deepeval_metrics as dm

            dm._SHORTCUT_TO_CLASS = None
            importlib.reload(dm)

            assert dm.DEEPEVAL_AVAILABLE is False
            with pytest.raises(ImportError, match="pip install.*traigent.*deepeval"):
                dm.DeepEvalScorer(["relevancy"])

            dm._SHORTCUT_TO_CLASS = None

    def test_import_error_is_surfaced(self):
        """The original import error should appear in the message."""
        with patch.dict(
            sys.modules,
            {"deepeval": None, "deepeval.metrics": None, "deepeval.test_case": None},
        ):
            import importlib

            import traigent.metrics.deepeval_metrics as dm

            dm._SHORTCUT_TO_CLASS = None
            importlib.reload(dm)

            # The error message should mention the original error if present
            if dm.DEEPEVAL_IMPORT_ERROR is not None:
                with pytest.raises(ImportError, match="Original error"):
                    dm.DeepEvalScorer(["relevancy"])

            dm._SHORTCUT_TO_CLASS = None


class TestShortcutResolution:
    """Test string shortcut → metric class resolution."""

    def test_relevancy_shortcut(self, mock_deepeval):
        dm, mocks = mock_deepeval
        scorer = dm.DeepEvalScorer(["relevancy"], model="gpt-4o-mini", threshold=0.7)
        funcs = scorer.to_metric_functions()

        assert "relevancy" in funcs
        # Verify the class was instantiated with correct kwargs
        metrics_mod = mocks["deepeval.metrics"]
        metrics_mod.AnswerRelevancyMetric.assert_called_once_with(
            threshold=0.7, model="gpt-4o-mini"
        )

    def test_answer_relevancy_alias(self, mock_deepeval):
        dm, mocks = mock_deepeval
        scorer = dm.DeepEvalScorer(["answer_relevancy"])
        funcs = scorer.to_metric_functions()

        assert "answer_relevancy" in funcs
        mocks["deepeval.metrics"].AnswerRelevancyMetric.assert_called_once()

    def test_unknown_shortcut_raises(self, mock_deepeval):
        dm, _ = mock_deepeval
        with pytest.raises(ValueError, match="Unknown DeepEval metric shortcut"):
            dm.DeepEvalScorer(["nonexistent_metric"])

    def test_unknown_shortcut_lists_available(self, mock_deepeval):
        dm, _ = mock_deepeval
        with pytest.raises(ValueError, match="relevancy"):
            dm.DeepEvalScorer(["bad_metric"])

    def test_case_insensitive(self, mock_deepeval):
        dm, _ = mock_deepeval
        scorer = dm.DeepEvalScorer(["Relevancy"])
        funcs = scorer.to_metric_functions()
        assert "relevancy" in funcs

    def test_no_model_kwarg_when_none(self, mock_deepeval):
        dm, mocks = mock_deepeval
        dm.DeepEvalScorer(["faithfulness"])
        call_kwargs = mocks["deepeval.metrics"].FaithfulnessMetric.call_args[1]
        assert "model" not in call_kwargs
        assert call_kwargs["threshold"] == 0.5


class TestPreConfiguredInstances:
    """Test passing pre-configured metric instances."""

    def test_instance_with_name_attr(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(name="custom_quality")
        scorer = dm.DeepEvalScorer([metric])
        funcs = scorer.to_metric_functions()
        assert "custom_quality" in funcs

    def test_instance_without_name_derives_from_class(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric()
        # MagicMock class name → derived snake_case
        type(metric).__name__ = "AnswerRelevancyMetric"
        scorer = dm.DeepEvalScorer([metric])
        funcs = scorer.to_metric_functions()
        assert "answer_relevancy" in funcs


class TestFieldMapping:
    """Test that Traigent fields map correctly to LLMTestCase."""

    def test_basic_mapping(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(score=0.9, name="test_metric")
        scorer = dm.DeepEvalScorer([metric])
        funcs = scorer.to_metric_functions()
        fn = funcs["test_metric"]

        fn(
            output="The answer is Paris",
            expected="Paris",
            input_data={"question": "What is the capital of France?"},
            metadata={},
        )

        # Verify LLMTestCase was called with correct fields
        tc_cls = mocks["deepeval.test_case"].LLMTestCase
        tc_cls.assert_called_once()
        call_kwargs = tc_cls.call_args[1]
        assert call_kwargs["input"] == "What is the capital of France?"
        assert call_kwargs["actual_output"] == "The answer is Paris"
        assert call_kwargs["expected_output"] == "Paris"

    def test_context_from_metadata(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="ctx_metric")
        scorer = dm.DeepEvalScorer([metric])
        fn = scorer.to_metric_functions()["ctx_metric"]

        fn(
            output="answer",
            input_data={"question": "q"},
            metadata={"context": ["doc1", "doc2"]},
        )

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["context"] == ["doc1", "doc2"]

    def test_context_from_input_data(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="ctx_metric")
        scorer = dm.DeepEvalScorer([metric])
        fn = scorer.to_metric_functions()["ctx_metric"]

        fn(
            output="answer",
            input_data={"question": "q", "context": ["doc1"]},
            metadata={},
        )

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["context"] == ["doc1"]

    def test_retrieval_context(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="ret_metric")
        scorer = dm.DeepEvalScorer([metric])
        fn = scorer.to_metric_functions()["ret_metric"]

        fn(
            output="answer",
            input_data={"question": "q"},
            metadata={"retrieval_context": ["ret1"]},
        )

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["retrieval_context"] == ["ret1"]

    def test_no_context_omitted(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="no_ctx")
        scorer = dm.DeepEvalScorer([metric])
        fn = scorer.to_metric_functions()["no_ctx"]

        fn(output="answer", input_data={"question": "q"}, metadata={})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert "context" not in call_kwargs
        assert "retrieval_context" not in call_kwargs

    def test_expected_none_omitted(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="no_exp")
        scorer = dm.DeepEvalScorer([metric])
        fn = scorer.to_metric_functions()["no_exp"]

        fn(output="answer", input_data={"question": "q"})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert "expected_output" not in call_kwargs


class TestDefensiveInputHandling:
    """Test that various input_data types are handled gracefully."""

    def test_input_data_none(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="answer", input_data=None)

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["input"] == ""

    def test_input_data_raw_string(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="answer", input_data="raw question")

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["input"] == "raw question"

    def test_input_data_dict_with_input_key(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="answer", input_data={"input": "from input key"})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["input"] == "from input key"

    def test_input_data_dict_with_prompt_key(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="answer", input_data={"prompt": "from prompt key"})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["input"] == "from prompt key"

    def test_output_none(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output=None, input_data={"question": "q"})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["actual_output"] == ""


class TestContextCoercion:
    """Test that context values are properly coerced to list[str]."""

    def test_string_context_to_list(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="a", metadata={"context": "single doc"})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["context"] == ["single doc"]

    def test_tuple_context_to_list(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="a", metadata={"context": ("doc1", "doc2")})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["context"] == ["doc1", "doc2"]

    def test_non_string_elements_coerced(self, mock_deepeval):
        dm, mocks = mock_deepeval
        metric = _make_mock_metric(name="m")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["m"]
        fn(output="a", metadata={"context": [1, 2, 3]})

        call_kwargs = mocks["deepeval.test_case"].LLMTestCase.call_args[1]
        assert call_kwargs["context"] == ["1", "2", "3"]


class TestScoreExtraction:
    """Test that metric scores are correctly extracted."""

    def test_score_returned(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(score=0.85, name="scored")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["scored"]
        result = fn(output="answer", input_data={"question": "q"})
        assert result == 0.85

    def test_score_converted_to_float(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(score=1, name="int_score")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["int_score"]
        result = fn(output="answer", input_data={"question": "q"})
        assert isinstance(result, float)
        assert result == 1.0


class TestSignatureCompatibility:
    """Test that generated functions have the correct signature for _invoke_metric_function."""

    def test_parameter_names(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(name="sig_test")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["sig_test"]

        params = list(inspect.signature(fn).parameters.keys())
        assert "output" in params
        assert "expected" in params
        assert "input_data" in params
        assert "metadata" in params

    def test_all_params_have_defaults_except_output(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(name="sig_test")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["sig_test"]

        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            if name == "output":
                continue
            assert (
                param.default is not inspect.Parameter.empty
            ), f"Parameter {name!r} should have a default value"


class TestThreadSafety:
    """Test the deepcopy fallback chain."""

    def test_deepcopy_is_used(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(score=0.5, name="cp")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["cp"]

        with patch.object(copy, "deepcopy", wraps=copy.deepcopy) as mock_dc:
            fn(output="answer", input_data={"question": "q"})
            mock_dc.assert_called_once()

    def test_fallback_to_copy_on_deepcopy_failure(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(score=0.5, name="cp")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["cp"]

        with (
            patch.object(copy, "deepcopy", side_effect=TypeError("not copyable")),
            patch.object(copy, "copy", wraps=copy.copy) as mock_cp,
        ):
            fn(output="answer", input_data={"question": "q"})
            mock_cp.assert_called_once()

    def test_raises_type_error_when_copy_impossible(self, mock_deepeval):
        dm, _ = mock_deepeval
        metric = _make_mock_metric(score=0.5, name="cp")
        fn = dm.DeepEvalScorer([metric]).to_metric_functions()["cp"]

        with (
            patch.object(copy, "deepcopy", side_effect=TypeError),
            patch.object(copy, "copy", side_effect=TypeError),
            pytest.raises(TypeError, match="cannot be copied"),
        ):
            fn(output="answer", input_data={"question": "q"})


class TestNamingCollisions:
    """Test handling of duplicate metric names."""

    def test_two_same_type_get_suffix(self, mock_deepeval):
        dm, _ = mock_deepeval
        m1 = _make_mock_metric(score=0.8, name="relevancy")
        m2 = _make_mock_metric(score=0.9, name="relevancy")
        scorer = dm.DeepEvalScorer([m1, m2])
        funcs = scorer.to_metric_functions()
        assert "relevancy" in funcs
        assert "relevancy_2" in funcs

    def test_three_same_type_get_incremental_suffix(self, mock_deepeval):
        dm, _ = mock_deepeval
        metrics = [_make_mock_metric(name="bias") for _ in range(3)]
        scorer = dm.DeepEvalScorer(metrics)
        funcs = scorer.to_metric_functions()
        assert "bias" in funcs
        assert "bias_2" in funcs
        assert "bias_3" in funcs


class TestEmptyMetrics:
    """Test edge cases."""

    def test_empty_list_raises(self, mock_deepeval):
        dm, _ = mock_deepeval
        with pytest.raises(ValueError, match="non-empty"):
            dm.DeepEvalScorer([])

    def test_multiple_metrics(self, mock_deepeval):
        dm, _ = mock_deepeval
        scorer = dm.DeepEvalScorer(["relevancy", "faithfulness", "toxicity"])
        funcs = scorer.to_metric_functions()
        assert len(funcs) == 3
        assert set(funcs.keys()) == {"relevancy", "faithfulness", "toxicity"}
