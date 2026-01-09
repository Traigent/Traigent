"""Tests for DSPy integration adapter.

These tests verify the DSPy adapter functionality without requiring
DSPy to be installed, using mocks where appropriate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Helper metric functions to avoid lambda assignments (ruff E731)
def _always_one_metric(example: Any, pred: Any) -> float:
    """Metric that always returns 1.0."""
    return 1.0


def _always_half_metric(example: Any, pred: Any) -> float:
    """Metric that always returns 0.5."""
    return 0.5


def _always_high_metric(example: Any, pred: Any) -> float:
    """Metric that always returns 0.8."""
    return 0.8


class TestDSPyAvailabilityDetection:
    """Test DSPy availability detection."""

    def test_dspy_available_flag_when_not_installed(self) -> None:
        """DSPY_AVAILABLE should be False when dspy is not installed."""
        # The actual value depends on whether dspy is installed in the test env
        from traigent.integrations.dspy_adapter import DSPY_AVAILABLE

        # This test just verifies the flag exists and is a boolean
        assert isinstance(DSPY_AVAILABLE, bool)

    def test_import_error_message_when_dspy_not_available(self) -> None:
        """DSPyPromptOptimizer should raise ImportError with helpful message."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", False):
            # Need to reload to pick up the patched value
            from traigent.integrations import dspy_adapter

            # Temporarily override the module-level check
            original_available = dspy_adapter.DSPY_AVAILABLE
            dspy_adapter.DSPY_AVAILABLE = False

            try:
                with pytest.raises(ImportError) as exc_info:
                    dspy_adapter.DSPyPromptOptimizer(method="mipro")

                assert "dspy-ai" in str(exc_info.value).lower()
            finally:
                dspy_adapter.DSPY_AVAILABLE = original_available


class TestPromptOptimizationResult:
    """Test the PromptOptimizationResult dataclass."""

    def test_result_dataclass_creation(self) -> None:
        """PromptOptimizationResult should store all fields correctly."""
        from traigent.integrations.dspy_adapter import PromptOptimizationResult

        mock_module = MagicMock()
        result = PromptOptimizationResult(
            optimized_module=mock_module,
            method="mipro",
            num_demos=4,
            trainset_size=100,
            best_score=0.85,
            metadata={"auto_setting": "medium"},
        )

        assert result.optimized_module is mock_module
        assert result.method == "mipro"
        assert result.num_demos == 4
        assert result.trainset_size == 100
        assert result.best_score == 0.85
        assert result.metadata == {"auto_setting": "medium"}

    def test_result_dataclass_defaults(self) -> None:
        """PromptOptimizationResult should have sensible defaults."""
        from traigent.integrations.dspy_adapter import PromptOptimizationResult

        mock_module = MagicMock()
        result = PromptOptimizationResult(
            optimized_module=mock_module,
            method="bootstrap",
        )

        assert result.num_demos == 0
        assert result.trainset_size == 0
        assert result.best_score is None
        assert result.metadata == {}


class TestDSPyPromptOptimizerInit:
    """Test DSPyPromptOptimizer initialization."""

    @pytest.fixture
    def mock_dspy_available(self):
        """Mock DSPy as available."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy") as mock_dspy:
                yield mock_dspy

    def test_init_with_mipro_method(self, mock_dspy_available) -> None:
        """Should initialize with mipro method."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        optimizer = DSPyPromptOptimizer(method="mipro")
        assert optimizer.method == "mipro"
        assert optimizer.auto_setting == "medium"

    def test_init_with_bootstrap_method(self, mock_dspy_available) -> None:
        """Should initialize with bootstrap method."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        optimizer = DSPyPromptOptimizer(method="bootstrap")
        assert optimizer.method == "bootstrap"

    def test_init_with_custom_auto_setting(self, mock_dspy_available) -> None:
        """Should accept custom auto_setting."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        optimizer = DSPyPromptOptimizer(method="mipro", auto_setting="light")
        assert optimizer.auto_setting == "light"

    def test_init_with_teacher_model(self, mock_dspy_available) -> None:
        """Should accept teacher_model parameter."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        optimizer = DSPyPromptOptimizer(method="mipro", teacher_model="gpt-4")
        assert optimizer.teacher_model == "gpt-4"


class TestCreateDSPyIntegration:
    """Test the factory function."""

    @pytest.fixture
    def mock_dspy_available(self):
        """Mock DSPy as available."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy"):
                yield

    def test_factory_creates_optimizer(self, mock_dspy_available) -> None:
        """create_dspy_integration should return DSPyPromptOptimizer."""
        from traigent.integrations.dspy_adapter import (
            DSPyPromptOptimizer,
            create_dspy_integration,
        )

        optimizer = create_dspy_integration(method="mipro")
        assert isinstance(optimizer, DSPyPromptOptimizer)

    def test_factory_passes_kwargs(self, mock_dspy_available) -> None:
        """create_dspy_integration should pass kwargs to constructor."""
        from traigent.integrations.dspy_adapter import create_dspy_integration

        optimizer = create_dspy_integration(
            method="bootstrap",
            teacher_model="claude-3",
            auto_setting="heavy",
        )
        assert optimizer.method == "bootstrap"
        assert optimizer.teacher_model == "claude-3"
        assert optimizer.auto_setting == "heavy"


class TestCreatePromptChoices:
    """Test the create_prompt_choices class method."""

    def test_returns_choices_when_dspy_unavailable(self) -> None:
        """Should return Choices object with base prompts when DSPy not available."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", False):
            from traigent.api.parameter_ranges import Choices
            from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

            base_prompts = ["Prompt 1: {q}", "Prompt 2: {q}"]
            result = DSPyPromptOptimizer.create_prompt_choices(
                base_prompts=base_prompts,
                trainset=[],
                metric=lambda x, y: 1.0,
            )

            assert isinstance(result, Choices)
            assert list(result.values) == base_prompts
            assert result.name == "prompt"

    def test_returns_list_when_return_choices_false(self) -> None:
        """Should return plain list when return_choices=False."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", False):
            from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

            base_prompts = ["Prompt 1: {q}", "Prompt 2: {q}"]
            result = DSPyPromptOptimizer.create_prompt_choices(
                base_prompts=base_prompts,
                return_choices=False,
            )

            assert result == base_prompts
            assert isinstance(result, list)

    def test_returns_choices_stub_implementation(self) -> None:
        """Current stub implementation returns Choices with base prompts."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            from traigent.api.parameter_ranges import Choices
            from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

            base_prompts = ["Answer: {question}", "Q: {question}\nA:"]
            result = DSPyPromptOptimizer.create_prompt_choices(
                base_prompts=base_prompts,
                trainset=[MagicMock()],
                metric=lambda x, y: 1.0,
            )

            # Current implementation returns Choices with base prompts
            assert isinstance(result, Choices)
            assert list(result.values) == base_prompts
            assert result.name == "prompt"

    def test_custom_name_parameter(self) -> None:
        """Should use custom name when provided."""
        from traigent.api.parameter_ranges import Choices
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        base_prompts = ["Prompt A", "Prompt B"]
        result = DSPyPromptOptimizer.create_prompt_choices(
            base_prompts=base_prompts,
            name="system_prompt",
        )

        assert isinstance(result, Choices)
        assert result.name == "system_prompt"


class TestOptimizePromptMethod:
    """Test the optimize_prompt method with mocked DSPy."""

    @pytest.fixture
    def mock_dspy(self):
        """Create comprehensive DSPy mock."""
        mock = MagicMock()

        # Mock MIPROv2
        mock_mipro_instance = MagicMock()
        mock_mipro_instance.compile.return_value = MagicMock(name="optimized_module")
        mock.MIPROv2.return_value = mock_mipro_instance

        # Mock BootstrapFewShot
        mock_bootstrap_instance = MagicMock()
        mock_bootstrap_instance.compile.return_value = MagicMock(
            name="optimized_module"
        )
        mock.BootstrapFewShot.return_value = mock_bootstrap_instance

        # Mock LM for teacher model
        mock.LM.return_value = MagicMock(name="teacher_lm")

        # Mock context manager
        mock.context.return_value.__enter__ = MagicMock()
        mock.context.return_value.__exit__ = MagicMock()

        return mock

    @pytest.fixture
    def setup_dspy_available(self, mock_dspy):
        """Setup DSPy as available with mock."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                yield mock_dspy

    def test_optimize_prompt_with_mipro(self, setup_dspy_available) -> None:
        """optimize_prompt should work with mipro method."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        mock_dspy = setup_dspy_available
        optimizer = DSPyPromptOptimizer(method="mipro")

        mock_module = MagicMock()
        trainset = [MagicMock(), MagicMock()]

        result = optimizer.optimize_prompt(
            module=mock_module,
            trainset=trainset,
            metric=_always_one_metric,
        )

        # Verify MIPROv2 was called
        mock_dspy.MIPROv2.assert_called_once()
        assert result.method == "mipro"
        assert result.trainset_size == 2

    def test_optimize_prompt_with_bootstrap(self, setup_dspy_available) -> None:
        """optimize_prompt should work with bootstrap method."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        mock_dspy = setup_dspy_available
        optimizer = DSPyPromptOptimizer(method="bootstrap")

        mock_module = MagicMock()
        trainset = [MagicMock(), MagicMock(), MagicMock()]

        result = optimizer.optimize_prompt(
            module=mock_module,
            trainset=trainset,
            metric=_always_half_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=10,
        )

        # Verify BootstrapFewShot was called with correct params
        mock_dspy.BootstrapFewShot.assert_called_once_with(
            metric=_always_half_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=10,
        )
        assert result.method == "bootstrap"
        assert result.trainset_size == 3

    def test_optimize_prompt_with_teacher_model(self, setup_dspy_available) -> None:
        """optimize_prompt should use teacher model when specified."""
        from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

        mock_dspy = setup_dspy_available
        optimizer = DSPyPromptOptimizer(method="mipro", teacher_model="gpt-4")

        mock_module = MagicMock()
        trainset = [MagicMock()]

        optimizer.optimize_prompt(
            module=mock_module,
            trainset=trainset,
            metric=_always_one_metric,
        )

        # Verify LM was created for teacher
        mock_dspy.LM.assert_called_once_with("gpt-4")
        # Verify context was used
        mock_dspy.context.assert_called()

    def test_optimize_prompt_raises_when_dspy_unavailable(self) -> None:
        """optimize_prompt should raise ImportError when DSPy unavailable."""
        from traigent.integrations import dspy_adapter

        original = dspy_adapter.DSPY_AVAILABLE
        dspy_adapter.DSPY_AVAILABLE = False

        try:
            # Create optimizer when DSPy appears available
            with patch.object(dspy_adapter, "DSPY_AVAILABLE", True):
                optimizer = dspy_adapter.DSPyPromptOptimizer(method="mipro")

            # Then make DSPy unavailable for optimize_prompt
            dspy_adapter.DSPY_AVAILABLE = False

            with pytest.raises(ImportError):
                optimizer.optimize_prompt(
                    module=MagicMock(),
                    trainset=[],
                    metric=lambda x, y: 1.0,
                )
        finally:
            dspy_adapter.DSPY_AVAILABLE = original


class TestCompileWithTeacher:
    """Test the _compile_with_teacher helper method."""

    @pytest.fixture
    def mock_dspy(self):
        """Create DSPy mock."""
        mock = MagicMock()
        mock.context.return_value.__enter__ = MagicMock()
        mock.context.return_value.__exit__ = MagicMock()
        return mock

    def test_compile_without_teacher(self, mock_dspy) -> None:
        """Should compile directly without teacher."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="mipro")
                mock_optimizer = MagicMock()
                mock_module = MagicMock()
                trainset = [MagicMock()]

                optimizer._compile_with_teacher(
                    mock_optimizer, mock_module, trainset, teacher=None
                )

                # Should compile directly without context
                mock_optimizer.compile.assert_called_once_with(
                    mock_module, trainset=trainset
                )
                mock_dspy.context.assert_not_called()

    def test_compile_with_teacher(self, mock_dspy) -> None:
        """Should use context manager with teacher."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="mipro")
                mock_optimizer = MagicMock()
                mock_module = MagicMock()
                trainset = [MagicMock()]
                teacher = MagicMock(name="teacher_lm")

                optimizer._compile_with_teacher(
                    mock_optimizer, mock_module, trainset, teacher=teacher
                )

                # Should use context with teacher
                mock_dspy.context.assert_called_once_with(lm=teacher)
                mock_optimizer.compile.assert_called_once()


class TestCountDemos:
    """Test the _count_demos helper method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with mocked DSPy."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy"):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                return DSPyPromptOptimizer(method="mipro")

    def test_count_demos_with_demos(self, optimizer) -> None:
        """Should count demos from module attributes."""

        # Create a simple class that has a predictor with demos
        class MockPredictor:
            demos = [1, 2, 3]  # 3 demos

        class MockModule:
            predict = MockPredictor()

        mock_module = MockModule()
        count = optimizer._count_demos(mock_module)
        assert count == 3

    def test_count_demos_no_demos(self, optimizer) -> None:
        """Should return 0 when no demos present."""
        mock_module = MagicMock(spec=[])  # No attributes with demos

        count = optimizer._count_demos(mock_module)
        assert count == 0

    def test_count_demos_handles_exceptions(self, optimizer) -> None:
        """Should handle exceptions gracefully."""
        mock_module = MagicMock()
        mock_module.__dir__ = MagicMock(side_effect=Exception("Error"))

        # Should not raise, should return 0
        count = optimizer._count_demos(mock_module)
        assert count == 0


class TestComputeBestScore:
    """Test the _compute_best_score helper method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with mocked DSPy."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy"):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                return DSPyPromptOptimizer(method="mipro")

    def test_compute_best_score_with_trainset(self, optimizer) -> None:
        """Should compute average score over trainset."""

        # Create mock examples with inputs method
        class MockExample:
            def __init__(self, question: str, answer: str):
                self._data = {"question": question, "answer": answer}

            def inputs(self):
                return {"question"}

            def items(self):
                return self._data.items()

        trainset = [
            MockExample("Q1", "A1"),
            MockExample("Q2", "A2"),
        ]

        # Mock module that returns predictions
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(answer="A1")

        # Metric that returns 1.0 for first, 0.0 for second
        call_count = [0]

        def metric(example, pred):
            call_count[0] += 1
            return 1.0 if call_count[0] == 1 else 0.0

        score = optimizer._compute_best_score(mock_module, trainset, metric)

        assert score == pytest.approx(0.5)  # Average of 1.0 and 0.0

    def test_compute_best_score_empty_trainset(self, optimizer) -> None:
        """Should return None for empty trainset."""
        score = optimizer._compute_best_score(MagicMock(), [], lambda x, y: 1.0)
        assert score is None

    def test_compute_best_score_handles_exceptions(self, optimizer) -> None:
        """Should return None on exception."""
        mock_module = MagicMock(side_effect=Exception("Error"))

        # Create mock example that will cause exception when accessed
        class BadExample:
            def inputs(self):
                raise ValueError("Bad inputs")

            def items(self):
                return {}.items()

        score = optimizer._compute_best_score(
            mock_module, [BadExample()], lambda x, y: 1.0
        )
        assert score is None


class TestRunMipro:
    """Test the _run_mipro internal method."""

    @pytest.fixture
    def mock_dspy(self):
        """Create DSPy mock for MIPRO tests."""
        mock = MagicMock()
        mock_mipro = MagicMock()
        mock_mipro.compile.return_value = MagicMock(name="mipro_optimized")
        mock.MIPROv2.return_value = mock_mipro
        mock.context.return_value.__enter__ = MagicMock()
        mock.context.return_value.__exit__ = MagicMock()
        return mock

    def test_run_mipro_basic(self, mock_dspy) -> None:
        """_run_mipro should configure and run MIPROv2."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="mipro", auto_setting="heavy")

                _, metadata = optimizer._run_mipro(
                    module=MagicMock(),
                    trainset=[MagicMock()],
                    metric=_always_one_metric,
                    num_candidates=5,
                    requires_permission_to_run=True,
                    teacher=None,
                )

                mock_dspy.MIPROv2.assert_called_once_with(
                    metric=_always_one_metric,
                    auto="heavy",
                    num_candidates=5,
                    requires_permission_to_run=True,
                )
                assert metadata["method"] == "mipro"
                assert metadata["auto_setting"] == "heavy"
                assert metadata["num_candidates"] == 5

    def test_run_mipro_with_teacher(self, mock_dspy) -> None:
        """_run_mipro should use teacher context when teacher is provided."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="mipro")
                teacher = MagicMock(name="teacher_lm")

                optimizer._run_mipro(
                    module=MagicMock(),
                    trainset=[MagicMock()],
                    metric=_always_one_metric,
                    num_candidates=5,
                    requires_permission_to_run=True,
                    teacher=teacher,
                )

                # Verify context was used with teacher
                mock_dspy.context.assert_called_with(lm=teacher)

    def test_run_mipro_returns_best_score(self, mock_dspy) -> None:
        """_run_mipro should populate best_score in metadata."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="mipro")

                # Patch _compute_best_score to return a known value
                with patch.object(optimizer, "_compute_best_score", return_value=0.85):
                    _, metadata = optimizer._run_mipro(
                        module=MagicMock(),
                        trainset=[MagicMock()],
                        metric=lambda x, y: 1.0,
                        num_candidates=5,
                        requires_permission_to_run=False,
                        teacher=None,
                    )

                    assert metadata["best_score"] == pytest.approx(0.85)


class TestRunBootstrap:
    """Test the _run_bootstrap internal method."""

    @pytest.fixture
    def mock_dspy(self):
        """Create DSPy mock for Bootstrap tests."""
        mock = MagicMock()
        mock_bootstrap = MagicMock()
        mock_bootstrap.compile.return_value = MagicMock(name="bootstrap_optimized")
        mock.BootstrapFewShot.return_value = mock_bootstrap
        mock.context.return_value.__enter__ = MagicMock()
        mock.context.return_value.__exit__ = MagicMock()
        return mock

    def test_run_bootstrap_basic(self, mock_dspy) -> None:
        """_run_bootstrap should configure and run BootstrapFewShot."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="bootstrap")

                _, metadata = optimizer._run_bootstrap(
                    module=MagicMock(),
                    trainset=[MagicMock(), MagicMock()],
                    metric=_always_high_metric,
                    max_bootstrapped_demos=6,
                    max_labeled_demos=12,
                    teacher=None,
                )

                mock_dspy.BootstrapFewShot.assert_called_once_with(
                    metric=_always_high_metric,
                    max_bootstrapped_demos=6,
                    max_labeled_demos=12,
                )
                assert metadata["method"] == "bootstrap"
                assert metadata["max_bootstrapped_demos"] == 6
                assert metadata["max_labeled_demos"] == 12

    def test_run_bootstrap_with_teacher(self, mock_dspy) -> None:
        """_run_bootstrap should use teacher model when provided."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="bootstrap")
                teacher = MagicMock(name="teacher_lm")

                optimizer._run_bootstrap(
                    module=MagicMock(),
                    trainset=[MagicMock()],
                    metric=lambda x, y: 1.0,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=8,
                    teacher=teacher,
                )

                # Verify context was used with teacher
                mock_dspy.context.assert_called_with(lm=teacher)

    def test_run_bootstrap_returns_best_score(self, mock_dspy) -> None:
        """_run_bootstrap should populate best_score in metadata."""
        with patch("traigent.integrations.dspy_adapter.DSPY_AVAILABLE", True):
            with patch("traigent.integrations.dspy_adapter.dspy", mock_dspy):
                from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

                optimizer = DSPyPromptOptimizer(method="bootstrap")

                # Patch _compute_best_score to return a known value
                with patch.object(optimizer, "_compute_best_score", return_value=0.92):
                    _, metadata = optimizer._run_bootstrap(
                        module=MagicMock(),
                        trainset=[MagicMock()],
                        metric=lambda x, y: 1.0,
                        max_bootstrapped_demos=4,
                        max_labeled_demos=8,
                        teacher=None,
                    )

                    assert metadata["best_score"] == pytest.approx(0.92)


class TestModuleExports:
    """Test that all expected symbols are exported."""

    def test_all_exports(self) -> None:
        """__all__ should contain expected exports."""
        from traigent.integrations.dspy_adapter import __all__

        expected = [
            "DSPyPromptOptimizer",
            "PromptOptimizationResult",
            "create_dspy_integration",
            "DSPY_AVAILABLE",
        ]
        assert set(__all__) == set(expected)

    def test_imports_from_integrations_package(self) -> None:
        """DSPy exports should be available from integrations package."""
        # This may raise ImportError if dspy not installed, which is fine
        try:
            from traigent.integrations import DSPY_INTEGRATION_AVAILABLE

            assert isinstance(DSPY_INTEGRATION_AVAILABLE, bool)
        except ImportError:
            pytest.skip("DSPy integration not available")
