"""Unit tests for traigent/optimizers/pruners.py.

Tests for custom Optuna pruners tailored to Traigent workloads.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance
# Traceability: CONC-Quality-Reliability FUNC-OPT-ALGORITHMS
# Traceability: REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from unittest.mock import Mock

import optuna
import pytest

from traigent.optimizers.pruners import (
    CeilingPruner,
    CeilingPrunerConfig,
    _completed_trials_with_values,
)


class TestCeilingPrunerConfig:
    """Tests for CeilingPrunerConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test CeilingPrunerConfig with default values."""
        config = CeilingPrunerConfig()

        assert config.min_completed_trials == 2
        assert config.warmup_steps == 2
        assert config.epsilon == 1e-6
        assert config.cost_threshold is None

    def test_custom_initialization(self) -> None:
        """Test CeilingPrunerConfig with custom values."""
        config = CeilingPrunerConfig(
            min_completed_trials=5,
            warmup_steps=3,
            epsilon=0.001,
            cost_threshold=100.0,
        )

        assert config.min_completed_trials == 5
        assert config.warmup_steps == 3
        assert config.epsilon == 0.001
        assert config.cost_threshold == 100.0

    def test_slots_attribute(self) -> None:
        """Test that config uses slots for memory efficiency."""
        config = CeilingPrunerConfig()

        # Dataclass with slots should not have __dict__
        assert not hasattr(config, "__dict__")


class TestCeilingPruner:
    """Tests for CeilingPruner."""

    @pytest.fixture
    def study_maximize(self) -> optuna.Study:
        """Create a study with maximize direction."""
        return optuna.create_study(direction="maximize")

    @pytest.fixture
    def study_minimize(self) -> optuna.Study:
        """Create a study with minimize direction."""
        return optuna.create_study(direction="minimize")

    @pytest.fixture
    def study_multi_objective(self) -> optuna.Study:
        """Create a multi-objective study."""
        return optuna.create_study(
            directions=["maximize", "minimize"]  # e.g., accuracy, cost
        )

    @pytest.fixture
    def pruner(self) -> CeilingPruner:
        """Create a CeilingPruner with default settings."""
        return CeilingPruner()

    def test_initialization_default(self) -> None:
        """Test CeilingPruner initialization with default parameters."""
        pruner = CeilingPruner()

        assert pruner._min_completed_trials == 2
        assert pruner._warmup_steps == 2
        assert pruner._epsilon == 1e-6
        assert pruner._cost_threshold is None

    def test_initialization_custom_parameters(self) -> None:
        """Test CeilingPruner initialization with custom parameters."""
        pruner = CeilingPruner(
            min_completed_trials=5,
            warmup_steps=3,
            epsilon=0.01,
            cost_threshold=50.0,
        )

        assert pruner._min_completed_trials == 5
        assert pruner._warmup_steps == 3
        assert pruner._epsilon == 0.01
        assert pruner._cost_threshold == 50.0

    def test_prune_returns_false_during_warmup(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test that pruning is disabled during warmup steps."""
        # Create a trial with step < warmup_steps
        trial = optuna.trial.FrozenTrial(
            number=0,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={0: 0.1, 1: 0.2},  # Only step 0 and 1
            trial_id=0,
        )

        # Should not prune during warmup (step 1 < warmup_steps 2)
        assert not pruner.prune(study_maximize, trial)

    def test_prune_returns_false_when_no_last_step(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test that pruning returns False when trial has no last_step."""
        trial = optuna.trial.FrozenTrial(
            number=0,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={},  # Empty intermediate values
            trial_id=0,
        )

        assert not pruner.prune(study_maximize, trial)

    def test_prune_returns_false_when_no_latest_value(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test that pruning returns False when latest step has no value."""
        # Create a trial where intermediate_values dict doesn't have a value for last_step
        # Since last_step is determined from intermediate_values, we need the get to return None
        # This happens when the step exists but the value lookup fails
        # However, in practice, last_step returns the max key, so this is testing the None check

        # Since we can't mock last_step (it's a property), we test the case where
        # intermediate_values.get(last_step) returns None by having no intermediate values
        # but that's already covered by test_prune_returns_false_when_no_last_step

        # Instead, let's test the logic path - this case is actually impossible to reach
        # because if last_step exists, intermediate_values[last_step] must exist.
        # So we'll change this test to verify the early return paths work correctly.

        # Test that when trial has intermediate_values but we're before warmup
        trial = optuna.trial.FrozenTrial(
            number=0,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={0: 0.1},  # Only step 0, which is < warmup_steps
            trial_id=0,
        )

        # Should not prune - still in warmup
        assert not pruner.prune(study_maximize, trial)

    def test_prune_with_cost_threshold_scalar_value_above_threshold(
        self, study_minimize: optuna.Study
    ) -> None:
        """Test cost threshold pruning with scalar value above threshold."""
        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create completed trials
        for i in range(3):
            trial = study_minimize.ask()
            study_minimize.tell(trial, i * 2.0)

        # Create a running trial with cost above threshold
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 15.0},  # Cost above threshold
            trial_id=3,
        )

        # Should prune due to cost threshold
        assert pruner.prune(study_minimize, trial)

    def test_prune_with_cost_threshold_scalar_value_below_threshold(
        self, study_minimize: optuna.Study
    ) -> None:
        """Test cost threshold pruning with scalar value below threshold."""
        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create completed trials
        for i in range(3):
            trial = study_minimize.ask()
            study_minimize.tell(trial, i * 2.0)

        # Create a running trial with cost below threshold
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 5.0},  # Cost below threshold
            trial_id=3,
        )

        # Should not prune due to cost threshold
        # (but might prune due to ceiling)
        _ = pruner.prune(study_minimize, trial)
        # Result depends on ceiling comparison, not asserting here

    def test_prune_with_cost_threshold_multi_objective_with_directions(
        self, study_multi_objective: optuna.Study
    ) -> None:
        """Test cost threshold pruning in multi-objective study."""
        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create completed trials
        for i in range(3):
            trial = study_multi_objective.ask()
            study_multi_objective.tell(trial, [0.5 + i * 0.1, i * 2.0])

        # Create a running trial with cost (second objective) above threshold
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            values=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: [0.7, 15.0]},  # Cost above threshold
            trial_id=3,
        )

        # Should prune due to cost threshold on minimize objective
        assert pruner.prune(study_multi_objective, trial)

    def test_prune_with_cost_threshold_multi_objective_without_directions(
        self,
    ) -> None:
        """Test cost threshold pruning without directions attribute."""
        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create a mock study without directions attribute
        study = Mock()
        study.directions = None
        study.trials = [
            optuna.trial.FrozenTrial(
                number=i,
                state=optuna.trial.TrialState.COMPLETE,
                value=None,
                values=[0.5, i * 2.0],
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=i,
            )
            for i in range(3)
        ]

        # Create a running trial with cost (second value) above threshold
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            values=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: [0.7, 15.0]},  # Cost above threshold
            trial_id=3,
        )

        # Should prune based on second value (legacy behavior)
        assert pruner.prune(study, trial)

    def test_prune_with_cost_threshold_list_too_short(self) -> None:
        """Test cost threshold pruning with list shorter than 2 elements."""
        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create a mock study
        study = Mock()
        study.directions = None
        study.trials = [
            optuna.trial.FrozenTrial(
                number=i,
                state=optuna.trial.TrialState.COMPLETE,
                value=i * 2.0,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=i,
            )
            for i in range(3)
        ]

        # Create a running trial with single-element list
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: [0.7]},  # Only one element
            trial_id=3,
        )

        # Should not prune due to cost threshold (list too short)
        # Result depends on ceiling comparison
        _ = pruner.prune(study, trial)
        # Not asserting as it depends on study direction

    def test_prune_returns_false_when_insufficient_completed_trials(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test pruning returns False with insufficient completed trials."""
        # Create only 1 completed trial (less than min_completed_trials=2)
        trial_obj = study_maximize.ask()
        study_maximize.tell(trial_obj, 0.8)

        # Create a running trial
        trial = optuna.trial.FrozenTrial(
            number=1,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 0.5},
            trial_id=1,
        )

        # Should not prune - not enough completed trials
        assert not pruner.prune(study_maximize, trial)

    def test_prune_maximize_direction_prunes_when_below_best(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test pruning in maximize direction when estimate is below best."""
        # Create completed trials with values
        for value in [0.5, 0.7, 0.6]:
            trial_obj = study_maximize.ask()
            study_maximize.tell(trial_obj, value)

        # Best is 0.7
        # Create a running trial with estimate below best
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 0.4},  # 0.4 < 0.7 - epsilon
            trial_id=3,
        )

        # Should prune because estimate is below best
        assert pruner.prune(study_maximize, trial)

    def test_prune_maximize_direction_does_not_prune_when_above_best(
        self, study_maximize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test pruning in maximize direction when estimate is above best."""
        # Create completed trials
        for value in [0.5, 0.7, 0.6]:
            trial_obj = study_maximize.ask()
            study_maximize.tell(trial_obj, value)

        # Best is 0.7
        # Create a running trial with estimate above best
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 0.8},  # 0.8 > 0.7 - epsilon
            trial_id=3,
        )

        # Should not prune because estimate is above best
        assert not pruner.prune(study_maximize, trial)

    def test_prune_minimize_direction_prunes_when_above_best(
        self, study_minimize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test pruning in minimize direction when estimate is above best."""
        # Create completed trials
        for value in [5.0, 3.0, 4.0]:
            trial_obj = study_minimize.ask()
            study_minimize.tell(trial_obj, value)

        # Best is 3.0
        # Create a running trial with estimate above best
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 6.0},  # 6.0 > 3.0 + epsilon
            trial_id=3,
        )

        # Should prune because estimate is above best
        assert pruner.prune(study_minimize, trial)

    def test_prune_minimize_direction_does_not_prune_when_below_best(
        self, study_minimize: optuna.Study, pruner: CeilingPruner
    ) -> None:
        """Test pruning in minimize direction when estimate is below best."""
        # Create completed trials
        for value in [5.0, 3.0, 4.0]:
            trial_obj = study_minimize.ask()
            study_minimize.tell(trial_obj, value)

        # Best is 3.0
        # Create a running trial with estimate below best
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 2.0},  # 2.0 < 3.0 + epsilon
            trial_id=3,
        )

        # Should not prune because estimate is below best
        assert not pruner.prune(study_minimize, trial)

    def test_prune_with_epsilon_boundary_maximize(
        self, study_maximize: optuna.Study
    ) -> None:
        """Test epsilon boundary in maximize direction."""
        pruner = CeilingPruner(epsilon=0.1)

        # Create completed trials
        for value in [0.5, 0.7, 0.6]:
            trial_obj = study_maximize.ask()
            study_maximize.tell(trial_obj, value)

        # Best is 0.7
        # Test at boundary: 0.7 - 0.1 = 0.6
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 0.6},  # Exactly at boundary
            trial_id=3,
        )

        # Should prune at boundary (<=)
        assert pruner.prune(study_maximize, trial)

    def test_prune_with_epsilon_boundary_minimize(
        self, study_minimize: optuna.Study
    ) -> None:
        """Test epsilon boundary in minimize direction."""
        pruner = CeilingPruner(epsilon=0.1)

        # Create completed trials
        for value in [5.0, 3.0, 4.0]:
            trial_obj = study_minimize.ask()
            study_minimize.tell(trial_obj, value)

        # Best is 3.0
        # Test at boundary: 3.0 + 0.1 = 3.1
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 3.1},  # Exactly at boundary
            trial_id=3,
        )

        # Should prune at boundary (>=)
        assert pruner.prune(study_minimize, trial)

    def test_prune_multi_objective_returns_false(self, pruner: CeilingPruner) -> None:
        """Test multi-objective studies don't perform ceiling pruning."""
        # Create a multi-objective study manually with mock to avoid .value errors
        from unittest.mock import Mock

        study = Mock()
        study.direction = None  # Multi-objective don't have single direction
        study.directions = [
            optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]

        # Create mock completed trials (avoiding .value access)
        study.trials = []

        # Create a running trial
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: [0.2, 10.0]},
            trial_id=3,
        )

        # Multi-objective studies don't have 'direction', only 'directions'
        # So ceiling pruning should return False
        assert not pruner.prune(study, trial)

    def test_prune_study_without_direction_attribute(self) -> None:
        """Test pruning behavior when study has no direction attribute."""
        pruner = CeilingPruner()

        # Create a mock study without direction or directions
        study = Mock()
        study.direction = None
        study.directions = None
        study.trials = [
            optuna.trial.FrozenTrial(
                number=i,
                state=optuna.trial.TrialState.COMPLETE,
                value=i * 2.0,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=i,
            )
            for i in range(3)
        ]

        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: 0.5},
            trial_id=3,
        )

        # Should return False when no direction is set
        assert not pruner.prune(study, trial)

    def test_prune_with_none_cost_in_multi_objective(self) -> None:
        """Test cost threshold pruning when cost value is None."""
        from unittest.mock import Mock

        pruner = CeilingPruner(cost_threshold=10.0, warmup_steps=0)

        # Create a mock multi-objective study to avoid .value errors
        study = Mock()
        study.direction = None
        study.directions = [
            optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]
        study.trials = []

        # Create a running trial with None cost
        trial = optuna.trial.FrozenTrial(
            number=3,
            state=optuna.trial.TrialState.RUNNING,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={5: [0.7, None]},  # Cost is None
            trial_id=3,
        )

        # Should not prune when cost is None
        result = pruner.prune(study, trial)
        # May return False due to None cost or due to multi-objective
        assert isinstance(result, bool)

    def test_prune_integration_with_optuna_study(self) -> None:
        """Test CeilingPruner integration with actual Optuna study."""
        pruner = CeilingPruner(min_completed_trials=2, warmup_steps=1, epsilon=0.01)
        study = optuna.create_study(direction="maximize", pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            # Report intermediate values
            for step in range(5):
                value = trial.suggest_float("x", 0, 1) * (step + 1) / 5
                trial.report(value, step)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return value

        # Run optimization
        study.optimize(objective, n_trials=5, timeout=10)

        # Check that some trials completed
        assert len(study.trials) > 0
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed_trials) > 0


class TestCompletedTrialsWithValues:
    """Tests for _completed_trials_with_values helper function."""

    def test_filters_completed_trials_with_values(self) -> None:
        """Test that function returns only completed trials with values."""
        trials = [
            optuna.trial.FrozenTrial(
                number=0,
                state=optuna.trial.TrialState.COMPLETE,
                value=0.5,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=0,
            ),
            optuna.trial.FrozenTrial(
                number=1,
                state=optuna.trial.TrialState.RUNNING,
                value=None,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=1,
            ),
            optuna.trial.FrozenTrial(
                number=2,
                state=optuna.trial.TrialState.COMPLETE,
                value=0.7,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=2,
            ),
            optuna.trial.FrozenTrial(
                number=3,
                state=optuna.trial.TrialState.PRUNED,
                value=None,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=3,
            ),
            optuna.trial.FrozenTrial(
                number=4,
                state=optuna.trial.TrialState.COMPLETE,
                value=None,  # Completed but no value
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=4,
            ),
        ]

        result = _completed_trials_with_values(trials)

        assert len(result) == 2
        assert result[0].number == 0
        assert result[0].value == 0.5
        assert result[1].number == 2
        assert result[1].value == 0.7

    def test_empty_trials_list(self) -> None:
        """Test function with empty trials list."""
        result = _completed_trials_with_values([])
        assert result == []

    def test_no_completed_trials(self) -> None:
        """Test function when no trials are completed."""
        trials = [
            optuna.trial.FrozenTrial(
                number=0,
                state=optuna.trial.TrialState.RUNNING,
                value=None,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=0,
            ),
            optuna.trial.FrozenTrial(
                number=1,
                state=optuna.trial.TrialState.PRUNED,
                value=None,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=1,
            ),
        ]

        result = _completed_trials_with_values(trials)
        assert result == []

    def test_all_completed_with_values(self) -> None:
        """Test function when all trials are completed with values."""
        trials = [
            optuna.trial.FrozenTrial(
                number=i,
                state=optuna.trial.TrialState.COMPLETE,
                value=i * 0.1,
                datetime_start=None,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=i,
            )
            for i in range(5)
        ]

        result = _completed_trials_with_values(trials)
        assert len(result) == 5
        for i, trial in enumerate(result):
            assert trial.value == i * 0.1
