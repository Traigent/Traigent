"""Bayesian optimization algorithm using Gaussian Process regression."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

from traigent.api.types import TrialResult
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. Bayesian optimization will not work. "
        "Install with: pip install scikit-learn scipy"
    )


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Process regression.

    This optimizer uses a Gaussian Process to model the objective function
    and an acquisition function to choose the next trial configuration.

    Args:
        config_space: Configuration space to search
        objectives: List of objectives to optimize
        acquisition_function: Acquisition function ("expected_improvement", "upper_confidence_bound")
        initial_random_samples: Number of random samples before starting Bayesian optimization
        xi: Exploration parameter for Expected Improvement
        kappa: Exploration parameter for Upper Confidence Bound
        random_seed: Random seed for reproducibility

    Example:
        >>> optimizer = BayesianOptimizer(
        ...     config_space={"x": (0.0, 1.0), "y": ["a", "b"]},
        ...     objectives=["accuracy"],
        ...     acquisition_function="expected_improvement",
        ...     initial_random_samples=5
        ... )
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        acquisition_function: str = "expected_improvement",
        initial_random_samples: int = 5,
        xi: float = 0.01,
        kappa: float = 2.576,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise OptimizationError(
                "Bayesian optimization requires scikit-learn and scipy. "
                "Install with: pip install scikit-learn scipy"
            )

        super().__init__(config_space, objectives, **kwargs)

        # Log config space for debugging
        logger.info("Bayesian optimizer initialized")
        logger.debug(f"Config space: {list(config_space.keys())}")
        if "model" in config_space:
            logger.debug(f"Model options: {config_space['model']}")
        logger.debug(f"Objectives: {objectives}")
        logger.debug(f"Initial random samples: {initial_random_samples}")

        # Validate acquisition function
        valid_acquisition = ["expected_improvement", "upper_confidence_bound"]
        result = CoreValidators.validate_choices(
            acquisition_function, "acquisition_function", valid_acquisition
        )
        validate_or_raise(result)

        self.acquisition_function = acquisition_function
        self.initial_random_samples = initial_random_samples
        self.xi = xi
        self.kappa = kappa
        self.random_seed = random_seed

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize Gaussian Process
        # Increased alpha (noise) to prevent overconfidence
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.01,  # Increased from 1e-6 to add more uncertainty
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_seed,
        )

        # Track parameter mapping
        self._param_mapping = self._create_parameter_mapping()
        self._continuous_dims = len(self._param_mapping["continuous"])
        self._categorical_dims = len(self._param_mapping["categorical"])

        logger.debug(
            f"Initialized Bayesian optimizer with {len(self._param_mapping['continuous'])} "
            f"continuous and {len(self._param_mapping['categorical'])} categorical parameters"
        )

    def _create_parameter_mapping(self) -> dict[str, Any]:
        """Create mapping between parameter names and optimization space."""
        continuous = []
        categorical = []
        fixed = []

        for param_name, param_values in self.config_space.items():
            if isinstance(param_values, (list, tuple)) and len(param_values) == 2:
                # Check for boolean parameters first (since bool is subclass of int)
                if all(isinstance(v, bool) for v in param_values):
                    categorical.append(
                        {
                            "name": param_name,
                            "values": list(param_values),
                            "type": "boolean",
                        }
                    )
                # Check if it's a continuous range
                elif all(isinstance(v, (int, float)) for v in param_values):
                    # Determine if this should be treated as integer
                    is_integer = all(
                        isinstance(v, int) for v in param_values
                    ) or param_name.lower() in [
                        "batch_size",
                        "num_layers",
                        "epochs",
                        "steps",
                        "size",
                    ]
                    continuous.append(
                        {
                            "name": param_name,
                            "bounds": param_values,
                            "type": "continuous",
                            "is_integer": is_integer,
                        }
                    )
                else:
                    # Categorical
                    categorical.append(
                        {
                            "name": param_name,
                            "values": list(param_values),
                            "type": "categorical",
                        }
                    )
            elif isinstance(param_values, list):
                # Handle boolean parameters first
                if all(isinstance(v, bool) for v in param_values):
                    categorical.append(
                        {"name": param_name, "values": param_values, "type": "boolean"}
                    )
                else:
                    # Regular categorical parameter
                    categorical.append(
                        {
                            "name": param_name,
                            "values": param_values,
                            "type": "categorical",
                        }
                    )
            else:
                # Single value - treat as fixed
                logger.debug(
                    f"Parameter {param_name} has single value, will be fixed at {param_values}"
                )
                fixed.append(
                    {"name": param_name, "value": param_values, "type": "fixed"}
                )

        return {"continuous": continuous, "categorical": categorical, "fixed": fixed}

    def _config_to_array(self, config: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Convert configuration dictionary to array for GP."""
        array_values = []

        # Add continuous parameters
        for param in self._param_mapping["continuous"]:
            value = config.get(param["name"])
            if value is None:
                # Use middle of range as default
                bounds = param["bounds"]
                value = (bounds[0] + bounds[1]) / 2

            # Normalize to [0, 1]
            bounds = param["bounds"]
            normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
            array_values.append(normalized)

        # Add categorical parameters (one-hot encoded)
        for param in self._param_mapping["categorical"]:
            value = config.get(param["name"])
            if value is None:
                value = param["values"][0]  # Default to first value

            # One-hot encoding
            one_hot = [0] * len(param["values"])
            if value in param["values"]:
                idx = param["values"].index(value)
                one_hot[idx] = 1
            else:
                one_hot[0] = 1  # Default to first if value not found

            array_values.extend(one_hot)

        return cast(np.ndarray[Any, Any], np.array(array_values))

    def _array_to_config(self, array: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Convert array back to configuration dictionary."""
        config = {}
        idx = 0

        # Extract continuous parameters
        for param in self._param_mapping["continuous"]:
            normalized_value = array[idx]
            bounds = param["bounds"]
            value = bounds[0] + normalized_value * (bounds[1] - bounds[0])

            # Handle integer parameters
            if param.get("is_integer", False):
                value = int(round(value))
                # Ensure we stay within bounds after rounding
                value = max(int(bounds[0]), min(int(bounds[1]), value))

            config[param["name"]] = value
            idx += 1

        # Extract categorical parameters
        for param in self._param_mapping["categorical"]:
            one_hot_size = len(param["values"])
            one_hot = array[idx : idx + one_hot_size]
            selected_idx = np.argmax(one_hot)
            value = param["values"][selected_idx]
            # Convert numpy types to native Python types for booleans
            if param.get("type") == "boolean" and hasattr(value, "item"):
                value = bool(value.item())
            config[param["name"]] = value
            idx += one_hot_size

        # Add fixed parameters
        for param in self._param_mapping.get("fixed", []):
            config[param["name"]] = param["value"]

        return config

    def _random_config(self) -> dict[str, Any]:
        """Generate a random configuration."""
        config = {}

        # Random continuous parameters
        for param in self._param_mapping["continuous"]:
            bounds = param["bounds"]
            if param.get("is_integer", False):
                # For integer parameters, sample uniformly from integer range
                value = np.random.randint(int(bounds[0]), int(bounds[1]) + 1)
            else:
                # For continuous parameters
                value = np.random.uniform(bounds[0], bounds[1])
            config[param["name"]] = value

        # Random categorical parameters
        for param in self._param_mapping["categorical"]:
            value = np.random.choice(param["values"])
            # Convert numpy types to native Python types
            if param.get("type") == "boolean" and hasattr(value, "item"):
                value = bool(value.item())
            config[param["name"]] = value

        # Add fixed parameters
        for param in self._param_mapping.get("fixed", []):
            config[param["name"]] = param["value"]

        return config

    def _expected_improvement(
        self, X: np.ndarray[Any, Any], y_best: float
    ) -> np.ndarray[Any, Any]:
        """Calculate Expected Improvement acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)

        sigma = sigma.reshape(-1, 1)

        # Calculate Expected Improvement
        with np.errstate(divide="ignore"):
            imp = mu - y_best - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return cast(np.ndarray[Any, Any], ei.flatten())

    def _upper_confidence_bound(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calculate Upper Confidence Bound acquisition function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)

        return cast(np.ndarray[Any, Any], mu + self.kappa * sigma)

    def _optimize_acquisition(self, y_best: float) -> np.ndarray[Any, Any]:
        """Find the point that maximizes the acquisition function."""

        def objective(x):
            if self.acquisition_function == "expected_improvement":
                return -self._expected_improvement(x, y_best)[0]
            else:  # upper_confidence_bound
                return -self._upper_confidence_bound(x)[0]

        # Number of dimensions
        n_dims = self._continuous_dims + sum(
            len(p["values"]) for p in self._param_mapping["categorical"]
        )

        # Try differential evolution first for global optimization
        try:
            result = differential_evolution(
                objective,
                bounds=[(0, 1)] * n_dims,
                seed=self.random_seed,
                maxiter=100,
                popsize=min(15, max(5, n_dims * 2)),  # Adaptive population size
                atol=1e-6,
                polish=True,  # Local refinement with L-BFGS-B
            )

            if result.success:
                logger.debug(f"Differential evolution succeeded with fun={result.fun}")
                return cast(np.ndarray[Any, Any], result.x)
            else:
                logger.debug(f"Differential evolution failed: {result.message}")
        except Exception as e:
            logger.debug(f"Differential evolution failed: {e}")

        # Fallback to multiple random starting points with L-BFGS-B
        best_x = None
        best_acquisition = float("inf")

        for _ in range(10):  # Try 10 random starting points
            x0 = np.random.rand(n_dims)

            try:
                result = minimize(
                    objective, x0, bounds=[(0, 1)] * n_dims, method="L-BFGS-B"
                )

                if result.success and result.fun < best_acquisition:
                    best_acquisition = result.fun
                    best_x = result.x
            except Exception as e:
                logger.debug(f"L-BFGS-B optimization failed for starting point: {e}")
                continue

        if best_x is None:
            # Final fallback to random point
            logger.warning(
                "All acquisition optimization methods failed, using random point"
            )
            return cast(np.ndarray[Any, Any], np.random.rand(n_dims))

        logger.debug(f"L-BFGS-B fallback succeeded with fun={best_acquisition}")
        return cast(np.ndarray[Any, Any], best_x)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest the next configuration to evaluate.

        Args:
            history: List of previous trial results

        Returns:
            Dictionary with suggested configuration
        """
        # Use random sampling for initial trials
        if len(history) < self.initial_random_samples:
            logger.debug(f"Using random sampling for trial {len(history) + 1}")
            config = self._random_config()
            logger.info(
                f"Bayesian: Random sampling (trial {len(history) + 1}/{self.initial_random_samples})"
            )
            logger.debug(f"Selected config: {config}")
            return config

        # Filter successful trials
        successful_trials = [trial for trial in history if trial.is_successful]

        if len(successful_trials) == 0:
            logger.warning("No successful trials found, using random sampling")
            return self._random_config()

        # Prepare data for Gaussian Process
        X = []
        y = []

        primary_objective = self.objectives[0]

        logger.info(f"Bayesian optimizer - Trial {len(history) + 1}")
        logger.debug(f"Primary objective: {primary_objective}")
        logger.debug(f"Successful trials so far: {len(successful_trials)}")

        for trial in successful_trials:
            config_array = self._config_to_array(trial.config)
            X.append(config_array)

            # Extract objective value
            objective_value = trial.metrics.get(primary_objective, 0.0)
            y.append(objective_value)

            # Log trial history
            model = trial.config.get("model", "unknown")
            logger.debug(
                f"Trial history - {model}: {primary_objective}={objective_value:.4f}"
            )

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid data for GP, using random sampling")
            return self._random_config()

        # Fit Gaussian Process
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}, using random sampling")
            return self._random_config()

        # Find best observed value
        # NOTE: We always maximize in the Bayesian optimizer
        # If the objective should be minimized, we need to negate the values
        # For now, we assume maximization (accuracy, f1-score, etc.)
        y_best: float = float(np.max(y))
        logger.info(f"Best observed {primary_objective}: {y_best:.4f}")

        # Evaluate acquisition function for all possible models (for categorical params)
        if len(self._param_mapping["continuous"]) == 0:
            # For categorical-only spaces, evaluate all possibilities
            logger.debug("Evaluating acquisition function for all models")

            # Get all unique models from config space
            model_values = self._param_mapping["categorical"][0][
                "values"
            ]  # Assuming 'model' is first

            candidate_info: dict[Any, dict[str, Any]] = {}
            for model_val in model_values:
                # Create config array for this model
                test_config = self._random_config()
                test_config["model"] = model_val
                test_x = self._config_to_array(test_config)

                # Predict with GP
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mu, sigma = self.gp.predict(test_x.reshape(1, -1), return_std=True)

                mu_scalar = float(mu[0])
                sigma_scalar = float(sigma[0])

                # Calculate Expected Improvement / UCB score
                if self.acquisition_function == "expected_improvement":
                    score_value = float(self._expected_improvement(test_x, y_best)[0])
                    improvement = mu_scalar - y_best
                    logger.debug(
                        f"{model_val}: EI={score_value:.6f}, predicted={mu_scalar:.4f}±{sigma_scalar:.4f}, improvement={improvement:.4f}"
                    )
                else:
                    score_value = float(self._upper_confidence_bound(test_x, y_best)[0])
                    logger.debug(
                        f"{model_val}: UCB={score_value:.6f}, predicted={mu_scalar:.4f}±{sigma_scalar:.4f}"
                    )

                candidate_info[model_val] = {
                    "config": dict(test_config),
                    "score": score_value,
                    "sigma": sigma_scalar,
                }

            # Show which model has best acquisition
            best_model = max(
                candidate_info, key=lambda name: candidate_info[name]["score"]
            )
            best_score = candidate_info[best_model]["score"]
            logger.info(
                f"Best acquisition score: {best_model} (score={best_score:.6f})"
            )

            # If all acquisition scores are essentially zero, we need more exploration
            if best_score < 1e-6:
                logger.warning(
                    "All acquisition scores near zero - falling back to uncertainty-based selection"
                )
                best_model = max(
                    candidate_info, key=lambda name: candidate_info[name]["sigma"]
                )
                best_sigma = candidate_info[best_model]["sigma"]
                logger.info(
                    f"Selected based on uncertainty: {best_model} (sigma={best_sigma:.6f})"
                )

        # For categorical-only problems with exhaustive evaluation above
        if len(self._param_mapping["continuous"]) == 0:
            # We've already selected best_model above (either from EI or uncertainty)
            selected_config = dict(candidate_info[best_model]["config"])
            logger.info(f"Final selected configuration: {selected_config}")
            return selected_config

        # Optimize acquisition function (for continuous or mixed spaces)
        try:
            best_x = self._optimize_acquisition(y_best)
            config = self._array_to_config(best_x)

            logger.debug(f"Bayesian optimization suggested config: {config}")
            logger.info(f"Final selected configuration: {config}")
            return config

        except Exception as e:
            logger.warning(
                f"Acquisition optimization failed: {e}, using random sampling"
            )
            return self._random_config()

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop.

        Args:
            history: List of trial results

        Returns:
            True if optimization should stop
        """
        # Basic stopping criterion
        if len(history) >= 100:  # Maximum trials
            return True

        # Check for convergence (no improvement in last 10 trials)
        if len(history) >= 20:
            successful_trials = [trial for trial in history if trial.is_successful]
            if len(successful_trials) >= 10:
                primary_objective = self.objectives[0]
                recent_scores = []
                for trial in successful_trials[-10:]:
                    score = trial.metrics.get(primary_objective, 0.0)
                    recent_scores.append(score)

                if len(recent_scores) >= 10:
                    improvement = max(recent_scores) - min(recent_scores)
                    if improvement < 0.01:  # Less than 1% improvement
                        logger.info("Convergence detected, stopping optimization")
                        return True

        return False
