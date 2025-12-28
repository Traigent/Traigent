"""Model bridges between Traigent SDK and OptiGen Backend.

This module provides conversion utilities to bridge the gap between
SDK cloud models and backend database entities, enabling seamless
integration between client-side optimization and server-side execution.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from traigent.evaluators.base import Dataset
from traigent.utils.hashing import generate_benchmark_hash, generate_experiment_hash
from traigent.utils.logging import get_logger

from .models import (
    AgentSpecification,
    OptimizationRequest,
    SessionCreationRequest,
    TrialSuggestion,
)

logger = get_logger(__name__)


@dataclass
class BackendExperimentRequest:
    """Request to create backend experiment from SDK optimization."""

    experiment_id: str
    name: str
    description: str
    agent_data: dict[str, Any]  # Agent creation data
    benchmark_data: dict[str, Any]  # Benchmark creation data
    example_set_data: dict[str, Any]  # Example set creation data
    model_parameters_data: dict[str, Any]  # Model parameters creation data
    measures: list[str]  # Measure IDs
    experiment_parameters: dict[str, Any]  # Additional experiment config
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendExperimentRunRequest:
    """Request to create backend experiment run from SDK session."""

    experiment_id: str
    run_id: str
    experiment_data: dict[str, Any]  # Complete experiment snapshot
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfigurationRunRequest:
    """Request to create backend configuration run from SDK trial."""

    experiment_run_id: str
    config_run_id: str
    experiment_parameters: dict[str, Any]  # Trial configuration
    trial_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionExperimentMapping:
    """Mapping between SDK session and backend experiment/run structure."""

    session_id: str
    experiment_id: str
    experiment_run_id: str
    function_name: str
    configuration_space: dict[str, Any]
    objectives: list[str]
    trial_mappings: dict[str, str] = field(
        default_factory=dict
    )  # trial_id -> config_run_id
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SDKBackendBridge:
    """Main bridge class for converting between SDK and backend models."""

    def __init__(self) -> None:
        """Initialize the bridge."""
        self._session_mappings: dict[str, SessionExperimentMapping] = {}
        self._agent_type_mappings: dict[str, Any] = {
            "conversational": "agent-type-1",
            "task": "agent-type-2",
            "analytical": "agent-type-3",
            "default": "agent-type-1",
        }

    def optimization_request_to_backend(
        self, request: OptimizationRequest, user_id: str | None = None
    ) -> BackendExperimentRequest:
        """Convert SDK OptimizationRequest to backend experiment structure.

        Args:
            request: SDK optimization request
            user_id: Optional user identifier

        Returns:
            Backend experiment request with all necessary components
        """
        # Extract dataset characteristics for consistent hashing
        dataset_characteristics = self._extract_dataset_characteristics(request.dataset)

        # Generate deterministic experiment ID based on optimization setup
        experiment_id = generate_experiment_hash(
            function_name=request.function_name,
            configuration_space=request.configuration_space,
            objectives=request.objectives,
            dataset_characteristics=dataset_characteristics,
        )

        # Generate agent data
        agent_data = self._create_agent_data_from_function(
            request.function_name, request.configuration_space, user_id
        )

        # Generate benchmark data with deterministic ID
        benchmark_data = self._create_benchmark_data_from_dataset(
            request.dataset, request.function_name
        )

        # Generate example set data
        example_set_data = self._create_example_set_data_from_dataset(
            request.dataset, f"{request.function_name}_examples"
        )

        # Generate model parameters data
        model_parameters_data = self._create_model_parameters_from_config_space(
            request.configuration_space,
            agent_spec=request.agent_specification,
        )

        # Map objectives to measures
        measures = self._map_objectives_to_measures(request.objectives)

        # Create experiment parameters
        experiment_parameters = {
            "max_trials": request.max_trials,
            "target_cost_reduction": request.target_cost_reduction,
            "billing_tier": request.billing_tier,
            "configuration_space": request.configuration_space,
            "objectives": request.objectives,
        }

        return BackendExperimentRequest(
            experiment_id=experiment_id,
            name=f"Optimization: {request.function_name}",
            description=f"SDK optimization for {request.function_name} function",
            agent_data=agent_data,
            benchmark_data=benchmark_data,
            example_set_data=example_set_data,
            model_parameters_data=model_parameters_data,
            measures=measures,
            experiment_parameters=experiment_parameters,
            metadata=request.metadata,
        )

    def session_creation_to_backend_run(
        self, session_request: SessionCreationRequest, experiment_id: str
    ) -> BackendExperimentRunRequest:
        """Convert SDK session creation to backend experiment run.

        Args:
            session_request: SDK session creation request
            experiment_id: Backend experiment ID

        Returns:
            Backend experiment run request
        """
        # Don't generate ID here - the backend will provide it via session endpoint
        # This method is deprecated and shouldn't be used
        run_id = experiment_id  # Just use experiment_id as placeholder

        experiment_data = {
            "function_name": session_request.function_name,
            "configuration_space": session_request.configuration_space,
            "objectives": session_request.objectives,
            "dataset_metadata": session_request.dataset_metadata,
            "max_trials": session_request.max_trials,
            "optimization_strategy": session_request.optimization_strategy,
            "user_id": session_request.user_id,
            "billing_tier": session_request.billing_tier,
            "run_id": run_id,
            "experiment_id": experiment_id,
        }

        return BackendExperimentRunRequest(
            experiment_id=experiment_id,
            run_id=run_id,
            experiment_data=experiment_data,
            metadata=session_request.metadata,
        )

    def trial_suggestion_to_config_run(
        self, trial: TrialSuggestion, experiment_run_id: str
    ) -> BackendConfigurationRunRequest:
        """Convert SDK trial suggestion to backend configuration run with enhanced mapping.

        Args:
            trial: SDK trial suggestion
            experiment_run_id: Backend experiment run ID

        Returns:
            Backend configuration run request with enhanced trial-to-config mapping
        """
        config_run_id = trial.trial_id

        # Enhanced experiment parameters with better OptiGen integration
        experiment_parameters = {
            # Core trial information
            "trial_number": trial.trial_number,
            "config": trial.config,
            # Enhanced dataset subset information for better UI visibility
            "dataset_subset": {
                "indices": (
                    trial.dataset_subset.indices
                    if hasattr(trial, "dataset_subset") and trial.dataset_subset
                    else []
                ),
                "selection_strategy": (
                    getattr(trial.dataset_subset, "selection_strategy", "privacy_mode")
                    if hasattr(trial, "dataset_subset") and trial.dataset_subset
                    else "privacy_mode"
                ),
                "confidence_level": (
                    getattr(trial.dataset_subset, "confidence_level", 0.0)
                    if hasattr(trial, "dataset_subset") and trial.dataset_subset
                    else 0.0
                ),
                "estimated_representativeness": (
                    getattr(trial.dataset_subset, "estimated_representativeness", 0.0)
                    if hasattr(trial, "dataset_subset") and trial.dataset_subset
                    else 0.0
                ),
                "subset_size": (
                    len(trial.dataset_subset.indices)
                    if hasattr(trial, "dataset_subset")
                    and trial.dataset_subset
                    and hasattr(trial.dataset_subset, "indices")
                    else 0
                ),
            },
            # Trial execution metadata
            "exploration_type": getattr(trial, "exploration_type", "optimization"),
            "priority": getattr(trial, "priority", 1.0),
            "estimated_duration": getattr(trial, "estimated_duration", 30.0),
            # Enhanced Traigent-specific parameters for better tracking
            "traigent_metadata": {
                "trial_id": trial.trial_id,
                "trial_number": trial.trial_number,
                "sdk_version": "1.1.0",
                "optimization_mode": "local_privacy",
                "model_config": (
                    {
                        "model": trial.config.get("model", "unknown"),
                        "temperature": trial.config.get("temperature", 0.7),
                        "max_tokens": trial.config.get("max_tokens", 1000),
                    }
                    if trial.config
                    else {}
                ),
                "created_at": datetime.now(UTC).isoformat(),
            },
        }

        # Enhanced trial metadata for better mapping visibility
        # Start with original trial metadata and add enhancements
        enhanced_trial_metadata = trial.metadata.copy() if trial.metadata else {}
        enhanced_trial_metadata.update(
            {
                "trial_id": trial.trial_id,
                "trial_number": trial.trial_number,
                "config": trial.config,
                "dataset_subset": experiment_parameters["dataset_subset"],
                "mapping_created_at": datetime.now(UTC).isoformat(),
                "optimization_session": "traigent_edge_analytics",
                "privacy_mode": True,
            }
        )

        return BackendConfigurationRunRequest(
            experiment_run_id=experiment_run_id,
            config_run_id=config_run_id,
            experiment_parameters=experiment_parameters,
            trial_metadata=enhanced_trial_metadata,
        )

    def agent_specification_to_backend(
        self, agent_spec: AgentSpecification, user_id: str | None = None
    ) -> dict[str, Any]:
        """Convert SDK AgentSpecification to backend agent data.

        Args:
            agent_spec: SDK agent specification
            user_id: Optional user identifier

        Returns:
            Backend agent creation data
        """
        return {
            "agent_id": agent_spec.id,
            "name": agent_spec.name,
            "description": "Agent generated from SDK specification",
            "agent_type_id": self._agent_type_mappings.get(
                agent_spec.agent_type or "default", self._agent_type_mappings["default"]
            ),
            "prompt_template": agent_spec.prompt_template,
            "reasoning": agent_spec.reasoning,
            "style": agent_spec.style,
            "tone": agent_spec.tone,
            "format": agent_spec.format,
            "persona": agent_spec.persona,
            "guidelines": agent_spec.guidelines,
            "response_validation": agent_spec.response_validation,
            "custom_tools": agent_spec.custom_tools,
            "metadata": agent_spec.metadata,
        }

    def create_session_mapping(
        self,
        session_id: str,
        experiment_id: str,
        experiment_run_id: str,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
    ) -> SessionExperimentMapping:
        """Create and store mapping between SDK session and backend experiment.

        Args:
            session_id: SDK session ID
            experiment_id: Backend experiment ID
            experiment_run_id: Backend experiment run ID
            function_name: Function being optimized
            configuration_space: Configuration space
            objectives: Optimization objectives

        Returns:
            Session mapping object
        """
        mapping = SessionExperimentMapping(
            session_id=session_id,
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            function_name=function_name,
            configuration_space=configuration_space,
            objectives=objectives,
        )

        self._session_mappings[session_id] = mapping
        logger.info(
            f"Created session mapping: {session_id} -> {experiment_id}/{experiment_run_id}"
        )

        return mapping

    def get_session_mapping(self, session_id: str) -> SessionExperimentMapping | None:
        """Get session mapping by session ID."""
        return self._session_mappings.get(session_id)

    def add_trial_mapping(
        self, session_id: str, trial_id: str, config_run_id: str
    ) -> None:
        """Add trial to configuration run mapping."""
        if session_id in self._session_mappings:
            self._session_mappings[session_id].trial_mappings[trial_id] = config_run_id
            logger.debug(f"Added trial mapping: {trial_id} -> {config_run_id}")

    def get_trial_mapping(self, session_id: str, trial_id: str) -> str | None:
        """Get configuration run ID for a trial."""
        mapping = self._session_mappings.get(session_id)
        if mapping:
            config_run_id = mapping.trial_mappings.get(trial_id)
            if config_run_id:
                logger.debug(
                    f"Found trial mapping: {trial_id} -> {config_run_id} (session: {session_id})"
                )
            return config_run_id
        logger.debug(f"No session mapping found for session: {session_id}")
        return None

    def get_all_trial_mappings(self, session_id: str) -> dict[str, str]:
        """Get all trial mappings for a session."""
        mapping = self._session_mappings.get(session_id)
        if mapping:
            return mapping.trial_mappings.copy()
        return {}

    def get_trial_count(self, session_id: str) -> int:
        """Get the number of trials mapped for a session."""
        mapping = self._session_mappings.get(session_id)
        if mapping:
            return len(mapping.trial_mappings)
        return 0

    def remove_trial_mapping(self, session_id: str, trial_id: str) -> bool:
        """Remove a trial mapping."""
        mapping = self._session_mappings.get(session_id)
        if mapping and trial_id in mapping.trial_mappings:
            removed_config_run_id = mapping.trial_mappings.pop(trial_id)
            logger.debug(
                f"Removed trial mapping: {trial_id} -> {removed_config_run_id}"
            )
            return True
        return False

    def get_session_mapping_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get a summary of session mapping for debugging and monitoring."""
        mapping = self._session_mappings.get(session_id)
        if mapping:
            return {
                "session_id": mapping.session_id,
                "experiment_id": mapping.experiment_id,
                "experiment_run_id": mapping.experiment_run_id,
                "function_name": mapping.function_name,
                "trial_count": len(mapping.trial_mappings),
                "trials": list(mapping.trial_mappings.keys()),
                "config_runs": list(mapping.trial_mappings.values()),
                "created_at": mapping.created_at.isoformat(),
                "objectives": mapping.objectives,
            }
        return None

    def _create_agent_data_from_function(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create agent data from function optimization request."""
        # Use function name as agent_id for better visibility in UI
        agent_id = function_name

        # Generate basic prompt template based on function name
        prompt_template = f"""You are an AI assistant optimized for {function_name} tasks.

Input: {{input}}

Instructions: Process the input according to the function requirements and provide an appropriate response.

Response:"""

        # Infer agent type from configuration space
        agent_type = "conversational"
        if "temperature" in configuration_space or "model" in configuration_space:
            agent_type = "conversational"

        return {
            "agent_id": agent_id,
            "name": f"Auto-generated agent for {function_name}",
            "description": f"Agent automatically created from SDK optimization of {function_name} function",
            "agent_type_id": self._agent_type_mappings.get(
                agent_type, self._agent_type_mappings["default"]
            ),
            "prompt_template": prompt_template,
            "response_validation": True,
            "chat_history": False,
            "custom_tools": [],
        }

    def _create_benchmark_data_from_dataset(
        self, dataset: Dataset, function_name: str
    ) -> dict[str, Any]:
        """Create benchmark data from SDK dataset with deterministic ID."""
        # Extract dataset info for consistent benchmark ID generation
        dataset_info = self._extract_dataset_characteristics(dataset)

        # Generate deterministic benchmark ID
        benchmark_id = generate_benchmark_hash(
            function_name=function_name, dataset_info=dataset_info
        )

        # Determine benchmark type from dataset
        benchmark_type = "qa"
        if hasattr(dataset, "examples") and dataset.examples:
            first_example = dataset.examples[0]
            if hasattr(first_example, "input_data") and hasattr(
                first_example, "expected_output"
            ):
                if isinstance(first_example.expected_output, str):
                    benchmark_type = "qa"
                elif isinstance(first_example.expected_output, (int, float)):
                    benchmark_type = "classification"

        return {
            "benchmark_id": benchmark_id,
            "benchmark_name": f"{function_name}_benchmark",
            "description": f"Benchmark auto-generated from {function_name} dataset",
            "type": benchmark_type,
            "agent_type_id": self._agent_type_mappings["default"],
            "label": f"SDK Generated Benchmark for {function_name}",
        }

    def _create_example_set_data_from_dataset(
        self, dataset: Dataset, name: str
    ) -> dict[str, Any]:
        """Create example set data from SDK dataset."""
        example_set_id = str(uuid.uuid4())

        # Determine example set type
        example_type = "input-output"
        if hasattr(dataset, "examples") and dataset.examples:
            first_example = dataset.examples[0]
            if (
                hasattr(first_example, "expected_output")
                and first_example.expected_output is not None
            ):
                example_type = "input-output"
            else:
                example_type = "input-only"

        return {
            "example_set_id": example_set_id,
            "name": name,
            "type": example_type,
            "description": f"Examples auto-generated from SDK dataset ({len(dataset.examples) if hasattr(dataset, 'examples') else 0} examples)",
            "examples": self._convert_dataset_to_examples(dataset),
        }

    def _convert_dataset_to_examples(self, dataset: Dataset) -> list[dict[str, Any]]:
        """Convert SDK dataset to backend example format."""
        examples = []

        if hasattr(dataset, "examples"):
            for _i, example in enumerate(dataset.examples):
                backend_example: dict[str, Any] = {
                    "example_id": f"EX{str(uuid.uuid4())[:6].upper()}",
                    "input": self._serialize_input_data(example.input_data),
                    "output": (
                        example.expected_output
                        if hasattr(example, "expected_output")
                        else None
                    ),
                    "explanation": "",
                    "tags": [],
                }

                # Add metadata as tags if available
                if hasattr(example, "metadata") and example.metadata:
                    backend_example["tags"] = [
                        f"{k}:{v}" for k, v in example.metadata.items()
                    ]

                examples.append(backend_example)

        return examples

    def _serialize_input_data(self, input_data: Any) -> str:
        """Serialize input data to string format for backend."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # For dict input, try to extract main content
            if "query" in input_data:
                return cast(str, input_data["query"])
            elif "question" in input_data:
                return cast(str, input_data["question"])
            elif "input" in input_data:
                return str(input_data["input"])
            else:
                # Return first string value or JSON representation
                for value in input_data.values():
                    if isinstance(value, str):
                        return value
                return str(input_data)
        else:
            return str(input_data)

    def _create_model_parameters_from_config_space(
        self,
        configuration_space: dict[str, Any],
        *,
        agent_spec: AgentSpecification | None = None,
    ) -> dict[str, Any]:
        """Create model parameters from configuration space."""
        model_params: dict[str, Any] = {}
        agent_defaults = dict(agent_spec.model_parameters or {}) if agent_spec else {}

        DEFAULT_MODEL = "gpt-3.5"
        DEFAULT_TEMPERATURE = 0.7
        DEFAULT_MAX_TOKENS = 150
        DEFAULT_TOP_P = 1.0

        model_value = self._extract_model_param(
            configuration_space,
            agent_defaults,
            "model",
            alias="model_id",
            default=DEFAULT_MODEL,
            allow_scalar_config=False,
        )
        if model_value is None:
            raise ValueError(
                "Configuration space or agent specification must define a model"
            )
        model_params["model_id"] = model_value

        temperature = self._extract_model_param(
            configuration_space,
            agent_defaults,
            "temperature",
            default=DEFAULT_TEMPERATURE,
        )
        model_params["temperature"] = self._safe_float(temperature)

        max_tokens = self._extract_model_param(
            configuration_space,
            agent_defaults,
            "max_tokens",
            alias="max_output_tokens",
            default=DEFAULT_MAX_TOKENS,
        )
        model_params["max_tokens"] = self._safe_int(max_tokens)

        top_p = self._extract_model_param(
            configuration_space,
            agent_defaults,
            "top_p",
            default=DEFAULT_TOP_P,
        )
        model_params["top_p"] = self._safe_float(top_p)

        # Set other parameters to defaults
        model_params["frequency_penalty"] = 0.0
        model_params["presence_penalty"] = 0.0

        return {"model_parameters_id": str(uuid.uuid4()), **model_params}

    def _extract_model_param(
        self,
        configuration_space: dict[str, Any],
        agent_defaults: dict[str, Any],
        key: str,
        *,
        alias: str | None = None,
        default: Any | None = None,
        allow_scalar_config: bool = True,
    ) -> Any | None:
        """Extract a parameter value from configuration space or agent defaults."""

        keys_to_check = [key]
        if alias:
            keys_to_check.append(alias)

        config_value: Any | None = None
        for candidate_key in keys_to_check:
            if candidate_key in configuration_space:
                config_value = configuration_space.get(candidate_key)
                if config_value is None:
                    return None
                break

        if isinstance(config_value, (list, tuple, set)) and not config_value:
            config_value = None

        if config_value is not None:
            candidate = self._first_config_value(config_value)
            if candidate is not None:
                config_is_sequence = isinstance(config_value, (list, tuple, set))
                if (
                    allow_scalar_config
                    or config_is_sequence
                    or isinstance(config_value, dict)
                ):
                    return candidate

        for candidate_key in keys_to_check:
            if candidate_key in agent_defaults:
                return agent_defaults[candidate_key]

        return default

    @staticmethod
    def _first_config_value(value: Any) -> Any | None:
        """Return a representative value from a config definition."""

        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                low, high = value
                return (low + high) / 2
            return value[0]
        if isinstance(value, dict):
            for key in ("value", "default", "initial", "low"):
                if key in value:
                    return value[key]
            return None
        return value

    @staticmethod
    def _safe_float(value: Any) -> float | Any | None:
        """Safely convert a value to float when possible."""

        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _safe_int(value: Any) -> int | Any | None:
        """Safely convert a value to int when possible."""

        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _map_objectives_to_measures(self, objectives: list[str]) -> list[str]:
        """Map SDK objectives to backend measure IDs."""
        objective_mapping = {
            "accuracy": "accuracy",
            "cost": "cost",
            "latency": "latency",
            "cost_efficiency": "cost",
            "response_time": "latency",
            "success_rate": "accuracy",
            "error_rate": "accuracy",  # Will be inverted
            "contextual_precision": "contextual_precision",
            "ragas": "ragas",
        }

        measures = []
        for objective in objectives:
            measure_id = objective_mapping.get(objective.lower(), "accuracy")
            if measure_id not in measures:
                measures.append(measure_id)

        # Ensure at least one measure
        if not measures:
            measures.append("accuracy")

        return measures

    def _extract_dataset_characteristics(self, dataset: Dataset) -> dict[str, Any]:
        """Extract key characteristics from dataset for consistent hashing.

        Args:
            dataset: The dataset to extract characteristics from

        Returns:
            Dict containing dataset characteristics like size, types, etc.
        """
        characteristics: dict[str, Any] = {}

        # Extract basic dataset properties
        if hasattr(dataset, "name"):
            characteristics["name"] = dataset.name

        if hasattr(dataset, "examples") and dataset.examples:
            characteristics["example_count"] = len(dataset.examples)

            # Sample first example to determine types
            first_example = dataset.examples[0]
            if hasattr(first_example, "input_data"):
                input_type = type(first_example.input_data).__name__
                characteristics["input_type"] = input_type

            if hasattr(first_example, "expected_output"):
                output_type = type(first_example.expected_output).__name__
                characteristics["output_type"] = output_type

            # Check if all examples have same structure (for consistency)
            if len(dataset.examples) > 1:
                # Sample a few examples to check consistency
                sample_size = min(5, len(dataset.examples))
                consistent = True
                for i in range(1, sample_size):
                    example = dataset.examples[i]
                    if hasattr(example, "input_data"):
                        if type(example.input_data).__name__ != input_type:
                            consistent = False
                            break
                characteristics["structure_consistent"] = consistent

        # Add dataset metadata if available
        if hasattr(dataset, "metadata") and dataset.metadata:
            # Only include stable metadata that affects dataset identity
            stable_keys = ["version", "source", "task_type", "domain"]
            for key in stable_keys:
                if key in dataset.metadata:
                    characteristics[f"meta_{key}"] = dataset.metadata[key]

        return characteristics


# Global bridge instance
bridge = SDKBackendBridge()
