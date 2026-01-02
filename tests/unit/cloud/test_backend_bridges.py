"""Tests for Traigent Cloud Backend Bridges."""

from datetime import datetime

import pytest

from traigent.cloud.backend_bridges import (
    BackendConfigurationRunRequest,
    BackendExperimentRequest,
    BackendExperimentRunRequest,
    SDKBackendBridge,
    SessionExperimentMapping,
    bridge,
)
from traigent.cloud.models import (
    AgentSpecification,
    DatasetSubsetIndices,
    OptimizationRequest,
    SessionCreationRequest,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"query": "What is machine learning?"},
            expected_output="Machine learning is a subset of artificial intelligence.",
            metadata={"difficulty": "easy"},
        ),
        EvaluationExample(
            input_data={"question": "Explain deep learning"},
            expected_output="Deep learning uses neural networks with multiple layers.",
            metadata={"difficulty": "medium"},
        ),
        EvaluationExample(
            input_data="What are transformers?",
            expected_output="Transformers are attention-based neural network architectures.",
            metadata={"difficulty": "hard"},
        ),
    ]
    return Dataset(
        examples=examples,
        name="test_dataset",
        description="Test dataset for bridge tests",
    )


@pytest.fixture
def optimization_request(sample_dataset):
    """Create optimization request for testing."""
    return OptimizationRequest(
        function_name="llm_response_function",
        dataset=sample_dataset,
        configuration_space={
            "model": ["gpt-3.5", "gpt-4"],
            "temperature": [0.0, 0.5, 1.0],
            "max_tokens": [100, 200, 300],
        },
        objectives=["accuracy", "cost"],
        max_trials=50,
        target_cost_reduction=0.65,
        user_id="test_user_123",
        billing_tier="premium",
        metadata={"experiment_type": "test"},
    )


@pytest.fixture
def session_creation_request():
    """Create session creation request for testing."""
    return SessionCreationRequest(
        function_name="test_function",
        configuration_space={"param1": [1, 2, 3], "param2": ["a", "b", "c"]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1000, "type": "qa"},
        max_trials=25,
        optimization_strategy={"algorithm": "random"},
        user_id="test_user",
        billing_tier="standard",
        metadata={"session_type": "test"},
    )


@pytest.fixture
def trial_suggestion():
    """Create trial suggestion for testing."""
    return TrialSuggestion(
        trial_id="trial_123",
        session_id="session_456",
        trial_number=1,
        config={"param1": 2, "param2": "b"},
        dataset_subset=DatasetSubsetIndices(
            indices=[0, 2, 4],
            selection_strategy="random",
            confidence_level=0.95,
            estimated_representativeness=0.8,
            metadata={"subset_info": "test"},
        ),
        exploration_type="exploration",
        priority=1,
        estimated_duration=30.0,
        metadata={"trial_info": "test"},
    )


@pytest.fixture
def agent_specification():
    """Create agent specification for testing."""
    return AgentSpecification(
        id="agent_123",
        name="Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Test prompt: {input}",
        model_parameters={"temperature": 0.7, "max_tokens": 150},
        reasoning="chain-of-thought",
        style="professional",
        tone="helpful",
        format="structured",
        persona="expert",
        guidelines=["be accurate", "be helpful"],
        response_validation=True,
        custom_tools=["calculator"],
        metadata={"agent_version": "1.0"},
    )


@pytest.fixture
def sdk_bridge():
    """Create fresh SDK bridge instance for testing."""
    return SDKBackendBridge()


class TestBackendDataClasses:
    """Test backend data classes."""

    def test_backend_experiment_request_creation(self):
        """Test BackendExperimentRequest creation."""
        request = BackendExperimentRequest(
            experiment_id="exp_123",
            name="Test Experiment",
            description="Test description",
            agent_data={"agent_id": "agent_123"},
            benchmark_data={"benchmark_id": "bench_123"},
            example_set_data={"example_set_id": "ex_123"},
            model_parameters_data={"model_id": "gpt-3.5"},
            measures=["accuracy", "cost"],
            experiment_parameters={"max_trials": 50},
            metadata={"test": "data"},
        )

        assert request.experiment_id == "exp_123"
        assert request.name == "Test Experiment"
        assert request.metadata == {"test": "data"}
        assert "accuracy" in request.measures

    def test_backend_experiment_run_request_creation(self):
        """Test BackendExperimentRunRequest creation."""
        request = BackendExperimentRunRequest(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_data={"function_name": "test_func"},
            metadata={"run_info": "test"},
        )

        assert request.experiment_id == "exp_123"
        assert request.run_id == "run_456"
        assert request.experiment_data["function_name"] == "test_func"

    def test_backend_configuration_run_request_creation(self):
        """Test BackendConfigurationRunRequest creation."""
        request = BackendConfigurationRunRequest(
            experiment_run_id="run_123",
            config_run_id="config_456",
            experiment_parameters={"config": {"param": "value"}},
            trial_metadata={"trial_info": "test"},
        )

        assert request.experiment_run_id == "run_123"
        assert request.config_run_id == "config_456"
        assert request.experiment_parameters["config"]["param"] == "value"

    def test_session_experiment_mapping_creation(self):
        """Test SessionExperimentMapping creation."""
        mapping = SessionExperimentMapping(
            session_id="session_123",
            experiment_id="exp_456",
            experiment_run_id="run_789",
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert mapping.session_id == "session_123"
        assert mapping.experiment_id == "exp_456"
        assert mapping.function_name == "test_func"
        assert mapping.trial_mappings == {}
        assert isinstance(mapping.created_at, datetime)

    def test_session_experiment_mapping_methods(self):
        """Test SessionExperimentMapping methods."""
        mapping = SessionExperimentMapping(
            session_id="session_123",
            experiment_id="exp_456",
            experiment_run_id="run_789",
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        # Test that mapping has trial_mappings attribute
        assert hasattr(mapping, "trial_mappings")
        assert isinstance(mapping.trial_mappings, dict)


class TestSDKBackendBridge:
    """Test SDK Backend Bridge functionality."""

    def test_bridge_initialization(self, sdk_bridge):
        """Test bridge initialization."""
        assert isinstance(sdk_bridge._session_mappings, dict)
        assert len(sdk_bridge._session_mappings) == 0
        assert "conversational" in sdk_bridge._agent_type_mappings
        assert "default" in sdk_bridge._agent_type_mappings

    def test_optimization_request_to_backend(self, sdk_bridge, optimization_request):
        """Test converting optimization request to backend format."""
        backend_request = sdk_bridge.optimization_request_to_backend(
            optimization_request, user_id="test_user"
        )

        assert isinstance(backend_request, BackendExperimentRequest)
        assert backend_request.name == "Optimization: llm_response_function"
        assert "llm_response_function" in backend_request.description
        assert backend_request.measures == ["accuracy", "cost"]
        assert backend_request.experiment_parameters["max_trials"] == 50
        assert backend_request.experiment_parameters["target_cost_reduction"] == 0.65
        assert backend_request.metadata == {"experiment_type": "test"}

        # Check agent data
        agent_data = backend_request.agent_data
        assert "agent_id" in agent_data
        assert agent_data["name"] == "Auto-generated agent for llm_response_function"
        assert "llm_response_function" in agent_data["description"]
        assert "llm_response_function tasks" in agent_data["prompt_template"]

        # Check benchmark data
        benchmark_data = backend_request.benchmark_data
        assert "benchmark_id" in benchmark_data
        assert benchmark_data["benchmark_name"] == "llm_response_function_benchmark"
        assert "llm_response_function" in benchmark_data["description"]

        # Check example set data
        example_set_data = backend_request.example_set_data
        assert "example_set_id" in example_set_data
        assert "llm_response_function_examples" in example_set_data["name"]
        assert len(example_set_data["examples"]) == 3

        # Check model parameters data
        model_params = backend_request.model_parameters_data
        assert "model_parameters_id" in model_params
        assert "model_id" in model_params
        assert "temperature" in model_params
        assert "max_tokens" in model_params

    def test_session_creation_to_backend_run(
        self, sdk_bridge, session_creation_request
    ):
        """Test converting session creation to backend run."""
        experiment_id = "exp_123"
        backend_run = sdk_bridge.session_creation_to_backend_run(
            session_creation_request, experiment_id
        )

        assert isinstance(backend_run, BackendExperimentRunRequest)
        assert backend_run.experiment_id == experiment_id
        # Since we don't generate IDs with timestamp prefixes anymore,
        # the run_id should just be the experiment_id
        assert backend_run.run_id == experiment_id

        experiment_data = backend_run.experiment_data
        assert experiment_data["function_name"] == "test_function"
        assert experiment_data["max_trials"] == 25
        assert experiment_data["user_id"] == "test_user"
        assert experiment_data["experiment_id"] == experiment_id
        assert backend_run.metadata == {"session_type": "test"}

    def test_trial_suggestion_to_config_run(self, sdk_bridge, trial_suggestion):
        """Test converting trial suggestion to configuration run."""
        experiment_run_id = "run_123"
        config_run = sdk_bridge.trial_suggestion_to_config_run(
            trial_suggestion, experiment_run_id
        )

        assert isinstance(config_run, BackendConfigurationRunRequest)
        assert config_run.experiment_run_id == experiment_run_id
        assert config_run.config_run_id == trial_suggestion.trial_id

        exp_params = config_run.experiment_parameters
        assert exp_params["trial_number"] == 1
        assert exp_params["config"] == {"param1": 2, "param2": "b"}
        assert exp_params["exploration_type"] == "exploration"
        assert exp_params["priority"] == 1
        assert exp_params["estimated_duration"] == 30.0

        # Check dataset subset conversion
        subset_data = exp_params["dataset_subset"]
        assert subset_data["indices"] == [0, 2, 4]
        assert subset_data["selection_strategy"] == "random"
        assert subset_data["confidence_level"] == 0.95

        # Check that original trial metadata is preserved
        assert "trial_info" in config_run.trial_metadata
        assert config_run.trial_metadata["trial_info"] == "test"

        # Check that enhanced metadata is also present
        assert "trial_id" in config_run.trial_metadata
        assert "mapping_created_at" in config_run.trial_metadata
        assert config_run.trial_metadata["trial_id"] == trial_suggestion.trial_id

    def test_agent_specification_to_backend(self, sdk_bridge, agent_specification):
        """Test converting agent specification to backend format."""
        backend_agent = sdk_bridge.agent_specification_to_backend(
            agent_specification, user_id="test_user"
        )

        assert backend_agent["agent_id"] == "agent_123"
        assert backend_agent["name"] == "Test Agent"
        assert backend_agent["description"] == "Agent generated from SDK specification"
        assert backend_agent["agent_type_id"] == "agent-type-1"  # conversational
        assert backend_agent["prompt_template"] == "Test prompt: {input}"
        assert backend_agent["reasoning"] == "chain-of-thought"
        assert backend_agent["style"] == "professional"
        assert backend_agent["tone"] == "helpful"
        assert backend_agent["format"] == "structured"
        assert backend_agent["persona"] == "expert"
        assert backend_agent["guidelines"] == ["be accurate", "be helpful"]
        assert backend_agent["response_validation"] is True
        assert backend_agent["custom_tools"] == ["calculator"]
        assert backend_agent["metadata"] == {"agent_version": "1.0"}

    def test_agent_type_mapping(self, sdk_bridge):
        """Test agent type mapping logic."""
        # Test known agent types
        for agent_type, expected_id in [
            ("conversational", "agent-type-1"),
            ("task", "agent-type-2"),
            ("analytical", "agent-type-3"),
            ("unknown_type", "agent-type-1"),  # default
        ]:
            agent_spec = AgentSpecification(
                id="test",
                name="Test",
                agent_type=agent_type,
                agent_platform="test",
                prompt_template="test",
                model_parameters={},
            )
            backend_agent = sdk_bridge.agent_specification_to_backend(agent_spec)
            assert backend_agent["agent_type_id"] == expected_id

    def test_session_mapping_operations(self, sdk_bridge):
        """Test session mapping CRUD operations."""
        session_id = "session_123"
        experiment_id = "exp_456"
        experiment_run_id = "run_789"

        # Create mapping
        mapping = sdk_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert isinstance(mapping, SessionExperimentMapping)
        assert mapping.session_id == session_id
        assert mapping.experiment_id == experiment_id
        assert mapping.experiment_run_id == experiment_run_id

        # Get mapping
        retrieved_mapping = sdk_bridge.get_session_mapping(session_id)
        assert retrieved_mapping == mapping

        # Test nonexistent session
        assert sdk_bridge.get_session_mapping("nonexistent") is None

        # Add trial mapping
        trial_id = "trial_123"
        config_run_id = "config_456"
        sdk_bridge.add_trial_mapping(session_id, trial_id, config_run_id)

        # Verify trial mapping
        assert sdk_bridge.get_trial_mapping(session_id, trial_id) == config_run_id
        assert sdk_bridge.get_trial_mapping(session_id, "nonexistent_trial") is None
        assert sdk_bridge.get_trial_mapping("nonexistent_session", trial_id) is None

    def test_create_agent_data_from_function(self, sdk_bridge):
        """Test agent data creation from function parameters."""
        agent_data = sdk_bridge._create_agent_data_from_function(
            function_name="test_function",
            configuration_space={"temperature": [0.1, 0.5], "model": ["gpt-3.5"]},
            user_id="test_user",
        )

        assert "agent_id" in agent_data
        assert agent_data["name"] == "Auto-generated agent for test_function"
        assert "test_function" in agent_data["description"]
        assert "test_function tasks" in agent_data["prompt_template"]
        assert agent_data["agent_type_id"] == "agent-type-1"  # conversational
        assert agent_data["response_validation"] is True
        assert agent_data["chat_history"] is False
        assert agent_data["custom_tools"] == []

    def test_create_benchmark_data_from_dataset(self, sdk_bridge, sample_dataset):
        """Test benchmark data creation from dataset."""
        benchmark_data = sdk_bridge._create_benchmark_data_from_dataset(
            sample_dataset, "test_function"
        )

        assert "benchmark_id" in benchmark_data
        assert benchmark_data["benchmark_name"] == "test_function_benchmark"
        assert "test_function" in benchmark_data["description"]
        assert benchmark_data["type"] == "qa"  # Based on string expected_output
        assert benchmark_data["agent_type_id"] == "agent-type-1"
        assert "test_function" in benchmark_data["label"]

    def test_create_example_set_data_from_dataset(self, sdk_bridge, sample_dataset):
        """Test example set data creation from dataset."""
        example_set_data = sdk_bridge._create_example_set_data_from_dataset(
            sample_dataset, "test_examples"
        )

        assert "example_set_id" in example_set_data
        assert example_set_data["name"] == "test_examples"
        assert example_set_data["type"] == "input-output"
        assert "3 examples" in example_set_data["description"]
        assert len(example_set_data["examples"]) == 3

        # Check example conversion
        first_example = example_set_data["examples"][0]
        assert "example_id" in first_example
        assert first_example["input"] == "What is machine learning?"
        assert (
            first_example["output"]
            == "Machine learning is a subset of artificial intelligence."
        )
        assert "difficulty:easy" in first_example["tags"]

    def test_serialize_input_data(self, sdk_bridge):
        """Test input data serialization."""
        # Test string input
        assert sdk_bridge._serialize_input_data("test string") == "test string"

        # Test dict with query
        dict_query = {"query": "test query", "other": "data"}
        assert sdk_bridge._serialize_input_data(dict_query) == "test query"

        # Test dict with question
        dict_question = {"question": "test question", "other": "data"}
        assert sdk_bridge._serialize_input_data(dict_question) == "test question"

        # Test dict with input
        dict_input = {"input": "test input", "other": "data"}
        assert sdk_bridge._serialize_input_data(dict_input) == "test input"

        # Test dict with string value
        dict_string_value = {"field1": "string value", "field2": 123}
        assert sdk_bridge._serialize_input_data(dict_string_value) == "string value"

        # Test dict with no string values
        dict_no_strings = {"field1": 123, "field2": 456}
        result = sdk_bridge._serialize_input_data(dict_no_strings)
        assert "field1" in result and "field2" in result

        # Test non-string, non-dict input
        assert sdk_bridge._serialize_input_data(123) == "123"
        assert sdk_bridge._serialize_input_data([1, 2, 3]) == "[1, 2, 3]"

    def test_create_model_parameters_from_config_space(self, sdk_bridge):
        """Test model parameters creation from configuration space."""
        config_space = {
            "model": ["gpt-4", "gpt-3.5"],
            "temperature": [0.3, 0.7, 1.0],
            "max_tokens": [100, 200],
            "top_p": [0.8, 1.0],
        }

        model_params = sdk_bridge._create_model_parameters_from_config_space(
            config_space
        )

        assert "model_parameters_id" in model_params
        assert model_params["model_id"] == "gpt-4"  # First in list
        assert model_params["temperature"] == 0.3  # First in list
        assert model_params["max_tokens"] == 100  # First in list
        assert model_params["top_p"] == 0.8  # First in list
        assert model_params["frequency_penalty"] == 0.0
        assert model_params["presence_penalty"] == 0.0

    def test_create_model_parameters_defaults(self, sdk_bridge):
        """Test model parameters with defaults."""
        config_space = {}  # Empty config space

        model_params = sdk_bridge._create_model_parameters_from_config_space(
            config_space
        )

        assert model_params["model_id"] == "gpt-3.5"
        assert model_params["temperature"] == 0.7
        assert model_params["max_tokens"] == 150
        assert model_params["top_p"] == 1.0
        assert model_params["frequency_penalty"] == 0.0
        assert model_params["presence_penalty"] == 0.0

    def test_create_model_parameters_single_values(self, sdk_bridge):
        """Test model parameters with single values."""
        config_space = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 200,
            "top_p": 0.9,
        }

        model_params = sdk_bridge._create_model_parameters_from_config_space(
            config_space
        )

        # Single string values that aren't lists fall back to default
        assert (
            model_params["model_id"] == "gpt-3.5"
        )  # Single string falls back to default
        assert model_params["temperature"] == 0.5
        assert model_params["max_tokens"] == 200
        assert model_params["top_p"] == 0.9

    def test_map_objectives_to_measures(self, sdk_bridge):
        """Test objective to measure mapping."""
        # Test standard mappings
        objectives = ["accuracy", "cost", "latency"]
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert set(measures) == {"accuracy", "cost", "latency"}

        # Test alternative mappings
        objectives = ["cost_efficiency", "response_time", "success_rate"]
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert set(measures) == {"cost", "latency", "accuracy"}

        # Test case insensitivity
        objectives = ["ACCURACY", "Cost", "LATENCY"]
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert set(measures) == {"accuracy", "cost", "latency"}

        # Test unknown objectives
        objectives = ["unknown_metric"]
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert measures == ["accuracy"]  # Default fallback

        # Test empty objectives
        objectives = []
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert measures == ["accuracy"]  # Default fallback

        # Test duplicate removal
        objectives = ["accuracy", "success_rate", "accuracy"]
        measures = sdk_bridge._map_objectives_to_measures(objectives)
        assert measures == ["accuracy"]  # No duplicates

    def test_convert_dataset_to_examples_edge_cases(self, sdk_bridge):
        """Test dataset to examples conversion with edge cases."""
        # Test dataset without examples
        dataset_no_examples = Dataset(examples=[], name="empty")
        examples = sdk_bridge._convert_dataset_to_examples(dataset_no_examples)
        assert examples == []

        # Test example without expected_output
        examples_input_only = [
            EvaluationExample(input_data="test input", expected_output=None)
        ]
        dataset_input_only = Dataset(examples=examples_input_only, name="input_only")
        examples = sdk_bridge._convert_dataset_to_examples(dataset_input_only)
        assert len(examples) == 1
        assert examples[0]["output"] is None

        # Test example without metadata
        examples_no_metadata = [
            EvaluationExample(input_data="test", expected_output="output")
        ]
        dataset_no_metadata = Dataset(examples=examples_no_metadata, name="no_meta")
        examples = sdk_bridge._convert_dataset_to_examples(dataset_no_metadata)
        assert len(examples) == 1
        assert examples[0]["tags"] == []


class TestGlobalBridgeInstance:
    """Test the global bridge instance."""

    def test_global_bridge_exists(self):
        """Test that global bridge instance exists."""
        assert bridge is not None
        assert isinstance(bridge, SDKBackendBridge)

    def test_global_bridge_functionality(self, optimization_request):
        """Test that global bridge instance works correctly."""
        backend_request = bridge.optimization_request_to_backend(optimization_request)
        assert isinstance(backend_request, BackendExperimentRequest)
        assert backend_request.name == "Optimization: llm_response_function"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_configuration_space(self, sdk_bridge):
        """Test handling of empty configuration space."""
        model_params = sdk_bridge._create_model_parameters_from_config_space({})
        assert model_params["model_id"] == "gpt-3.5"
        assert model_params["temperature"] == 0.7

    def test_malformed_configuration_space(self, sdk_bridge):
        """Test handling of malformed configuration space."""
        config_space = {
            "model": [],  # Empty list
            "temperature": None,  # None value
            "max_tokens": "invalid",  # String instead of number
        }

        model_params = sdk_bridge._create_model_parameters_from_config_space(
            config_space
        )
        assert model_params["model_id"] == "gpt-3.5"  # Default
        assert model_params["temperature"] is None  # None value passed through
        assert model_params["max_tokens"] == "invalid"  # Takes as-is for non-list

    def test_empty_lists_use_defaults(self, sdk_bridge):
        """Empty configuration lists should fall back to defaults without crashing."""
        config_space = {
            "model": ["gpt-test"],
            "temperature": [],
            "max_tokens": [],
            "top_p": [],
        }

        model_params = sdk_bridge._create_model_parameters_from_config_space(
            config_space
        )

        assert model_params["model_id"] == "gpt-test"
        assert model_params["temperature"] == 0.7
        assert model_params["max_tokens"] == 150
        assert model_params["top_p"] == 1.0

    def test_experiment_id_generation(self, sdk_bridge, optimization_request):
        """Test deterministic experiment ID generation."""
        # Generate backend request
        backend_request = sdk_bridge.optimization_request_to_backend(
            optimization_request
        )

        # Verify experiment_id format is deterministic hash
        assert backend_request.experiment_id.startswith("exp_")
        assert len(backend_request.experiment_id) == 20  # "exp_" + 16 hex chars

        # Verify same request generates same ID (deterministic)
        backend_request2 = sdk_bridge.optimization_request_to_backend(
            optimization_request
        )
        assert backend_request.experiment_id == backend_request2.experiment_id

    def test_dataset_type_inference(self, sdk_bridge):
        """Test dataset type inference for benchmark creation."""
        # Test with numeric expected_output (classification)
        numeric_examples = [
            EvaluationExample(input_data="input1", expected_output=1),
            EvaluationExample(input_data="input2", expected_output=0),
        ]
        numeric_dataset = Dataset(examples=numeric_examples, name="numeric")
        benchmark_data = sdk_bridge._create_benchmark_data_from_dataset(
            numeric_dataset, "func"
        )
        assert benchmark_data["type"] == "classification"

        # Test with string expected_output (qa)
        string_examples = [
            EvaluationExample(input_data="input1", expected_output="output1"),
            EvaluationExample(input_data="input2", expected_output="output2"),
        ]
        string_dataset = Dataset(examples=string_examples, name="string")
        benchmark_data = sdk_bridge._create_benchmark_data_from_dataset(
            string_dataset, "func"
        )
        assert benchmark_data["type"] == "qa"

    def test_example_set_type_inference(self, sdk_bridge):
        """Test example set type inference."""
        # Test with expected_output (input-output)
        io_examples = [EvaluationExample(input_data="input", expected_output="output")]
        io_dataset = Dataset(examples=io_examples, name="io")
        example_set_data = sdk_bridge._create_example_set_data_from_dataset(
            io_dataset, "test"
        )
        assert example_set_data["type"] == "input-output"

        # Test without expected_output (input-only)
        input_examples = [EvaluationExample(input_data="input", expected_output=None)]
        input_dataset = Dataset(examples=input_examples, name="input")
        example_set_data = sdk_bridge._create_example_set_data_from_dataset(
            input_dataset, "test"
        )
        assert example_set_data["type"] == "input-only"
