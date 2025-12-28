# Traigent Cloud Integration Module

This module provides the cloud integration capabilities for Traigent SDK, enabling communication with the OptiGen backend for optimization tracking and analytics.

## Components

### DTOs (`dtos.py`)
Data Transfer Objects based on `optigen_schemas` that provide type-safe communication with the backend while preserving privacy:

- **ExperimentDTO**: Represents optimization experiments with privacy-preserving defaults
- **ExperimentRunDTO**: Tracks individual optimization runs
- **ConfigurationRunDTO**: Manages configuration trials and results

Key features:
- Nullable fields for sensitive data (agent_id, benchmark_id, etc.)
- Empty dataset indices to protect data content
- Privacy metadata flags
- Optional schema validation

### Backend Client (`backend_client.py`)
The `BackendIntegratedClient` provides:
- HTTP communication with OptiGen backend
- Privacy-preserving session management
- Asynchronous and synchronous interfaces
- Automatic fallback for offline operation

### Models (`models.py`)
Core data models for optimization requests, sessions, and results.

### Optimizer Client (`optimizer_client.py`)
Direct integration with the OptiGen Optimizer service for advanced optimization strategies.

## Privacy-Preserving Architecture

All DTOs implement privacy protection:

```python
# Example: Creating a Edge Analytics experiment
experiment = create_local_experiment(
    experiment_id="exp_123",
    name="My Experiment",
    description="Testing privacy mode",
    configuration_space={...},
    max_trials=10,
    dataset_size=100
)

# Sensitive fields are automatically set to None
assert experiment.agent_id is None
assert experiment.benchmark_id is None

# Privacy metadata is included
assert experiment.metadata["privacy_mode"] is True
assert experiment.metadata["execution_mode"] == "edge_analytics"
```

## Usage

The cloud module is automatically used by the Traigent orchestrator. No direct interaction is typically needed:

```python
from traigent import optimize

@optimize(
    model="gpt-3.5-turbo",
    optimization_mode="edge_analytics"  # Works with all modes
)
def my_function(prompt: str) -> str:
    return "response"

# Metadata is automatically submitted to backend
result = my_function("test")
```

## Configuration

### Backend URL
- Default: `http://localhost:5000`
- Environment variable: `TRAIGENT_BACKEND_URL`
- Optional API base override: `TRAIGENT_API_URL`
- Config parameter: `backend_base_url`

### API Key
- Required for authentication
- Set via `TRAIGENT_API_KEY` environment variable
- Or pass to `@optimize` decorator

## Testing

Run tests specific to cloud integration:
```bash
pytest tests/unit/cloud/
pytest tests/integration/test_backend_integration.py
```

## Development

When adding new DTOs:
1. Base them on existing optigen_schemas
2. Include privacy-preserving defaults
3. Add validation methods
4. Update tests

Example:
```python
@dataclass
class NewFeatureDTO:
    id: str
    name: str
    # Nullable for privacy
    sensitive_field: Optional[str] = None

    # Privacy metadata
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "privacy_mode": True,
        "execution_mode": "edge_analytics"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
```
