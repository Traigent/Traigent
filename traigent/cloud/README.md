# Traigent Cloud Integration Module

This module provides backend integration capabilities for Traigent SDK, including portal-tracked hybrid sessions, privacy-preserving session/result submission, authentication, and analytics.

Remote cloud execution is intentionally not implemented yet. `execution_mode="cloud"` fails closed with: “Cloud remote execution is not available yet; use hybrid for portal-tracked optimization.”

## Components

### DTOs (`dtos.py`)
Data Transfer Objects that provide type-safe communication with the backend while preserving privacy:

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
- HTTP communication with Traigent backend
- Privacy-preserving session management
- Asynchronous and synchronous interfaces
- Fail-closed behavior for backend-required operations when dependencies or credentials are missing

### Models (`models.py`)
Core data models for optimization requests, sessions, and results.

### Optimizer Client (`optimizer_client.py`)
Direct metric submission for hybrid mode, where the SDK executes trials locally while the backend tracks sessions and results.

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

The backend integration module is automatically used by the Traigent orchestrator when portal tracking is enabled. No direct interaction is typically needed:

```python
from traigent import optimize

@optimize(
    model="gpt-3.5-turbo",
    execution_mode="hybrid"  # Local trials plus backend/portal tracking
)
def my_function(prompt: str) -> str:
    return "response"

# Session and trial metrics are submitted to the backend.
result = my_function("test")
```

## Configuration

### Backend URL
- Default: production portal/backend configuration, unless explicitly overridden for local SDK development
- Environment variable: `TRAIGENT_BACKEND_URL`
- Optional API base override: `TRAIGENT_API_URL`
- Config parameter: `backend_base_url`

### API Key
- Required for authenticated portal tracking
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
1. Base them on existing Traigent schemas
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
