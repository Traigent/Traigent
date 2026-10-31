# Traigent Cloud Integration Module

This module provides backend integration capabilities for Traigent SDK, including portal-tracked hybrid sessions, session/result submission, authentication, and analytics.

This module supports backend and portal integration for SDK-run trials. Users
should select optimization behavior with `algorithm` and `offline`; the legacy
execution-mode field is kept only for compatibility.

## Components

### DTOs (`dtos.py`)
Data Transfer Objects that provide type-safe communication with the backend:

- **ExperimentDTO**: Represents optimization experiments with conservative defaults
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
- Session management
- Asynchronous and synchronous interfaces
- Fail-closed behavior for backend-required operations when dependencies or credentials are missing

### Models (`models.py`)
Core data models for optimization requests, sessions, and results.

### Optimizer Client (`optimizer_client.py`)
Direct metric submission for hybrid mode, where the SDK executes trials locally while the backend tracks sessions and results.

## DTO defaults

DTOs null out some identifier fields by default. This is a payload-shaping default,
not a privacy guarantee: Traigent observability is not privacy-preserving, and
the `privacy_mode` metadata flag is a legacy marker, not a data-protection control.
`offline=True` stops submission to the Traigent backend and, for the duration of the
run, also switches off OpenTelemetry trace export, the langfuse client, workflow-trace
ingestion and the standalone API clients (they refuse with `OfflineModeError`, or
return a failed result for trace ingestion). It does not keep everything on the
machine: model-provider calls your own function makes use your own provider
credentials and are not gated by it. Traigent observability is not privacy-preserving.
The switch is process-wide while any `offline=True` run is in flight (issue #2240):

```python
# Example: Creating a Edge Analytics experiment
experiment = create_local_experiment(
    experiment_id="exp_123",
    name="My Experiment",
    description="Testing DTO defaults",
    configuration_space={...},
    max_trials=10,
    dataset_size=100
)

# Sensitive fields are automatically set to None
assert experiment.agent_id is None
assert experiment.benchmark_id is None

# Legacy metadata flag (informational only)
assert experiment.metadata["privacy_mode"] is True
assert experiment.metadata["execution_mode"] == "local"
```

## Usage

The backend integration module is automatically used by the Traigent orchestrator when portal tracking is enabled. No direct interaction is typically needed:

```python
from traigent import optimize

@optimize(
    model="gpt-5.6-terra",
    algorithm="auto",  # Backend suggestions with SDK-run trials
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
2. Include conservative (nullable) defaults
3. Add validation methods
4. Update tests

Example:
```python
@dataclass
class NewFeatureDTO:
    id: str
    name: str
    # Nullable by default
    sensitive_field: Optional[str] = None

    # Legacy metadata flag (informational only)
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "privacy_mode": True,
        "execution_mode": "local"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
```
