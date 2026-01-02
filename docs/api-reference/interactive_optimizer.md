# Interactive Optimizer API

The `InteractiveOptimizer` enables hybrid optimization: a remote service
proposes configurations and dataset subsets while execution stays local.
All interactions are async.

## Import

```python
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer
```

## Constructor

```python
InteractiveOptimizer(
    config_space: dict[str, Any],
    objectives: list[str],
    remote_service: RemoteGuidanceService,
    dataset_metadata: dict[str, Any] | None = None,
    optimization_strategy: dict[str, Any] | None = None,
    context: TraigentConfig | None = None,
    **kwargs: Any,
) -> InteractiveOptimizer
```

| Parameter | Type | Description |
| --- | --- | --- |
| `config_space` | `dict[str, Any]` | Search space for optimization. |
| `objectives` | `list[str]` | Objectives to optimize (e.g., `["accuracy", "cost"]`). |
| `remote_service` | `RemoteGuidanceService` | Async service that returns trial suggestions. |
| `dataset_metadata` | `dict[str, Any] \| None` | Dataset hints (size, type, etc.). |
| `optimization_strategy` | `dict[str, Any] \| None` | Strategy hints for the remote service. |
| `context` | `TraigentConfig \| None` | Optional SDK config context. |
| `**kwargs` | `Any` | Passed to `BaseOptimizer`. |

## Remote Guidance Interface

The remote service must implement the `RemoteGuidanceService` protocol:

```python
class RemoteGuidanceService(Protocol):
    async def create_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse: ...

    async def get_next_trial(self, request: NextTrialRequest) -> NextTrialResponse: ...

    async def submit_result(self, result: TrialResultSubmission) -> None: ...

    async def finalize_session(
        self, request: OptimizationFinalizationRequest
    ) -> OptimizationFinalizationResponse: ...
```

## Methods

### `initialize_session`

```python
await optimizer.initialize_session(
    function_name: str,
    max_trials: int,
    user_id: str | None = None,
    billing_tier: str = "standard",
) -> OptimizationSession
```

Creates a remote optimization session and stores session state locally.

### `get_next_suggestion`

```python
await optimizer.get_next_suggestion(
    dataset_size: int,
    previous_results: list[TrialResultSubmission] | None = None,
) -> TrialSuggestion | None
```

Requests the next configuration and dataset subset. Returns `None` when
optimization is complete.

### `report_results`

```python
await optimizer.report_results(
    trial_id: str,
    metrics: dict[str, float],
    duration: float,
    status: TrialStatus = TrialStatus.COMPLETED,
    outputs_sample: list[Any] | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None
```

Sends trial metrics back to the remote service and updates local session state.

### `get_optimization_status`

```python
await optimizer.get_optimization_status() -> dict[str, Any]
```

Returns session status, progress, and best metrics/config if available.

### `finalize_optimization`

```python
await optimizer.finalize_optimization(
    include_full_history: bool = False,
) -> OptimizationFinalizationResponse
```

Finalizes the session and returns the remote summary.

### `suggest_next_trial`

```python
optimizer.suggest_next_trial(history: list[TrialResult]) -> dict[str, Any]
```

Not supported. This raises `NotImplementedError`; use `get_next_suggestion`.

## Example

```python
optimizer = InteractiveOptimizer(
    config_space={"temperature": (0.0, 1.0)},
    objectives=["accuracy"],
    remote_service=remote_service,
    dataset_metadata={"size": 1000},
)

await optimizer.initialize_session(function_name="answer", max_trials=20)

while True:
    suggestion = await optimizer.get_next_suggestion(dataset_size=1000)
    if suggestion is None:
        break

    metrics = run_local_evaluation(
        suggestion.config,
        suggestion.dataset_subset.indices,
    )
    await optimizer.report_results(
        trial_id=suggestion.trial_id,
        metrics=metrics,
        duration=metrics.get("duration", 0.0),
    )

summary = await optimizer.finalize_optimization()
print(summary.best_config)
```
