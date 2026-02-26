# Phase 3 - Backend Status

## Branch
- `observability-backend-phase-3-broken-example-intelligence`

## Progress
- status: done
- completed_tasks:
  - Added stable broken-example DTO schema definitions in `src/schemas/observability_broken_examples.py`.
  - Added backend service methods for broken-example ranking and drilldown evidence:
    - `ObservabilityService.list_run_broken_examples`
    - `ObservabilityService.get_run_broken_example_evidence`
  - Added deterministic filter handling (`metric`, `min_samples`, `max_items`, `segment`, `direction`) with confidence metadata.
  - Added tenant-safe trial visibility gates and bounded query protections for broken-example collection.
  - Added routes:
    - `GET /api/observability/runs/<run_id>/broken-examples`
    - `GET /api/observability/runs/<run_id>/broken-examples/<example_id>/evidence`
  - Added service and route tests for broken-example filters, evidence, and not-found handling.
  - Aligned backend parsing to prefer SDK `failure_classification_detail` when provided.
- smoke_tests:
  - `.venv/bin/pytest tests/unit/services/test_observability_correlation_service.py::test_list_run_broken_examples_returns_ranked_items tests/unit/services/test_observability_correlation_service.py::test_get_run_broken_example_evidence_returns_sorted_points tests/unit/routes/test_observability_v2_routes.py::test_broken_examples_route_passes_filters tests/unit/routes/test_observability_v2_routes.py::test_broken_example_evidence_route_not_found_returns_404 -q` -> PASS (`4 passed`)
- tests:
  - `.venv/bin/ruff check src/services/observability_service.py src/routes/observability_routes.py src/schemas/observability_broken_examples.py tests/unit/services/test_observability_correlation_service.py tests/unit/routes/test_observability_v2_routes.py` -> PASS
  - `.venv/bin/pytest tests/unit/services/test_observability_correlation_service.py tests/unit/routes/test_observability_v2_routes.py -q` -> PASS (`24 passed`)
- blockers:
  - None.
- residual_risks:
  - Large-run broken-example analysis still bounded in-memory; tuned row caps guard worker memory but may require future pagination for very large tenants.
- notes:
  - latest_commit: `698cbd452118e1bc5af975423185fa6f581b0870`
