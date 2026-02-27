# Observability Smoke Run

Date: 2026-02-26

## Environment checks
- Backend health: `http://localhost:5000/health` -> 200
- Frontend dev server: `http://localhost:3000` -> 200

## SDK run command
From `/home/nimrodbu/Traigent_enterprise/Traigent`:

```bash
source .venv/bin/activate
set -a && source .env && set +a
export TRAIGENT_OFFLINE_MODE=false
export TRAIGENT_TRACES_ENABLED=true
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py
```

## Observed IDs from run logs
- `experiment_id=361afbabf6aebabbf38d82d97e815cc1`
- `run_id=1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee`

## Endpoint validation (with `X-API-Key`)
- `GET /api/observability/runs/1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee?include_spans=true` -> 200
  - trials=5
  - total_spans=5
  - graph_nodes=3
- `GET /api/observability/runs/1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee/trials?include_spans=true&limit=50` -> 200
  - items=5
- `GET /api/observability/runs/1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee/attribution` -> 200
  - items=0
- `GET /api/observability/dashboard/runs/1c64efa5-a13e-4aa5-be5a-ec2ba4db80ee/cost-latency?granularity=hour&limit=100` -> 200
  - items=1
- `GET /api/observability/dashboard/run-health?limit=10` -> 200
  - items=10

## FE URLs to inspect
- Experiment view: `http://localhost:3000/experiments/view/361afbabf6aebabbf38d82d97e815cc1`
- Dashboard route (if enabled in nav/router): `http://localhost:3000/observability`
