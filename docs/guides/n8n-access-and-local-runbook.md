# n8n Access and Local Runbook

This runbook covers two goals:

1. Understand how to run n8n with Traigent integration locally.
2. Track external access dependencies for Yossi's n8n environment.

## Scope

This guide is for the `n8n-nodes-traigent` community node and Hybrid Mode API flows.

It does not change DTOs or API contracts.

## Prerequisites

1. Python environment with Traigent repository checked out.
2. Node.js + npm.
3. Access to these repositories:
   - `Traigent`
   - `n8n-nodes-traigent` (GitHub repo: `Traigent/traigent-n8n`)

## Local Bring-Up (Verified Path)

### Step 1: Start a local Traigent-wrapped service

From `Traigent`:

```bash
cd examples/experimental/hybrid_api_demo
python app.py
```

In another terminal, verify the service:

```bash
python test_mastra_js_api.py
```

Expected: endpoint tests pass for health/capabilities/config-space/execute/evaluate.

### Step 2: Build and test the n8n node package

From `n8n-nodes-traigent`:

```bash
npm install
npm run build
npm test
```

Expected: build succeeds and test suite passes.

### Step 3: Install node into n8n

Option A (recommended for end users):

1. In n8n UI: `Settings -> Community Nodes`.
2. Install package: `n8n-nodes-traigent`.

Option B (local developer path):

```bash
# in n8n-nodes-traigent
npm link

# in your n8n custom folder
npm link n8n-nodes-traigent

# start n8n
npx n8n start
```

### Step 4: Configure credentials

1. In n8n: `Credentials -> New`.
2. Choose `Traigent API`.
3. Set `Service URL` to local service (example: `http://localhost:8080`).
4. Add API key only if service requires auth.

### Step 5: Import and run a smoke workflow

Import one of these from `n8n-nodes-traigent/examples/`:

1. `demo-1-health-check.json`
2. `demo-3-discover-and-run.json`

Smoke criteria:

1. `Health Check` returns healthy status.
2. `Get Capabilities` and `Get Config Space` return non-empty payloads.
3. `Execute` returns `outputs` and `operational_metrics`.
4. If evaluate is enabled, `Evaluate` returns per-example metrics.

## Yossi Environment Access Checklist (External Dependency)

Use this checklist before attempting project-level optimization work:

1. Workspace access granted (owner + date recorded).
2. Runtime target confirmed (`n8n cloud` vs `self-hosted`).
3. Environment variable contract documented (keys, location, rotation owner).
4. Network allow-list/proxy requirements confirmed.
5. One known workflow selected for first Traigent integration.

Status template:

```text
workspace_access: pending|granted
runtime_target: cloud|self_hosted
secrets_contract: pending|defined
first_workflow: <name>
owner: <name>
```

## Definition of Done for Access + Understanding

`#203` can be closed when both are true:

1. Local bring-up succeeds end-to-end using this runbook.
2. Yossi environment access checklist is complete (not pending).
