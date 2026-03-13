# Enterprise Beta API Reference

This document covers the enterprise-beta surfaces added on top of the existing Traigent APIs:

- project-scoped analytics dashboards
- pricing and export history
- project membership and governance policies

All endpoints documented here are **project-scoped** and assume the backend tenant/project enforcement kernel is active.

## Auth and Scope

Authentication:
- browser/JWT session, or
- API key

Required scope context:
- tenant is resolved by the backend auth/context chain
- project is explicit in the URL path

Headers commonly used:
- `Authorization: Bearer <token>`
- `X-API-Key: <key>`
- `X-Tenant-Id: <tenant_id>` when tenant selection is needed in browser/API flows

Canonical path shape:

```text
/api/v1beta/projects/{project_id}/...
```

Authorization rules:
- `viewer`: read-only dashboards, summaries, export-job history
- `editor`: export generation and other write-like operational actions
- `admin`: membership and governance policy management

## Analytics Dashboards

Base path:

```text
/api/v1beta/projects/{project_id}/analytics
```

### GET `/summary`
Returns aggregate-safe project analytics.

Query params:
- `days`

Response highlights:
- `context.privacy_classification`
- `entity_counts`
- `status_breakdowns`
- `usage_summary`
- `measure_summaries`

### GET `/pricing-catalog`
Returns the project-visible pricing catalog and fallback metadata.

Response highlights:
- `catalog_source`
- `total_providers`
- `total_models`
- `providers[*].models[*]`

### GET `/dashboards/optimization-overview`
Returns the optimization overview dashboard.

Query params:
- `days`
- `limit`

Response highlights:
- `summary_cards`
- `cost_source_breakdown`
- `recent_experiments`

### GET `/dashboards/evaluator-quality`
Returns evaluator quality and throughput trends.

Query params:
- `days`
- `limit`

Response highlights:
- `summary_cards`
- `score_trend`
- `top_evaluators`

### GET `/dashboards/project-usage`
Returns project-level usage, cost, token, and latency summaries.

Query params:
- `days`
- `limit`

Response highlights:
- `summary_cards`
- `run_volume_trend`
- `top_experiments`

### GET `/dashboards/observability-summary`
Returns aggregate observability activity for the active project.

Query params:
- `days`
- `limit`

Response highlights:
- `summary_cards`
- `trace_volume_trend`
- `top_traces`

### GET `/trends/run-volume`
Returns run-volume time series.

Query params:
- `days`
- `bucket`
- `experiment_id`

### GET `/distributions/measures/{measure_key}`
Returns a measure histogram/distribution.

Query params:
- `bins`
- `experiment_id`

## Export APIs

### GET `/exports/fine-tuning.manifest`
Generates a fine-tuning manifest export.

Authorization:
- requires project `editor`

Query params:
- `experiment_id`
- `experiment_run_id`
- `limit`
- `include_content`

Behavior:
- privacy-safe manifest export is the default
- if `include_content=true` and project policy disallows materialized export, the route returns `403`
- successful and failed export attempts create audit events
- artifacts are persisted through the provider-neutral artifact-store interface

Response highlights:
- `job_id`
- `context.privacy_classification`
- `privacy_mode`
- `include_content`
- `record_count`
- `records`

### GET `/export-jobs`
Lists export jobs for the active project.

Authorization:
- requires project `viewer`

Query params:
- `page`
- `per_page`

Response highlights:
- `items[*].job_id`
- `items[*].status`
- `items[*].retention_category`
- `items[*].expires_at`
- `items[*].error_message`

### GET `/export-jobs/{job_id}`
Returns a single export-job record if it belongs to the active project.

Authorization:
- requires project `viewer`

## Project Governance APIs

Base path:

```text
/api/v1beta/projects
```

### Memberships

#### GET `/{project_id}/memberships`
Lists memberships for a project.

Authorization:
- requires project `admin`

Query params:
- `page`
- `per_page`
- `role`
- `status`

#### POST `/{project_id}/memberships`
Creates a project membership.

Authorization:
- requires project `admin`

Notes:
- the target user must already have an **active tenant membership**
- membership changes are audited

#### PATCH `/{project_id}/memberships/{membership_id}`
Updates membership role and/or status.

Authorization:
- requires project `admin`

Notes:
- the service prevents removal of the last effective project admin

### Rate-limit policy

#### GET `/{project_id}/policies/rate-limits`
Returns the current beta rate-limit policy.

Authorization:
- requires project `admin`

#### PATCH `/{project_id}/policies/rate-limits`
Updates the beta rate-limit policy.

Authorization:
- requires project `admin`

### Retention policy

#### GET `/{project_id}/policies/retention`
Returns the current retention policy.

Authorization:
- requires project `admin`

Beta retention categories:
- `export_artifact`
- `materialized_export`

#### PATCH `/{project_id}/policies/retention`
Updates retention windows for the beta-supported categories.

Authorization:
- requires project `admin`

### Export policy

#### GET `/{project_id}/policies/export`
Returns export-policy settings for the project.

Authorization:
- requires project `admin`

#### PATCH `/{project_id}/policies/export`
Updates export-policy settings, including materialized-export gating.

Authorization:
- requires project `admin`

## Privacy Mode Expectations

All analytics and export APIs in beta must function correctly in privacy-safe mode.

Expected behavior:
- dashboards use only aggregate-safe values
- exports default to manifest/reference mode
- raw content is returned only when explicitly allowed by policy and request mode
- `privacy_classification` is present on relevant analytics/export responses

## Error Conventions

Common patterns:
- `403`: in-scope resource exists but caller lacks required project role
- `404`: project-scoped resource is not visible in the active tenant/project
- `422`: request validation failure

Examples:
- `PROJECT_EXPORT_POLICY_DENIED`
- `EXPORT_FAILED`
- `EXPORT_STORAGE_FAILED`

## Related Documents
- [enterprise_beta_execution_plan.md](../feature_matrices/enterprise_beta_execution_plan.md)
- [enterprise_beta_execution_tracker.md](../feature_matrices/enterprise_beta_execution_tracker.md)
- [enterprise_beta_threat_model.md](../architecture/enterprise_beta_threat_model.md)
