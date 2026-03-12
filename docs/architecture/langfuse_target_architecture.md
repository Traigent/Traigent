# Langfuse Replacement Target Architecture

Date: 2026-03-12

## Summary

This document defines the architectural constraints for the remaining Langfuse
replacement work. It is not a roadmap. It is the target operating model every later
wave must respect.

## Enterprise Operating Model

The required backend-enforced chain is:

`authenticate -> resolve tenant -> resolve project -> authorize -> scope data -> audit`

Interpretation:

- authentication proves identity
  - JWT/cookie for users
  - API keys for services
  - SSO/MFA where configured
- tenant and project context are resolved centrally in backend
- authorization is enforced centrally in backend
- data scoping is enforced in backend, not only in the UI
- sensitive actions are auditable

## Canonical Entities

These concepts are fixed and should not be duplicated:

- `Measure` is the only canonical metric or score-definition entity
- `EvaluatorDefinition` is the only canonical automated scorer-definition entity
- `ScoreRecord` is the only canonical measured outcome entity

Derived rule:

- dashboards are query outputs, not domain entities

## Deployment Model

- strategic target: one shared deployment with backend-enforced `tenant -> project -> resource` isolation
- compatibility mode: existing single-tenant deployments continue to work as one default tenant plus one default project
- tenant-per-deployment may remain a supported ops choice, but it is not the strategic architecture target

## Privacy Mode

Privacy mode remains a default product behavior and must stay supported everywhere.

Safe by default:

- ids
- hashes
- local or customer-controlled refs
- aggregates
- scores
- token, cost, and latency summaries

Explicit-only:

- raw input/output export
- plaintext materialization outside a customer-controlled environment

Machine-checkable requirement:

- analytics and export contracts must carry schema-level privacy metadata such as `privacy_classification`

## Architectural Red Lines

- no new public wire contract outside `TraigentSchema`
- no route may bypass the tenant/project auth chain
- no new score-like entity besides `ScoreRecord`
- no new metric-definition entity besides `Measure`
- no persisted dashboard entity
- no raw-content export by default
- browser feedback may not introduce a second feedback/score taxonomy when existing entities can represent it
