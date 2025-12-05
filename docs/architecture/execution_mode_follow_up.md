# Execution Mode Follow-Up Plan

This note captures the remaining work streams for execution-mode standardization that require broader coordination or non-trivial refactors. It is intended as a living document for post-refactor tracking.

---

## 1. Backend Protocol Alignment (Short-Term)

**Goal:** ensure Traigent clients and backend services share the same canonical execution-mode vocabulary.

- **Owner:** SDK & Backend teams  
- **Success Criteria:** backend accepts / returns enum-aligned values (`edge_analytics`, `hybrid`, `standard`, `cloud`) without special-casing raw legacy strings.

### Tasks
1. **Inventory current payloads**  
   - Document every endpoint (`trial_operations`, `session_operations`, etc.) returning mode strings.
   - Capture existing variants (e.g. `"privacy"`, `"private"`).
2. **Draft a migration plan**  
   - Introduce versioned API fields or graceful fallbacks.
   - Communicate expected rollout timeline to SDK consumers.
3. **Implement dual-read / dual-write**  
   - During transition, accept both old and new values, emitting warnings when legacy strings arrive.
4. **Update SDK**  
   - Replace the remaining string comparisons in `traigent/cloud/trial_operations.py` with enum-aware handling once backend changes are live.
5. **Monitor & clean up**  
   - Remove legacy fallbacks after sufficient bake time.

---

## 2. Concept & Documentation Consolidation (Medium-Term)

**Goal:** clarify the relationship between execution modes, deployment modes, and SLA tiers to avoid future drift.

- **Owner:** SDK Architecture / Docs  
- **Success Criteria:** single authoritative description of what each mode implies (feature matrix, privacy expectations, storage behaviour).

### Tasks
1. **Review duplicate enums / constants**  
   - Compare `ExecutionMode` with `DeploymentMode` and SLA constants under `traigent/security/`.
   - Decide whether to consolidate or explicitly document their differing scopes.
2. **Publish a mode capability matrix**  
   - Highlight supported features (local storage, backend sync, privacy guarantees, offline operation) per mode.
   - Surface the current limitations of `HYBRID` and the sunset status of the `PRIVACY` alias.
3. **Update API docs & examples**  
   - Replace raw string defaults with enum references where applicable.
   - Add guidance on selecting an execution mode based on deployment requirements.
4. **Document migration guidance**  
   - Provide a clear recipe for teams moving from legacy `privacy` usage to the `HYBRID + privacy_enabled` pattern.

---

## 3. Tracking

| Task | Priority | Owner | Status |
|------|----------|-------|--------|
| Backend accepts enum payloads | Short-term | Backend Platform | ☐ |
| SDK switches trial operations to enums | Short-term | SDK | ☐ |
| Publish execution-mode feature matrix | Medium-term | Docs | ☐ |
| Align / document deployment vs execution enums | Medium-term | SDK Architecture | ☐ |

> **Note:** check this document into version control along with issue tracker links once tickets are created. Update the status table as work completes.
