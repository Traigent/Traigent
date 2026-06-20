# Execution Model Follow-Up Plan

This note tracks the remaining follow-up work after the execution-mode
consolidation shipped on `develop`. The public model is now `algorithm` plus
`offline`; legacy `execution_mode`, `privacy_enabled`, and
`cloud_fallback_policy` are deprecated compatibility inputs only.

---

## 1. Backend Protocol Alignment (Short-Term)

**Goal:** ensure Traigent clients and backend services share the same canonical
execution-policy vocabulary and provenance labels.

- **Owner:** SDK & Backend teams  
- **Success Criteria:** backend contracts align on the current public model:
  `algorithm`, `offline`, `TRAIGENT_REQUIRE_CLOUD` behavior, and result
  `source` values (`cloud_brain`, `local_fallback`, `explicit_local`,
  `offline`).

### Tasks
1. **Inventory current payloads**  
   - Document every endpoint (`trial_operations`, `session_operations`, etc.)
     that still emits or consumes legacy mode strings.
   - Capture where the backend should instead rely on `algorithm`/`offline`
     semantics or provenance `source`.
2. **Draft a migration plan**  
   - Introduce versioned API fields or graceful fallbacks where backend payloads
     still expose the old taxonomy.
   - Communicate expected rollout timeline to SDK consumers.
3. **Implement dual-read / dual-write**  
   - During transition, accept deprecated legacy strings on compatibility paths
     while writing canonical policy/provenance fields.
4. **Update SDK**  
   - Replace remaining backend-facing legacy mode assumptions with
     policy-aware handling once the backend contract is live.
5. **Monitor & clean up**  
   - Remove legacy fallbacks after sufficient bake time.

---

## 2. Concept & Documentation Consolidation (Medium-Term)

**Goal:** clarify the relationship between optimization policy, deployment
concerns, and SLA tiers so the deprecated mode taxonomy does not drift back into
docs or contracts.

- **Owner:** SDK Architecture / Docs  
- **Success Criteria:** single authoritative description of what
  `algorithm`, `offline`, external evaluators, privacy guarantees, and backend
  fallback imply.

### Tasks
1. **Review duplicate enums / constants**  
   - Compare `ExecutionMode` with `DeploymentMode` and SLA constants under `traigent/security/`.
   - Decide whether to consolidate or explicitly document their differing scopes.
2. **Publish a mode capability matrix**  
   - Highlight supported features for the current surface: cloud-first
     `algorithm="auto"`, explicit local algorithms, `offline=True`, and
     external evaluator dispatch.
   - Keep the deprecated mapping table explicit so `execution_mode="privacy"`
     is not misread as a no-egress guarantee.
3. **Update API docs & examples**  
   - Replace stale `execution_mode`, `privacy_enabled`, `cloud_fallback_policy`,
     and flat `hybrid_api_*` examples with the current public surface.
   - Add guidance on selecting `algorithm`/`offline` based on deployment
     requirements.
4. **Document migration guidance**  
   - Provide a clear recipe for teams moving from legacy
     `execution_mode`/`privacy_enabled` usage to `algorithm` plus `offline`.

---

## 3. Tracking

| Task | Priority | Owner | Status |
|------|----------|-------|--------|
| Backend accepts canonical execution-policy payloads | Short-term | Backend Platform | ☐ |
| SDK removes backend-facing legacy mode assumptions | Short-term | SDK | ☐ |
| Publish algorithm/offline capability matrix | Medium-term | Docs | ☐ |
| Align or document deployment vs execution-policy concepts | Medium-term | SDK Architecture | ☐ |

> **Note:** the public documentation should treat the consolidated
> `algorithm`/`offline` model as canonical. This note exists only for remaining
> backend and architecture follow-up work, not as an alternate user model.
