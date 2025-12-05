# Traceability Tagging – Multi-Agent Collaboration Protocol

Purpose: enable multiple AI agents to update tags safely across specs and code while keeping traceability intact and avoiding conflicting edits.

## Tagging Rules (must follow)
- Closed vocabularies live in `docs/traceability/taxonomy_reference.md`.
- Exactly one `CONC-Layer-*`; up to three `CONC-Quality-*`; optional `CONC-View-*`, `CONC-Lifecycle-*` (if used), `CONC-Domain-*`, `CONC-Compliance-*`, `CONC-ML-*`.
- Use symbol-level entries in `trace_links.json` if a file spans layers; no stacked layer tags.
- Deprecated: `CONC-Quality-Compliance`. Use a normal quality + `CONC-Compliance-*` when needed.
- Tests/Experimental only if explicitly in scope; otherwise leave untagged per policy.

## Safe Change Protocol
1) Classify change: add/update/deprecate requirement; add/update/deprecate functionality; retag code only.
2) Edit specs first: update `docs/traceability/requirements.yml`, then `docs/traceability/functionalities.yml` if references change. Never delete IDs; deprecate instead.
3) Propagate: align `trace_links.json` (or symbol entries) with spec changes.
4) Retag code: apply Layer/Quality tags (and optional tags) per taxonomy; mixed-purpose files go to `trace_links.json`.
5) Validate: run CodeSync scan; confirm zero orphans, no deprecated/unknown tags; spot-check `gaps.json`.
6) Review: ensure PR notes include rationale and that CI/lint gates pass (one layer, ≤3 qualities, closed vocab).

## Ownership & Tracking
- Tracking file: `docs/traceability/tagging_tracking.md`.
- Agents self-assign with a UUID; set status to `in-progress` when working; update `last_update` each edit.
- Status values: `todo`, `in-progress`, `blocked`, `completed`.
- Only modify rows you own unless clearing a `blocked` item with a note.

## Conflict Avoidance
- Claim before editing: update `tagging_tracking.md` to `in-progress` with your agent_id.
- For Core vs Infra vs Integration disputes, use the decision tree (taxonomy reference); pause and note in `notes` if unclear.
- Do not bulk search/replace IDs; scope changes and verify intent.

## Governance
- New vocab entries require an RFC and updates to taxonomy reference plus CodeSync validators (handled by CodeSync team).
- Keep `docs/traceability/tagging_guidelines.md` and `taxonomy_reference.md` in sync with any protocol tweaks.
- When cleaning up the `docs/traceability` folder, archive or delete obsolete tracking/RFC files only after confirming no active work references them.

## Cleanup Guidance
- Archive superseded tracking docs to `docs/traceability/archive/` (create if absent) with a datestamped filename.
- Do not delete active tracking or RFC files referenced in open work.
- Update `tagging_tracking.md` to remove/close rows that are done before archiving.

## Quick Agent Checklist
- Claim scope in `tagging_tracking.md` (set `in-progress`, add agent_id).
- Apply tags per taxonomy (Layer 1, Qualities ≤3, optional closed vocab).
- Update specs first if IDs change; then trace metadata; then code tags.
- Run CodeSync scan; check `gaps.json`; note results in `tagging_tracking.md`.
- Set status to `completed` (or `blocked` with notes) when done.
