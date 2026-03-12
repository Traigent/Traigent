# Hybrid Session API

The JS SDK talks to the typed interactive session API:

- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}/next-trial`
- `POST /sessions/{session_id}/results`
- `POST /sessions/{session_id}/finalize`
- `DELETE /sessions/{session_id}`

Public JS helpers for this surface:

- `createOptimizationSession(request, options?)`
- `getNextOptimizationTrial(sessionId, options?)`
- `submitOptimizationTrialResult(sessionId, result, options?)`
- `listOptimizationSessions(options?)`
- `checkOptimizationServiceStatus(options?)`
- `getOptimizationSessionStatus(sessionId, options?)`
- `finalizeOptimizationSession(sessionId, options?)`
- `deleteOptimizationSession(sessionId, options?)`

The helpers normalize both raw backend payloads and wrapped
`{ success, data }` envelopes into one JS-facing DTO shape.

For session status and session-list entries, the normalized DTO also lifts the
current backend's known detail fields to top-level JS-friendly properties when
present:

- `createdAt`
- `functionName`
- `datasetSize`
- `objectives`
- `experimentId`
- `experimentRunId`

The original backend `metadata` object is still preserved.

This is distinct from:

- the legacy `/api/v1/sessions` create payload that expects `problem_statement`,
  `dataset`, `search_space`, and `optimization_config`
- the round-based `/api/v1/hybrid/sessions` workflow
