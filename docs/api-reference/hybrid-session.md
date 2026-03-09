# Hybrid Session API

The JS SDK talks to the typed interactive session API:

- `POST /sessions`
- `POST /sessions/{session_id}/next-trial`
- `POST /sessions/{session_id}/results`
- `POST /sessions/{session_id}/finalize`

This is distinct from:

- the legacy `/api/v1/sessions` create payload that expects `problem_statement`,
  `dataset`, `search_space`, and `optimization_config`
- the round-based `/api/v1/hybrid/sessions` workflow
