# WI-B — Python SDK economics telemetry emission — result

Worktree: `worktrees/econ-model-2026-07-17/Traigent`, branch
`feature/econ-model-python-sdk`, base `1563e9296ee7aa7cda2bd8552768073393bd6e0c`.
Left **uncommitted** for the captain. Authoritative contract: TraigentSchema
`c0a70a1d759774cb23f7f3d0943fbfc022d37f39`; accepted backend route
TraigentBackend `dc3d40354268f19fe9d24120ccfff926c13f08cd`.

## Scope delivered

The smallest safe public surface to construct and submit the authoritative
closed-Schema request for `POST /api/v1/economics/telemetry`:

- Builds the batch envelope (contract id/version, source, batch_id, sent_at,
  events) and a **caller-stable Idempotency-Key that matches the body**; the
  header and body key are identical by construction.
- Enforces the **client-side characterization egress rules before transmission**
  (withheld ⇒ absent; every transmitted value ⇔ exactly one `shared` field
  report; duplicate/contradictory field reports fail locally; evidence-pointer
  egress closed). A violation raises locally and nothing is sent.
- **Requires the exact economics Schema** and validates every request before any
  bytes are serialized; fails closed (no transport) when the schema is
  absent/old/raises, and rejects arbitrary extra keys (`additionalProperties`).
- Canonical auth + `X-Project-Id` project scoping (mirrors `analytics_client`);
  retry-safe transient retries (bounded backoff, bounded/finite `Retry-After`,
  no sleep after the final attempt).
- **Honest results**: parses/validates the response (identity, version,
  status↔replayed agreement, count reconciliation, rejection shape) and never
  coerces a malformed response or an all-rejected 422 into apparent success;
  `fully_accepted` requires fresh writes (an all-duplicate batch is not
  `fully_accepted`; a replay is signalled by `replayed`).
- **Cross-call recovery**: `prepare()` returns an immutable
  `PreparedTelemetryBatch` (exact bytes + key); `submit_prepared()` replays it.
  Documented that cross-process replay requires resubmitting the same prepared
  batch or the same complete stable tuple (events, batch_id, sent_at,
  idempotency_key).
- No payload values, evidence pointers, raw survey data, or secrets are logged.

Out of scope and NOT implemented (WI-C/WI-D): survey submission, calculator,
recommendations, MCP tooling, preflight economics, credits, incentives, dollar
gates, pricing.

## Changed / added files

New public package `traigent/economics/`:
- `__init__.py` — public surface + documented subpackage-only policy.
- `contract.py` — frozen contract constants / closed vocabularies.
- `errors.py` — typed, payload-free error hierarchy.
- `egress.py` — client-side characterization egress enforcement.
- `payload.py` — envelope construction + idempotency + funnel_eligible_event.
- `schema.py` — fail-closed exact-Schema request validation.
- `result.py` — fail-closed response parsing + honest acceptance semantics.
- `client.py` — `EconomicsTelemetryClient` + `PreparedTelemetryBatch` transport.

New tests `tests/unit/economics/`:
- `conftest.py`, `test_egress.py`, `test_payload.py`, `test_schema.py`,
  `test_client.py`, `test_contract_drift.py`, `test_public_surface.py`.

Modified:
- `pyproject.toml` — `internal_schema` extra now exact-Git-pins
  `traigent-schema @ git+…@c0a70a1…` (Backend's pattern; no sibling path).
- `uv.lock` — regenerated via `uv lock`: `traigent-schema` → git source
  `…?rev=c0a70a1…` (4.10.0). (Incidental: recorded self-version 0.22.0→0.23.0.)
- `traigent/__init__.py` — comment documenting why the emitter is intentionally
  NOT a root export (schema-owned parity manifest); no functional change.

## Captain-verification blockers — remediation

1. Offline-env test failures — fixed: `test_client.py` autouse `_online`
   fixture sets `TRAIGENT_OFFLINE_MODE=false` for mocked-transport tests; the
   explicit `test_offline_mode_fails_closed` overrides it to true.
2. Ruff — fixed: removed unused `CONTRACT_ID`, unquoted annotations, `datetime.UTC`,
   `ruff format` applied. `ruff check`/`format --check` clean.
3. Mypy — fixed: `_auth_headers` annotated `dict[str, str]`. `mypy traigent/economics`
   clean.
4. Schema pin — fixed: exact Git commit in `pyproject.toml` + `uv.lock`.
5. Response fail-closed — fixed: `TelemetryIngestResult.from_response` validates
   contract/version/identity/status↔replayed/count-reconciliation/rejection
   shape and raises `EconomicsResponseError`; body never logged.
6. Idempotency retry stability + cross-call recovery — fixed: retries send
   identical pre-serialized bytes/key (tested); `PreparedTelemetryBatch` +
   documented stable-tuple requirement.
7. `__pycache__`/`.pyc` removed from the untracked source tree.

Terra pre-review remediations:
- Closed-pipe validation fail-closed when the exact Schema is absent/old/raises,
  independent of the generic strict toggle; extra keys rejected; zero transport
  on every failure mode (tested in `test_schema.py`,
  `test_client.py::test_schema_failure_blocks_before_transport`).
- Honest cross-call recovery via immutable prepared batch (tested identical
  bytes/key; documented tuple).
- `fullmatch`/true end anchors for ShortLabel + IdempotencyKey; `allow_nan=False`
  everywhere serialization happens; serialization failures wrapped as typed
  contract errors (tested non-finite rejection).
- `Retry-After` parsed finite and clamped to `_MAX_RETRY_AFTER_SECONDS`; no
  sleep after the final failed attempt (tested).
- All-duplicate batch is not `fully_accepted` (distinct from a fresh write;
  tested).
- Contract drift **fails rather than skips** and covers endpoint/header, batch
  limit, key grammar, and 200/201 replay-status bindings against the installed
  exact Schema.
- Public client: documented **subpackage-only** policy (see Owner decisions),
  with an import-contract test.

## Commands and results

Run with the repo `.venv` (Python 3.13). Exact economics Schema installed into
the `.venv` from the local worktree (== `c0a70a1`) so runtime validation and
contract tests use the exact contract:
`uv pip install --python .venv/bin/python ../TraigentSchema` → `traigent-schema==4.10.0`.

- Focused suite: `.venv/bin/python -m pytest tests/unit/economics/ -q`
  → **93 passed**.
- Ruff: `.venv/bin/ruff check traigent/economics tests/unit/economics traigent/__init__.py`
  → **All checks passed**;
  `.venv/bin/ruff format --check traigent/economics tests/unit/economics` → **16 already formatted**.
- Mypy: `.venv/bin/mypy traigent/economics` → **Success: no issues found in 8 source files**.
- Compile: `.venv/bin/python -m py_compile traigent/economics/*.py traigent/__init__.py tests/unit/economics/*.py`
  → **OK**.
- Public surface / parity: `pytest tests/unit/economics tests/cross_sdk_oracles/test_js_public_parity_manifest.py tests/unit/test_init_imports.py`
  → **119 passed** (parity manifest test passes — no unclassified root symbol).
- Broader lane: `.venv/bin/python -m pytest tests/unit/cloud -q`
  → **1884 passed, 2 skipped** (pre-existing env skips: httpx-absent guard;
  memory-variance test).

## Residuals

- The production dependency is the Git pin in `pyproject.toml` / `uv.lock`. In
  this dev environment the exact Schema was realized by installing the local
  worktree (identical commit) into the `.venv`; the captain's CI realizes the
  same via `uv sync`. No sibling-path dependency is declared in project metadata.
- `uv lock` also corrected the lockfile's recorded self-version (0.22.0→0.23.0)
  — incidental, matches `traigent.__version__`.

## Owner decisions

### Public-export placement for the economics telemetry client

**Context.** The new emitter is a Python-only WI-B surface. The Traigent SDK's
`traigent` root exports are governed by a schema-owned Python/JS parity manifest
(`TraigentSchema parity/python-js-sdk.json`, pinned to a target SHA); a root
symbol not classified there fails the parity gate in CI. There is no JS
counterpart for this emitter yet.

**Decision needed.** Where should `EconomicsTelemetryClient` live publicly now?

**Options.**
- A (chosen, smallest reversible): ship as `traigent.economics.EconomicsTelemetryClient`
  (subpackage-only), documented, with an import-contract test. Reversible;
  promotion later is additive.
- B: also add a `traigent` root lazy export now. Requires a coordinated
  TraigentSchema parity-manifest classification at the SDK target SHA (a
  different, captain-owned repo) or CI parity goes red; out of this packet's
  scope.
- C: root export without touching the manifest. Rejected — knowingly ships a
  CI-red parity gate.

**Recommendation.** Keep A now. It is the smallest reversible, compatible option
and keeps this packet inside the Traigent worktree. Promote to a root export in a
coordinated follow-up that lands the Schema parity-manifest classification
alongside — the load-bearing reason is that the root surface is contractually
owned by TraigentSchema, so a root add is only correct as a cross-repo change the
captain sequences, not a unilateral SDK edit.
