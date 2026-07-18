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

---

# Terra final review remediation (on captain commit 23e171a3)

Fresh detached Terra review BLOCKED `1563e929..23e171a3`; remediated on top,
left uncommitted. Files changed this round: `traigent/economics/schema.py`,
`traigent/economics/result.py`, `traigent/economics/client.py`,
`tests/unit/economics/test_schema.py`, `tests/unit/economics/test_client.py`,
`tests/unit/economics/test_contract_drift.py`.

## HIGH 1 — canonical key vs wire bytes
`client._serialize` now emits the SAME canonical (sorted-key, compact) JSON the
idempotency key and batch id are derived from (`payload.canonical_json`). Two
mappings equal in content but differing in insertion order now produce identical
batch id, idempotency key, AND wire bytes — the key can no longer agree while the
bytes diverge. Regression: `test_reordered_events_yield_identical_batch_key_and_wire_bytes`,
`test_reordered_batch_replays_with_identical_wire_bytes` (retries/cross-call replay
identical), plus the existing `test_retries_send_identical_bytes_and_key`.

## HIGH 2 — response validated against the exact schema before parsing
`TelemetryIngestResult.from_response` now calls `schema.validate_response_or_fail`
FIRST, validating every 200/201/422 body against the exact per-status response
schema (200→replay, 201/422→initial). That rejects unknown top-level/count/
rejection keys (`additionalProperties:false`) and malformed timestamps
(`UtcTimestamp`). Semantic reconciliation is preserved and extended: duplicate
rejection `event_index` is rejected, and a supplied rejection `event_id` must
equal the request event at that index. `PreparedTelemetryBatch` now carries an
immutable ordered `event_ids` tuple (identifiers only, no payload) for this.
Tests: `test_response_unknown_top_level_key_fails_closed`,
`test_response_malformed_timestamp_fails_closed`,
`test_duplicate_rejection_index_fails_closed`,
`test_rejection_event_id_mismatch_fails_closed`,
`test_rejection_with_matching_event_id_is_accepted`; schema-level
`test_response_*` cases; drift `test_response_status_schema_map_matches_endpoint_bindings`.

## HIGH 3 — exact-Schema content fingerprint (not name-only)
`schema.compute_economics_schema_fingerprint` folds the canonical JSON of the 11
economics contract files actually used (request + all response variants + every
referenced event/definition schema + endpoint bindings) into one SHA-256.
`_load_bundle` verifies it against the pinned
`EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT`
(`0ad15f00ff6d5a467f144cd29c9f663f67adec51fc95421872b43e31de96d9a9`, computed from
the accepted c0a70a1 material) BEFORE trusting the validator, and fails closed on
any mismatch even when the schema names exist. The version string is never trusted
alone. Errors carry only public contract hashes. Bump procedure documented in the
module docstring. Tests: `test_installed_fingerprint_matches_pinned_expected`,
`test_fingerprint_mismatch_fails_closed`, `test_fingerprint_is_content_sensitive`,
client `test_fingerprint_mismatch_blocks_before_transport` (zero transport), drift
`test_runtime_fingerprint_binds_to_local_c0a70a1_material`.

## MEDIUM 4 — Retry-After (delta-seconds + HTTP-date, injectable clock)
`_parse_retry_after(value, now)` now parses both delta-seconds and an HTTP-date
(`email.utils.parsedate_to_datetime`); a past date yields zero, invalid/non-finite
yields fallback (None → computed backoff), and `_backoff` clamps the honored delay
to a finite `_MAX_RETRY_AFTER_SECONDS` (30s) and never sleeps after the final
attempt. Time is injectable via `client._utcnow` for deterministic tests. Tests:
`test_retry_after_parsing_variants` (numeric, oversized, inf/nan/garbage, future
date, past→zero), `test_future_http_date_retry_after_is_clamped`, existing
`test_bounded_retry_after_is_clamped`, `test_no_sleep_after_final_attempt`.

## Consequences addressed
- Response schema unavailability/mismatch/validator exceptions fail closed
  (`EconomicsSchemaUnavailable` / `EconomicsResponseError`); a malformed response
  is never coerced into an accepted batch, and no body/secret/evidence is logged.
- Exact Git pin unchanged (`pyproject.toml` / `uv.lock` at c0a70a1); compatibility
  stays subpackage-only (`traigent.economics`); no root export added this round.

## Commands and results (this round)
- Focused: `.venv/bin/python -m pytest tests/unit/economics/ -q` → **112 passed**
  (was 93).
- Ruff: `ruff check traigent/economics tests/unit/economics` → **All checks passed**;
  `ruff format --check …` → **16 files already formatted**.
- Mypy: `mypy traigent/economics` → **Success: no issues found in 8 source files**.
- Compile: `py_compile traigent/economics/*.py tests/unit/economics/*.py` → **OK**.
- Parity/init: `pytest tests/cross_sdk_oracles/test_js_public_parity_manifest.py
  tests/unit/test_init_imports.py` → **29 passed**.
- Broader lane: `pytest tests/unit/cloud -q` → **1884 passed, 2 skipped**
  (pre-existing env skips).
- `git diff --check` → clean. Generated `__pycache__`/`.pyc` removed before handoff.

## Residuals (this round)
- The runtime fingerprint is realized against the installed exact-commit
  `traigent-schema` (git-pinned in metadata; installed into `.venv` from the local
  worktree == c0a70a1 for this environment). If the exact Git pin is ever bumped,
  recompute `EXPECTED_ECONOMICS_SCHEMA_FINGERPRINT` per the schema-module docstring
  in the same change.

---

# Terra final review remediation (on captain commit 627f3e8a)

Fresh detached gpt-5.6-terra xhigh review of the committed range BLOCKED on three
findings; remediated on top of `627f3e8af3d13cb1a548daf7c7c80535a1799269`, left
uncommitted. Files changed this round: `traigent/economics/client.py`,
`traigent/economics/result.py`, `tests/unit/economics/test_client.py`,
`tests/unit/economics/test_contract_drift.py`.

## CRITICAL 1 — PreparedTelemetryBatch is untrusted at the POST boundary
`PreparedTelemetryBatch` stays a public, constructible/`dataclasses.replace`-able
dataclass (API compatibility preserved), but `submit_prepared` no longer trusts
its precomputed fields. New `EconomicsTelemetryClient._reverify_prepared` runs
immediately before every POST and, from the batch BODY: (a) re-runs
characterization egress on every `run_economics` event; (b) re-validates the body
against the exact economics Schema (fails closed when the Schema is absent/
mismatched); (c) re-checks project binding (every `project_ref` == scope);
(d) re-derives canonical wire bytes, event ids, key, batch id, and submitted count
and requires the object's claimed values to match exactly — otherwise a
payload-free `EconomicsTelemetryContractError`/`EgressPolicyError`/
`EconomicsSchemaUnavailable`. The bytes transmitted are the RE-DERIVED canonical
bytes, never the object's `content`, so no forged byte reaches transport.
Adversarial tests: `test_directly_constructed_non_schema_batch_is_refused`,
`test_directly_constructed_withheld_value_batch_is_refused`,
`test_replace_content_mismatch_is_refused`,
`test_replace_body_with_arbitrary_json_is_refused`,
`test_replace_idempotency_key_mismatch_is_refused`,
`test_replace_event_ids_mismatch_is_refused`,
`test_replace_project_id_mismatch_is_refused`,
`test_prepared_submit_fails_closed_when_schema_unavailable` (all assert
`post` not called).

## HIGH 2 — all-rejected must be 422, never a fresh 201
`TelemetryIngestResult.from_response` now rejects a 201 whose
`rejected == submitted > 0` (`EconomicsResponseError`). A 200 all-rejected stays
valid (it replays a prior 422 ingest); partial/fresh 201 and duplicate-only 200
are preserved. Tests: `test_all_rejected_fresh_201_is_refused`,
`test_partial_rejection_201_is_accepted`,
`test_duplicate_only_replay_200_is_accepted`.

## MEDIUM 3 — 422 drift bound to the endpoint, not a literal
`test_response_status_schema_map_matches_endpoint_bindings` now derives all three
statuses from the endpoint: 200/201 bind to their response `$ref` stems; 422 reads
`responses["422"]` (a `KeyError` fails the test if the endpoint drops it) and, since
the endpoint declares 422 as a bare all-rejected status with no body `$ref`, asserts
the SDK maps it to the same replayed=false initial schema as 201 and that the
endpoint description states the all-rejected semantics. If the Schema later gives
422 its own body `$ref`, the test binds to it and fails on drift.

## Surrounding security premise re-checked (not narrow patching)
- Canonical exact transmitted bytes: POST now sends re-derived canonical bytes.
- Per-status response validation + installed-Schema content fingerprint: unchanged,
  still enforced (and now also re-enforced on the prepared path).
- Redirect/credential safety: `httpx.AsyncClient(follow_redirects=False)` set
  explicitly so a 3xx never re-sends the auth credential; a redirect surfaces as a
  transport error. Tests: `test_redirect_is_not_treated_as_success`,
  `test_http_client_does_not_follow_redirects`.
- No sleep after the final retry attempt; payload-free errors/logs: unchanged. No
  gate weakened.

## Commands and results (this round)
- Focused: `pytest tests/unit/economics/ -q` → **125 passed** (was 112).
- Ruff `check` + `format --check` → clean (16 files formatted).
- Mypy `traigent/economics` → **Success: no issues found in 8 source files**.
- Compile `py_compile` → OK.
- Parity/init → **29 passed**. Broader `tests/unit/cloud` → **1884 passed, 2 skipped**
  (pre-existing env skips). `git diff --check` → clean. Caches removed before handoff.

## Residual risk (this round)
- `PreparedTelemetryBatch.headers` property still reflects the object's (possibly
  tampered) claimed fields; it is informational only and is NOT used by the
  transport path (headers are rebuilt from the re-validated body in
  `_reverify_prepared`). Left public for API compatibility.
