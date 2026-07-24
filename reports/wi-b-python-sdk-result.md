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
  **[SUPERSEDED by Fix round 2, below.]** The "no JS counterpart / subpackage-only"
  rationale was stale: the JS SDK does root-export `EconomicsTelemetryClient`, and
  the parity manifest now classifies it `matched`. Fix round 2 adds the Python root
  lazy export; this bullet's original claim no longer holds.

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

> **[SUPERSEDED 2026-07-18 — Fix round 2.]** This decision's load-bearing premise —
> "There is no JS counterpart for this emitter yet" — was **false**. The accepted JS
> SDK (`traigent-js` `08d8931`, `src/index.ts`) root-exports `EconomicsTelemetryClient`,
> and TraigentSchema `c27a034` classifies the symbol `matched` and lists it in
> `javascript.requiredRootExports`. Option A was chosen on a stale fact. Fix round 2
> implements the true-parity outcome (manifest `matched` classification in the Schema
> branch + Python root **lazy** export), so the emitter is now
> `traigent.EconomicsTelemetryClient` **and** `traigent.economics.EconomicsTelemetryClient`.
> The text below is retained as the original (now-corrected) record.

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

---

# Terra final review remediation (on captain commit aaacd1c7)

Fresh detached gpt-5.6-terra xhigh full-range review BLOCKED one critical issue on
`aaacd1c7925e6777b80eafefd489aa245966c869`; remediated on top, left uncommitted.
Files changed this round: `traigent/economics/client.py`,
`tests/unit/economics/test_client.py`.

## CRITICAL — a fully self-consistent hand-built batch was still submittable
The prior `_reverify_prepared` compared canonical content, key, batch id, event
ids, count, and project — but a directly-constructed (or `dataclasses.replace`-d)
`PreparedTelemetryBatch` can supply ALL of those consistently, so it passed and
reached POST.

Fix — an internal, non-constructor PROVENANCE capability issued only by
`prepare()`:
- Added a module-private sentinel `client._PREPARE_PROVENANCE = object()` and a
  dataclass field `_provenance: Any = field(default=None, init=False,
  compare=False, repr=False)`. The class stays publicly constructible (the field
  is not a constructor argument; equality/repr are unaffected), but a direct
  instance defaults to the invalid `None`.
- `prepare()` is the ONLY place that grants it, via
  `object.__setattr__(prepared, "_provenance", _PREPARE_PROVENANCE)` after
  construction.
- `submit_prepared` calls `_require_provenance` FIRST (before any field work or
  transport): a batch whose `_provenance` is not the exact module sentinel is
  refused with a payload-free `EconomicsTelemetryContractError`, POST not called.
- `dataclasses.replace` produces a new instance whose `init=False` provenance
  field is re-defaulted to `None` (verified: not copied), so any replace — even
  one that recomputes every public/body/content/key/id/count/scope field
  self-consistently — loses the capability and is non-submittable.

Trust-boundary scope (documented in code): this closes normal public construction
and replacement; it does NOT claim cryptographic defense against a caller who
reaches in with `object.__setattr__` or module introspection. The existing
re-derivation checks (content/key/ids/project/schema-fingerprint/egress) remain as
defense-in-depth for a provenanced-but-inconsistent object.

In-process, cross-client rule (smallest compatible): the sentinel is module-level,
so a batch prepared by one `EconomicsTelemetryClient` instance is submittable by
another instance in the SAME process. It does not survive pickling or crossing a
process boundary (a fresh import gets a new sentinel object) — acceptable because
cross-call recovery is defined as resubmitting the exact in-process object or
rebuilding via `prepare()`.

Replay preserved: the exact object returned by `prepare()` carries provenance and
submits; retries replay byte-identical content and key.

## Decisive tests (all assert POST not called on refusal)
Primary gate: `test_direct_construction_fully_valid_is_non_submittable`,
`test_replace_with_no_changes_loses_provenance`,
`test_replace_all_fields_recomputed_loses_provenance`,
`test_provenance_refusal_is_payload_free` (asserts the sensitive evidence pointer
and magnitude appear nowhere in the error or its cause/context chain),
`test_exact_prepared_object_submits_and_retries_identical`.
Defense-in-depth (white-box, provenance minted then a field tampered):
`test_provenanced_content_tamper_is_refused`,
`test_provenanced_arbitrary_body_is_refused`,
`test_provenanced_withheld_value_body_is_refused`,
`test_provenanced_idempotency_key_tamper_is_refused`,
`test_provenanced_event_ids_tamper_is_refused`,
`test_provenanced_project_tamper_is_refused`,
`test_prepared_submit_fails_closed_when_schema_unavailable`.

All prior protections (revalidation, egress, Schema fingerprint, project/status/
Retry-After/redirect) are unchanged and still enforced.

## Commands and results (this round)
- Focused: `pytest tests/unit/economics/ -q` → **129 passed** (was 125).
- Ruff `check` + `format --check` → clean (16 files). Mypy `traigent/economics` →
  **Success: no issues found in 8 source files**. Compile → OK.
- Parity/init → **29 passed**. Broader `tests/unit/cloud` → **1884 passed, 2 skipped**
  (pre-existing env skips). `git diff --check` → clean. Caches removed before handoff.

## Residual risk (this round)
- Provenance is an API trust boundary, not a cryptographic seal: a caller who
  deliberately runs `object.__setattr__(batch, "_provenance", <the module
  sentinel>)` can forge it. This is out of scope by the finding's own terms; the
  re-derivation + egress + Schema checks still bound what such a forged batch
  could actually transmit (schema-valid, egress-clean, self-consistent only).
- `PreparedTelemetryBatch.headers` remains an informational property reflecting the
  object's claimed fields; the transport path rebuilds headers from the
  re-validated body and never uses it.

---

# Terra final review remediation (on captain commit aaacd1c7 — round 2)

Fresh Terra final review BLOCKED two issues; remediated on top, uncommitted.
Files changed this round: `traigent/economics/client.py`,
`traigent/economics/schema.py`, `traigent/economics/payload.py`,
`tests/unit/economics/test_client.py`, `tests/unit/economics/test_schema.py`,
`tests/unit/economics/test_payload.py`.

## CRITICAL — copy.copy(prepared) carried the sentinel and could submit
The prior fix marked a per-instance `_provenance` field; `copy.copy`/`copy.deepcopy`
copy the instance dict, so a copy carried the sentinel and passed. Replaced the
field-sentinel with an in-process **identity issuance registry**:
- `client._ISSUED_BATCHES: weakref.WeakValueDictionary[int, PreparedTelemetryBatch]`.
  `prepare()` records the EXACT returned instance (`_ISSUED_BATCHES[id(prepared)] =
  prepared`) — the only place identity is granted. The `_provenance` field is
  removed.
- `submit_prepared` gates on `_ISSUED_BATCHES.get(id(prepared)) is prepared`
  (identity, never equality). A directly-constructed, `copy.copy`d,
  `copy.deepcopy`d, `dataclasses.replace`d, or unpickled batch is a different
  object with a different id → absent from the registry → refused (payload-free),
  POST not called.
- Leak-free + stale-id-safe: the weak value is dropped the instant the batch is
  GC'd, so the registry never grows unbounded AND a later object reusing that id
  resolves to nothing rather than a stale batch. Not equality-based.
- Copy/pickle policy (defined + tested): `copy.copy`/`copy.deepcopy`/`pickle`
  round-trip all yield a non-issued identity (or cannot be reconstructed at all —
  the read-only `MappingProxyType` body cannot be deep-copied/pickled) → never
  submittable. In-process cross-client rule: the registry is module-level, so a
  batch prepared by one client instance submits via another in the SAME process;
  it does not cross a process boundary. The exact `prepare()` object still submits
  and retries byte-identical content/key.
- Trust boundary (documented): a caller who deliberately registers a forged object
  under its own id can still submit — out of scope by the finding's terms; the
  re-derivation/egress/Schema checks still bound what it could transmit.
- Tests: `test_copy_copy_of_prepared_is_non_submittable`,
  `test_deepcopy_of_prepared_is_non_submittable`,
  `test_pickle_roundtrip_of_prepared_is_non_submittable`,
  `test_cross_client_issued_batch_is_submittable`,
  `test_issuance_registry_is_leak_free`, plus the existing direct-construction /
  replace primary-gate tests (now via the registry).

## HIGH — boundary exceptions chained raw payload via cause/context
`response.json()` (JSONDecodeError.doc) and schema-validator exceptions were
chained with `from exc`, exposing the raw payload through
`__cause__`/`__context__`. Sanitized every payload-bearing boundary by confining
the third-party exception to its handler and RAISING the typed error OUTSIDE it
(sentinel pattern), so `__cause__` is None and `__context__` is not populated:
- `client._interpret`: invalid-JSON → `EconomicsResponseError`, chain payload-free.
- `schema.validate_request_or_fail` / `validate_response_or_fail`: validator
  exception → `EconomicsSchemaUnavailable` / `EconomicsResponseError`, chain
  payload-free (the invalid-body path already reported only an error count).
- `payload.canonical_json` / `_json_roundtrip`: `json` serialization error →
  `EconomicsTelemetryContractError`, chain payload-free.
- Tests: `test_invalid_json_response_error_is_payload_free` (JSONDecodeError.doc
  carries a sensitive marker; asserts absent from the whole chain and no
  JSONDecodeError in it), `test_request_validator_exception_is_payload_free`,
  `test_response_validator_exception_is_payload_free`,
  `test_serialization_error_is_payload_free`.

Exact Schema / egress / retry / replay behavior unchanged; WI-B-only.

## Commands and results (this round)
- Focused: `pytest tests/unit/economics/ -q` → **138 passed** (was 129).
- Ruff `check` + `format --check` → clean (16 files). Mypy `traigent/economics` →
  **Success: no issues found in 8 source files**. Compile → OK.
- Parity/init → **29 passed**. Broader `tests/unit/cloud` → **1884 passed, 2 skipped**
  (pre-existing env skips). `git diff --check` → clean. Caches removed.

## Residual risk (this round)
- Identity registry is an API trust boundary, not a cryptographic seal: a caller
  who deliberately does `client._ISSUED_BATCHES[id(x)] = x` can forge issuance.
  Out of scope per the finding; re-derivation + egress + Schema still bound what
  such a batch could transmit.
- Provenance/identity is per-process (module-level registry); it deliberately does
  not survive pickling or a process boundary. Cross-process "recovery" means
  rebuilding via `prepare()` from the same stable tuple, not shipping the object.

---

# Terra final review remediation — closure issuance + UTF-8 chokepoint (on HEAD 4cc29cb6)

Fresh Terra full-range review BLOCKED two issues on `4cc29cb6`; remediated,
left **uncommitted**. Files changed this round: `traigent/economics/client.py`,
`traigent/economics/payload.py`, `tests/unit/economics/test_client.py`,
`tests/unit/economics/test_payload.py`. Exact TraigentSchema pin/fingerprint,
egress, retry, replay, redirect, and project-scoping behavior are unchanged
(WI-B-only); the client diff touches no preserved-invariant line
(`git diff` grep for `follow_redirects|Retry-After|fingerprint|
validate_request_or_fail|enforce_characterization|guard_project|max_retries`
returns nothing).

## CRITICAL — module-exposed mutable `_ISSUED_BATCHES` minting authority
The prior identity-registry fix left a **module-global, writable** registry:
`client._ISSUED_BATCHES: weakref.WeakValueDictionary`. The exact reproducer —
`import ...client as m; m._ISSUED_BATCHES[id(forged)] = forged` on a
fully-consistent hand-built `PreparedTelemetryBatch`, then `submit_prepared` —
minted issuance and reached POST. Any ordinary import held the minting authority.

Fix — an **opaque, non-public, identity-bound issuance capability** with no
module-reachable mint (`_install_issuance()` in `client.py`):
- A process-random `secrets.token_bytes(32)` secret is created inside the factory
  and captured by two closures: the stamping `prepare` (installed onto the class)
  and a **verify-only** function returned as `_verify_issuance`. The secret, the
  token function, and the stamping wrapper are all **locals of the factory** —
  none is bound to module globals. `_ISSUED_BATCHES` and the old `_PREPARE_PROVENANCE`
  sentinel are gone; there is **no writable registry and no module-level mint/stamp
  callable** to write to.
- The token is `HMAC(secret, str(id(batch)))` — bound to the exact object identity.
  `prepare()` is the ONLY minting site; it stamps the token via
  `object.__setattr__` and is inlined in the closure so no standalone "stamp
  arbitrary batch" callable exists. `submit_prepared` calls `_require_issued`
  FIRST, which invokes the verify-only closure; verification is the sole exposed
  capability.
- Identity binding defeats copy/replace/deepcopy/pickle/equality: a
  `copy.copy`/`copy.deepcopy`/`dataclasses.replace`/unpickled/equal-by-value twin
  has a different `id()`, so a copied token no longer matches and an absent token
  fails outright — refused payload-free, POST not called. `dataclasses.replace`
  (`init=False`)/copy/pickle all lose valid issuance.
- **No retained state / leak-free / lifetime**: the token lives on the batch
  instance (fixed-size bytes) and is collected with it; the secret is one
  fixed-size constant. There is no per-batch module bookkeeping, so nothing to
  leak. In-process cross-client acceptance is preserved (module-scoped secret); it
  does not survive pickling or a process boundary.

**Introspection boundary (stated honestly).** The secret is not a module
attribute (an ordinary import cannot read it), but it IS recoverable by
deliberately walking `_verify_issuance.__closure__` (proven by
`_recover_issuance_secret` in the tests). This is an API trust boundary, exactly
as `object.__setattr__` can stamp any attribute — not a cryptographic seal
against a caller who reaches into interpreter internals. It DOES close the
normal-import mint path the finding requires; the re-derivation + egress + Schema
checks still bound what any forged-provenance batch could transmit.

Adversarial tests (all assert POST not called on refusal):
- No authority exists: `test_no_module_global_issuance_registry_or_mint`
  (asserts `_ISSUED_BATCHES`/`_PREPARE_PROVENANCE` gone, **no** module global is a
  mutable `dict`/`list`/`set`/weak-map, verify returns False for an unissued
  batch), `test_issuance_secret_is_not_a_module_global` (secret is not a module
  attribute yet is reachable only via closure introspection — the honest boundary).
- Exact reproducer dead:
  `test_exact_import_and_register_reproducer_cannot_post`
  (`m._ISSUED_BATCHES[id(forged)] = forged` raises `AttributeError`; the forged
  batch is then refused before transport).
- Construction/replace/copy/deepcopy/pickle/equality:
  `test_direct_construction_fully_valid_is_non_submittable`,
  `test_replace_with_no_changes_loses_provenance`,
  `test_replace_all_fields_recomputed_loses_provenance`,
  `test_copy_copy_of_prepared_is_non_submittable`,
  `test_deepcopy_of_prepared_is_non_submittable`,
  `test_pickle_roundtrip_of_prepared_is_non_submittable`,
  `test_equal_but_distinct_batch_is_non_submittable` (a value-`==` twin with a
  distinct identity is refused).
- Lifetime/leak-free: `test_prepared_batch_is_not_retained_by_module` (the batch
  is GC'd once the caller drops it — nothing pins it), and
  `test_repeated_prepare_grows_no_module_state` (50 prepares add zero module
  attributes and all batches remain collectable).
- Positive/cross-client/defense-in-depth:
  `test_exact_prepared_object_submits_and_retries_identical`,
  `test_cross_client_issued_batch_is_submittable`, and the six
  `test_provenanced_*_is_refused` white-box tests (issuance forged via closure
  introspection, then a single field tampered — the re-derivation checks still
  refuse).

## HIGH — lone surrogate leaks the canonical payload via `UnicodeEncodeError`
`canonical_json(..., ensure_ascii=False)` lets a lone surrogate (e.g. `"\ud800"`,
a valid `str` but invalid UTF-8) pass `json.dumps`; the later `.encode("utf-8")`
raised a raw `UnicodeEncodeError` whose `.object` carries the ENTIRE canonical
payload, and it propagated outside `canonical_json`'s handler.

Fix — a single payload-safe UTF-8 chokepoint (`payload.canonical_json_bytes`):
- Wraps `json.dumps` **and** `.encode("utf-8")` in one `try`, catching
  `(TypeError, ValueError, UnicodeError)`, and raises a fresh
  `EconomicsTelemetryContractError` **outside** the handler (sentinel/`pass`-then-
  raise), so `__cause__` and `__context__` are both `None` and no
  `UnicodeError.object` rides along.
- Every bytes-producing path routes through it so wire bytes cannot diverge:
  batch-id derivation (`_derive_batch_id`), idempotency-key derivation
  (`_derive_idempotency_key`), and wire serialization (`client._serialize`,
  and the re-derivation in `_reverify_prepared`).

Tests (sensitive marker + lone surrogate, complete attributes/cause/context/logs,
zero POST):
- `test_canonical_json_bytes_lone_surrogate_is_payload_free` (helper unit),
- `test_batch_id_derivation_lone_surrogate_is_payload_free` (no `batch_id` → derive
  path; asserts clean `caplog`),
- `test_idempotency_key_derivation_lone_surrogate_is_payload_free` (explicit
  `batch_id` isolates the key-derivation path; asserts clean `caplog`),
- `test_wire_serialize_lone_surrogate_is_payload_free` (`client._serialize`),
- `test_submit_lone_surrogate_event_fails_closed_zero_post` (end-to-end
  `client.submit`: `post` not called, error chain + `caplog` carry no marker,
  `__cause__`/`__context__` both `None`).

## Note on starting state / lint
The two code fixes (closure issuance in `client.py`, `canonical_json_bytes` in
`payload.py`) were already present in the working tree at session start
(uncommitted, atop `4cc29cb6`, whose committed source still had the vulnerable
`_ISSUED_BATCHES`). This round: (a) completed the client fix with a documented
`# noqa: B010` on the deliberate `setattr(EconomicsTelemetryClient, "prepare",
...)` method-rebind — `setattr` is required because plain assignment trips mypy's
`method-assign`, and the closure keeps the mint non-module-visible; (b) rewrote
the 7 tests that referenced the removed `_ISSUED_BATCHES` (the white-box helper
now mints via honest closure introspection); (c) added the adversarial coverage
above.

## Commands and results (this round)
Repo `.venv` (Python 3.13), exact economics Schema installed (`traigent-schema==4.10.0`).
- Focused: `.venv/bin/python -m pytest tests/unit/economics/ -q` → **148 passed**
  (was 138; +11 adversarial tests, −1 obsolete registry test).
- Econ + parity + init: `pytest tests/unit/economics
  tests/cross_sdk_oracles/test_js_public_parity_manifest.py
  tests/unit/test_init_imports.py` → **177 passed**.
- Ruff: `ruff check traigent/economics tests/unit/economics traigent/__init__.py`
  → **All checks passed**; `ruff format --check traigent/economics
  tests/unit/economics` → **16 files already formatted**.
- Mypy: `mypy traigent/economics` → **Success: no issues found in 8 source files**.
- Compile: `py_compile traigent/economics/*.py traigent/__init__.py
  tests/unit/economics/*.py` → **OK**.
- Broader lane: `pytest tests/unit/cloud -q` → **1884 passed, 2 skipped**
  (pre-existing env skips: httpx-absent guard; memory-variance test).
- `git diff --check` → clean. `__pycache__`/`.pyc` removed before handoff.

## Residual risk (this round)
- The issuance capability is an **API trust boundary, not a cryptographic seal**:
  a caller who deliberately walks `_verify_issuance.__closure__` to recover the
  secret (or uses `object.__setattr__`) can forge a token. Out of scope by the
  finding's own terms; the re-derivation + egress + exact-Schema checks still
  bound what such a batch could transmit (schema-valid, egress-clean,
  self-consistent only), and no ordinary import can reach the mint.
- Issuance is per-process (module-scoped closure secret); it deliberately does not
  survive pickling or a process boundary. Cross-process "recovery" means
  rebuilding via `prepare()` from the same stable tuple, not shipping the object.
- `PreparedTelemetryBatch.headers` remains an informational property reflecting the
  object's claimed fields; the transport path rebuilds headers from the
  re-validated body and never uses it.
- The standalone `payload.canonical_json` (str) remains public for callers, but no
  production wire/derivation path uses it followed by a raw `.encode`; all
  bytes-producing paths go through `canonical_json_bytes`.

---

# Captain verification remediation — normal-call mint bypass + module-global verifier

Captain verification found a remaining CRITICAL bypass in the closure-issuance
repair. Remediated narrowly, left **uncommitted**. Files changed this round:
`traigent/economics/client.py`, `tests/unit/economics/test_client.py`. The
surrogate fix (`payload.py` + its tests) and all prior cases are preserved
unchanged; the `client.py` diff touches no preserved-invariant line
(schema pin/fingerprint, egress, retry/Retry-After, replay, redirect, project
scoping, `_reverify_prepared` — grep confirms none changed). WI-B-only.

## CRITICAL — normal-call mint bypass via dynamically-dispatched `self._build_prepared`
The issuing `prepare` (installed by `_install_issuance`) called
`self._build_prepared(...)`. Because `prepare` is a plain function, an attacker
could call it unbound with a hostile `self`:

```python
forged = <fully consistent forged PreparedTelemetryBatch>
class EvilIssuer:
    def _build_prepared(self, events, **kw):
        return forged
EconomicsTelemetryClient.prepare(EvilIssuer(), events, project_id="proj-1")
```

The wrapper dispatched to `EvilIssuer._build_prepared`, so it received the
attacker's forged object and HMAC-stamped it: `same_object=True`,
verification `True`, submittable. A subclass overriding `_build_prepared` was the
same hole.

Fix (exactly as directed) — the mint only ever stamps a batch from a
**captured, dispatch-free** builder:
- Inside `_install_issuance`, the GENUINE builder is captured BEFORE any wrapper
  is installed: `_genuine_build = EconomicsTelemetryClient._build_prepared`. The
  issuing `prepare` calls `_genuine_build(self, events, ...)` directly — never a
  dynamically-dispatched `self._build_prepared`. A caller-supplied `self` or a
  subclass override can no longer substitute the object that gets stamped; the
  mint stamps only a batch produced by the captured builder.
- Result: the exact duck-typed reproducer now raises (the genuine builder rejects
  a `self` lacking the real client machinery) and the forged object is never
  returned/stamped; a subclass whose `_build_prepared` returns a forged batch gets
  a **genuinely-built** batch back instead — the injected object stays
  non-submittable.

## CRITICAL (same finding) — module-global `_verify_issuance` reassignment surface
`_verify_issuance` was a module global that `_require_issued` called, so
`client_mod._verify_issuance = lambda b: True` would monkeypatch an always-accept
bypass.

Fix — the verifier is installed as a **class-attribute closure**, not a module
global:
- `_install_issuance` now defines a verifier-backed `_require_issued` closure over
  the secret and installs it on the class via
  `setattr(EconomicsTelemetryClient, "_require_issued", staticmethod(...))`. It
  **returns nothing** — no verifier, secret, key, mint, or registry is bound to
  module scope. The `_verify_issuance` module global is removed entirely.
- The in-class `_require_issued` is now a fail-closed FALLBACK that refuses
  unconditionally (if installation were skipped, every submission is refused).
- Introspection boundary (restated honestly): the verifier/secret are reachable
  only by deliberately walking the installed class-attribute closures
  (`EconomicsTelemetryClient.prepare.__closure__` /
  `._require_issued.__closure__`) — an API trust boundary, not a crypto seal;
  ordinary import/reassignment cannot reach it.

## Tests (all assert zero POST on refusal)
- Exact reproducer / subclass regression:
  `test_evil_issuer_normal_call_cannot_mint_forged_object` (the unbound-`prepare`
  duck-typed `EvilIssuer` reproducer — prepare never returns/stamps the forged
  object; `forged` stays non-issued; `submit_prepared(forged)` refused, POST not
  called), `test_evil_subclass_prepare_uses_genuine_builder_not_override` (subclass
  override bypassed → a genuine batch is returned and IS issued, the injected
  `forged` is not, POST not called).
- Module-global reassignment/bypass audit:
  `test_no_module_global_verifier_or_mint_to_reassign` (no `_verify_issuance` /
  `_ISSUED_BATCHES` / `_PREPARE_PROVENANCE`; no module global is the 32-byte
  secret or a verifier/mint-named callable; the verifier is reachable only via
  class-closure introspection). `test_no_module_global_issuance_registry_or_mint`
  updated to assert `not hasattr(client_mod, "_verify_issuance")` and verify an
  unissued batch through the installed class verifier (`_is_issued`).
- White-box helpers updated: `_recover_issuance_secret` now walks the installed
  class-attribute closures (not the removed module global); new `_is_issued`
  checks issuance through `EconomicsTelemetryClient._require_issued`. All six
  `test_provenanced_*_is_refused` defense-in-depth cases and the
  copy/deepcopy/pickle/replace/equality/lifetime cases are preserved and pass.

## Commands and results (this round)
Repo `.venv` (Python 3.13), exact economics Schema (`traigent-schema==4.10.0`).
- Focused: `pytest tests/unit/economics/ -q` → **151 passed** (was 148; +3
  regression/audit tests).
- Econ + parity + init: `pytest tests/unit/economics
  tests/cross_sdk_oracles/test_js_public_parity_manifest.py
  tests/unit/test_init_imports.py` → **180 passed**.
- Ruff: `ruff check traigent/economics tests/unit/economics traigent/__init__.py`
  → **All checks passed**; `ruff format --check …` → **16 files already formatted**
  (2 files reformatted during the round, then clean).
- Mypy: `mypy traigent/economics` → **Success: no issues found in 8 source files**.
- Compile: `py_compile` → **OK**.
- Broader lane: `pytest tests/unit/cloud -q` → **1884 passed, 2 skipped**
  (pre-existing env skips). `git diff --check` → clean. Caches removed.

## Residual risk (this round)
- Issuance remains an **API trust boundary, not a cryptographic seal**: a caller
  who deliberately walks `EconomicsTelemetryClient.prepare.__closure__` /
  `._require_issued.__closure__` (or uses `object.__setattr__`, or overrides
  `_require_issued`/`_reverify_prepared` on a subclass they fully control and then
  drives the whole transport themselves) can still forge. Out of scope by the
  finding's terms — an ordinary import/normal call can no longer mint, and the
  re-derivation + egress + exact-Schema checks still bound what any forged batch
  could transmit.
- Issuance is per-process (module-scoped closure secret installed on the class);
  it deliberately does not survive pickling or a process boundary.

---

# Captain verification remediation — one-shot installer self-removal

Captain verification found one remaining exact bypass. The prior handoff claim
that "no module mint/installer is bound" was **false**: `_install_issuance`
itself stayed in module globals after initialization, so an attacker could
`EconomicsTelemetryClient._build_prepared = lambda ...: forged; client_mod.
_install_issuance()` to reinstall the mint around the hostile builder, then
`prepare(object(), ...)` returned and HMAC-stamped `forged`
(`installer_exported=True, same_object=True, verifies=True`).

Remediated narrowly, left **uncommitted**. Files changed this round:
`traigent/economics/client.py` (one-shot `del` + docstrings) and
`tests/unit/economics/test_client.py`. No transport/behavior logic changed; the
surrogate fix and all prior cases are preserved.

## Fix — delete the installer after its single import-time call
`client.py` now runs the installer exactly once and immediately removes the name:

```python
_install_issuance()
del _install_issuance
```

After import there is no `_install_issuance` attribute on the module, so an
ordinary import cannot re-invoke it to recapture a monkeypatched `_build_prepared`
into the mint. The capability lives only in the class-attribute closures
(`prepare`, `_require_issued`); nothing at module scope can re-mint.

## Test — exact monkeypatch + reinstall regression
`test_installer_is_not_reinvokable_after_import`: asserts the module has no
`_install_issuance` attribute; monkeypatches `_build_prepared` to return a forged
batch, then asserts `client_mod._install_issuance()` raises `AttributeError` (the
recapture cannot happen); `forged` remains unissued (`_is_issued` False) and
`submit_prepared(forged)` is refused with **zero POST**. The
`test_no_module_global_verifier_or_mint_to_reassign` audit is extended to assert
`not hasattr(client_mod, "_install_issuance")` and that no module-global name
contains `install_issuance`.

## Claims corrected (stop overclaiming)
- The exact statement now proven and made true: after import there is **no
  `_install_issuance`, `_verify_issuance`, `_ISSUED_BATCHES`, or
  `_PREPARE_PROVENANCE` in module globals**, and no module-global verifier / mint /
  installer / 32-byte secret. The capability is only on the class closures.
- What is NOT claimed: this is an **API trust boundary, not a cryptographic seal**.
  A caller who deliberately walks `EconomicsTelemetryClient.prepare.__closure__` /
  `._require_issued.__closure__`, uses `object.__setattr__`, or replaces
  `prepare`/`_require_issued`/`_build_prepared` on the class or a subclass they
  fully control can still forge — those require reaching into interpreter/class
  internals, not an ordinary import or normal call. The re-derivation + egress +
  exact-Schema checks still bound what any forged batch could transmit.

## Commands and results (this round)
Repo `.venv` (Python 3.13), exact economics Schema (`traigent-schema==4.10.0`).
- Focused: `pytest tests/unit/economics/ -q` → **152 passed** (was 151; +1 regression).
- Econ + parity + init: `pytest tests/unit/economics
  tests/cross_sdk_oracles/test_js_public_parity_manifest.py
  tests/unit/test_init_imports.py` → **181 passed**.
- Ruff: `ruff check …` → **All checks passed**; `ruff format --check …` →
  **16 files already formatted**.
- Mypy: `mypy traigent/economics` → **Success: no issues found in 8 source files**.
- Compile: `py_compile` → **OK**.
- Broader lane (module-init `del` executes at import, so run for safety):
  `pytest tests/unit/cloud -q` → **1884 passed, 2 skipped** (pre-existing env
  skips). `git diff --check` → clean. Caches removed.

## Residual risk (this round)
- Unchanged from above: API trust boundary, not a cryptographic seal; per-process;
  re-derivation/egress/Schema still bound a forged batch. The remaining forge paths
  all require deliberate closure/class-internal introspection, not import or a
  normal call.

---

# Fix round 2 — Terra final-gate findings (on HEAD f2d5caaf)

Terra's final gate (`runs/econ-model-2026-07-17/python-sdk-terra-final-verdict.md`,
all three findings captain-verified real) BLOCKED with three items. Remediated on
top of `f2d5caaf`, left **uncommitted**. Repo `.venv` (Python 3.13), exact
economics Schema installed (`traigent-schema==4.10.0`). Files changed this round:
`traigent/economics/client.py`, `traigent/economics/result.py`,
`traigent/economics/__init__.py`, `traigent/__init__.py`,
`tests/unit/economics/test_client.py`, `tests/unit/economics/test_public_surface.py`.
No behavior changed beyond the three fixes; all prior sealing / egress / idempotency
/ retry / response-reconciliation invariants and their tests remain intact and
unmodified (no prior test asserted the removed `detail` field or the root lockout).

## P1 (finding 1) — PreparedTelemetryBatch repr leaked payload / evidence
`PreparedTelemetryBatch` used the default dataclass repr, so `content` (raw wire
bytes) and `body` (raw payload mapping, incl. shared evidence pointers) appeared in
`repr()`, `str()`, `f"{batch!r}"`, and `%r` log records. Fix: `content` and `body`
are now `field(repr=False)` (import `field` added). This changes the repr ONLY —
the fields stay ordinary init/eq fields, wire bytes are unchanged, and the
identity-bound issuance token (an instance attribute set outside the dataclass
fields via `object.__setattr__`) is untouched, so issuance/sealing semantics and
their tests are preserved. Identifiers (project_id, idempotency_key, batch_id,
submitted, event_ids) remain visible. New test
`test_prepared_batch_repr_does_not_leak_payload_or_evidence`: a batch built from the
sensitive `_FULL_RUN_EVENT` — its unique evidence-pointer sentinel is genuinely in
`prepared.content` (wire preserved) — asserts the sentinel is in NONE of
`repr()`/`str()`/`f"{batch!r}"` nor a `%r` logging call (`caplog`), while identifiers
still render. Existing byte-identity tests (`test_reordered_events_*`,
`test_retries_send_identical_bytes_and_key`, `test_exact_prepared_object_submits_and_retries_identical`)
pass unmodified.

## P1 (finding 2) — backend `detail` retained and became loggable
`result.py` stored the response rejection `detail` string on the public `Rejection`
dataclass. Per the response schema, `detail` is `x-privacy-classification:
user_content` with a prose-only (not machine-enforced) no-payload-echo rule, so a
`detail` quoting a withheld value leaked through the public result / its repr / logs.
Fix: `detail`'s shape is still validated exactly as before (string-or-absent, else
`EconomicsResponseError`) but is then **DISCARDED** — the `detail` field is removed
from `Rejection` entirely (matching the accepted JS SDK, which drops `detail` from
its public rejection type: `traigent-js` `src/economics/telemetry-client.ts`
`EconomicsTelemetryRejection`). `event_index` / `reason` / `event_id` are kept. No
existing test referenced `detail` (grep-confirmed), so none needed updating. New
tests: `test_rejection_detail_is_discarded_and_never_surfaced` (a schema-valid 422
whose rejection `detail` carries a unique sentinel yields a result where the sentinel
is in NO public field, `repr`, `str`, or `rejection_reasons`, `hasattr(rej,"detail")`
is False, while index/reason/event_id are surfaced correctly) and
`test_malformed_rejection_detail_still_fails_closed` (a non-string `detail` still
fails closed).

## P2 (finding 3) — Python/JS root-export parity incorrectly diverged
The SDK claimed subpackage-only with a stale "no JS counterpart" rationale and a
test that LOCKED OUT the root import, but the JS SDK root-exports
`EconomicsTelemetryClient` (`traigent-js` `08d8931`, `src/index.ts:566`) and
TraigentSchema `c27a034` now classifies it `matched` (and lists it in
`javascript.requiredRootExports`). Fix (true parity): added `EconomicsTelemetryClient`
as a `traigent` root **lazy** export following the existing `_LAZY_EXPORTS` pattern
exactly — entry `("traigent.economics", "EconomicsTelemetryClient")` plus a
`traigent.__all__` membership — so plain `import traigent` still does not import
`traigent.economics` until the symbol is accessed. Rewrote the stale
`traigent.economics` module docstring (now: matched-manifest + JS-root-export
rationale; subpackage import still supported). Replaced
`test_client_is_not_a_root_export_by_policy` with three tests:
`test_client_is_a_root_export` (in `traigent.__all__`),
`test_root_client_resolves_to_the_same_object_as_the_subpackage` (lazy
`traigent.EconomicsTelemetryClient is traigent.economics.EconomicsTelemetryClient`),
and `test_root_import_does_not_eagerly_import_economics_subpackage` (fresh-interpreter
subprocess asserting `traigent.economics` absent from `sys.modules` after
`import traigent`, present only after attribute access — mirrors
`test_init_imports.py::test_main_module_import_stays_cold`). The stale report
rationale flagged by Terra is corrected inline above (changed-files bullet + Owner
decisions SUPERSEDED note).

## Commands and results (this round)
- Focused: `.venv/bin/python -m pytest tests/unit/economics -q`
  → **157 passed** (was 152; +5: repr-leak, detail-discard, malformed-detail, and
  net +2 in public-surface — removed 1 lockout test, added 3 parity tests).
- Econ + parity + init:
  `.venv/bin/python -m pytest tests/unit/economics tests/cross_sdk_oracles/test_js_public_parity_manifest.py tests/unit/test_init_imports.py -q`
  → **186 passed** (0 skipped — the parity-manifest classification test PASSES, not
  skips: `EconomicsTelemetryClient` is now both in `traigent.__all__` and `matched`
  in the resolved sibling manifest at `c27a034`).
- Broader lane (root `__init__` changed): `.venv/bin/python -m pytest tests/unit/cloud -q`
  → **1884 passed, 2 skipped** (pre-existing env skips: httpx-absent guard;
  memory-variance test).
- Ruff: `.venv/bin/python -m ruff check traigent/economics tests/unit/economics traigent/__init__.py`
  → **All checks passed**;
  `.venv/bin/ruff format --check traigent/economics tests/unit/economics traigent/__init__.py`
  → **17 files already formatted**.
- Mypy: `.venv/bin/python -m mypy traigent/economics`
  → **Success: no issues found in 8 source files**.
- `git diff --check` → clean. Generated `__pycache__` removed before handoff.

## Residuals (this round)
- The parity-manifest classification lives in the **sibling TraigentSchema branch**
  (`c27a034`, read-only here). This Python packet's root export is correct only once
  that Schema change lands on the matching branch; the captain sequences the
  cross-repo merge. The local parity test resolves the sibling worktree manifest and
  passes; if a stale manifest (no economics classification) were resolved instead,
  `test_python_public_root_symbols_are_classified_in_parity_manifest` would `skip`
  locally (CI would fail), not silently pass.
- The `detail` shape check in `_parse_rejections` is now defense-in-depth: the
  per-status response schema (`detail` typed `string`) already rejects a non-string
  `detail` at `validate_response_or_fail`, before the hand-parse check. It is kept
  (validate-exactly-as-today per the finding), and the malformed-detail test
  exercises the end-to-end fail-closed behavior.

## Owner decisions
None — all three fixes had a single clearly-correct implementation (true parity was
the captain-chosen resolution recorded in the Terra verdict); no genuine owner
decision arose this round.
