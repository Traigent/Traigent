# Implementing a ClientStore (guide for coding agents)

Your task: implement `ClientStore` for the customer's own storage (S3, GCS, a git repo, a NAS) so the SDK can keep a client-side copy of each pillar version's representation and restore it later.

Source of truth is the code: `traigent/legend/store.py`, `traigent/legend/recovery.py`. Runnable reference: `examples/advanced/legend-client-store/directory_store.py`.

## Hard rules

1. Content stays on the customer side. Traigent receives only fingerprints, closed statuses and opaque refs. Your adapter must not send bytes, keys or error text anywhere except the customer's storage.
2. There is no delete. The protocol has no delete method. Do not add one, and do not "clean up" conformance entries.
3. The key is a scheme-qualified recovery-manifest fingerprint (`rmf0:sha256:<64 hex>`), supplied by the SDK. Never derive or invent keys. Reserved synthetic `rmfc0:sha256:...` keys appear only during the conformance check. Call `check_key` (`store.py`) first; it raises `LegendError("invalid_request")` for anything else.
4. Never overwrite a different capture under the same key. Identical bytes again is fine; different bytes is `store_key_conflict`.
5. A successful `put` means `stored_unverified`, never more. Only the SDK's isolated round trip (get into a fresh directory, re-hash) makes a capture `round_trip_verified`.

## The protocol (`traigent/legend/store.py`)

```python
class ClientStore(Protocol):
    def put(self, kind: str, manifest_fingerprint: str, stream: BinaryIO) -> PutStatus: ...
    def get(self, kind: str, manifest_fingerprint: str, dest_dir: Path) -> GetStatus: ...
    def has(self, kind: str, manifest_fingerprint: str) -> HasStatus: ...
```

| Method | Closed statuses | Mapping |
| --- | --- | --- |
| `put` (`PutStatus`) | `stored_unverified`, `store_key_conflict`, `unreachable`, `permission_denied` | stored or identical bytes already there / different bytes at the key / network or I/O fault / auth or ACL refusal |
| `get` (`GetStatus`) | `retrieved`, `missing`, `truncated`, `corrupted`, `unreachable`, `permission_denied` | write the bytes to `dest_dir / BUNDLE_NAME` (`"bundle.rmf"`), then report how it went |
| `has` (`HasStatus`) | `available`, `missing`, `unknown` | `unknown` when you cannot tell (outage); never guess |

Return the status; do not raise for storage faults and do not put exception text, paths or keys in any message. The SDK treats an adapter exception as a closed failure with the text dropped (`restore` maps it to `unreachable`).

## Recipe

1. **Implement** `put`, `get`, `has`. Read the whole `stream` in `put`; store the bytes plus their SHA-256 and length; on `get`, re-check both before returning `retrieved` (return `truncated` if shorter than recorded, `corrupted` on any other mismatch).
2. **Self-test.** `from traigent.legend import conformance_check; outcome = conformance_check(store)`. It runs, for every pillar kind, put, `has`, get into a fresh temp dir, byte comparison, then a second `put` of different bytes under the same key that must not succeed and must not change what `get` returns. `outcome.passed` is required: `capture` refuses with `conformance_failed` otherwise. Each run leaves small synthetic entries (no delete), so run it once per store, not in a loop.
3. **Capture.** `capture(version_ref, source, store=store, conformance=outcome)` returns a `CaptureRecord` with `recovery_state` (`stored_unverified` or `round_trip_verified`) and `coverage` (`status`, `covered`, `missing`).
4. **Restore into an isolated destination.** `restore(record, "/new/empty/dir", store=store)` never touches live state. Read `RestoreResult.grade`:
   - `exact`: representations and pillar fingerprint re-verified, coverage complete, dependencies satisfied.
   - `partial`: coverage incomplete, or dependencies not confirmed (pass `dependency_present=` to confirm them).
   - `substituted`: bytes verified but the pillar fingerprint could not be re-derived client-side (see backend-keyed kinds below).
   - `failed`: bytes absent or unverifiable; `store_status` says which closed status (`missing`, `truncated`, `corrupted`, `unreachable`, `permission_denied`). `dest` is `None`.
   Also check `RestoreResult.coverage["missing"]`: it lists what the capture could not hold.
5. **Activate separately.** `activate(result, target)` copies a restored directory into place. It requires `exact`; a `partial` or `substituted` result needs `accept="partial"` or `accept="substituted"` and keeps its grade. A `failed` result raises `LegendError("restore_failed")`. Capture and restore never write into the live tree.

## Pitfalls

- **Partial writes.** Write to a temp name in the same location, flush and fsync, then publish atomically (rename or hard link; on object stores a single PUT, or a conditional write). Never expose a half-written key. A crash must leave nothing visible or a complete entry.
- **Eventual consistency.** `put` then `has` must say `available` (conformance checks it). On stores with delayed visibility, wait or read through a consistent path before acknowledging; otherwise return `unreachable`.
- **No-overwrite needs an atomic primitive.** Use create-if-absent (`If-None-Match: *`, `ifGenerationMatch=0`, `link`, `O_EXCL`), not check-then-write. On a lost race, compare digests: equal means `stored_unverified`, different means `store_key_conflict`.
- **Credentials.** Take them from the customer's own secret store or workload identity. Never hard-code them, never log keys or fingerprints with them, never return them in statuses.
- **Concurrency.** Several captures may `put` at once; keys are content-derived, so safe concurrency is exactly the atomic create-if-absent above.
- **Large blobs.** `put` receives a stream; prefer multipart or streamed upload when the storage supports it, and keep the length and digest to check in `get`. A short read on download is `truncated`.
- **Backend-keyed kinds.** A `search_space`, and an `evaluator` whose authoritative format is keyed or unknown, have fingerprints derived with a key held by the Backend, so the client cannot recompute them and restore tops out at `substituted` without a server-origin match report (`traigent/legend/recovery.py`, grade selection in `restore`). Your store is not at fault; do not try to "fix" the grade.

## Check your work

```bash
python -m pytest tests/unit/legend/test_directory_store_example.py -q
python examples/advanced/legend-client-store/directory_store.py
```

Copy the example's test, swap in your adapter, and keep it network-free by using a local emulator or a fake client for the storage SDK.
