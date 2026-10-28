# Storing the execution snapshot

An execution snapshot is one verifiable bundle of everything an optimization
used: the evaluation dataset rows, the configuration and configuration space,
the agent's source, the evaluator's source, and (only if you ask) your
uncommitted git changes. You write it to storage **you own** (a local
directory, an S3, Google Cloud Storage or Azure Blob container, or a private GitHub
repository) and can verify it later.

Nothing in `traigent.snapshot` contacts Traigent. The snapshot is built and
stored on your side.

## What a v1 snapshot does and does not say

A v1 snapshot is an **unbound client declaration**. It says "these bytes were
captured". It does not say "this is what ran": it is not linked to a run, a
trial, or the configuration that was actually selected. Linking a snapshot to a
run, and putting its digest into a signed certificate, are separate, later
features. Treat the manifest digest as the snapshot's name, and keep it
somewhere you trust.

## Capture and store

```python
from traigent.snapshot import (
    EvaluatorInputs, build_execution_snapshot, publish_snapshot, verify_snapshot,
)
from traigent.snapshot.backends import LocalDirStore  # or S3Store, GCSStore, AzureBlobStore, GitHubStore

snapshot = build_execution_snapshot(
    dataset=examples,                      # a list / dataset object, not a generator
    config={"model": "gpt-5.6-luna", "temperature": 0.2},
    config_space=config_space,
    agent=my_agent,
    agent_id="support-agent",              # optional: adds the agent build digest
    evaluator=EvaluatorInputs(metric_functions={"accuracy": accuracy}),
    declared_assets={"prompts": {"system": SYSTEM_PROMPT}, "tool_definitions": {}},
)

store = LocalDirStore("/data/snapshots")
ref = publish_snapshot(store, snapshot)    # SnapshotRef(manifest_digest, locator)
assert verify_snapshot(store, ref.manifest_digest).integrity == "ok"
```

For S3 install the extra and pass a bucket and prefix:

```bash
pip install "traigent[snapshot-s3]"
```

```python
from traigent.snapshot.backends import S3Store

store = S3Store("my-bucket", "traigent/snapshots")  # uses your default AWS credentials
```

Remote endpoints must use HTTPS. For an S3-compatible service pass your own
boto3 client built with `endpoint_url=...`; the store then tests, once before
the first write, that the endpoint honours conditional writes
(`If-None-Match: *`) and refuses it otherwise; only a `412` answer counts as
proof (a `409` conflict is retried later, never remembered as support). Only
endpoints that pass this check are supported. The client must expose its
endpoint (`client.meta.endpoint_url`), or the store refuses it. The extra
requires `boto3>=1.35.2`, the first release whose `PutObject` accepts
`IfNoneMatch`. Plain `http://` is refused unless you pass
`allow_insecure_local=True` **and** the host is `localhost`, `127.0.0.1` or
`::1`.

### Backends

| Backend | Install | Destination | Atomic create |
|---------|---------|-------------|---------------|
| `LocalDirStore(root)` | none | `file://<root>` | hard link, no clobber |
| `S3Store(bucket, prefix)` | `traigent[snapshot-s3]` | `s3://<bucket>/<prefix>` | `If-None-Match: *` |
| `GCSStore(bucket, prefix)` | `traigent[snapshot-gcs]` | `gs://<bucket>/<prefix>` | `if_generation_match=0` |
| `AzureBlobStore(container_client, prefix)` or `AzureBlobStore.from_account(account_url, container, credential, prefix)` | `traigent[snapshot-azure]` | `azblob://<account host>/<container>/<prefix>` | `upload_blob(overwrite=False)` |
| `GitHubStore(repo, branch, prefix)` | none (uses `requests`) | `github://<owner>/<repo>@<branch>/<prefix>` | Contents API `PUT` without `sha` |

Every backend refuses to overwrite: if the object already exists with the same
bytes the write is a no-op, with different bytes it raises `CONFLICT`. Locators
never contain a query string, user name or token; for Azure any SAS token in
the URL is dropped from the locator and from the store's own attributes (the
credential stays inside the `ContainerClient`). Custom GCS emulator endpoints and
Azure account URLs follow the same HTTPS rule as S3 (plain `http://` only for
loopback with `allow_insecure_local=True`).

```python
from traigent.snapshot.backends import AzureBlobStore, GCSStore

gcs = GCSStore("my-bucket", "traigent/snapshots")  # Application Default Credentials
azure = AzureBlobStore.from_account(
    "https://myaccount.blob.core.windows.net", "snapshots", credential, "traigent"
)
```

S3-compatible services are supported only if they pass the conditional-write
probe described above; the probe creates and deletes one `<prefix>/.probe/<id>`
object on first write, so the credential needs `DeleteObject` there (a failed
delete is ignored and leaves that one small object).

### GitHub repository store

```python
from traigent.snapshot.backends import GitHubStore

store = GitHubStore("my-org/eval-snapshots", branch="main", prefix="traigent")
# the token is read from the GITHUB_TOKEN environment variable (token_env=...) on every call
```

Safety rules:

- **Private repositories only by default.** The repository's visibility is
  checked once, before the first request that touches content; a public repository
  raises `UNSAFE_DESTINATION` unless you pass `allow_public=True` (and then
  anyone can read the snapshot, including your source and dataset rows).
- **The branch must already exist.** A missing branch, or an empty repository
  with no commits, raises `NOT_FOUND`. Create the branch (one initial commit)
  first; the store never creates branches or repositories.
- **Append-only.** Objects are created with the Contents API without a `sha`, so
  GitHub refuses to replace a file; an existing file is read and compared. Each
  object is one commit on the branch, writes are serialized, and a branch-advance
  conflict (409) is retried at most 3 times with 1, 2 and 4 second backoff.
- **50 MiB per part.** The SDK refuses any part over 50 MiB before writing
  anything (GitHub warns above 50 MiB and rejects above 100 MiB). Each snapshot
  is a handful of small commits, so GitHub suits small snapshots; use S3, GCS or
  Azure for large datasets.
- **Rate limits.** `Retry-After` and `x-ratelimit-reset` are honoured. A
  secondary rate limit without those headers (a `429`, or a `403` whose message
  names a secondary limit) waits the documented minimum of 60 seconds. One
  operation (`read` or `create_if_absent`, including its preflight, conflict
  rereads and backoff) sleeps at most 60 seconds in total, after which
  `BACKEND_UNAVAILABLE` is raised. A plain permission `403` is `AUTH`. Every
  request has a 30 second timeout.
- **A 404 on an object is checked.** After a missing-object `404` the store
  re-checks the branch: a deleted branch or lost repository access raises
  `NOT_FOUND` instead of reading as "object absent".
- **The token goes only to the API host.** Redirects are followed only within
  the same scheme and host as `api_url`; a redirect anywhere else, or one with
  userinfo or a backslash, raises `UNSAFE_DESTINATION` and the token is not
  sent. The URL checked is the prepared URL that is actually sent. The token is never stored, shown
  in `repr` or put in an error. For GitHub Enterprise pass
  `api_url="https://<host>/api/v3"` (HTTPS required).

## What each part contains

| Part | Contents |
|------|----------|
| `dataset/examples.jsonl` | One JSON row per example (`input`, `expected`, and `id` / `metadata` when present), in order. Decoding it gives the same dataset fingerprint as the original. A generator or other non-materialized input is refused, never consumed. A value that is not plain JSON makes the part unavailable (`NOT_SERIALIZABLE`); nothing is stringified. |
| `config/config.json`, `config/config_space.json` | Strict JSON. |
| `agent/callable.py` | The agent callable's source. |
| `agent/source/<project path>` | Each project source file the SDK finds loaded in the agent's project (the helper modules). |
| `agent/assets/prompts.json`, `agent/assets/tool_definitions.json` | The prompts and tool definitions you declare through `declared_assets`. The SDK cannot discover these. |
| `agent/bound_state.json` | Partial arguments, closure cells and instance attributes, when they serialize canonically. |
| `agent/git.json` | Commit, dirty flag, project-relative entry path and a count of untracked files left out. Never a remote URL. |
| `agent/git_diff.patch` | Only with `capture_dirty_diff=True` (below). |
| `evaluator/<role>/<name>.py` | Source of the custom evaluator, scoring function and each metric function. External evaluators have no source (`EXTERNAL_EVALUATOR`). |

The manifest (`tes1`, canonical JSON) lists every part with its SHA-256 and
size, the legacy fingerprints (`dataset_fp1`, `config_space_fp1`, `agent_fp1`,
`evaluator_fp1`), the evaluator code identity, the agent build digest, and
`snapshot_coverage.agent` (`complete` only when callable, helper modules,
prompts, tool definitions and bound state were all stored). The manifest digest
is `sha256:` plus the SHA-256 of the manifest bytes.

A part that cannot be captured is recorded as unavailable with a closed reason
code, so a partial snapshot says exactly what is missing.

## Uncommitted changes (opt in)

`capture_dirty_diff` is off by default and the part is recorded as
`NOT_REQUESTED`. Turn it on only if you accept that the diff can contain
anything you changed in your working tree, including secrets you edited:

```python
snapshot = build_execution_snapshot(agent=my_agent, capture_dirty_diff=True)
```

It is `git diff --no-ext-diff --no-textconv --binary HEAD` for the agent's
project (staged and unstaged changes against `HEAD`), capped at 10 MiB
(`TOO_LARGE` beyond that). The diff is read as a stream and git is stopped as
soon as the cap is crossed, so a huge diff never sits in memory. Untracked files are never included, only counted.

## Verify levels

`verify_snapshot(store, manifest_digest)` returns two independent answers:

- `integrity`: `ok`, `manifest_mismatch` (the stored manifest is not the one
  you asked for), `part_missing` or `part_corrupt`. `problems` lists which
  parts.
- `completeness`: `complete` or `partial`; `unavailable` lists each missing
  part with its reason.

A missing manifest raises `SnapshotStoreError(NOT_FOUND)`. Authentication and
transport failures always raise and are never reported as "missing".
`load_snapshot` returns the snapshot only after validating every byte string.

Republishing the same snapshot writes nothing new. Rebuilding later produces a
new snapshot (the manifest has a new `created_at`), but unchanged parts are
stored once.

## Idempotent puts

`SnapshotArtifactStore(store).put(snapshot, idem_key)` adapts the store to the
SDK's `ArtifactStore` port. The same key with the same snapshot returns the
same reference; the same key with a different snapshot raises
`SnapshotStoreError(CONFLICT)`. The key is stored only as a SHA-256 hash.

## Errors

`SnapshotStoreError` carries only a closed `code`, the part path and the
backend name. No provider message, URL, credential or content is attached, and
errors are raised without a chained cause.
