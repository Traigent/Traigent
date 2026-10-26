# Storing the execution snapshot

An execution snapshot is one verifiable bundle of everything an optimization
used: the evaluation dataset rows, the configuration and configuration space,
the agent's source, the evaluator's source, and (only if you ask) your
uncommitted git changes. You write it to storage **you own** (a local
directory or an S3 bucket) and can verify it later.

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
from traigent.snapshot.backends import LocalDirStore  # or S3Store

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
(`If-None-Match: *`) and refuses it otherwise. Only endpoints that pass this
check are supported. Plain `http://` is refused unless you pass
`allow_insecure_local=True` **and** the host is `localhost`, `127.0.0.1` or
`::1`.

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
(`TOO_LARGE` beyond that). Untracked files are never included, only counted.

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
