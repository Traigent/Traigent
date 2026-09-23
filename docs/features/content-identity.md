# Content Identity (opt-in)

## Overview

Content identity gives every evaluation example a keyed, tenant-scoped id
(`ex1:…`) and version (`exv1:…`), and every run a dataset root, so the Backend
can tell which examples a run actually evaluated. The ids are minted with
purpose keys that only your tenant's Backend can issue; the SDK never derives or
invents a key.

In this release content identity is **off by default**. With it off, the SDK
sends exactly the same payloads as before.

## Enabling it

Either set the environment variable:

```bash
export TRAIGENT_CONTENT_IDENTITY=1   # also accepts true / yes / on
```

or pass it explicitly on the config (an explicit value overrides the env var):

```python
from traigent.config.types import TraigentConfig

config = TraigentConfig(content_identity=True)   # force on
config = TraigentConfig(content_identity=False)  # force off, even if the env var is set
```

`content_identity=None` (the default) defers to `TRAIGENT_CONTENT_IDENTITY`.

## What happens when it is on

When a run talks to a Backend (not `offline=True` / `TRAIGENT_OFFLINE=1`, and
not privacy mode), `optimize()` makes one request before creating the session:

```
POST /api/v1/content-identity/purpose-keys
```

using the same API key / JWT headers the SDK already sends. The Backend answers
with the tenant's purpose-key grant (`tenant_id`, `kid`, two hex keys). The SDK
validates it strictly (`ContentIdentityKeys.from_grant`) and scopes it to that
run only: it is never installed process-wide, so another `optimize()` run in the
same process (another tenant's, or one with the switch off) never sees it, and
it is dropped when the run ends. The session and every trial then
carry a `content_identity` object with `key_status: "available"` and the
grant's `kid`.

- **Keys stay in memory.** They are never written to disk, cached, logged, sent
  in telemetry or included in any payload; only the derived ids and the public
  `kid` leave the process.
- **Your own grant wins.** If you installed keys yourself with
  `traigent.identity.set_content_identity_keys(...)`, no request is made and
  your keys are used.
- **Privacy mode and offline runs** make no request.

## When the fetch fails

Any failure — network error, HTTP 401 / 403 / 404 / 429 / 503
(`content_identity_keys_unavailable`), or a malformed grant — never fails the
run. The run proceeds exactly as if content identity were off: no
`content_identity` object is sent. One warning names the failure type only
(for example `HTTP 503`); the response body is never logged.
