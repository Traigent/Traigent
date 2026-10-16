# Portal-dev laptop agent setup

Use this when running Traigent from this laptop against the dev backend.

## Backend URL

Use the API host, not the portal host:

```bash
export TRAIGENT_BACKEND_URL=https://api-dev.traigent.ai
export TRAIGENT_API_URL=https://api-dev.traigent.ai
```

`portal-dev.traigent.ai` is the browser application. SDK and agent code should call
`api-dev.traigent.ai`.

## API key

Use a key that validates against `POST /api/v1/keys/validate`. Do not paste keys into shell
history. Store them in the SDK's encrypted credential store, which needs
`TRAIGENT_MASTER_PASSWORD`. Enter that password at a hidden prompt (this works in bash
and zsh), then log in:

```bash
printf 'Encrypted-store password: '; IFS= read -r -s TRAIGENT_MASTER_PASSWORD; printf '\n'
export TRAIGENT_MASTER_PASSWORD
traigent auth login --backend-url https://api-dev.traigent.ai
```

To store an existing API key instead of logging in, run the first two lines above, then
`traigent auth configure` with `TRAIGENT_BACKEND_URL` exported as shown earlier. Check
that the current backend it shows is `https://api-dev.traigent.ai` and answer `n` to
"Change backend URL?". At "Select authentication method", choose `2` (API Key) and
paste the key at the hidden "Enter API Key" prompt; the key is stored with the current
backend URL. If it prints "Failed to store API key", nothing was saved: check that
`TRAIGENT_MASTER_PASSWORD` is exported and is not a weak or placeholder value.

Later CLI and SDK processes need the same `TRAIGENT_MASTER_PASSWORD` to unlock the store;
keep it in a secret manager rather than in a file or your shell history.

The SDK ignores a plaintext `~/.traigent/credentials.json` by default.
`TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS=true` is only a one-time migration aid; see
[Migrating from `~/.traigent/credentials.json`](../features/authentication.md#migrating-from-traigentcredentialsjson).

## Smoke checks

```bash
traigent auth whoami
```

`whoami` validates the API key the SDK will send, prints where it came from (never the key),
and exits non-zero when no key resolves.

If a newly minted key is listed in the UI but `whoami` or `/keys/validate` returns 401,
mint a new key after the backend fix that verifies generated keys before returning them.
Do not keep retrying an unpersisted one-time key; it cannot be recovered after creation.
