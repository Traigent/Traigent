# Nous provider (auth flow unverified)

Traigent's Nous Portal provider (`traigent/integrations/llms/nous_auth.py`)
mints a short-lived bearer token from a refresh token, or returns
`NOUS_API_KEY` as-is.

**Status: UNVERIFIED against the live service.** These parts were never
confirmed against the real Nous Portal or `hermes-agent` output:

- the token endpoint (`TRAIGENT_NOUS_TOKEN_URL` overrides the default),
- the `~/.hermes/auth.json` schema (`TRAIGENT_NOUS_AUTH_FILE` points elsewhere),
- the refresh request shape (JSON body with `refresh_token`).

The provider logs one value-free warning at first use. Using `NOUS_API_KEY`
avoids the refresh flow, but the inference base URL and model-id spelling are
unverified too. Report mismatches so the constants can be corrected.
