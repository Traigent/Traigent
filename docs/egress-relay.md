# Egress relay: see what the SDK sends to the Traigent backend

`traigent egress relay` runs a small forwarding proxy on your own machine. Point the
SDK at it and you get a local log of every request: method, route shape, body size,
body SHA-256 and response status. The log never holds values, header names or values,
query strings, body content or JSON key names.

```bash
traigent egress relay --upstream https://YOUR-TRAIGENT-BACKEND --exec "python my_script.py"
traigent egress list
traigent egress show 1
```

Without `--exec` the relay prints its address and the line to export
(`TRAIGENT_BACKEND_URL=http://127.0.0.1:PORT` and `TRAIGENT_EGRESS_RELAY=1`), then serves
until interrupted. With `--exec` it sets `TRAIGENT_BACKEND_URL`, `TRAIGENT_API_URL` and
`TRAIGENT_EGRESS_RELAY=1` for the child only,
stops when the child exits and returns the child's exit code. The log is
`.traigent/egress/relay.jsonl` (mode 0600) unless `--log` says otherwise.

## What it guarantees

- Binds `127.0.0.1` only; there is no flag to widen it.
- The upstream must be `https://` and TLS is always verified against the default trust
  store. Plain `http://` is accepted for a loopback IP literal only (tests).
- The request line is written and fsynced before the first byte is forwarded. If it
  cannot be written the relay answers 502 and forwards nothing.
- Request and response bytes pass through unchanged. Redirects are never followed.
  Interim `1xx` responses (such as `103 Early Hints`) are consumed, not forwarded; a
  `101 Switching Protocols` is refused with 502 (upgrades are unsupported) and logged
  with `outcome: upstream_unsupported`. Response bodies are passed on as they arrive.
- Request framing is never rewritten. A request with a `Transfer-Encoding` other than
  exactly `chunked` is refused with 501, and one that carries both `Transfer-Encoding`
  and `Content-Length` (or two `Content-Length` headers) with 400. A refused request is
  not forwarded, so it gets no log line.
- A log file that ends with an incomplete line is refused at start (move or repair it);
  appending would glue a new record onto the torn one.
- Stopping the relay (child exit, Ctrl-C) waits about 3 seconds for running exchanges,
  then cuts the rest; their response lines carry `outcome: relay_stopped`.
- If a response line cannot be written after the exchange, the response is still
  delivered, one fixed line goes to stderr, and every further request gets 502.
- Route shape is a heuristic: path segments that are not lowercase words (or `v1`-style
  version markers) become `{}`. A lowercase word in a path is logged as written.

## Limits

- The SDK refuses plain `http://` backends outside development and test environments
  (`validate_cloud_base_url`). `TRAIGENT_EGRESS_RELAY` adds one narrow exception, so a
  production-configured SDK can use the relay without switching to dev mode. The switch
  is on only when its value is exactly `1` or `true` (case-insensitive). It admits a
  plain-`http` backend URL only when all of these hold: the URL's origin (scheme, host,
  port) equals the origin of `TRAIGENT_BACKEND_URL` (no fallback to `TRAIGENT_API_URL`
  or stored credentials); the host is a loopback IP literal (`127.0.0.0/8` or `::1`;
  never `localhost` or another name, and no DNS lookup is involved); the URL names an
  explicit port; and the transport gate is not `TRAIGENT_TRANSPORT_GATE=off` (redirects
  are refused for that origin). For that one origin it also allows the loopback
  destination, as `TRAIGENT_ALLOW_PRIVATE_BACKEND` would, without needing that switch.
  It relaxes nothing else: metadata and link-local blocking, path, whitespace and
  userinfo checks, redirect refusal, and the https requirement for every other origin
  (including other loopback ports) are unchanged, and `https://` URLs and dev/test
  environments behave as before.
- The relay sees only traffic sent to it. Anything the SDK sends to another host is not
  covered.
- A response line is missing if the log could not be written after the exchange.
