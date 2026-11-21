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
(`TRAIGENT_BACKEND_URL=http://127.0.0.1:PORT`, `TRAIGENT_EGRESS_RELAY=1` and
`NO_PROXY=...,127.0.0.1`, see below), then serves
until interrupted. With `--exec` it sets `TRAIGENT_BACKEND_URL`, `TRAIGENT_API_URL` and
`TRAIGENT_EGRESS_RELAY=1` for the child only, and appends `127.0.0.1` to the child's
`NO_PROXY` and `no_proxy` (existing values are kept),
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
- Route shape keeps a path segment only when it is a word from a fixed allowlist (the
  static route words the SDK itself uses, plus `v1`-style version markers); every other
  segment, including a lowercase id or name you chose, becomes `{}`. Request lines say
  `route_source: "allowlist"` (logs from the first version say `"heuristic"` and still
  read fine with `list` and `show`).

## Limits

- The SDK refuses plain `http://` backends outside development and test environments
  (`validate_cloud_base_url`). `TRAIGENT_EGRESS_RELAY` adds one narrow exception, so a
  production-configured SDK can use the relay without switching to dev mode. The switch
  is on only when its value is exactly `1` or `true` (case-insensitive). It admits a
  plain-`http` backend URL only when all of these hold: the URL's origin (scheme, host,
  port) equals the origin of `TRAIGENT_BACKEND_URL` (no fallback to `TRAIGENT_API_URL`
  or stored credentials); the host is a loopback IP literal (`127.0.0.0/8` or `::1`;
  never `localhost` or another name, and no DNS lookup is involved); the URL names an
  explicit port other than `80` or `0` (a relay on port 80 is never needed, and some HTTP
  libraries drop a default port from the URL they print); the URL has no userinfo
  (not even an empty `@`), no backslash, no query or fragment; and the transport gate is not `TRAIGENT_TRANSPORT_GATE=off` (redirects
  are refused for that origin). For that one origin it also allows the loopback
  destination, as `TRAIGENT_ALLOW_PRIVATE_BACKEND` would, without needing that switch.
  It relaxes nothing else: metadata and link-local blocking, path, whitespace and
  userinfo checks, redirect refusal, and the https requirement for every other origin
  (including other loopback ports) are unchanged, and `https://` URLs and dev/test
  environments behave as before.
- An environment proxy must not carry traffic meant for the relay. The SDK's HTTP
  clients honour `HTTP_PROXY`, `http_proxy` and `ALL_PROXY`; with one set (an `http` or
  `all` entry as `urllib.request.getproxies()` reports it; `HTTPS_PROXY` alone never
  carries plain http), the switch is refused with a fixed message unless at least one of `NO_PROXY` /
  `no_proxy` is set, every one that is set (an empty one counts, because the lower-case
  spelling wins in urllib, aiohttp and httpx and an empty value removes the bypass) is
  `*` or lists the relay host as a bare IPv4 literal (`127.0.0.1`), and the effective
  value `getproxies()` reports agrees. For an IPv6 relay address only the whole value
  `*` is accepted (urllib compares `[::1]`, requests and aiohttp compare `::1`); the
  relay itself binds `127.0.0.1` only, so the IPv4 address is the supported relay
  address. `host:port`, CIDR ranges and suffix tokens are not accepted, because
  requests, aiohttp, httpx and urllib disagree on them. Add
  the relay host to `NO_PROXY` (`--exec` does it for the child). The check runs each time
  the SDK validates its backend URL (when a client or config is constructed, and before
  credential-bearing requests that re-validate); a proxy variable set after a client
  validated its URL is not re-checked.
- The route allowlist cannot tell a customer-chosen segment that equals an allowlisted
  route word (an agent named `billing`, say) from the route word itself: it is logged as
  that word, because the log has no per-route templates yet. Contract-derived route
  templates will replace the allowlist.
- The relay sees only traffic sent to it. Anything the SDK sends to another host is not
  covered.
- A response line is missing if the log could not be written after the exchange.
