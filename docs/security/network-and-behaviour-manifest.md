# Python SDK network and behaviour manifest

What the Traigent Python SDK does on a customer machine that a firewall, proxy,
antivirus or EDR product may notice, and how to turn each item off. Every claim
cites where it comes from:

- **[R]** the PEP 578 recorder (`scripts/security/behaviour_audit.py`), with the
  run tables in [behaviour-audit-evidence-2026-09-25.md](behaviour-audit-evidence-2026-09-25.md)
  (origin/develop `e2a04f85`, CPython 3.12.13, litellm 1.98.0 from `uv.lock`);
- **[S]** reading the source at the cited `file:line` (static; the recorder
  workloads did not reach that code).

Re-check before a customer deployment:

```bash
pytest tests/security/test_egress_allowlist.py -m egress_audit -n 0 -rxX
python scripts/security/behaviour_audit.py run optimize_mock token_count_llama --pins
```

## For your security team: allowlist summary

| Destination | Port | Needed? | Contacted when | Turn off with |
|---|---|---|---|---|
| `portal.traigent.ai` (or your `TRAIGENT_BACKEND_URL`) | 443 | Only for cloud features | A connected run: the SDK resolves the host when it validates the URL (`traigent/cloud/url_security.py:270`), then calls `POST /api/v1/keys/validate` and session endpoints [R] | `TRAIGENT_OFFLINE_MODE=true`, or no `TRAIGENT_API_KEY` |
| `raw.githubusercontent.com` | 443 | No | **Not contacted by default** as of `traigent/__init__.py:72-77` / `traigent/utils/cost_calculator.py:30-46`: `import traigent` pins `LITELLM_LOCAL_MODEL_COST_MAP=True` before litellm's own import-time fetch runs, for every code path (including `from traigent.api.decorators import EvaluationOptions`, which used to re-set the pin unconditionally and mask a broken opt-out — fixed). Opt back in with `TRAIGENT_LITELLM_LIVE_PRICES=1` [R]. Residual: if the *application* imports `litellm` before `traigent`, the pin cannot retroactively apply — see "Residual risk — import order" below |
| `raw.githubusercontent.com` (Anthropic beta-header config) | 443 | No | Not observed in any workload on litellm 1.98.0 [R]; litellm has the fetch (`litellm/__init__.py:416-419`, `anthropic_beta_headers_manager.py`); pinned the same way and by the same default as the price table above | `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True` (also on by default) |
| `huggingface.co` | 443 | No | **Not contacted by default**: pricing/counting tokens for a Llama-family model (`traigent/utils/cost_calculator.py:397` → `litellm.token_counter` → `huggingface_hub`) no longer downloads a tokenizer just to count tokens (fixed) [R] | `HF_HUB_OFFLINE=1` (redundant with the fix, kept as defense in depth) |
| LLM provider endpoints (OpenAI, Anthropic, …) | 443 | Yes, for real runs | Only when your own code calls a model; mock mode sends nothing [R] | n/a (your provider choice) |
| Optional integrations: Langfuse (`cloud.langfuse.com`, `LANGFUSE_HOST`), a hybrid-API agent service you configure | 443 | Only if you enable them | `traigent/integrations/langfuse/client.py:230`; hybrid heartbeats every 30 s to *your* service when it advertises keep-alive (`traigent/evaluators/hybrid_api.py:437-441`) [S] | Do not configure them |

**Recommended environment for locked-down hosts** (all four together gave zero
third-party contact in every workload [R]):

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
export LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# and, for no Traigent cloud traffic at all:
export TRAIGENT_OFFLINE_MODE=true
```

**Residual risk — import order.** If the application imports `litellm` *before*
`traigent`, litellm fetches its price table at its own import, before the SDK
can set any default [R: workload `import_litellm_first`]. Only the environment
variable set before the process starts prevents that; the SDK's own default
pin cannot, because it only runs when `traigent/__init__.py` first executes,
which is already too late. The egress test keeps this as a strict xfail
(`test_litellm_imported_first_reaches_no_third_party_host`).

**What happens when the network is blocked.** In `docker run --network none`
every workload completed [R, evidence §5]. A refused DNS lookup costs litellm
well under a second (default vs pinned `optimize_mock`: 11.9 s vs 11.3/11.7 s),
so the 5 s litellm timeout was not observed; it would apply only where a
firewall silently *drops* packets — unmeasured. The Hugging Face path is
different: 8 attempts and about 22 s added (32.5 s vs 10.3 s with
`HF_HUB_OFFLINE=1`). With the host refused on a normal network it added about
24 s (29.0 s vs 4.8 s) [R].

## Network behaviour observed per workload [R]

| Workload | Default settings | With the four pins |
|---|---|---|
| Keyless quickstart (`traigent quickstart`) | no external host | no external host |
| `@traigent.optimize`, mock LLM, dummy key, loopback stub backend | no external host (default pin, `--pins` not needed) [R]; `raw.githubusercontent.com:443` only if `TRAIGENT_LITELLM_LIVE_PRICES=1` | stub only |
| Same, `injection_mode="seamless"` | same as above | stub only |
| `import litellm` before `import traigent` | `raw.githubusercontent.com:443` (residual — see above; pins/opt-out cannot help here) | stub only |
| Llama token count (`calculate_prompt_cost`) | no external host (fixed: no longer downloads a tokenizer to count tokens) [R] | none |
| Mock run with built-in backend, production policy | `portal.traigent.ai` only (the `raw.githubusercontent.com` contact this row used to show is gone with the default pin) [R] | `portal.traigent.ai` only |

Also seen at import, not traffic: `urllib3` binds a socket to `::1` port 0 to
test IPv6 support (first seen via `traigent/integrations/observability/workflow_traces.py:77`
importing `requests`) [R]. No UDP sends and no listening sockets were recorded; idle-time traffic
was not measured.

## Child processes

**Known finding, fix in flight:** a pinned `optimize_mock` / `optimize_seamless`
/ `import_litellm_first` run spawns one `subprocess.Popen: uname` [R,
`test_pinned_run_spawns_no_process_and_loads_no_foreign_library`, currently
failing on this branch]. Root cause: `import mlflow`
(`traigent/integrations/observability/mlflow.py:13`), eagerly imported via
`traigent/integrations/__init__.py:77` — mlflow's own import shells out to
`uname` on Linux/macOS. No other workload launched a process [R]. Being fixed
on a separate branch, `fix/lazy-mlflow-import` (make the mlflow integration a
lazy import so it is not paid by every `import traigent`); this document
should drop this paragraph once that lands and the test above passes again.

The SDK source can also launch these [S]:

| Call site | Command (argv[0]) | Why | Trigger |
|---|---|---|---|
| `traigent/identity/agent_build.py:316` | `git -C <dir> rev-parse` / `status --porcelain` | Record the code revision of the optimized function for content identity | Connected run whose backend issued a content-identity grant (`traigent/core/orchestrator.py:3317`); no shell, fixed argv, timeout |
| `traigent/cli/onboard_commands.py:125` | `traigent mcp --help` | Check the MCP server entry point works | `traigent onboard` CLI only |
| `traigent/cli/onboard_commands.py:144` | `traigent mcp register --agent <name>` | Register the MCP server with a coding agent | `traigent onboard` CLI, after the user confirms |
| `traigent/examples/providers/_bootstrap.py:98` | `python -m pip install <pkg>` | Install a missing example dependency | Packaged provider examples only, after an interactive `[y/N]` prompt |

Audit hooks do not see inside child processes; each launch above would need its
own audit.

## Files written [R]

| Location | What | Code |
|---|---|---|
| `./.traigent/optimization_logs/…` (cwd) | Run logs: `.json`, `.jsonl`, `.txt`, a `.gitignore`; atomic `*.json.tmp` + rename | `traigent/utils/optimization_logger.py:507,642,665-667,681,870` |
| `~/.traigent/` | Usage records and local session storage, atomic write + rename | `traigent/cloud/billing.py:212`, `traigent/storage/local_storage.py:256,944`, `traigent/utils/secure_path.py:212` |
| System temp dir | A lock/staging dir created through `huggingface_hub`'s `filelock` when counting Llama tokens, also with `HF_HUB_OFFLINE=1` | via `traigent/utils/cost_calculator.py:397` |
| `site-packages/…/__pycache__` | Ordinary Python bytecode cache on first import | the import system |

Relocate with `TRAIGENT_RESULTS_FOLDER` (`traigent/cloud/billing.py:205`,
`traigent/storage/local_storage.py:246`). Other temp-file writers [S]:
`traigent/security/crypto_utils.py:334` (secure atomic JSON write, mode 0600),
`traigent/utils/diagnostics.py:640` (write probe in `traigent doctor`),
`traigent/utils/batch_processing.py:374` (0700 checkpoint dir),
`traigent/utils/optimization_logger.py:487` (log fallback when cwd is unusable).
No file was made executable and no `.py` file was written in any workload [R].

## Code generation, explained plainly

`injection_mode="seamless"` (opt-in; the default is `"context"`) lets the SDK
change a hard-coded value such as `model = "gpt-4o-mini"` inside your function
for each trial. It does this entirely in memory:

1. `inspect.getsource(func)` reads your function's source (`traigent/config/providers.py:804`);
2. `ast.parse` turns it into a syntax tree (`providers.py:821`);
3. `ConfigTransformer` swaps the configured literals and `SafeASTCompiler.validate_ast`
   rejects imports, `global`, and calls such as `exec`, `eval`, `open`
   (`traigent/config/ast_transformer.py:327-349`);
4. `compile(tree, filename, "exec")` (`ast_transformer.py:517`) produces a code object,
   and `types.FunctionType` wraps the inner function with your original globals.

Nothing is written to disk and nothing is run from a file: the recorder saw
exactly this compile (3 per seamless run) and no `.py` write [R]. It is the
same class of activity as pytest's assertion rewriting. The SDK itself never
calls `exec()` [S: `grep` of `traigent/`]. Other in-memory compiles:
`traigent/tvl/spec_loader.py:1219` compiles AST-validated TVL constraint
expressions (`mode="eval"`) [S]. The ~1,000 further compile/exec events per run
come from the standard library generating `__init__`/`__eq__` for dataclasses and
typing helpers when classes are defined [R].

A customer run failed with an exception mentioning `_ast.py` and no traceback.
Seamless mode's parse step (steps 1–2) is the SDK path that reaches it; when the
source is unavailable (frozen, obfuscated, or zipped apps) it fails with a
`ConfigurationError`. This is a hypothesis, not confirmed without the traceback.

## Loading code and native libraries

- `ctypes`: the only native-library event in any run was `import ctypes` opening
  the running process (`ctypes.dlopen(None)`), triggered by litellm [R]. The SDK
  uses `ctypes.memset` to zero its own credential buffer
  (`traigent/security/credentials.py:250-255`) [S]; no other library is loaded.
- User modules by path: `traigent/cli/main.py:259-266` and
  `traigent/cli/function_discovery.py:75-80` (the `traigent` CLI imports the script
  you name), `traigent/mcp/tools.py:559-571` (MCP tool, same), and
  `traigent/integrations/plugin_registry.py:457-463` (plugins from a trusted
  directory). Each registers in `sys.modules`; none writes the file it runs [S].
- `importlib.reload` is never called; `traigent/api/decorators.py:533` only
  handles objects created by a user's reload [S].

## Findings that could matter to AV/EDR

1. Third-party fetches on import (`raw.githubusercontent.com` for litellm's price
   table and Anthropic beta-header config; `huggingface.co` for Llama tokenizer
   downloads): both now pinned off by default, fixed. Two residuals remain:
   (a) the application importing `litellm` before `traigent` (import order —
   the SDK's own pin cannot help, see "Residual risk — import order"), and
   (b) the documented opt-out `TRAIGENT_LITELLM_LIVE_PRICES=1`, which
   deliberately restores the `raw.githubusercontent.com` fetch on request.
2. A pinned run spawns `subprocess.Popen: uname` via mlflow's own import
   (`traigent/integrations/observability/mlflow.py:13`, eagerly imported
   through `traigent/integrations/__init__.py:77`). Fix in flight on
   `fix/lazy-mlflow-import` (lazy-import the mlflow integration).
3. Seamless mode's in-memory AST rewrite + `compile` of user source: benign, but
   behaviour monitors that flag runtime code generation may notice it. The
   `context` injection mode avoids it.
4. `git` subprocess during connected runs with content identity; `pip install`
   subprocess in examples (prompted).
5. `ctypes.memset` on an in-process buffer (credential wipe).
6. Hybrid-API heartbeats every 30 s: periodic, but only to a service you configure.

None observed: writing then executing a file, spawning a shell, `chmod +x`,
loading a non-system native library, or periodic traffic during the recorded
workloads (runs of 5–30 s; longer idle periods were not measured).
