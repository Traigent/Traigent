# Native crashes: what survives and how to recover

A native crash (for example a segfault in a C or Rust extension that your
optimized function or evaluator calls) terminates the whole Python process.
Python `try`/`except`, `finally` blocks and `atexit` handlers do not run, so
Traigent cannot catch it, finalize the session, or print a summary. Crash
isolation (running trials in a separate process so one crash does not end the
run) is not implemented yet; this page describes what you can rely on today.

## What survives

- **Trials that had already completed.** After each trial completes, the SDK
  attempts to record it locally before any remote submission
  (`BackendSessionManager._persist_trial_locally` in
  `traigent/core/backend_session_manager.py`). Offline runs write to the local
  storage directory (`TRAIGENT_RESULTS_FOLDER` if set, otherwise
  `~/.traigent`); connected runs record through the backend client's local
  fallback storage.
- **Nothing from the trial that was running when the process died.** Only
  completed trials are persisted, so the crashed trial and any trials that were
  in flight in parallel are lost.

## What is not guaranteed

Local persistence is best-effort, not a durability guarantee:

- If the session is missing from local storage, or the write raises (full
  disk, permissions, storage unavailable), the SDK logs a warning and skips the
  local record for that trial. A connected run in that state depends on the
  remote submission, which a crash can also interrupt.
- A crash can interrupt a write in progress. Treat the last recorded trial as
  possibly incomplete and verify it.
- The session will not be finalized, so its status and summary fields reflect
  whatever was last written, not a finished run.
- Costs of the lost trial are not recorded. Check your provider dashboard if
  spend matters.

## Inspect what was saved

```bash
# Use the same storage location as the crashed run
export TRAIGENT_RESULTS_FOLDER="<that path>"   # skip to use ~/.traigent

traigent results list
traigent results show <result_name> --trials
```

`traigent results list` also prints the directory it read local sessions from,
which confirms you are looking at the right place. Verify the trial count
against what you expected before trusting any "best" value: a run that never
finished has not compared every configuration.

`traigent results show <result_name> --trials` is a summary. It truncates each
configuration at 40 characters and shows at most 50 trials. For complete
configurations, open the saved trials file in the same storage directory:
`<storage directory>/<result_name>/trials.json.gz`. It is gzip-compressed JSON
with one object per trial, each holding its full `config` (read it with
`gunzip -c`). The `metadata.json` file beside it holds the best configuration
and summary fields. The storage directory is `TRAIGENT_RESULTS_FOLDER` when set,
otherwise `~/.traigent`.

## Recover and re-run

There is no resume command: `resume_from` does not exist in the SDK yet, and a
re-run starts a new optimization session. To avoid repeating work:

1. Read the configurations already evaluated from `traigent results show
   <result_name> --trials`.
2. Remove those configurations from `configuration_space` (or narrow it to the
   untried region) and run again.
3. Find the crashing configuration (it is usually the one tried next, or one
   of the in-flight ones) and exclude it, or run it on its own to reproduce
   the crash. Fix or pin the native dependency before re-including it.

To contain the blast radius meanwhile, run the risky native call in its own
subprocess inside your function and return an error value when the child
exits abnormally, so a crash fails one trial instead of the whole process.
