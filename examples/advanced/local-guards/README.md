# Block destructive tool calls with local guards

A tool call to `infra.delete_volume` is refused unless an approval was recorded
first, and one approval covers one delete. The check runs in your process, before
the tool body runs. No API keys, no network.

Files:

- `delete_needs_approval.runtime.yml`: the TVL runtime module (requirement, monitor, response)
- `token-map.json`: maps the module's names to opaque tokens for the compiler
- `run.py`: compiles the module, loads the guards, runs three scenarios

Run it:

    pip install "traigent[runtime-tvl]"
    python examples/advanced/local-guards/run.py

Expected output: scenario 1 (no approval) is refused and no volume is deleted;
scenario 2 (approve, then delete) deletes `vol-1`; scenario 3 (one approval, two
deletes) deletes `vol-1` and refuses `vol-2`.

The full walkthrough is in `docs/guides/local-guards.md`.
