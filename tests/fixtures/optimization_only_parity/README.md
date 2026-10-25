# optimization_only parity fixtures

Pins the SDK port in `traigent/utils/optimization_only_egress.py` to the backend projector.

- Backend: TraigentBackend `src/services/optimization_only_metadata_contract.py`, contract
  `METADATA_CONTRACT_VERSION = 1`, commit **2d5cceda5**.
- `session_create_typed.json`, `trial_results.json`, `finalize.json`, `content_identity_*.json`
  are copied unchanged from backend `tests/fixtures/sdk_payloads/` at that commit (provenance in
  `BACKEND_SOURCE_README.md`; produced from SDK commit a861521256be, no customer text, no secrets).
- `cases.json`: 41 inputs (the five captured fixtures plus hand-written legacy, results,
  finalize, configuration-run, parent-run, constraints and tuned-variable cases, including
  injected free text, bad identities, bool/NaN/inf, oversize). Authored once with a throwaway
  script; edit it by hand if a case is added.
- `expected.json`: what the BACKEND projector returned for each case (`{"ok": ...}` or
  `{"violation": true}`). Generated, not hand-written, by `generate_expected.py`:

      git -C <TraigentBackend> show 2d5cceda5:src/services/optimization_only_metadata_contract.py \
          > /tmp/be/optimization_only_metadata_contract.py
      python -I tests/fixtures/optimization_only_parity/generate_expected.py \
          /tmp/be/optimization_only_metadata_contract.py

  The script stubs the backend's tenant guard (the only non-stdlib import) and calls the
  backend's own `project_*` functions; nothing is computed by the SDK port.
- `tests/unit/utils/test_optimization_only_egress.py::test_parity_with_backend_projection`
  asserts SDK output == `expected.json` for every case.

Boundaries with no backend counterpart (next-trial, optimizer metrics, intermediate report,
hybrid session create, best-config publish) are SDK-side extensions covered by SDK tests only.
To move the pin: re-dump the backend module at the new SHA, rerun the generator, review the diff.
