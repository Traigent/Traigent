# SDK payload fixtures

Real request bodies produced by calling the Python SDK's own builders (no hand-written shapes):
`ApiOperations._build_typed_session_payload`, `compute_portal_name`, `filter_agent_configuration`,
`promotion_policy_to_wire`/`tvl_governance_to_wire` (through the request), `build_backend_metadata`,
`TrialOperations._build_trial_result_data`, `prepare_content_identity_run().session_wire/trial_wire`
(with fixed, non-secret test keys) and the finalize body of `session_operations.finalize_session`.

Produced by Traigent SDK commit `a861521256be731f61b921a598c89c7bd7bb74a0` (develop, SDK 0.31.0.dev0)
with `generate_fixtures.py.txt` (SDK venv, `python -I`, `TRAIGENT_OFFLINE_MODE=true`, no network).
The decorated function is a stub (`qa_agent_mod.py.txt`); there is no customer text and no secrets.

Note the real `function_name`: `opt:accuracy,cost · model,temperature (ab12cd34)` (not an identifier).
