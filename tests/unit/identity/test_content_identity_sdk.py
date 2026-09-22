"""SDK behaviour of content identity v1: keys, examples, roots, agent builds, observations.

Success-certificate properties exercised here (goal lines 3a/3b/3c and 6):
reorder -> same dataset_root; duplicate row -> different root; edited expected
output -> same example_id, new example_version; edited input -> new id; same
example in two datasets -> same id; other tenant -> different id; helper /
prompt / config change -> new build_digest; dirty tree without a source digest
-> no build version.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.identity import content_identity as ci
from traigent.identity import agent_build
from traigent.identity.agent_build import (
    afp2_source_digest,
    candidate_agent_version,
    collect_agent_build_base,
    declare_agent_assets,
)
from traigent.identity.examples import (
    build_evaluated_set,
    identify_dataset,
    result_identity_fields,
    stamped_identity,
)
from traigent.identity.keys import (
    ContentIdentityKeys,
    clear_content_identity_keys,
    get_content_identity_keys,
    set_content_identity_keys,
)
from traigent.identity.trial import build_trial_content_identity
from traigent.utils.langchain_interceptor import (
    capture_langchain_response,
    capture_observed_response,
    capture_scope,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "content_identity_v1_vectors.json").read_text(
        encoding="utf-8"
    )
)


def _grant(tenant: str) -> ContentIdentityKeys:
    row = next(k for k in VECTORS["key_derivation"] if k["tenant"] == tenant)
    return ContentIdentityKeys.from_grant(
        {
            "tenant_id": row["tenant_id"],
            "kid": row["key_id"],
            "example_id_key": row["example_id_key_hex"],
            "example_version_key": row["example_version_key_hex"],
        }
    )


@pytest.fixture
def tenant_a() -> Any:
    keys = _grant("tenant_a")
    set_content_identity_keys(keys)
    yield keys
    clear_content_identity_keys()


@pytest.fixture(autouse=True)
def _no_keys_leak() -> Any:
    clear_content_identity_keys()
    yield
    clear_content_identity_keys()


def _dataset(rows: list[tuple[Any, Any]], **metadata: Any) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data=i, expected_output=e, metadata=dict(metadata))
            for i, e in rows
        ],
        name="d",
    )


ROWS = [({"q": "a"}, "1"), ({"q": "b"}, "2"), ({"q": "c"}, "3")]


# ---------------------------------------------------------------------------
# Keys: fail closed, never leak
# ---------------------------------------------------------------------------


def test_no_grant_means_no_ids() -> None:
    dataset = _dataset(ROWS)
    assert get_content_identity_keys() is None
    assert identify_dataset(dataset) is None
    assert all(stamped_identity(example) is None for example in dataset.examples)
    fields = result_identity_fields(dataset.examples[0], "example_0")
    assert fields == {"example_id": "example_0", "external_id": None}


def test_key_material_never_in_repr() -> None:
    keys = _grant("tenant_a")
    text = repr(keys) + str(keys) + repr(keys.as_tenant_keys())
    assert keys.example_id_key.hex() not in text
    assert keys.example_version_key.hex() not in text
    assert keys.kid in text


@pytest.mark.parametrize(
    "grant",
    [
        {},
        {
            "tenant_id": "t\n",
            "kid": "k" + "0" * 16,
            "example_id_key": "a" * 64,
            "example_version_key": "b" * 64,
        },
        {
            "tenant_id": "t",
            "kid": "K" + "0" * 16,
            "example_id_key": "a" * 64,
            "example_version_key": "b" * 64,
        },
        {
            "tenant_id": "t",
            "kid": "k" + "0" * 16,
            "example_id_key": "A" * 64,
            "example_version_key": "b" * 64,
        },
        {
            "tenant_id": "t",
            "kid": "k" + "0" * 16,
            "example_id_key": "a" * 64,
            "example_version_key": "a" * 64,
        },
    ],
)
def test_malformed_grants_are_rejected(grant: dict[str, Any]) -> None:
    with pytest.raises(ci.ContentIdentityError):
        ContentIdentityKeys.from_grant(grant)


# ---------------------------------------------------------------------------
# Examples and dataset roots (goal lines 3a-3c)
# ---------------------------------------------------------------------------


def test_reordering_keeps_root_and_duplicate_changes_it(tenant_a: Any) -> None:
    base = identify_dataset(_dataset(ROWS))
    reordered = identify_dataset(_dataset(list(reversed(ROWS))))
    duplicated = identify_dataset(_dataset([*ROWS, ROWS[0]]))
    assert base is not None and reordered is not None and duplicated is not None
    assert base.dataset_root == reordered.dataset_root
    assert duplicated.dataset_root != base.dataset_root
    assert duplicated.multiset.total_count == 4
    assert duplicated.multiset.distinct_count == 3


def test_expected_edit_keeps_id_input_edit_changes_it(tenant_a: Any) -> None:
    base = identify_dataset(_dataset(ROWS))
    relabelled = identify_dataset(_dataset([(ROWS[0][0], "changed"), *ROWS[1:]]))
    retyped = identify_dataset(_dataset([({"q": "a!"}, "1"), *ROWS[1:]]))
    assert base and relabelled and retyped
    assert relabelled.examples[0].example_id == base.examples[0].example_id
    assert relabelled.examples[0].example_version != base.examples[0].example_version
    assert retyped.examples[0].example_id != base.examples[0].example_id


def test_same_example_two_datasets_same_id_other_tenant_different(
    tenant_a: Any,
) -> None:
    one = identify_dataset(_dataset(ROWS[:2]))
    two = identify_dataset(_dataset(ROWS[1:]))
    assert one and two
    assert one.examples[1].example_id == two.examples[0].example_id
    other = identify_dataset(_dataset(ROWS[:2]), _grant("tenant_b"))
    assert other is not None
    assert other.examples[1].example_id != one.examples[1].example_id


def test_annotations_move_to_external_id_and_never_change_identity(
    tenant_a: Any,
) -> None:
    plain = identify_dataset(_dataset(ROWS[:1]))
    tagged = identify_dataset(
        _dataset(ROWS[:1], example_id="user-row-7", tags=["x"], split="test")
    )
    assert plain and tagged
    assert tagged.examples[0].example_id == plain.examples[0].example_id
    assert tagged.examples[0].example_version == plain.examples[0].example_version
    assert tagged.examples[0].external_id == "user-row-7"


def test_unidentifiable_example_disables_identity_for_the_whole_dataset(
    tenant_a: Any,
) -> None:
    dataset = _dataset([*ROWS, ({"n": 2**53}, "x")])
    assert identify_dataset(dataset) is None
    assert all(stamped_identity(example) is None for example in dataset.examples)


def test_result_fields_use_content_identity_and_keep_user_id(tenant_a: Any) -> None:
    dataset = _dataset(ROWS[:1], example_id="user-7")
    identity = identify_dataset(dataset)
    assert identity is not None
    fields = result_identity_fields(dataset.examples[0], "user-7")
    assert fields["example_id"] == identity.examples[0].example_id
    assert fields["example_id"].startswith("ex1:")
    assert fields["example_version"].startswith("exv1:")
    assert fields["external_id"] == "user-7"


def test_stale_stamp_is_ignored_after_the_grant_changes(tenant_a: Any) -> None:
    dataset = _dataset(ROWS)
    assert identify_dataset(dataset) is not None
    set_content_identity_keys(_grant("tenant_b"))
    assert stamped_identity(dataset.examples[0]) is None
    clear_content_identity_keys()
    assert stamped_identity(dataset.examples[0]) is None


def _result(fields: dict[str, Any]) -> ExampleResult:
    return ExampleResult(
        input_data={},
        expected_output=None,
        actual_output=None,
        metrics={},
        execution_time=0.0,
        success=True,
        **fields,
    )


def test_evaluated_set_full_subset_and_recomputable(tenant_a: Any) -> None:
    dataset = _dataset(ROWS)
    identity = identify_dataset(dataset)
    assert identity is not None
    full = build_evaluated_set(
        [_result(result_identity_fields(e, "x")) for e in dataset.examples], identity
    )
    assert full is not None
    assert full["evaluated_root"] == identity.dataset_root
    subset = build_evaluated_set(
        [_result(result_identity_fields(dataset.examples[0], "x"))], identity
    )
    assert subset is not None
    assert subset["evaluated_root"] != identity.dataset_root
    assert subset["dataset_root"] == identity.dataset_root
    # A consumer recomputes the stated root from the member list (spec section 5).
    recomputed = ci.compute_multiset_root(
        [(m["example_id"], m["example_version"], m["count"]) for m in subset["members"]]
    )
    assert recomputed.root == subset["evaluated_root"]
    evaluated = ci.compute_multiset_root(
        [(m["example_id"], m["example_version"], m["count"]) for m in subset["members"]]
    )
    assert ci.is_sub_multiset(evaluated, identity.multiset)


def test_evaluated_set_withheld_when_any_result_lacks_identity(tenant_a: Any) -> None:
    dataset = _dataset(ROWS)
    identity = identify_dataset(dataset)
    results = [_result(result_identity_fields(e, "x")) for e in dataset.examples]
    results.append(_result({"example_id": "example_3"}))
    assert build_evaluated_set(results, identity) is None


def test_example_result_serialization_is_unchanged_without_identity() -> None:
    result = _result({"example_id": "example_0"})
    payload = result.to_dict()
    assert "example_version" not in payload and "external_id" not in payload
    keyed = _result(
        {"example_id": "ex1:x", "example_version": "exv1:y", "external_id": "u"}
    )
    round_trip = ExampleResult.from_dict(keyed.to_dict())
    assert (round_trip.example_version, round_trip.external_id) == ("exv1:y", "u")


# ---------------------------------------------------------------------------
# Agent build manifest (goal line 6)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(repo),
        },
    )


def _agent_project(tmp_path: Path, *, helper: str = "RULE = 1\n") -> Path:
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "helper_mod_ci.py").write_text(helper, encoding="utf-8")
    (repo / "agent_mod_ci.py").write_text(
        "import helper_mod_ci\n\n\ndef agent(q):\n    return helper_mod_ci.RULE\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _load_agent(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(repo))
    for name in ("agent_mod_ci", "helper_mod_ci"):
        sys.modules.pop(name, None)
    module = importlib.import_module("agent_mod_ci")
    monkeypatch.setattr(agent_build, "_SDK_ROOT", Path("/nonexistent-sdk-root"))
    return module.agent


def test_clean_commit_with_declared_assets_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _load_agent(_agent_project(tmp_path), monkeypatch)
    declare_agent_assets(agent, prompts={"system": "Be terse."}, tool_definitions={})
    base = collect_agent_build_base(agent, agent_id="agent_1")
    assert base is not None
    assert base.coverage == "complete", base.gaps
    assert base.code_revision is not None and base.code_revision["dirty"] is False
    assert set(base.asset_digests["helper_modules"]) == {
        "agent_mod_ci.py",
        "helper_mod_ci.py",
    }
    version = candidate_agent_version(base, {"alpha": 1})
    assert version is not None
    manifest = version["manifest"]
    assert manifest["runtime"]["language"] == "python"
    assert manifest["runtime"]["sdk_version"]
    assert "Be terse." not in json.dumps(manifest)  # digests only, no content
    assert (
        ci.compute_agent_build_digest(manifest, certifiable=True)
        == (version["build_digest"])
    )
    other = candidate_agent_version(base, {"alpha": 2})
    assert other is not None and other["build_digest"] != version["build_digest"]


def test_undeclared_prompts_and_tools_make_coverage_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _load_agent(_agent_project(tmp_path), monkeypatch)
    base = collect_agent_build_base(agent, agent_id="agent_1")
    assert base is not None
    assert base.coverage == "partial"
    assert {"prompts_not_declared", "tool_definitions_not_declared"} <= set(base.gaps)
    version = candidate_agent_version(base, {})
    assert version is not None
    with pytest.raises(ci.ContentIdentityError):
        ci.compute_agent_build_digest(version["manifest"], certifiable=True)


def test_helper_or_prompt_change_changes_build_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _agent_project(tmp_path)
    agent = _load_agent(repo, monkeypatch)
    declare_agent_assets(agent, prompts={"system": "v1"}, tool_definitions={})
    first = candidate_agent_version(collect_agent_build_base(agent, agent_id="a"), {})
    declare_agent_assets(agent, prompts={"system": "v2"})
    second = candidate_agent_version(collect_agent_build_base(agent, agent_id="a"), {})
    (repo / "helper_mod_ci.py").write_text("RULE = 2\n", encoding="utf-8")
    third = candidate_agent_version(collect_agent_build_base(agent, agent_id="a"), {})
    assert first and second and third
    assert (
        len({first["build_digest"], second["build_digest"], third["build_digest"]}) == 3
    )
    # The helper edit dirtied the tree; the enumerated helper covers it.
    assert third["manifest"]["code_revision"]["dirty"] is True
    assert "source_digest" in third["manifest"]


def test_dirty_tree_without_source_digest_yields_no_build_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _agent_project(tmp_path)
    agent = _load_agent(repo, monkeypatch)
    (repo / "notes.txt").write_text("untracked", encoding="utf-8")
    monkeypatch.setattr(agent_build, "afp2_source_digest", lambda func: None)
    assert collect_agent_build_base(agent, agent_id="agent_1") is None


def test_dirty_file_outside_the_manifest_makes_coverage_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _agent_project(tmp_path)
    agent = _load_agent(repo, monkeypatch)
    declare_agent_assets(agent, prompts={}, tool_definitions={})
    (repo / "prompt.txt").write_text("unlisted prompt file", encoding="utf-8")
    base = collect_agent_build_base(agent, agent_id="agent_1")
    assert base is not None
    assert base.coverage == "partial"
    assert "dirty_files_outside_manifest" in base.gaps


def test_no_representable_agent_id_yields_no_build_version() -> None:
    def agent(q: str) -> str:
        return q

    assert collect_agent_build_base(agent, agent_id=None) is None
    assert collect_agent_build_base(agent, agent_id="has spaces") is None


def test_afp2_is_unknown_for_unserializable_bound_state() -> None:
    client = object()

    def with_client(q: str) -> str:
        return str(client) + q

    def plain(q: str) -> str:
        return q

    assert afp2_source_digest(with_client) is None
    digest = afp2_source_digest(plain)
    assert digest is not None and digest.startswith("sha256:")


def _plain_agent() -> Any:
    def agent(q: str) -> str:
        return q

    return agent


def _decorated_agent() -> Any:
    @declare_agent_assets(prompts={})
    def agent(q: str) -> str:
        return q

    return agent


def test_decorator_lines_do_not_change_afp2() -> None:
    plain = afp2_source_digest(_plain_agent())
    assert plain is not None
    assert afp2_source_digest(_decorated_agent()) == plain


def test_git_status_ignores_bytecode_caches() -> None:
    assert agent_build._is_bytecode_cache("__pycache__/")
    assert agent_build._is_bytecode_cache("pkg/__pycache__/m.cpython-313.pyc")
    assert agent_build._is_bytecode_cache("legacy.pyc")
    assert not agent_build._is_bytecode_cache("pkg/pycache_notes.py")


# ---------------------------------------------------------------------------
# Observed provider versions
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def test_observations_are_recorded_from_real_responses_only() -> None:
    with capture_scope() as bucket:
        capture_observed_response(
            _Response(model="gpt-4o-2024-08-06", system_fingerprint="fp_1"),
            provider="openai",
            requested_model="gpt-4o",
        )
        capture_observed_response(
            _Response(model="gpt-4o-2024-08-06", system_fingerprint="fp_1"),
            provider="openai",
            requested_model="gpt-4o",
        )
        capture_observed_response(
            _Response(), provider="anthropic", requested_model=None
        )
        capture_langchain_response(_Response(model="mock-model"))  # mock path
        observed = bucket.observed_provider_versions()
    assert observed == [
        {
            "provider": "anthropic",
            "requested_model": "unknown",
            "response_model": None,
            "call_count": 1,
        },
        {
            "provider": "openai",
            "requested_model": "gpt-4o",
            "response_model": "gpt-4o-2024-08-06",
            "call_count": 2,
            "system_fingerprint": "fp_1",
        },
    ]


def test_langchain_style_metadata_is_read() -> None:
    with capture_scope() as bucket:
        capture_observed_response(
            _Response(response_metadata={"model_name": "claude-x-20260101"}),
            provider="anthropic",
            requested_model="claude-x",
        )
        observed = bucket.observed_provider_versions()
    assert observed[0]["response_model"] == "claude-x-20260101"


# ---------------------------------------------------------------------------
# Trial block
# ---------------------------------------------------------------------------


def test_trial_block_omits_what_cannot_be_stated() -> None:
    assert (
        build_trial_content_identity(
            dataset_identity=None,
            example_results=None,
            candidate=None,
            observed_provider_versions=[],
        )
        is None
    )
    block = build_trial_content_identity(
        dataset_identity=None,
        example_results=None,
        candidate={"agent_id": "a", "build_digest": "sha256:" + "0" * 64},
        observed_provider_versions=None,
    )
    assert block == {
        "scheme": ci.SCHEME,
        "candidate": {"agent_id": "a", "build_digest": "sha256:" + "0" * 64},
        "observed_provider_versions": [],
    }
