"""E2E-1: the Python SDK against a REAL backend, through the whole production requirement loop.

declare -> apply -> traces -> window evaluation -> breach opens -> watcher alerts -> human-approved
fix runs -> healthy windows -> breach resolves -> watcher reports the resolution.

Nothing sits between the SDK and the backend: no fake transport, no stubbed route. The only thing
the harness does that the product does not is choose WHICH windows are judged, by handing the
backend's own scheduled-task function a controlled clock (``TRAIGENT_E2E_EVALUATE_CMD``); every
requirement, trace, evaluation and breach row is written by the backend's own routes and task.

The suite SKIPS (with the reason) unless the backend harness is exported, so it never runs in the
default unit run or in CI::

    # in the TraigentBackend worktree
    scripts/e2e/production_loop_backend.sh up
    source "${TMPDIR:-/tmp}/traigent-e2e-prodloop/env.sh"
    # in the SDK worktree
    TRAIGENT_E2E_RECEIPT_DIR=/path/to/receipts \\
      pytest tests/e2e/test_runtime_requirements_live_backend.py -p no:xdist -o addopts="" -s

``tvl-spec`` (the TVL compiler, Python >= 3.12) is needed only for the TVL-fragment step; without it
that one step is reported ``skipped`` in the receipt, never passed.

Each run writes a JSON receipt (heads, step results, ids, timings) to ``TRAIGENT_E2E_RECEIPT_DIR``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import pytest

REQUIRED_ENV = (
    "TRAIGENT_E2E_BACKEND_URL",
    "TRAIGENT_E2E_API_KEY",
    "TRAIGENT_E2E_PROJECT_ID",
    "TRAIGENT_E2E_AGENT_ID",
    "TRAIGENT_E2E_AGENT_KEY",
    "TRAIGENT_E2E_OTHER_API_KEY",
    "TRAIGENT_E2E_EVALUATE_CMD",
)
_MISSING = [name for name in REQUIRED_ENV if not os.environ.get(name)]

pytestmark = [
    pytest.mark.e2e,
    # Connected on purpose: this suite opens real sockets to the harness backend. The marker only
    # keeps the autouse offline fixture from forcing offline mode; it is not a transport mock.
    pytest.mark.backend_online,
    pytest.mark.skipif(
        bool(_MISSING),
        reason="live-backend E2E: export the harness env first "
        "(scripts/e2e/production_loop_backend.sh up; missing: "
        + ", ".join(_MISSING)
        + ")",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_MS = 2000
THETA = 0.2
WINDOW_H = 1
GRACE_MIN = 30  # the sugar does not expose grace; the backend default applies
MIN_SAMPLES = 40
CLOSURE_WINDOWS = 2  # backend default
VIOLATING = 60
HEALTHY = 100
CEILING_USD = 1.0
#: Stored-profile digest both SDK suites must reproduce for this one declaration (parity check).
#: Computed over the backend's stored profile fields, not over any SDK-side object.
PARITY_PROFILE_DIGEST = (
    "8edcf83d28a69b42f02c9b0d12f27ba3d1c7656b2fb6ba18b74b62be1789c765"
)

TVL_FRAGMENT = """\
tvl_version: "1.3"
requirements:
  - { name: latency_3s, kind: rate_bound, unit: trace,
      violation: { metric: wall_clock_latency_ms, exceeds: 3000 }, max_rate: 0.05 }
monitors:
  - { name: latency_3s_prod, requirement: latency_3s, phase: runtime, profile: backend_v1,
      evidence: { source: backend_trace_metadata, deployment_environment: production },
      decision: { method: clopper_pearson_exact_v1 } }
responses:
  - name: latency_ladder
    rules:
      - { id: alert, trigger: { monitor: latency_3s_prod, event: episode_opened }, do: notify, severity: warning }
      - { id: ok, trigger: { monitor: latency_3s_prod, event: episode_resolved }, do: notify, severity: info }
"""


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Steps:
    """Runs named steps, records pass / fail / skipped with timings, never hides a failure."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.status: dict[str, str] = {}

    def run(
        self, name: str, fn: Callable[[], Any], *, requires: tuple[str, ...] = ()
    ) -> Any:
        blocked = [r for r in requires if self.status.get(r) != "passed"]
        if blocked:
            self.status[name] = "skipped"
            self.rows.append(
                {"step": name, "status": "skipped", "reason": f"requires {blocked}"}
            )
            return None
        started = time.monotonic()
        try:
            value = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-asserted at the end
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.status[name] = "failed"
            self.rows.append(
                {
                    "step": name,
                    "status": "failed",
                    "seconds": round(time.monotonic() - started, 2),
                    "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                }
            )
            return None
        self.status[name] = "passed"
        self.rows.append(
            {
                "step": name,
                "status": "passed",
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        return value if value is not None else True

    def skip(self, name: str, reason: str) -> None:
        self.status[name] = "skipped"
        self.rows.append({"step": name, "status": "skipped", "reason": reason})

    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "failed"]


def _ceil_hour(moment: datetime) -> datetime:
    floor = moment.replace(minute=0, second=0, microsecond=0)
    return floor if floor == moment else floor + timedelta(hours=1)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _traces(
    tag: str, window_start: datetime, *, total: int, slow: int, agent_key: str
) -> list[dict[str, Any]]:
    gap = timedelta(hours=WINDOW_H) / (total + 1)
    rows = []
    for index in range(total):
        started = window_start + gap * (index + 1)
        duration = timedelta(milliseconds=THRESHOLD_MS + 500 if index < slow else 200)
        rows.append(
            {
                "id": f"e2e-{tag}-{index:04d}",
                "name": "e2e",
                "status": "completed",
                "environment": "production",
                "started_at": started.isoformat(),
                "ended_at": (started + duration).isoformat(),
                "execution_context": {"agent_id": agent_key},
            }
        )
    return rows


def _profile_digest(head: dict[str, Any]) -> str:
    fields = (
        "kind",
        "latency_threshold_ms",
        "environment",
        "max_violation_rate",
        "confidence",
        "window_hours",
        "grace_minutes",
        "min_samples",
        "max_unresolved_fraction",
        "closure_windows",
        "closure_max_age_hours",
        "false_alarm_budget",
        "horizon_days",
    )
    blob = json.dumps(
        {f: head.get(f) for f in fields}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _tvl_compiler_available() -> bool:
    try:
        import tvl.runtime_compiler  # noqa: F401
    except ImportError:
        return False
    return True


def test_production_requirement_loop_against_real_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    import traigent
    from traigent.api.decorators import EvaluationOptions, ExecutionOptions
    from traigent.runtime import manifest
    from traigent.runtime.client import RuntimeClient
    from traigent.runtime.model import RuntimeEvent, RuntimeRequirementError
    from traigent.runtime.respond import OccurrenceUsed, respond
    from traigent.runtime.watch import poll_once

    url = os.environ["TRAIGENT_E2E_BACKEND_URL"].rstrip("/")
    api_key = os.environ["TRAIGENT_E2E_API_KEY"]
    project = os.environ["TRAIGENT_E2E_PROJECT_ID"]
    agent = os.environ["TRAIGENT_E2E_AGENT_ID"]
    agent_key = os.environ["TRAIGENT_E2E_AGENT_KEY"]
    other_key = os.environ["TRAIGENT_E2E_OTHER_API_KEY"]
    evaluate_cmd = shlex.split(os.environ["TRAIGENT_E2E_EVALUATE_CMD"])

    state = tmp_path / "state"
    monkeypatch.setenv("TRAIGENT_RUNTIME_STATE_DIR", str(state))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", url)
    monkeypatch.setenv("TRAIGENT_API_KEY", api_key)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"input": {"question": f"q{i}"}, "output": f"a{i}"})
            for i in (1, 2, 3)
        )
    )

    client = RuntimeClient(project, base_url=url, api_key=api_key)
    steps = Steps()
    facts: dict[str, Any] = {}
    events: list[RuntimeEvent] = []
    optimize_calls: list[float] = []

    # -- the declared function: a tiny local agent, $0 evaluator, no model call -----------------
    @traigent.requirement(
        latency_ms=THRESHOLD_MS,
        max_violation_rate=THETA,
        window=f"{WINDOW_H}h",
        min_samples=MIN_SAMPLES,
        on_breach="fix",
        approval="human",
        max_fix_cost_usd=CEILING_USD,
        environment="production",
        on_event=events.append,
    )
    @traigent.optimize(
        configuration_space={"mode": ["a", "b"]},
        objectives=["accuracy"],
        evaluation=EvaluationOptions(eval_dataset=str(dataset)),
        execution=ExecutionOptions(offline=True),
        max_trials=2,
    )
    def answer(question: str) -> Any:
        optimize_calls.append(1.0)
        cfg = traigent.get_config()
        text = f"a{question[1:]}" if cfg.get("mode") == "b" else "zzz"
        return traigent.with_usage(
            text, total_cost=0.0, input_tokens=0, output_tokens=0
        )

    req = answer.requirements[0]

    def ingest(rows: list[dict[str, Any]]) -> int:
        sent = 0
        for offset in range(0, len(rows), 100):
            response = requests.post(
                f"{url}/api/v1beta/observability/ingest",
                json={"traces": rows[offset : offset + 100]},
                headers={"X-API-Key": api_key},
                timeout=60,
            )
            assert response.status_code == 202, response.status_code
            sent += len(rows[offset : offset + 100])
        return sent

    def evaluate(at: datetime) -> dict[str, Any]:
        proc = subprocess.run(
            [*evaluate_cmd, at.isoformat()],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"evaluate exit {proc.returncode}: {proc.stderr[-300:]}"
        )
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
        assert summary["outcome"] == "success", summary
        return summary  # type: ignore[no-any-return]

    def status_item(requirement_head: str | None = None) -> dict[str, Any]:
        code, body = client.request("GET", f"/agents/{agent}/production-status")
        assert code == 200, code
        items = body["items"]
        key = requirement_head or facts["head_id"]
        mine = [i for i in items if i["agent_head_id"] == key]
        assert len(mine) == 1, [i["agent_head_id"] for i in items]
        return mine[0]  # type: ignore[no-any-return]

    def open_breaches() -> list[dict[str, Any]]:
        rows, more = client.list_breaches(agent)
        assert more is False
        return [
            r
            for r in rows
            if r["latency_threshold_ms"] == THRESHOLD_MS and r["outcome"] is None
        ]

    def poll() -> Any:
        return poll_once([req], project_id=project, agent_id=agent, client=client)

    # ---- 1. declare ---------------------------------------------------------------------------
    def declare() -> None:
        # Offline validation of the declaration itself (no compiler needed).
        assert [i for i in req.validate() if i.severity == "error"] == []
        kinds = [
            (r.event, r.action, r.severity) for r in req.rules if r.action == "notify"
        ]
        assert ("episode_opened", "notify", "warning") in kinds
        assert ("episode_resolved", "notify", "info") in kinds
        assert any(r.action == "optimize" and r.approval == "human" for r in req.rules)
        assert req.max_fix_cost_usd == CEILING_USD
        assert "episode_opened" in answer.requirements.to_tvl()

    steps.run("1a declare (decorator args)", declare)

    tvl_path = tmp_path / "latency_3s.tvl.yml"
    tvl_path.write_text(TVL_FRAGMENT)
    tvl_fn: Any = None

    def declare_tvl() -> None:
        nonlocal tvl_fn

        @traigent.requirement(
            tvl=str(tvl_path), max_fix_cost_usd=CEILING_USD, agent_id=agent
        )
        def tvl_agent(question: str) -> str:
            return question

        tvl_fn = tvl_agent
        assert [
            i for i in tvl_fn.requirements[0].validate() if i.severity == "error"
        ] == []
        assert tvl_fn.requirements[0].latency_threshold_ms == 3000

    if _tvl_compiler_available():
        steps.run("1b declare (TVL fragment)", declare_tvl)
    else:
        steps.skip(
            "1b declare (TVL fragment)", "tvl-spec (tvl.runtime_compiler) not installed"
        )

    def compiler_lint() -> None:
        # RequirementList.validate() adds the TVL compiler's lints when tvl-spec is installed.
        errors = [i for i in answer.requirements.validate() if i.severity == "error"]
        assert errors == [], [(i.code, i.path) for i in errors]

    if _tvl_compiler_available():
        steps.run("1c compiler lint of the full policy (tvl-spec)", compiler_lint)
    else:
        steps.skip(
            "1c compiler lint of the full policy (tvl-spec)", "tvl-spec not installed"
        )

    # ---- 2. apply -----------------------------------------------------------------------------
    def apply_kwargs() -> None:
        receipt = answer.requirements.apply(project, agent)
        assert (
            receipt is not None
            and receipt.created is True
            and receipt.provenance == "api"
        )
        facts.update(head_id=receipt.head_id, revision_id=receipt.revision_id)
        stored = manifest.read_manifest(state, agent)["requirements"][req.name]
        assert stored["receipt"]["head_id"] == receipt.head_id
        listed = [
            h for h in client.list_requirements() if h["head_id"] == receipt.head_id
        ]
        assert len(listed) == 1 and listed[0]["status"] == "active"
        assert listed[0]["latency_threshold_ms"] == THRESHOLD_MS
        facts["stored_profile_digest"] = _profile_digest(listed[0])
        assert facts["stored_profile_digest"] == PARITY_PROFILE_DIGEST

    steps.run(
        "2a apply (decorator args)",
        apply_kwargs,
        requires=("1a declare (decorator args)",),
    )

    def apply_again() -> None:
        again = answer.requirements.apply(project, agent)
        assert again is not None and again.created is False
        assert again.head_id == facts["head_id"]

    steps.run(
        "2b apply is idempotent", apply_again, requires=("2a apply (decorator args)",)
    )

    def apply_tvl() -> None:
        receipt = tvl_fn.requirements.apply(project, agent)
        assert receipt is not None and receipt.provenance == "tvl_spec"
        assert tvl_fn.requirements.apply(project, agent).created is False  # type: ignore[union-attr]
        facts["tvl_head_id"] = receipt.head_id

    steps.run(
        "2c apply (TVL fragment, distinct key)",
        apply_tvl,
        requires=("1b declare (TVL fragment)", "2a apply (decorator args)"),
    )

    # ---- 3. traces + evaluation -> breach ------------------------------------------------------
    def before_evaluation() -> None:
        item = status_item()
        assert item["headline_state"] == "NOT_YET_EVALUATED"
        assert item["last_evaluation"] is None
        facts["w1"] = _ceil_hour(_parse(item["effective_from"]) + timedelta(seconds=5))

    steps.run(
        "3a status before any evaluation",
        before_evaluation,
        requires=("2a apply (decorator args)",),
    )

    def tick(window_start: datetime) -> datetime:
        return window_start + timedelta(hours=WINDOW_H, minutes=GRACE_MIN + 1)

    def ingest_violating() -> int:
        return ingest(
            _traces(
                "w1", facts["w1"], total=VIOLATING, slow=VIOLATING, agent_key=agent_key
            )
        )

    steps.run(
        "3b ingest violating traces for W1 (real route)",
        ingest_violating,
        requires=("3a status before any evaluation",),
    )

    def evaluate_w1() -> None:
        evaluate(tick(facts["w1"]))
        item = status_item()
        assert item["headline_state"] == "BREACHED", item["headline_state"]
        last = item["last_evaluation"]
        assert last["outcome"] == "BREACHED"
        assert (last["counts"]["n"], last["counts"]["k"]) == (VIOLATING, VIOLATING)
        facts["evaluation_ids"] = [last["evaluation_id"]]
        assert item["breach"]["open_breach_count"] == 1
        facts["breach_id"] = item["breach"]["open_breach_id"]
        assert [b["breach_id"] for b in open_breaches()] == [facts["breach_id"]]

    steps.run(
        "3c evaluate W1 via the backend task -> breach open, status BREACHED",
        evaluate_w1,
        requires=("3b ingest violating traces for W1 (real route)",),
    )
    breach_open = "3c evaluate W1 via the backend task -> breach open, status BREACHED"

    # ---- 4. watcher alerts --------------------------------------------------------------------
    def first_poll() -> None:
        result = poll()
        assert result.error is None and result.coverage == "complete"
        assert [e.event for e in result.events] == ["episode_opened"]
        assert result.events[0].severity == "warning"
        assert result.events[0].breach_id == facts["breach_id"]
        assert [e.event for e in events] == ["episode_opened"]  # on_event hook ran once

    steps.run(
        "4a poll_once -> exactly one episode_opened",
        first_poll,
        requires=(breach_open,),
    )

    def repeat_poll() -> None:
        result = poll()
        assert result.events == [] and result.coverage == "complete"
        assert [e.event for e in events] == ["episode_opened"]

    steps.run(
        "4b repeat poll -> no duplicate",
        repeat_poll,
        requires=("4a poll_once -> exactly one episode_opened",),
    )

    def next_step_is_restore() -> None:
        payload = client.get_next_step(agent)
        assert payload["step"]["code"] == "RESTORE_PRODUCTION_REQUIREMENT"
        assert (
            payload["basis"]["production_breach"]["open_breach_ref"]
            == facts["breach_id"]
        )

    steps.run(
        "4c next step is RESTORE_PRODUCTION_REQUIREMENT",
        next_step_is_restore,
        requires=(breach_open,),
    )

    # ---- 5. respond: negatives first (nothing may start), then the real run -------------------
    def run_respond(**kwargs: Any) -> Any:
        base: dict[str, Any] = {
            "project_id": project,
            "agent_id": agent,
            "confirm": lambda facts_: True,
            "interactive": True,
            "client": client,
            "state_dir": state,
        }
        base.update(kwargs)
        return asyncio.run(respond(answer, req, **base))

    def expect_code(code: str, **kwargs: Any) -> None:
        before = len(optimize_calls)
        with pytest.raises(RuntimeRequirementError) as caught:
            run_respond(**kwargs)
        assert caught.value.code == code, caught.value.code
        assert len(optimize_calls) == before, "a held respond must not start the run"

    steps.run(
        "5a non-interactive session holds (approval_required)",
        lambda: expect_code("approval_required", interactive=False),
        requires=("4c next step is RESTORE_PRODUCTION_REQUIREMENT",),
    )
    steps.run(
        "5b a declined approval holds",
        lambda: expect_code("approval_required", confirm=lambda facts_: False),
        requires=("4c next step is RESTORE_PRODUCTION_REQUIREMENT",),
    )

    def expired_approval() -> None:
        skew = {"next": False}

        def confirm(_facts: dict[str, Any]) -> bool:
            skew["next"] = True
            return True

        def clock() -> datetime:
            now = datetime.now(UTC)
            if skew["next"]:
                skew["next"] = False  # the approval is stamped two days ago ...
                return now - timedelta(days=2)
            return now  # ... and found expired when it is about to be used

        expect_code("approval_expired", confirm=confirm, clock=clock)

    steps.run(
        "5c an expired approval holds",
        expired_approval,
        requires=("4c next step is RESTORE_PRODUCTION_REQUIREMENT",),
    )

    def real_respond() -> None:
        asked: list[dict[str, Any]] = []

        def confirm(asked_facts: dict[str, Any]) -> bool:
            asked.append(asked_facts)
            return True

        before = len(optimize_calls)
        result = run_respond(confirm=confirm)
        assert result.breach_id == facts["breach_id"]
        assert result.occurrence == 0 and result.cost_ceiling_usd == CEILING_USD
        assert asked and asked[0]["breach_id"] == facts["breach_id"]
        assert asked[0]["cost_ceiling_usd"] == CEILING_USD
        assert len(optimize_calls) > before, "the real optimize ran the local function"
        facts["run_id"] = result.run_id
        key = manifest.occurrence_key(facts["breach_id"], "fix", 0)
        record = manifest.read_occurrence(
            state,
            agent,
            key,
            now=datetime.now(UTC),
            lease=manifest.OCCURRENCE_LEASE,
        )
        assert record is not None and record.get("finished") is True

    steps.run(
        "5d approved respond runs a real optimize (occurrence 0)",
        real_respond,
        requires=("5c an expired approval holds",),
    )

    def second_respond() -> None:
        before = len(optimize_calls)
        with pytest.raises(OccurrenceUsed):
            run_respond()
        assert len(optimize_calls) == before

    steps.run(
        "5e a second respond -> OccurrenceUsed",
        second_respond,
        requires=("5d approved respond runs a real optimize (occurrence 0)",),
    )

    # ---- 6. healthy windows -> closure progress -> resolved -----------------------------------
    def healthy(index: int) -> Callable[[], None]:
        def go() -> None:
            window = facts["w1"] + timedelta(hours=WINDOW_H * index)
            ingest(
                _traces(
                    f"w{index + 1}", window, total=HEALTHY, slow=0, agent_key=agent_key
                )
            )
            evaluate(tick(window))
            item = status_item()
            last = item["last_evaluation"]
            assert last["outcome"] == "MET", last["outcome"]
            facts["evaluation_ids"].append(last["evaluation_id"])
            if index < CLOSURE_WINDOWS:
                assert item["closure"]["met_in_a_row"] == index
                assert item["closure"]["needed"] == CLOSURE_WINDOWS
                assert item["breach"]["open_breach_count"] == 1
                assert len(open_breaches()) == 1, (
                    "still open until all K windows are healthy"
                )
            else:
                assert item["breach"]["open_breach_count"] == 0

        return go

    steps.run(
        "6a W2 healthy -> closure 1 of 2, breach still open",
        healthy(1),
        requires=(breach_open,),
    )

    def poll_nothing() -> None:
        result = poll()
        assert result.events == [] and len(events) == 1

    steps.run(
        "6b poll while recovering -> no event",
        poll_nothing,
        requires=("6a W2 healthy -> closure 1 of 2, breach still open",),
    )
    steps.run(
        "6c W3 healthy -> closure 2 of 2",
        healthy(2),
        requires=("6a W2 healthy -> closure 1 of 2, breach still open",),
    )

    def resolved() -> None:
        rows, _ = client.list_breaches(agent)
        mine = [r for r in rows if r["breach_id"] == facts["breach_id"]]
        assert len(mine) == 1
        assert mine[0]["outcome"] == "resolved" and mine[0]["closed_at"] is not None
        assert len(mine[0]["closing_evaluation_ids"]) == CLOSURE_WINDOWS
        assert open_breaches() == []
        assert status_item()["headline_state"] == "MET"

    steps.run(
        "6d breach resolved, status MET",
        resolved,
        requires=("6c W3 healthy -> closure 2 of 2",),
    )

    # ---- 7. watcher reports the resolution ----------------------------------------------------
    def resolution_event() -> None:
        result = poll()
        assert [e.event for e in result.events] == ["episode_resolved"]
        assert (
            result.events[0].severity == "info"
            and result.events[0].outcome == "resolved"
        )
        assert [e.event for e in events] == ["episode_opened", "episode_resolved"]
        assert poll().events == [], "a repeat poll must not re-report"
        assert len(events) == 2

    steps.run(
        "7 poll_once -> exactly one episode_resolved, repeat -> none",
        resolution_event,
        requires=("6d breach resolved, status MET",),
    )

    # ---- 8. isolation: another tenant's key sees a closed 404 ---------------------------------
    def cross_project() -> None:
        intruder = RuntimeClient(project, base_url=url, api_key=other_key)
        seen: dict[str, int] = {}
        for method, suffix, body in (
            ("GET", "/production-requirements", None),
            ("GET", f"/agents/{agent}/production-status", None),
            ("GET", f"/agents/{agent}/production-breaches", None),
            ("POST", "/production-requirements", req.native_body(agent)),
        ):
            code, payload = intruder.request(method, suffix, body)
            seen[f"{method} {suffix}"] = code
            # Closed refusal, no data. The design expected 404; the backend answers 403
            # authorization_denied on some routes. Both are recorded in the receipt.
            assert code in (403, 404), (method, suffix, code)
            assert isinstance(payload, dict)
            assert set(payload) <= {"error_code", "error", "message", "success"}, (
                sorted(payload)
            )
            assert payload.get("error_code") in ("not_found", "authorization_denied"), (
                payload
            )
            assert facts["breach_id"] not in json.dumps(payload)
        facts["cross_project_statuses"] = seen

    steps.run(
        "8 a key from another project is refused (403/404) on every route",
        cross_project,
    )

    # ---- receipt ------------------------------------------------------------------------------
    receipt_dir = Path(os.environ.get("TRAIGENT_E2E_RECEIPT_DIR") or tmp_path)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = receipt_dir / f"python-sdk-live-backend-{stamp}.json"
    receipt_path.write_text(
        json.dumps(
            {
                "suite": "E2E-1 python-sdk <-> real backend",
                "command": "pytest tests/e2e/test_runtime_requirements_live_backend.py "
                "-p no:xdist -o addopts= -s",
                "python": sys.version.split()[0],
                "heads": {
                    "sdk": _git_head(REPO_ROOT),
                    "backend": _git_head(Path(os.environ["TRAIGENT_E2E_BACKEND_REPO"]))
                    if os.environ.get("TRAIGENT_E2E_BACKEND_REPO")
                    else None,
                },
                "backend_url": url,
                "ids": {
                    "project_id": project,
                    "agent_id": agent,
                    "head_id": facts.get("head_id"),
                    "revision_id": facts.get("revision_id"),
                    "tvl_head_id": facts.get("tvl_head_id"),
                    "breach_id": facts.get("breach_id"),
                    "evaluation_ids": facts.get("evaluation_ids"),
                    "run_id": facts.get("run_id"),
                },
                "stored_profile_digest": facts.get("stored_profile_digest"),
                "cross_project_statuses": facts.get("cross_project_statuses"),
                "events_seen": [e.event for e in events],
                "steps": steps.rows,
                "passed": sum(r["status"] == "passed" for r in steps.rows),
                "failed": sum(r["status"] == "failed" for r in steps.rows),
                "skipped": sum(r["status"] == "skipped" for r in steps.rows),
            },
            indent=2,
        )
    )
    print(f"E2E-1 receipt: {receipt_path}")  # noqa: T201 - the pointer to the evidence
    assert not steps.failed(), json.dumps(steps.failed(), indent=2)
