"""Isolated real optimization lifecycle probe for shutdown interrupts."""

import asyncio
import json
import os
import sys

os.environ["TRAIGENT_OFFLINE_MODE"] = "true"
os.environ["TRAIGENT_COST_APPROVED"] = "true"
os.environ["TRAIGENT_SKIP_DOTENV"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from tests.unit.core.test_orchestrator import MockEvaluator, MockOptimizer
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.callbacks import CallbackManager, OptimizationCallback

kind, stage, deadline = sys.argv[1:]
interrupt_type = KeyboardInterrupt if kind == "keyboard" else SystemExit


class Callback(OptimizationCallback):
    fired = False

    def interrupt(self, here):
        if stage == here and not self.fired:
            self.fired = True
            raise interrupt_type("intentional-test-interrupt")

    def on_optimization_start(self, *args):
        self.interrupt("initialize")

    def on_trial_start(self, *args):
        self.interrupt("start")

    def on_trial_complete(self, *args):
        self.interrupt("complete")

    def on_optimization_complete(self, *args):
        pass


async def main():
    owner = OptimizationOrchestrator(
        optimizer=MockOptimizer({"param1": [0, 1]}, ["accuracy"]),
        evaluator=MockEvaluator(),
        max_trials=2,
        parallel_trials=2,
        timeout=30 if deadline == "deadline" else None,
    )
    owner.callback_manager = CallbackManager(
        [Callback()], timeout=0 if kind == "keyboard" else 0.5
    )
    finalized = []
    finalize_local = owner.backend_session_manager.finalize_local_session

    def record_finalize(session_id, status, **kwargs):
        finalized.append(status.value)
        return finalize_local(session_id, status, **kwargs)

    owner.backend_session_manager.finalize_local_session = record_finalize

    async def agent(input_data, **config):
        return "local response"

    try:
        result = await owner.optimize(
            agent, Dataset([EvaluationExample({"query": "local"}, "local response")])
        )
        outcome = {"result_status": result.status.value}
    except SystemExit:
        outcome = {"exception": "SystemExit"}
    status = owner.cost_enforcer.get_status()
    outcome.update(
        trials=len(owner._trials),
        accounted_trials=status.trial_count,
        in_flight=status.in_flight_count,
        reserved=status.reserved_cost_usd,
        finalized=finalized,
    )
    owner.callback_manager.close()
    return outcome


try:
    print(json.dumps(asyncio.run(main())), flush=True)
except BaseException as exc:
    print(json.dumps({"runner_exception": type(exc).__name__}), flush=True)
    raise SystemExit(23) from None
