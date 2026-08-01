#!/usr/bin/env python3
"""Build an offline cold-start evaluation set, then hand it to Traigent.

The default command only constructs and validates local artifacts.  Pass
``--run-optimize`` to invoke the existing ``@traigent.optimize`` flow; this
example deliberately does not contain a second configuration/evaluation runner.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import traigent
from traigent.generation.coldstart import (
    CallableOracle,
    ColdStartOptions,
    ColdStartOutcome,
    assert_optimizer_eligible,
    generate_eval_set,
)


@traigent.optimize(
    eval_dataset=None,
    objectives=["accuracy"],
    configuration_space={"implementation": ["correct", "predictably-wrong"]},
    injection_mode="seamless",
    offline=True,
)
def optimized_square(number: int) -> int:
    """The one real agent; its two local arms are distinguishable by accuracy."""
    implementation = str(traigent.get_config().get("implementation", "correct"))
    if implementation == "correct":
        return number * number
    if implementation == "predictably-wrong":
        return number * number + 1
    raise ValueError(f"unsupported local implementation: {implementation!r}")


def _oracle(inputs: dict[str, object]) -> int:
    """Independent local ground truth; it is not the target callable."""
    number = inputs["number"]
    if not isinstance(number, int):
        raise TypeError("number must be an integer")
    return number * number


def build(output_dir: Path):
    """Construct and hand off a tune-only set without calling the agent."""
    result = generate_eval_set(
        func=optimized_square,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_oracle, oracle_id="local_square_oracle.v1"),
        output_dir=output_dir,
        options=ColdStartOptions(
            num_candidates=5,
            max_files=1,
            include_globs=(Path(__file__).relative_to(REPO_ROOT).as_posix(),),
        ),
    )
    if result.outcome != ColdStartOutcome.EVAL_SET or result.tuning_path is None:
        raise RuntimeError(f"Cold-start discovery-only result: {list(result.gaps)}")
    assert_optimizer_eligible(result.tuning_path)
    optimized_square.set_eval_dataset_override(
        traigent.Dataset.from_jsonl(str(result.tuning_path))
    )
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "coldstart-eval-output",
        help=(
            "Approved local directory for artifacts. It must be under the current "
            "working directory, or under TRAIGENT_DATASET_ROOT if that is set."
        ),
    )
    command.add_argument(
        "--run-optimize",
        action="store_true",
        help="Run the existing offline @traigent.optimize handoff after construction.",
    )
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = build(arguments.output_dir)
    assert result.tuning_path is not None

    print(f"Wrote {result.tuning_path}")
    print(f"Wrote {result.audit_path}")
    print(f"Wrote {result.manifest_path}")
    print("Canonical handoff: await optimized_square.optimize(max_trials=2)")

    if arguments.run_optimize:
        optimization = asyncio.run(optimized_square.optimize(max_trials=2))
        print(f"Optimization completed with best score: {optimization.best_score}")
    else:
        print(
            "Construction is complete. Re-run with --run-optimize to invoke the "
            "SDK's existing offline optimizer; no local runner was reimplemented."
        )


if __name__ == "__main__":
    main()
