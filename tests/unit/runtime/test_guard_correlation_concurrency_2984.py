"""Concurrent tasks sharing one unit keep their own correlation id (#2984)."""

from __future__ import annotations

import asyncio

from traigent.runtime.guards import LocalGuardSet, guarded


def _stamped(unit) -> dict[str, str | None]:
    return {
        ev["tool"]: ev.get("correlation_id")
        for ev in unit._history
        if ev.get("type") == "tool_call"
    }


def test_gathered_tasks_are_not_mis_stamped() -> None:
    gate_c = asyncio.Event()
    gate_d = asyncio.Event()
    c_called = asyncio.Event()

    @guarded(tool="tool.c")
    async def call_c() -> None:
        return None

    @guarded(tool="tool.d")
    async def call_d() -> None:
        return None

    async def task_c(unit) -> None:
        with unit.correlated("C"):
            gate_c.set()
            await gate_d.wait()  # D has entered its own block meanwhile
            await call_c()
            c_called.set()

    async def task_d(unit) -> None:
        await gate_c.wait()
        with unit.correlated("D"):
            gate_d.set()
            await c_called.wait()  # C's call happens while D's block is open
            await call_d()

    async def main():
        with LocalGuardSet([]).unit() as unit:
            await asyncio.gather(task_c(unit), task_d(unit))
            return unit

    unit = asyncio.run(main())
    assert _stamped(unit) == {"tool.c": "C", "tool.d": "D"}
    assert unit.correlation_id is None


def test_non_lifo_exit_leaves_no_stale_id() -> None:
    async def main() -> str | None:
        with LocalGuardSet([]).unit() as unit:
            a_in, b_in, a_out = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def a() -> None:
                with unit.correlated("A"):
                    a_in.set()
                    await b_in.wait()
                a_out.set()

            async def b() -> None:
                await a_in.wait()
                with unit.correlated("B"):
                    b_in.set()
                    await a_out.wait()

            await asyncio.gather(a(), b())
            return unit.correlation_id

    assert asyncio.run(main()) is None
