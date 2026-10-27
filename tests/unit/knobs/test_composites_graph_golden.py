"""Golden characterization for ``required_parents`` and ``_detect_cycle``."""

from __future__ import annotations

import pytest

from traigent.knobs.composites import (
    AcceptDecl,
    AggregateDecl,
    AggregateKind,
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    CompositeProgram,
    EnsembleBody,
    GateDecl,
    GateKind,
    LoopBody,
    Placement,
    SignalUse,
    StageArm,
    StatKind,
    StopDecl,
    StopKind,
    _detect_cycle,
    required_parents,
)


def _loop(name: str, ref: str) -> CompositeNode:
    return CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=CompositeArm(ref),
            stop=StopDecl(kind=StopKind.EXHAUSTED),
            max_iters=2,
        ),
    )


def _casc(name: str, arms: tuple, gates: tuple = ()) -> CompositeNode:
    return CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(arms=arms, gates=gates),
    )


def _code(exc: pytest.ExceptionInfo[ValueError]) -> str:
    return str(exc.value).split(":", 1)[0]


# ------------------------------ cycle detection -----------------------------


def test_no_cycle_for_empty_chain_and_diamond() -> None:
    _detect_cycle(CompositeProgram(composites={}))
    leaf = _casc("leaf", (StageArm("s"),))
    a = _casc("a", (CompositeArm("leaf"),))
    b = _casc("b", (CompositeArm("leaf"),))
    top = _casc(
        "top",
        (CompositeArm("a"), CompositeArm("b")),
        (GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
    )
    _detect_cycle(
        CompositeProgram(composites={"top": top, "a": a, "b": b, "leaf": leaf})
    )


def test_self_cycle_and_indirect_cycle() -> None:
    with pytest.raises(ValueError, match=r"cycle through composite\('A'\)") as e:
        _detect_cycle(CompositeProgram(composites={"A": _loop("A", "A")}))
    assert _code(e) == "composite_cycle"
    prog = CompositeProgram(
        composites={"A": _loop("A", "B"), "B": _loop("B", "C"), "C": _loop("C", "A")}
    )
    with pytest.raises(ValueError, match=r"cycle through composite\('A'\)"):
        _detect_cycle(prog)


def test_missing_ref_reported_even_when_reached_through_other_node() -> None:
    prog = CompositeProgram(composites={"A": _loop("A", "B"), "B": _loop("B", "ghost")})
    with pytest.raises(ValueError, match=r"composite\('ghost'\) resolves") as e:
        _detect_cycle(prog)
    assert _code(e) == "missing_composite_ref"


def test_arm_order_decides_which_fault_is_reported_first() -> None:
    gates = (GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),)
    cyc_first = _casc("X", (CompositeArm("X"), CompositeArm("ghost")), gates)
    with pytest.raises(ValueError) as e:
        _detect_cycle(CompositeProgram(composites={"X": cyc_first}))
    assert _code(e) == "composite_cycle"
    ghost_first = _casc("X", (CompositeArm("ghost"), CompositeArm("X")), gates)
    with pytest.raises(ValueError) as e:
        _detect_cycle(CompositeProgram(composites={"X": ghost_first}))
    assert _code(e) == "missing_composite_ref"


def test_ensemble_judge_arm_is_traversed() -> None:
    ens = CompositeNode(
        name="E",
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=(StageArm("a"), StageArm("b")),
            aggregate=AggregateDecl(
                kind=AggregateKind.JUDGE_MAX, judge=CompositeArm("E")
            ),
        ),
    )
    with pytest.raises(ValueError) as e:
        _detect_cycle(CompositeProgram(composites={"E": ens}))
    assert _code(e) == "composite_cycle"


def test_cycle_found_in_later_root_after_clean_root() -> None:
    clean = _casc("clean", (StageArm("s"),))
    prog = CompositeProgram(composites={"clean": clean, "A": _loop("A", "A")})
    with pytest.raises(ValueError) as e:
        _detect_cycle(prog)
    assert _code(e) == "composite_cycle"


# ------------------------------ required_parents ----------------------------


def test_post_cascade_repeated_threshold_unions_prefixes() -> None:
    node = _casc(
        "c",
        (StageArm("a", ("m1",)), StageArm("b", ("m2",)), StageArm("c", ("m3",))),
        (
            GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),
            GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),
        ),
    )
    prog = CompositeProgram(composites={"c": node}, tvars=frozenset({"m1", "m2", "m3"}))
    assert required_parents(prog, node) == {"t": frozenset({"m1", "m2"})}


def test_pre_cascade_each_gate_covers_only_its_arm() -> None:
    node = CompositeNode(
        name="r",
        kind=CompositeKind.CASCADE,
        body=CascadeBody(
            arms=(StageArm("a", ("m1",)), StageArm("b", ("m2",)), StageArm("c")),
            gates=(
                GateDecl(
                    kind=GateKind.SIGNAL_BELOW,
                    threshold="t1",
                    signal=SignalUse("s"),
                ),
                GateDecl(
                    kind=GateKind.SIGNAL_BELOW,
                    threshold="t1",
                    signal=SignalUse("s"),
                ),
            ),
            placement=Placement.PRE,
        ),
    )
    prog = CompositeProgram(composites={"r": node}, tvars=frozenset({"m1", "m2"}))
    assert required_parents(prog, node) == {"t1": frozenset({"m1", "m2"})}


def test_post_single_arm_cascade_has_no_obligations() -> None:
    node = _casc("c", (StageArm("a", ("m1",)),))
    assert required_parents(CompositeProgram(composites={"c": node}), node) == {}


def _ens(
    accept: AcceptDecl | None, cardinality: str | None, arms: tuple
) -> CompositeNode:
    return CompositeNode(
        name="e",
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=arms,
            aggregate=AggregateDecl(kind=AggregateKind.MAJORITY_VOTE, accept=accept),
            cardinality=cardinality,
        ),
    )


def test_ensemble_without_accept_has_no_obligations() -> None:
    node = _ens(None, None, (StageArm("a", ("m1",)), StageArm("b")))
    prog = CompositeProgram(composites={"e": node}, tvars=frozenset({"m1"}))
    assert required_parents(prog, node) == {}


def test_ensemble_accept_unions_arms_and_tvar_cardinality() -> None:
    accept = AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold="th")
    node = _ens(accept, "k", (StageArm("a", ("m1",)),))
    prog = CompositeProgram(
        composites={"e": node}, tvars=frozenset({"m1", "k"}), cvars=frozenset({"th"})
    )
    assert required_parents(prog, node) == {"th": frozenset({"m1", "k"})}
    # cardinality that is not a TVAR is intersected out
    prog = CompositeProgram(
        composites={"e": node}, tvars=frozenset({"m1"}), cvars=frozenset({"th", "k"})
    )
    assert required_parents(prog, node) == {"th": frozenset({"m1"})}
    # committee form: no cardinality term at all
    committee = _ens(accept, None, (StageArm("a", ("m1",)), StageArm("b", ("m2",))))
    prog = CompositeProgram(
        composites={"e": committee}, tvars=frozenset({"m1", "m2", "k"})
    )
    assert required_parents(prog, committee) == {"th": frozenset({"m1", "m2"})}


def test_ensemble_accept_includes_judge_arm_leaves() -> None:
    accept = AcceptDecl(stat=StatKind.VOTE_MARGIN, threshold="th")
    node = CompositeNode(
        name="e",
        kind=CompositeKind.ENSEMBLE,
        body=EnsembleBody(
            arms=(StageArm("a", ("m1",)), StageArm("b")),
            aggregate=AggregateDecl(
                kind=AggregateKind.JUDGE_MAX,
                judge=StageArm("j", ("mj",)),
                accept=accept,
            ),
        ),
    )
    prog = CompositeProgram(composites={"e": node}, tvars=frozenset({"m1", "mj"}))
    assert required_parents(prog, node) == {"th": frozenset({"m1", "mj"})}


def test_loop_obligations() -> None:
    stop = StopDecl(
        kind=StopKind.SIGNAL_ACCEPT, threshold="th", signal=SignalUse("s", ("k",))
    )
    node = CompositeNode(
        name="l",
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm("a", ("m1",)), stop=stop, state_keys=("k",), max_iters=3
        ),
    )
    prog = CompositeProgram(composites={"l": node}, tvars=frozenset({"m1"}))
    assert required_parents(prog, node) == {"th": frozenset({"m1"})}
    exhausted = _loop("x", "l")
    prog = CompositeProgram(composites={"l": node, "x": exhausted})
    assert required_parents(prog, exhausted) == {}
    ext = CompositeNode(
        name="ext",
        kind=CompositeKind.LOOP,
        body=LoopBody(
            body=StageArm("a", ("m1",)),
            stop=StopDecl(kind=StopKind.EXTERNAL_ACCEPT, predicate="p"),
            max_iters=2,
        ),
    )
    assert required_parents(CompositeProgram(composites={"ext": ext}), ext) == {}


def test_nested_composite_arm_contributes_nested_leaves() -> None:
    inner = _casc("inner", (StageArm("a", ("m1",)),))
    outer = _casc(
        "outer",
        (CompositeArm("inner"), StageArm("b", ("m2",))),
        (GateDecl(kind=GateKind.MARGIN_BELOW, threshold="t"),),
    )
    prog = CompositeProgram(
        composites={"inner": inner, "outer": outer}, tvars=frozenset({"m1", "m2"})
    )
    assert required_parents(prog, outer) == {"t": frozenset({"m1"})}
