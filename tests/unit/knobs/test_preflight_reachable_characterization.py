"""Characterization tests pinning ``_preflight_reachable`` (fail-closed preflight)."""

from __future__ import annotations

from traigent.knobs.composites import (
    CascadeBody,
    CompositeArm,
    CompositeKind,
    CompositeNode,
    GateDecl,
    GateKind,
    LoopBody,
    Placement,
    SignalUse,
    StageArm,
    StopDecl,
    StopKind,
)
from traigent.knobs.runtime import ResultKind, _preflight_reachable


def _post(name, arms):
    arm_objs = tuple(StageArm(a) if isinstance(a, str) else a for a in arms)
    gates = tuple(
        GateDecl(kind=GateKind.MARGIN_BELOW, threshold=f"t{i}")
        for i in range(len(arm_objs) - 1)
    )
    return CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(arms=arm_objs, gates=gates, placement=Placement.POST),
    )


def _pre(name, arms, signals):
    arm_objs = tuple(StageArm(a) if isinstance(a, str) else a for a in arms)
    gates = tuple(
        GateDecl(
            kind=GateKind.SIGNAL_BELOW,
            threshold=f"t{i}",
            signal=SignalUse(signal=sig),
        )
        for i, sig in enumerate(signals)
    )
    return CompositeNode(
        name=name,
        kind=CompositeKind.CASCADE,
        body=CascadeBody(arms=arm_objs, gates=gates, placement=Placement.PRE),
    )


def _loop(name, stop, body="lb"):
    return CompositeNode(
        name=name,
        kind=CompositeKind.LOOP,
        body=LoopBody(body=StageArm(body), stop=stop, state_keys=(), max_iters=2),
    )


def _run(root, stages=(), registry=None, signals=(), predicates=()):
    reg = registry if registry is not None else {root.name: root}
    result = _preflight_reachable(
        root,
        {s: (lambda *_a: None) for s in stages},
        reg,
        {s: (lambda *_a: None) for s in signals},
        {p: (lambda *_a: None) for p in predicates},
    )
    if result is None:
        return None
    assert result.result_kind is ResultKind.ERROR
    assert result.output is None and result.measures == {}
    return result.error


def test_everything_present_returns_none():
    root = _post("root", ["a", "b"])
    assert _run(root, stages=["a", "b"]) is None


def test_missing_stage_message_sorted():
    root = _post("root", ["b", "a", "c"])
    assert _run(root, stages=["c"]) == (
        "composite-runtime: missing stage callable(s) ['a', 'b'] for composite "
        "'root' (fail-closed: a missing stage fails the item's evaluation)"
    )


def test_missing_stage_reported_before_missing_ref():
    root = _post("root", ["a", CompositeArm("ghost")])
    assert "missing stage callable(s) ['a']" in _run(root)


def test_missing_ref_message():
    root = _post("root", ["a", CompositeArm("ghost")])
    assert _run(root, stages=["a"]) == (
        "composite-runtime: missing composite ref(s) ['ghost'] for composite "
        "'root' (fail-closed: an unresolved nested arm fails the item's evaluation)"
    )


def test_loop_signal_accept_missing_signal():
    loop = _loop(
        "lp",
        StopDecl(
            kind=StopKind.SIGNAL_ACCEPT,
            threshold="t",
            signal=SignalUse(signal="sig", inputs=()),
        ),
    )
    assert _run(loop, stages=["lb"]) == (
        "composite-runtime: signal 'sig' is not provided for composite 'lp' "
        "(fail-closed: a missing signal fails the item's evaluation)"
    )
    assert _run(loop, stages=["lb"], signals=["sig"]) is None


def test_loop_external_accept_missing_predicate():
    loop = _loop("lp", StopDecl(kind=StopKind.EXTERNAL_ACCEPT, predicate="pred"))
    assert _run(loop, stages=["lb"]) == (
        "composite-runtime: predicate 'pred' is not provided for composite 'lp' "
        "(fail-closed: a missing predicate fails the item's evaluation)"
    )
    assert _run(loop, stages=["lb"], predicates=["pred"]) is None
    # a signal of the same name does not satisfy a predicate
    assert _run(loop, stages=["lb"], signals=["pred"]) is not None


def test_loop_exhausted_needs_nothing():
    loop = _loop("lp", StopDecl(kind=StopKind.EXHAUSTED))
    assert _run(loop, stages=["lb"]) is None


def test_pre_cascade_gate_signal_missing():
    node = _pre("disp", ["a", "b", "c"], ["s0", "s1"])
    assert _run(node, stages=["a", "b", "c"], signals=["s0"]) == (
        "composite-runtime: signal 's1' is not provided for composite 'disp' "
        "(fail-closed: a missing dispatch signal fails the item's evaluation)"
    )
    # first missing gate signal wins
    assert "'s0'" in _run(node, stages=["a", "b", "c"])
    assert _run(node, stages=["a", "b", "c"], signals=["s0", "s1"]) is None


def test_post_cascade_never_checks_signals():
    node = _post("root", ["a", "b"])
    assert _run(node, stages=["a", "b"]) is None


def test_nested_child_checked_even_if_unselected():
    inner = _post("inner", ["z"])
    outer = _post("outer", ["a", CompositeArm("inner")])
    reg = {"outer": outer, "inner": inner}
    assert _run(outer, stages=["a"], registry=reg) == (
        "composite-runtime: missing stage callable(s) ['z'] for composite "
        "'inner' (fail-closed: a missing stage fails the item's evaluation)"
    )
    assert _run(outer, stages=["a", "z"], registry=reg) is None


def test_children_visited_last_arm_first():
    c1 = _post("c1", ["m1"])
    c2 = _post("c2", ["m2"])
    root = _post("root", [CompositeArm("c1"), CompositeArm("c2")])
    reg = {"root": root, "c1": c1, "c2": c2}
    # stack pops the last pushed child first, so c2 is reported before c1
    assert "composite 'c2'" in _run(root, registry=reg)
    assert "composite 'c1'" in _run(root, stages=["m2"], registry=reg)


def test_self_reference_terminates():
    root = _post("root", ["a", CompositeArm("root")])
    assert _run(root, stages=["a"]) is None


def test_mutual_reference_terminates_and_reports_once():
    a = _post("a", ["x", CompositeArm("b")])
    b = _post("b", ["y", CompositeArm("a")])
    reg = {"a": a, "b": b}
    assert _run(a, stages=["x", "y"], registry=reg) is None
    assert "composite 'b'" in _run(a, stages=["x"], registry=reg)


def test_nested_pre_arm_signal_checked():
    inner = _pre("inner", ["ia", "ib"], ["needed"])
    outer = _post("outer", ["a", CompositeArm("inner")])
    reg = {"outer": outer, "inner": inner}
    err = _run(outer, stages=["a", "ia", "ib"], registry=reg)
    assert err is not None and "'needed'" in err and "composite 'inner'" in err
