"""Resource-bounded bottom-up enumerator with semantic memoization and learned priorities.

Scheduling invariant (proof in docs/architecture.md):

* terminals have priority cost 1;
* a composition has ``base = 1 + sum(child priority costs)`` and is stored at priority cost
  ``base + floor(lam * (1 - p))`` with ``p`` in [0, 1] (``p = 1`` when unscored), so a stored
  state's cost is always ``>= base``;
* the level loop processes base costs in increasing order; *bucket k is final once base
  cost k has been processed*, because only base costs <= k can write to bucket k;
* a composition of base cost c reads only buckets <= c - 1, which are therefore final;
* every state of AST size s has priority cost <= (1 + lam) * s - lam, so
  ``c_max = (1 + lam) * max_nodes - lam`` is a sufficient level bound.

AST size and priority cost are tracked separately; the node limit is always applied to AST
size.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from operator import add
from typing import Protocol

import numpy as np

from .. import dsl
from ..dsl import Invalid, Node
from ..tasks import SearchTask

SCORE_BATCH = 256
RSS_CHECK_EVERY = 4096

OUTCOMES = ("solved", "timeout", "candidate_limit", "state_limit", "exhausted", "invalid_task", "resource_limit")


class Scorer(Protocol):
    """Usefulness scorer: maps a batch of candidate behaviours to p in [0, 1]."""

    name: str

    def begin(self, task: SearchTask) -> None: ...

    def score(self, outs: list[tuple[str, ...]], sizes: list[int]) -> np.ndarray: ...

    def stats(self) -> dict: ...


@dataclass
class Counters:
    terminals: int = 0
    terminal_duplicates: int = 0
    candidates: int = 0  # attempted compositions executed on the example batch
    invalid: int = 0
    duplicates: int = 0  # exact (outputs, size) memo hits
    dominated: int = 0  # dropped: a smaller state with identical outputs already exists
    noop_skipped: int = 0  # Concat with a constant-empty child, skipped before execution
    interpreter_calls: int = 0  # per-example operator applications (= candidates * examples)
    states: int = 0  # retained semantic states, terminals included
    solutions_seen: int = 0
    levels: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SearchResult:
    outcome: str
    program: Node | None
    program_size: int | None
    counters: Counters
    seconds: float
    scoring: dict = field(default_factory=dict)  # model_calls, rows, cache_hits, featurize_s, model_s
    lam: int = 0
    policy: str = "uniform"
    priority_cost: int | None = None
    max_priority_level: int = 0
    trace: dict = field(default_factory=dict)
    milestones: list[list[float]] = field(default_factory=list)  # [candidates, seconds] per level end
    states: list[tuple[tuple[str, ...], int]] | None = None  # collect_states=True only
    solved_outs: set[tuple[str, ...]] | None = None
    detail: str = ""

    @property
    def solved(self) -> bool:
        return self.outcome == "solved"

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "program": dsl.to_json(self.program) if self.program is not None else None,
            "program_text": dsl.to_str(self.program) if self.program is not None else None,
            "program_size": self.program_size,
            "priority_cost": self.priority_cost,
            "counters": self.counters.to_dict(),
            "seconds": self.seconds,
            "scoring": self.scoring,
            "lam": self.lam,
            "policy": self.policy,
            "max_priority_level": self.max_priority_level,
            "milestones": self.milestones,
            "trace": self.trace,
            "detail": self.detail,
        }


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Solved(Exception):
    def __init__(self, op: str, params: tuple, kids: tuple[int, ...]) -> None:
        self.op, self.params, self.kids = op, params, kids


def priority_level_bound(max_nodes: int, lam: int) -> int:
    """Sufficient bound on the priority cost of any state with AST size <= max_nodes."""
    return (1 + lam) * max_nodes - lam


def _invalid_result(task: SearchTask, detail: str, policy: str, lam: int) -> SearchResult:
    return SearchResult("invalid_task", None, None, Counters(), 0.0, lam=lam, policy=policy, detail=detail)


def search(
    task: SearchTask,
    scorer: Scorer | None = None,
    lam: int = 0,
    *,
    memo: str = "semantic",
    prune_dominated: bool = True,
    stop_on_solution: bool = True,
    collect_states: bool = False,
    check_visited: bool = False,
    trace_detail: bool = False,
) -> SearchResult:
    """Run the bottom-up search.

    ``memo``: ``"semantic"`` keys states by (output vector, AST size); ``"none"`` disables
    memoization (tiny controls only).  ``prune_dominated`` additionally drops a candidate when a
    *smaller* AST with identical outputs is already retained (sound for the node budget; see
    docs/architecture.md).  ``stop_on_solution=False`` keeps exploring past solutions (used only
    to collect training negatives).
    """
    t0 = time.perf_counter()
    policy = scorer.name if scorer is not None else "uniform"
    if lam < 0 or int(lam) != lam:
        raise ValueError("lam must be a nonnegative integer")
    if lam > 0 and scorer is None:
        raise ValueError("lam > 0 requires a scorer")
    if memo not in ("semantic", "none"):
        raise ValueError("memo must be 'semantic' or 'none'")
    spec, vocab, bud = task.spec, task.vocab, task.budgets
    if bud.max_nodes < 1 or bud.max_candidates < 1 or bud.seconds <= 0:
        return _invalid_result(task, "nonpositive budget", policy, lam)
    if not 1 <= len(spec.examples) <= 6 or vocab.n_inputs != spec.n_inputs:
        return _invalid_result(task, "bad example count or vocabulary/schema mismatch", policy, lam)
    ML = bud.max_string_len
    if any(len(s) > ML for e in spec.examples for s in (*e.inputs, e.output)):
        return _invalid_result(task, "example string longer than max_string_len", policy, lam)

    n_ex = len(spec.examples)
    targets = tuple(e.output for e in spec.examples)
    N = bud.max_nodes
    deadline = t0 + bud.seconds
    pc = time.perf_counter
    cmax = priority_level_bound(N, lam)
    cn = Counters()

    # ---- state storage (struct-of-arrays; ASTs are rebuilt lazily only for solutions)
    outs_l: list[tuple[str, ...]] = []
    size_l: list[int] = []
    cost_l: list[int] = []
    op_l: list[str] = []
    par_l: list[tuple] = []
    kid_l: list[tuple[int, ...]] = []
    p_l: list[float] = []
    empty_ids: set[int] = set()
    buckets: dict[int, dict[int, list[int]]] = {}
    seen: set[tuple[tuple[str, ...], int]] = set()
    min_size: dict[tuple[str, ...], int] = {}
    visited: set | None = set() if check_visited else None
    solved_outs: set[tuple[str, ...]] = set()
    pending: list[int] = []
    trace_events: list[dict] = []
    milestones: list[list[float]] = []
    trace_truncated = False
    proc = None
    ncand = 0
    use_score = scorer is not None and lam > 0
    if scorer is not None:
        scorer.begin(task)

    def new_state(outs: tuple[str, ...], size: int, op: str, params: tuple, kids: tuple[int, ...]) -> int:
        if len(outs_l) >= bud.max_states:
            raise _Stop("state_limit")
        i = len(outs_l)
        outs_l.append(outs)
        size_l.append(size)
        cost_l.append(-1)
        op_l.append(op)
        par_l.append(params)
        kid_l.append(kids)
        p_l.append(1.0)
        if all(s == "" for s in outs):
            empty_ids.add(i)
        return i

    def place(i: int, cost: int) -> None:
        cost_l[i] = cost
        buckets.setdefault(cost, {}).setdefault(size_l[i], []).append(i)

    def flush(base: int) -> None:
        if not pending:
            return
        ids = pending[:]
        pending.clear()
        ps = scorer.score([outs_l[i] for i in ids], [size_l[i] for i in ids])  # type: ignore[union-attr]
        ps = np.clip(np.nan_to_num(np.asarray(ps, dtype=np.float64), nan=0.0), 0.0, 1.0)
        for i, p in zip(ids, ps.tolist()):
            p_l[i] = p
            place(i, base + int(math.floor(lam * (1.0 - p))))

    def consider(outs: tuple[str, ...], size: int, op: str, params: tuple, kids: tuple[int, ...], base: int) -> None:
        if outs == targets:
            cn.solutions_seen += 1
            if stop_on_solution:
                raise _Solved(op, params, kids)
            solved_outs.add(outs)
            return
        if memo == "semantic":
            key = (outs, size)
            if key in seen:
                cn.duplicates += 1
                return
            if prune_dominated:
                ms = min_size.get(outs)
                if ms is not None and ms < size:
                    cn.dominated += 1
                    return
                if ms is None or size < ms:
                    min_size[outs] = size
            seen.add(key)
        i = new_state(outs, size, op, params, kids)
        if use_score:
            pending.append(i)
            if len(pending) >= SCORE_BATCH:
                if pc() > deadline:
                    raise _Stop("timeout")
                flush(base)
        else:
            place(i, base)

    def attempt_guard() -> None:
        nonlocal ncand, proc
        if ncand >= bud.max_candidates:
            raise _Stop("candidate_limit")
        ncand += 1
        if not (ncand & 31) and pc() > deadline:
            raise _Stop("timeout")
        if not (ncand % RSS_CHECK_EVERY):
            if proc is None:
                import psutil

                proc = psutil.Process()
            if proc.memory_info().rss / 2**20 > bud.max_rss_mb:
                raise _Stop("resource_limit")

    def build(i: int) -> Node:
        op, par, kids = op_l[i], par_l[i], kid_l[i]
        return _build_node(op, par, [build(k) for k in kids])

    outcome = "exhausted"
    program: Node | None = None
    priority_cost: int | None = None
    detail = ""
    level = 0
    try:
        # ---- terminals (priority cost 1, never scored)
        terminals: list[tuple[tuple[str, ...], str, tuple]] = []
        for j in range(spec.n_inputs):
            terminals.append((tuple(e.inputs[j] for e in spec.examples), "Input", (j,)))
        for lit in vocab.literals:
            terminals.append(((lit,) * n_ex, "Literal", (lit,)))
        for outs, op, params in terminals:
            cn.terminals += 1
            if outs == targets:
                cn.solutions_seen += 1
                if stop_on_solution:
                    raise _Solved(op, params, ())
                solved_outs.add(outs)
                continue
            key = (outs, 1)
            if memo == "semantic":
                if key in seen:
                    cn.terminal_duplicates += 1
                    continue
                seen.add(key)
                min_size.setdefault(outs, 1)
            place(new_state(outs, 1, op, params, ()), 1)

        unary = [(op, par, dsl.unary_fn(op, par, ML)) for op, par in vocab.unary_ops()]

        for level in range(2, cmax + 1):
            if pc() > deadline:
                raise _Stop("timeout")
            c0, n0, st0 = ncand, 0, len(outs_l)
            cn.levels += 1
            # unary compositions: child priority cost level-1
            b1 = buckets.get(level - 1)
            if b1:
                for s in sorted(b1):
                    if s + 1 > N:
                        continue
                    for sid in b1[s]:
                        co = outs_l[sid]
                        for op, par, fn in unary:
                            attempt_guard()
                            if visited is not None:
                                k = (op, par, (sid,))
                                assert k not in visited, f"composition scheduled twice: {k}"
                                visited.add(k)
                            res = tuple(map(fn, co))
                            cn.interpreter_calls += n_ex
                            if any(r.__class__ is Invalid for r in res):
                                cn.invalid += 1
                                continue
                            consider(res, s + 1, op, par, (sid,), level)  # type: ignore[arg-type]
            # concat compositions: ordered (left, right) priority costs summing to level-1
            for ca in range(1, level - 1):
                cb = level - 1 - ca
                ba, bb = buckets.get(ca), buckets.get(cb)
                if not ba or not bb:
                    continue
                for sa in sorted(ba):
                    for sb in sorted(bb):
                        if sa + sb + 1 > N:
                            continue
                        for ia in ba[sa]:
                            if ia in empty_ids:
                                cn.noop_skipped += len(bb[sb])
                                continue
                            oa = outs_l[ia]
                            for ib in bb[sb]:
                                if ib in empty_ids:
                                    cn.noop_skipped += 1
                                    continue
                                attempt_guard()
                                if visited is not None:
                                    k = ("Concat", (), (ia, ib))
                                    assert k not in visited, f"composition scheduled twice: {k}"
                                    visited.add(k)
                                res = tuple(map(add, oa, outs_l[ib]))
                                cn.interpreter_calls += n_ex
                                if max(map(len, res)) > ML:
                                    cn.invalid += 1
                                    continue
                                consider(res, sa + sb + 1, "Concat", (), (ia, ib), level)
            if use_score:
                flush(level)  # scored states must be visible before any later level
            milestones.append([ncand, round(pc() - t0, 6)])
            if len(trace_events) < bud.max_trace_events:
                ev = {
                    "level": level,
                    "attempted": ncand - c0,
                    "new_states": len(outs_l) - st0,
                    "elapsed_s": round(pc() - t0, 6),
                }
                if use_score and len(outs_l) > st0:
                    ps = p_l[st0:]
                    ev["mean_p"] = round(float(np.mean(ps)), 4)
                    ev["max_p"] = round(float(np.max(ps)), 4)
                trace_events.append(ev)
            else:
                trace_truncated = True
    except _Solved as sol:
        outcome = "solved"
        program = _build_node(sol.op, sol.params, [build(k) for k in sol.kids])
        if sol.kids:
            priority_cost = 1 + sum(cost_l[k] for k in sol.kids)
        else:
            priority_cost = 1
    except _Stop as stop:
        outcome = stop.reason
    seconds = pc() - t0
    cn.states = len(outs_l)
    cn.candidates = ncand
    stats = scorer.stats() if scorer is not None else {}
    trace: dict = {"events": trace_events, "truncated": trace_truncated, "c_max": cmax}
    if outcome == "solved" and program is not None:
        trace["lineage"] = _lineage(program, task, trace_detail)
    return SearchResult(
        outcome=outcome,
        program=program,
        program_size=dsl.ast_size(program) if program is not None else None,
        counters=cn,
        seconds=seconds,
        scoring=stats,
        lam=lam,
        policy=policy,
        priority_cost=priority_cost,
        max_priority_level=level,
        trace=trace,
        milestones=milestones,
        states=[(outs_l[i], size_l[i]) for i in range(len(outs_l)) if kid_l[i]] if collect_states else None,
        solved_outs=solved_outs if collect_states else None,
        detail=detail,
    )


def _build_node(op: str, params: tuple, kids: list[Node]) -> Node:
    if op == "Input":
        return dsl.Input(params[0])
    if op == "Literal":
        return dsl.Literal(params[0])
    if op == "Concat":
        return dsl.Concat(kids[0], kids[1])
    return dsl.make_unary(op, params, kids[0])


def _lineage(program: Node, task: SearchTask, detail: bool) -> list[dict]:
    """Post-order list of every sub-expression of the solution with its behaviour on the
    supplied examples (real values recomputed by the interpreter, not stored estimates)."""
    rows = [e.inputs for e in task.spec.examples]
    out: list[dict] = []

    def rec(n: Node) -> None:
        for c in n.children():
            rec(c)
        vals = dsl.evaluate_rows(n, rows, task.budgets.max_string_len)
        out.append(
            {
                "expr": dsl.to_str(n),
                "size": dsl.ast_size(n),
                "values": [v if isinstance(v, str) else f"<invalid:{v.reason}>" for v in vals],
            }
        )

    rec(program)
    return out
