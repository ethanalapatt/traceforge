import itertools
import math

import numpy as np
import pytest

from conftest import spec_of, task_of, tiny_vocab
from traceforge import dsl
from traceforge.dsl import Concat, Input, Literal, Lower, Slice, Token, Trim, Upper, evaluate
from traceforge.evaluation import RunSpec, run_one
from traceforge.search import priority_level_bound, search
from traceforge.tasks import Budgets, EvalTask, Example, SearchTask, make_search_task


def _const_empty(node, rows, max_len=256):
    return all(evaluate(node, r, max_len) == "" for r in rows)


def brute_force_keys(vocab, rows, max_nodes, max_len=256, include_terminals=False, skip_empty_concat=True):
    """All (output vector, size) keys over every valid AST with <= max_nodes nodes, by plain
    recursive enumeration that shares no code with the search engine's scheduler.  The engine
    skips Concat operands that are constant-empty on the task rows (a lossless no-op: such a
    Concat only re-derives its other operand at a larger size); the oracle mirrors that rule."""
    by_size: dict[int, list[dsl.Node]] = {1: [Input(i) for i in range(vocab.n_inputs)] + [Literal(l) for l in vocab.literals]}
    for n in range(2, max_nodes + 1):
        nodes = [dsl.make_unary(op, par, c) for op, par in vocab.unary_ops() for c in by_size[n - 1]]
        for a in range(1, n - 1):
            for l in by_size[a]:
                for r in by_size[n - 1 - a]:
                    if skip_empty_concat and (_const_empty(l, rows, max_len) or _const_empty(r, rows, max_len)):
                        continue
                    nodes.append(Concat(l, r))
        by_size[n] = nodes
    keys = set()
    for n, nodes in by_size.items():
        if n == 1 and not include_terminals:
            continue
        for node in nodes:
            vals = [evaluate(node, r, max_len) for r in rows]
            if all(isinstance(v, str) for v in vals):
                keys.add((tuple(vals), n))
    return keys


ROWS = [("ab-cd",), ("Ef-gh",), ("ij-KL",)]


def unreachable_task(vocab, max_nodes, **kw):
    spec = spec_of([(r, "zzzz-never") for r in ROWS])
    return SearchTask(spec, vocab, Budgets(max_nodes=max_nodes, max_candidates=10**7, seconds=60, **kw))


class HashScorer:
    """Deterministic pseudo-random usefulness that depends only on (outputs, size)."""

    name = "hash"

    def __init__(self, p_fn=None):
        self.p_fn = p_fn
        self.calls = 0

    def begin(self, task):
        pass

    def score(self, outs, sizes):
        self.calls += 1
        if self.p_fn is not None:
            return np.asarray([self.p_fn(o, s) for o, s in zip(outs, sizes)])
        return np.asarray([(hash((o, s)) % 1000) / 999.0 for o, s in zip(outs, sizes)])

    def stats(self):
        return {"model_calls": self.calls}


@pytest.fixture(autouse=True)
def _fixed_hash(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")


def test_unlearned_search_agrees_with_brute_force_oracle():
    vocab = tiny_vocab()
    N = 4
    oracle = brute_force_keys(vocab, ROWS, N)
    res = search(unreachable_task(vocab, N), prune_dominated=False, stop_on_solution=False, collect_states=True)
    assert set(res.states) == oracle
    assert len(res.states) == len(set(res.states))  # memoization: one representative per key
    assert res.outcome == "exhausted"
    assert res.counters.duplicates > 0  # memoization actually did work


def test_dominance_pruning_preserves_minimal_size_per_output_vector():
    vocab = tiny_vocab()
    N = 5
    oracle = brute_force_keys(vocab, ROWS, N, include_terminals=True)
    min_oracle = {}
    for outs, size in oracle:
        min_oracle[outs] = min(size, min_oracle.get(outs, 99))
    res = search(unreachable_task(vocab, N), prune_dominated=True, stop_on_solution=False, collect_states=True)
    terminals = [Input(i) for i in range(vocab.n_inputs)] + [Literal(l) for l in vocab.literals]
    min_found = {tuple(evaluate(t, r) for r in ROWS): 1 for t in terminals}  # terminals are not in res.states
    for outs, size in res.states:
        min_found[outs] = min(size, min_found.get(outs, 99))
    assert min_found == min_oracle
    full = search(unreachable_task(vocab, N), prune_dominated=False, stop_on_solution=False, collect_states=True)
    assert len(res.states) < len(full.states)


@pytest.mark.parametrize("lam", [0, 1, 3])
def test_penalized_schedule_is_exhaustive_within_node_budget(lam):
    """c_max = (1+lam)*N - lam must be sufficient: with arbitrary key-dependent penalties the
    same (output, size) keys are reachable as in the unpenalized brute force."""
    vocab = tiny_vocab()
    N = 4
    oracle = brute_force_keys(vocab, ROWS, N)
    scorer = HashScorer() if lam else None
    res = search(unreachable_task(vocab, N), scorer, lam, prune_dominated=False, stop_on_solution=False, collect_states=True)
    assert set(res.states) == oracle
    assert res.max_priority_level <= priority_level_bound(N, lam)


def test_priority_bound_formula():
    assert priority_level_bound(7, 0) == 7
    assert priority_level_bound(4, 3) == 13
    assert priority_level_bound(1, 5) == 1


def test_worst_case_penalty_state_still_found_at_cmax():
    """With p=0 everywhere (max penalty) every non-terminal costs base+lam; a size-N program is
    still found: its cost is exactly the bound."""
    rows = [(("ab",), "AB-ab"), (("cd",), "CD-cd"), (("ef",), "EF-ef")]
    vocab = tiny_vocab()
    vocab = vocab.__class__(1, ("-", "."), vocab.delimiters, vocab.token_indices, vocab.slice_pairs, vocab.replace_pairs)
    spec = spec_of(rows)
    N = 6  # Concat(Upper(I0), Concat(L("-"), I0)) has 1+2+(1+1+1)=6 nodes
    task = SearchTask(spec, vocab, Budgets(max_nodes=N, max_candidates=10**7, seconds=60))
    base = search(task)
    assert base.solved
    lam = 3
    res = search(task, HashScorer(lambda o, s: 0.0), lam)
    assert res.solved
    assert res.program_size <= N
    assert res.priority_cost is not None and res.priority_cost >= res.program_size
    assert res.priority_cost <= priority_level_bound(N, lam) + lam


def test_priority_cost_is_distinct_from_ast_size_and_node_limit_uses_size():
    rows = [(("ab-cd",), "ab"), (("Ef-gh",), "ef"), (("ij-KL",), "ij")]  # Lower(Token(I0, "-", 0)): size 3
    task = SearchTask(spec_of(rows), tiny_vocab(), Budgets(max_nodes=3, max_candidates=10**6, seconds=30))
    plain = search(task)
    pen = search(task, HashScorer(lambda o, s: 0.0), 4)
    assert plain.solved and pen.solved
    assert plain.priority_cost == plain.program_size  # lam = 0: cost == size
    # the solution is returned on generation: its reported cost is its base cost, 1 + the stored
    # (penalized) cost of its child, which exceeds the AST size once the child was penalized
    assert pen.priority_cost > pen.program_size and pen.program_size == 3
    assert pen.program_size <= 3  # the node limit still bounds AST size, not cost
    res = search(unreachable_task(tiny_vocab(), 3), HashScorer(lambda o, s: 0.0), 4, stop_on_solution=False, collect_states=True)
    assert all(s <= 3 for _, s in res.states)


def test_concat_order_and_repeated_subtrees():
    task = task_of([(("ab", "cd"), "cdab"), (("x", "y"), "yx"), (("12", "34"), "3412")], cols=("a", "b"))
    r = search(task)
    assert r.solved and r.program == Concat(Input(1), Input(0))
    rep = task_of([(("ab",), "abab"), (("x",), "xx"), (("12",), "1212")])
    r2 = search(rep)
    assert r2.solved and r2.program_size == 3 and r2.program == Concat(Input(0), Input(0))  # repeated Input counted twice


def test_visited_composition_bookkeeping_never_repeats():
    res = search(unreachable_task(tiny_vocab(), 4), check_visited=True, stop_on_solution=False)
    assert res.outcome == "exhausted"  # the in-engine assertion would have raised on a repeat


def test_solution_is_verified_by_independent_interpreter():
    task = task_of([(("  Ada Lovelace  ",), "ada.lovelace"), (("Grace  Hopper",), "grace..hopper"), ((" Alan Turing",), "alan.turing")], max_nodes=5)
    r = search(task)
    assert r.solved
    for e in task.spec.examples:
        assert evaluate(r.program, e.inputs) == e.output


def test_budget_candidate_limit_counts_attempts_exactly():
    task = unreachable_task(tiny_vocab(), 6)
    task = SearchTask(task.spec, task.vocab, Budgets(max_nodes=6, max_candidates=137, seconds=30))
    r = search(task)
    assert r.outcome == "candidate_limit" and r.counters.candidates == 137


def test_budget_time_limit():
    task = SearchTask(unreachable_task(tiny_vocab(), 7).spec, tiny_vocab(), Budgets(max_nodes=7, max_candidates=10**9, seconds=1e-9))
    r = search(task)
    assert r.outcome == "timeout"


def test_budget_state_limit():
    task = SearchTask(unreachable_task(tiny_vocab(), 7).spec, tiny_vocab(), Budgets(max_nodes=7, max_candidates=10**9, seconds=30, max_states=25))
    r = search(task)
    assert r.outcome == "state_limit" and r.counters.states <= 25


def test_budget_string_length_gives_invalid_task_or_invalid_candidates():
    spec = spec_of([(("abcdefgh",), "x"), (("abc",), "y"), (("zz",), "w")])
    r = search(SearchTask(spec, tiny_vocab(), Budgets(max_nodes=3, max_candidates=100, seconds=5, max_string_len=5)))
    assert r.outcome == "invalid_task"
    # candidates that exceed the bound become *invalid*, never an unbounded allocation
    spec2 = spec_of([(("abcd",), "zz"), (("efgh",), "q"), (("ijkl",), "r")])
    t2 = SearchTask(spec2, tiny_vocab(), Budgets(max_nodes=4, max_candidates=100000, seconds=5, max_string_len=6))
    r2 = search(t2, stop_on_solution=False)
    assert r2.counters.invalid > 0


def test_budget_trace_events_are_bounded():
    task = SearchTask(unreachable_task(tiny_vocab(), 5).spec, tiny_vocab(), Budgets(max_nodes=5, max_candidates=10**6, seconds=30, max_trace_events=2))
    r = search(task)
    assert len(r.trace["events"]) == 2 and r.trace["truncated"] is True


def test_budget_process_resource_limit():
    from traceforge.data.authored import authored_tasks

    t = authored_tasks()[13]  # a task that needs many candidates
    st_ = make_search_task(t.spec, Budgets(max_nodes=9, max_candidates=50_000, seconds=30, max_rss_mb=1.0))
    r = search(st_)
    assert r.outcome == "resource_limit"


def test_exhausted_means_finite_grammar_exhausted():
    r = search(unreachable_task(tiny_vocab(), 3))
    assert r.outcome == "exhausted" and r.program is None


def test_search_receives_no_hidden_information_and_is_label_independent():
    """Changing withheld labels must not change candidate-budget-limited search behaviour."""
    assert {f for f in SearchTask.__dataclass_fields__} == {"spec", "vocab", "budgets"}
    spec = spec_of([(("ab-cd",), "ab_cd"), (("Ef-gh",), "Ef_gh"), (("ij-KL",), "ij_KL")])
    hid_a = tuple(Example((f"q{i}-z",), f"q{i}_z") for i in range(20))
    hid_b = tuple(Example((f"q{i}-z",), f"WRONG{i}") for i in range(20))
    ta = EvalTask(spec, hid_a, suite="s")
    tb = EvalTask(spec, hid_b, reference=Lower(Input(0)), suite="s")  # even a different 'reference'
    bud = Budgets(max_nodes=5, max_candidates=300, seconds=60)  # candidate-limited -> deterministic
    ra, rb = run_one(ta, RunSpec("uniform"), bud, None), run_one(tb, RunSpec("uniform"), bud, None)
    det = ("outcome", "candidates", "invalid", "duplicates", "dominated", "states", "interpreter_calls", "program", "levels")
    assert {k: ra[k] for k in det} == {k: rb[k] for k in det}
    assert ra["hidden_correct"] != rb["hidden_correct"] or ra["outcome"] != "solved"  # labels only affect scoring


def test_oracle_property_random_tiny_tasks_solved_iff_oracle_finds_target():
    rng = np.random.default_rng(0)
    vocab = tiny_vocab()
    N = 5  # Concat(Slice(Lower(I0), 0, 1), Literal(".")) = 3 + 1 + 1 nodes
    for _ in range(12):
        rows = [("".join(rng.choice(list("abAB-"), size=rng.integers(2, 6))),) for _ in range(3)]
        prog = Concat(Slice(Lower(Input(0)), 0, 1), Literal("."))
        outs = [evaluate(prog, r) for r in rows]
        spec = spec_of(list(zip(rows, outs)))
        r = search(SearchTask(spec, vocab, Budgets(max_nodes=N, max_candidates=10**6, seconds=30)))
        assert r.solved and r.program_size <= ast_size_of(prog)
        assert all(evaluate(r.program, row) == o for row, o in zip(rows, outs))


def ast_size_of(p):
    return dsl.ast_size(p)
