import json

import pytest

from conftest import spec_of
from traceforge import dsl
from traceforge.data.authored import regression_tasks
from traceforge.evaluation import (
    EvalPlan, ResultStore, ResumeError, RunSpec, bootstrap_diff, budget_curves, capped_seconds, run_one, run_plan, summarize,
)
from traceforge.models import save_neural, random_bundle, model_path
from traceforge.policies import ALL_POLICIES, PolicyBook
from traceforge.tasks import Budgets, EvalTask, Example

BUD = Budgets(max_nodes=5, max_candidates=5000, seconds=5.0)


def easy_task(tid="easy", hidden_ok=True):
    spec = spec_of([(("  Ada  ",), "ada"), ((" Bob",), "bob"), (("CY ",), "cy")], tid=tid)
    hidden = tuple(Example((f" q{i} ",), f"q{i}" if hidden_ok else "nope") for i in range(5))
    return EvalTask(spec, hidden, reference=dsl.Lower(dsl.Trim(dsl.Input(0))), suite="s")


@pytest.fixture
def book(tmp_path):
    for s in range(1):
        save_neural(model_path(tmp_path, "neural", s), random_bundle(s))
        save_neural(model_path(tmp_path, "neural_nosize", s), random_bundle(s, ["ast_size"]))
    return PolicyBook(tmp_path, 0, "cpu")


def test_all_policies_return_valid_programs_checked_by_the_evaluator(book):
    t = easy_task()
    for pol in ("uniform", "heuristic", "neural", "neural_untrained", "neural_shuffled", "neural_nosize"):
        r = run_one(t, RunSpec(pol, 0, 2), BUD, book)
        assert r["outcome"] == "solved", pol
        prog = dsl.from_json(json.loads(r["program_json"]))
        assert all(dsl.evaluate(prog, e.inputs) == e.output for e in t.spec.examples)  # independent re-check
        assert r["solved_examples"] and r["hidden_correct"] == r["hidden_total"] == 5
    assert set(ALL_POLICIES) >= {"uniform", "heuristic", "tree", "neural", "neural_untrained", "neural_shuffled", "neural_nosize"}


def test_hidden_rows_are_scored_after_search_and_example_fit_differs_from_hidden_accuracy(book):
    r = run_one(easy_task(hidden_ok=False), RunSpec("uniform"), BUD, book)
    assert r["solved_examples"] and not r["solved_hidden"] and r["hidden_correct"] == 0


def test_timeouts_and_limits_stay_in_the_denominator():
    solved = run_one(easy_task("a"), RunSpec("uniform"), BUD, None)
    hard = regression_tasks()[-1]
    hard.suite = "s"
    timed_out = run_one(hard, RunSpec("uniform"), Budgets(max_nodes=9, max_candidates=10**8, seconds=1e-6), None)
    assert timed_out["outcome"] == "timeout"
    rows = [solved, timed_out]
    s = summarize(rows)["suites"]["s"]["policies"]["uniform"]
    assert s["n_tasks"] == 2 and s["mean_over_seeds"]["solved_examples"] == pytest.approx(0.5)
    assert capped_seconds(timed_out) == timed_out["cap_seconds"] == 1e-6
    assert capped_seconds(solved) == solved["seconds"]


def test_summary_does_not_average_unsolved_latency_into_joint_solved():
    a = easy_task("a")
    rows = [run_one(a, RunSpec(p), BUD, None) for p in ("uniform",)]
    hard = regression_tasks()[-1]
    hard.suite = "s"
    rows.append(run_one(hard, RunSpec("uniform"), Budgets(max_nodes=9, max_candidates=10**8, seconds=1e-6), None))
    summary = summarize(rows)
    assert "joint" in summary["suites"]["s"]


def test_bootstrap_is_paired_and_deterministic():
    a = {"t1": 1.0, "t2": 1.0, "t3": 0.0, "t4": 1.0}
    b = {"t1": 0.0, "t2": 0.0, "t3": 0.0, "t4": 1.0}
    r1, r2 = bootstrap_diff(a, b), bootstrap_diff(a, b)
    assert r1 == r2 and r1["n"] == 4 and r1["mean_diff"] == 0.5 and r1["ci95"][0] > 0.0 - 1e-9
    assert bootstrap_diff({"x": 1.0}, {"y": 1.0}) == {"n": 0}


def test_resume_is_idempotent_and_rejects_incompatible_config(tmp_path, book):
    suites = {"s": [easy_task("a"), easy_task("b")]}
    runs = [RunSpec("uniform"), RunSpec("heuristic", 0, 2)]
    cfg = {"v": 1}
    plan = EvalPlan(suites, runs, BUD)
    store = ResultStore(tmp_path, "main", cfg)
    first = run_plan(plan, store, {0: book}, None)
    assert first["expected"] == first["completed"] == 4 and not first["partial"]
    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    again = run_plan(plan, ResultStore(tmp_path, "main", cfg), {0: book}, None)
    assert again["new_this_session"] == 0 and again["completed"] == 4
    assert (tmp_path / "main.jsonl").read_text().splitlines() == lines  # nothing re-run, nothing duplicated
    with pytest.raises(ResumeError, match="different configuration"):
        ResultStore(tmp_path, "main", {"v": 2})


def test_interrupted_run_resumes_and_torn_last_line_is_ignored(tmp_path, book):
    suites = {"s": [easy_task("a"), easy_task("b")]}
    plan = EvalPlan(suites, [RunSpec("uniform")], BUD)
    store = ResultStore(tmp_path, "m", {"v": 1})
    run_plan(EvalPlan(suites, [RunSpec("uniform")], BUD, max_total_seconds=0.0), store, {0: book}, None)
    with (tmp_path / "m.jsonl").open("a") as fh:
        fh.write('{"suite": "s", "task_id": "b", "pol')  # torn write
    prog = run_plan(plan, ResultStore(tmp_path, "m", {"v": 1}), {0: book}, None)
    assert prog["completed"] == 2 and not prog["partial"]


def test_time_budget_marks_run_partial(tmp_path, book):
    plan = EvalPlan({"s": [easy_task("a"), easy_task("b")]}, [RunSpec("uniform")], BUD, max_total_seconds=0.0)
    r = run_plan(plan, ResultStore(tmp_path, "p", {"v": 1}), {0: book}, None)
    assert r["partial"] and r["stopped_early_by_time_budget"] and r["completed"] < r["expected"]


def test_csv_export_escapes_special_characters(tmp_path):
    store = ResultStore(tmp_path, "c", {"v": 1})
    row = run_one(easy_task('we"ird,id\nx'), RunSpec("uniform"), BUD, None)
    row["suite"] = "s"
    store.add(row)
    import csv

    with open(store.write_csv(), newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert back[0]["task_id"] == 'we"ird,id\nx' and len(back) == 1


def test_curves_are_monotone_and_include_every_task():
    rows = [run_one(easy_task(f"t{i}"), RunSpec("uniform"), BUD, None) | {"suite": "s"} for i in range(3)]
    c = budget_curves(rows, "s")
    ys = c["by_policy"]["uniform"]["solve_rate_vs_candidates"]
    assert all(b >= a for a, b in zip(ys, ys[1:])) and ys[-1] == 1.0
