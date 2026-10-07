"""Evaluation harness: identical frozen tasks for every policy, raw JSONL/CSV results,
resume, lambda selection on validation tasks, timeout-aware paired statistics.

The solver only ever receives a :class:`SearchTask` (spec + vocabulary + budgets).  Hidden rows
are read *after* a program has been selected.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import dsl
from .policies import PolicyBook
from .search import search
from .tasks import Budgets, EvalTask, fingerprint, make_search_task

ROW_KEY = ("suite", "task_id", "policy", "model_seed", "repeat")


class ResumeError(RuntimeError):
    """Raised when resuming a run whose saved configuration differs from the requested one."""


# ------------------------------------------------------------------- resource sampling


class RssSampler:
    """Background sampler of process RSS (unified memory on Apple silicon: MPS allocations are
    *part of* this figure, so they must not be added to it).  Sampling interval is explicit; peaks
    shorter than one interval can be missed."""

    def __init__(self, interval: float = 0.05) -> None:
        import psutil

        self.interval = interval
        self._proc = psutil.Process()
        self._lock = threading.Lock()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            r = self._proc.memory_info().rss
            with self._lock:
                self._peak = max(self._peak, r)
            self._stop.wait(self.interval)

    def reset(self) -> None:
        with self._lock:
            self._peak = self._proc.memory_info().rss

    def peak_mb(self) -> float:
        with self._lock:
            self._peak = max(self._peak, self._proc.memory_info().rss)
            return self._peak / 2**20

    def close(self) -> None:
        self._stop.set()


def mps_allocated_mb() -> float | None:
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**20
    except Exception:  # pragma: no cover - platform dependent
        pass
    return None


# ----------------------------------------------------------------------------- one run


@dataclass(frozen=True)
class RunSpec:
    policy: str
    model_seed: int = 0
    lam: int = 0
    repeat: int = 0
    memo: str = "semantic"
    prune_dominated: bool = True
    label: str = ""  # row label when it differs from the policy (e.g. "uniform_nomemo")


def score_hidden(program: dsl.Node | None, task: EvalTask, max_len: int) -> tuple[int, int]:
    """(number of hidden rows matched, number of hidden rows).  Called only after search."""
    if program is None:
        return 0, len(task.hidden)
    ok = sum(dsl.evaluate(program, h.inputs, max_len) == h.output for h in task.hidden)
    return ok, len(task.hidden)


def run_one(
    task: EvalTask,
    run: RunSpec,
    budgets: Budgets,
    book: PolicyBook | None,
    sampler: RssSampler | None = None,
    device: str = "cpu",
) -> dict:
    t_wall = time.perf_counter()
    scorer = book.get(run.policy) if (book is not None and run.policy != "uniform") else None
    if run.policy != "uniform" and scorer is None:
        raise ValueError(f"policy {run.policy!r} needs a PolicyBook")
    st = make_search_task(task.spec, budgets)  # the ONLY thing the solver receives
    if sampler:
        sampler.reset()
    try:
        res = search(st, scorer, run.lam if scorer is not None else 0, memo=run.memo, prune_dominated=run.prune_dominated)
    except MemoryError:  # pragma: no cover
        from .search.engine import Counters, SearchResult

        res = SearchResult("resource_limit", None, None, Counters(), time.perf_counter() - t_wall, policy=run.policy)
    rss = sampler.peak_mb() if sampler else None
    prog = res.program
    # independent verification with the interpreter (not the search's own bookkeeping)
    consistent = prog is not None and all(
        dsl.evaluate(prog, e.inputs, budgets.max_string_len) == e.output for e in task.spec.examples
    )
    if res.outcome == "solved" and not consistent:
        raise AssertionError(f"search returned an inconsistent program for {task.id}")
    ok, n_h = score_hidden(prog, task, budgets.max_string_len)
    c = res.counters
    sc = res.scoring or {}
    return {
        "suite": task.suite,
        "task_id": task.id,
        "family": task.family,
        "ref_size": task.meta.get("ref_size"),
        "policy": run.label or run.policy,
        "model_seed": run.model_seed,
        "repeat": run.repeat,
        "lam": run.lam if scorer is not None else 0,
        "memo": run.memo,
        "outcome": res.outcome,
        "solved_examples": bool(consistent),
        "solved_hidden": bool(consistent and ok == n_h),
        "hidden_correct": ok,
        "hidden_total": n_h,
        "accuracy": ok / n_h if n_h else None,
        "program": dsl.to_str(prog) if prog is not None else None,
        "program_json": dsl.canonical_json(prog) if prog is not None else None,
        "program_size": res.program_size,
        "candidates": c.candidates,
        "terminals": c.terminals,
        "invalid": c.invalid,
        "duplicates": c.duplicates,
        "dominated": c.dominated,
        "noop_skipped": c.noop_skipped,
        "interpreter_calls": c.interpreter_calls,
        "states": c.states,
        "levels": c.levels,
        "seconds": res.seconds,
        "wall_seconds": time.perf_counter() - t_wall,
        "model_calls": sc.get("model_calls", 0),
        "scored_rows": sc.get("rows", 0),
        "score_cache_hits": sc.get("cache_hits", 0),
        "featurize_s": sc.get("featurize_s", 0.0),
        "model_s": sc.get("model_s", 0.0),
        "scoring_s": sc.get("featurize_s", 0.0) + sc.get("model_s", 0.0),
        "inference_device": sc.get("device", "none"),
        "rss_peak_mb": rss,
        "mps_allocated_mb": mps_allocated_mb() if device == "mps" else None,
        "max_nodes": budgets.max_nodes,
        "cap_candidates": budgets.max_candidates,
        "cap_seconds": budgets.seconds,
        "milestones": _compact(res.milestones),
    }


def _compact(ms: list[list[float]], k: int = 24) -> list[list[float]]:
    if len(ms) <= k:
        return ms
    step = len(ms) / k
    return [ms[int(i * step)] for i in range(k)] + [ms[-1]]


# ------------------------------------------------------------------- results file + resume


class ResultStore:
    """Append-only JSONL with a config fingerprint; idempotent resume."""

    def __init__(self, directory: str | Path, name: str, config: dict) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.jsonl"
        self.meta_path = self.dir / f"{name}.meta.json"
        self.config = config
        self.fp = fingerprint(config)
        if self.meta_path.exists():
            saved = json.loads(self.meta_path.read_text())
            if saved.get("fingerprint") != self.fp:
                raise ResumeError(
                    f"{self.meta_path} was created with a different configuration "
                    f"({saved.get('fingerprint')} != {self.fp}); use a new --output directory or delete it"
                )
        else:
            self.meta_path.write_text(json.dumps({"fingerprint": self.fp, "config": config}, indent=2, sort_keys=True))
        self.rows: dict[tuple, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted write
                    self.rows.setdefault(tuple(r[k] for k in ROW_KEY), r)

    def done(self, key: tuple) -> bool:
        return key in self.rows

    def add(self, row: dict) -> None:
        key = tuple(row[k] for k in ROW_KEY)
        if key in self.rows:
            return
        self.rows[key] = row
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()

    def all_rows(self) -> list[dict]:
        return list(self.rows.values())

    def write_csv(self) -> Path:
        out = self.path.with_suffix(".csv")
        rows = self.all_rows()
        if not rows:
            out.write_text("")
            return out
        cols = [c for c in rows[0] if c != "milestones"]
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return out


@dataclass
class EvalPlan:
    suites: dict[str, list[EvalTask]]
    runs: list[RunSpec]
    budgets: Budgets
    repeats: int = 1
    max_total_seconds: float | None = None


def run_plan(
    plan: EvalPlan,
    store: ResultStore,
    books: dict[int, PolicyBook],
    sampler: RssSampler | None,
    device: str = "cpu",
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run every (suite, task, run) triple not already in ``store``.  Stops gracefully when the
    total time budget is exhausted and reports completed/expected counts (partial runs are marked
    partial by the caller)."""
    t0 = time.perf_counter()
    expected = sum(len(ts) for ts in plan.suites.values()) * len(plan.runs) * plan.repeats
    completed = new = 0
    stopped_early = False
    for suite, tasks in plan.suites.items():
        for task in tasks:
            for run in plan.runs:
                for rep in range(plan.repeats):
                    r = RunSpec(run.policy, run.model_seed, run.lam, rep, run.memo, run.prune_dominated, run.label)
                    key = (suite, task.id, r.label or r.policy, r.model_seed, r.repeat)
                    if store.done(key):
                        completed += 1
                        continue
                    if plan.max_total_seconds is not None and time.perf_counter() - t0 > plan.max_total_seconds:
                        stopped_early = True
                        break
                    row = run_one(task, r, plan.budgets, books.get(r.model_seed), sampler, device)
                    row["suite"] = suite
                    store.add(row)
                    completed += 1
                    new += 1
                    if log and new % 50 == 0:
                        log(f"  {completed}/{expected} runs ({time.perf_counter() - t0:.0f}s)")
                if stopped_early:
                    break
            if stopped_early:
                break
        if stopped_early:
            break
    return {
        "expected": expected,
        "completed": completed,
        "new_this_session": new,
        "partial": completed < expected,
        "stopped_early_by_time_budget": stopped_early,
    }


# ------------------------------------------------------------------ lambda selection


def tune_lambda(
    tasks: Sequence[EvalTask],
    policy: str,
    book: PolicyBook,
    grid: Sequence[int],
    budgets: Budgets,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Pick the penalty strength on *validation* tasks only.  Criterion: most tasks solved
    (example-consistent) within budget, then fewer mean candidates, then smaller lambda."""
    results = []
    for lam in grid:
        solved = 0
        cands = 0
        secs = 0.0
        for t in tasks:
            r = run_one(t, RunSpec(policy, book.seed, lam), budgets, book)
            solved += r["solved_examples"]
            cands += r["candidates"]
            secs += r["seconds"]
        results.append({"lam": lam, "solved": solved, "mean_candidates": cands / max(len(tasks), 1), "mean_seconds": secs / max(len(tasks), 1)})
        if log:
            log(f"  tune {policy} lam={lam}: solved {solved}/{len(tasks)}, mean cand {results[-1]['mean_candidates']:.0f}")
    best = min(results, key=lambda r: (-r["solved"], r["mean_candidates"], r["lam"]))
    return {"policy": policy, "chosen": best["lam"], "grid": results, "n_tasks": len(tasks)}


# ------------------------------------------------------------------------ statistics


def _by_task(rows: Iterable[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in rows:
        d.setdefault(r["task_id"], []).append(r)
    return d


def task_values(rows: list[dict], policy: str, suite: str | None, metric: Callable[[dict], float]) -> dict[str, float]:
    """Per-task value of ``metric`` for ``policy``, averaged over model seeds and timing repeats."""
    sel = [r for r in rows if r["policy"] == policy and (suite is None or r["suite"] == suite)]
    return {tid: float(np.mean([metric(r) for r in rs])) for tid, rs in _by_task(sel).items()}


def capped_seconds(r: dict) -> float:
    """Timeout-aware runtime: solved -> measured seconds; unsolved -> the wall-clock cap."""
    return r["seconds"] if r["outcome"] == "solved" else float(r["cap_seconds"])


def bootstrap_diff(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over tasks of mean(a - b)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    d = np.asarray([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"n": len(ids), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def summarize(rows: list[dict], baseline: str = "uniform") -> dict:
    """Machine-readable summary per suite and policy (plus 'ALL')."""
    out: dict = {"suites": {}, "baseline": baseline}
    suites = sorted({r["suite"] for r in rows}) + ["ALL"]
    for suite in suites:
        sr = rows if suite == "ALL" else [r for r in rows if r["suite"] == suite]
        policies = sorted({r["policy"] for r in sr})
        entry: dict = {"policies": {}, "paired_vs_baseline": {}, "joint": {}}
        task_ids = sorted({r["task_id"] for r in sr})
        for pol in policies:
            pr = [r for r in sr if r["policy"] == pol]
            seeds = sorted({r["model_seed"] for r in pr})
            per_seed = []
            for s in seeds:
                ps = [r for r in pr if r["model_seed"] == s]
                per_seed.append(
                    {
                        "model_seed": s,
                        "n_runs": len(ps),
                        "solved_examples": float(np.mean([r["solved_examples"] for r in ps])),
                        "solved_hidden": float(np.mean([r["solved_hidden"] for r in ps])),
                        "mean_accuracy": float(np.mean([r["accuracy"] or 0.0 for r in ps])),
                        "mean_candidates": float(np.mean([r["candidates"] for r in ps])),
                        "mean_seconds": float(np.mean([r["seconds"] for r in ps])),
                        "capped_mean_seconds": float(np.mean([capped_seconds(r) for r in ps])),
                    }
                )
            outcomes: dict[str, int] = {}
            for r in pr:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            keys = ["solved_examples", "solved_hidden", "mean_accuracy", "mean_candidates", "mean_seconds", "capped_mean_seconds"]
            entry["policies"][pol] = {
                "n_tasks": len({r["task_id"] for r in pr}),
                "n_runs": len(pr),
                "per_seed": per_seed,
                "mean_over_seeds": {k: float(np.mean([p[k] for p in per_seed])) for k in keys},
                "sd_over_seeds": {k: float(np.std([p[k] for p in per_seed])) for k in keys} if len(per_seed) > 1 else None,
                "outcomes": outcomes,
                "mean_scoring_s": float(np.mean([r["scoring_s"] for r in pr])),
                "mean_model_calls": float(np.mean([r["model_calls"] for r in pr])),
                "peak_rss_mb": max((r["rss_peak_mb"] or 0.0) for r in pr),
            }
        if baseline in policies:
            for pol in policies:
                if pol == baseline:
                    continue
                entry["paired_vs_baseline"][pol] = {
                    "solved_examples": bootstrap_diff(
                        task_values(sr, pol, None, lambda r: r["solved_examples"]),
                        task_values(sr, baseline, None, lambda r: r["solved_examples"]),
                    ),
                    "solved_hidden": bootstrap_diff(
                        task_values(sr, pol, None, lambda r: r["solved_hidden"]),
                        task_values(sr, baseline, None, lambda r: r["solved_hidden"]),
                    ),
                    "candidates": bootstrap_diff(
                        task_values(sr, pol, None, lambda r: r["candidates"]),
                        task_values(sr, baseline, None, lambda r: r["candidates"]),
                    ),
                    "capped_seconds": bootstrap_diff(
                        task_values(sr, pol, None, capped_seconds),
                        task_values(sr, baseline, None, capped_seconds),
                    ),
                }
        # jointly solved latency (shows how many tasks the comparison excludes)
        solved_sets = {}
        for pol in policies:
            solved_sets[pol] = {
                tid for tid in task_ids
                if all(r["solved_examples"] for r in sr if r["policy"] == pol and r["task_id"] == tid)
                and any(r["policy"] == pol and r["task_id"] == tid for r in sr)
            }
        if baseline in solved_sets:
            for pol in policies:
                if pol == baseline:
                    continue
                joint = solved_sets[pol] & solved_sets[baseline]
                if joint:
                    a = task_values(sr, pol, None, lambda r: r["seconds"])
                    b = task_values(sr, baseline, None, lambda r: r["seconds"])
                    ca = task_values(sr, pol, None, lambda r: r["candidates"])
                    cb = task_values(sr, baseline, None, lambda r: r["candidates"])
                    entry["joint"][pol] = {
                        "n_joint": len(joint),
                        "n_tasks": len(task_ids),
                        "excluded": len(task_ids) - len(joint),
                        "mean_seconds": float(np.mean([a[t] for t in joint])),
                        "baseline_mean_seconds": float(np.mean([b[t] for t in joint])),
                        "mean_candidates": float(np.mean([ca[t] for t in joint])),
                        "baseline_mean_candidates": float(np.mean([cb[t] for t in joint])),
                    }
        out["suites"][suite] = entry
    return out


def budget_curves(rows: list[dict], suite: str | None = None, n_points: int = 40) -> dict:
    """Solve-rate vs candidate budget and vs wall-clock budget, derived from single runs:
    a run solved at ``a`` attempts / ``t`` seconds counts as solved for every budget >= a / t."""
    sr = [r for r in rows if suite is None or suite == "ALL" or r["suite"] == suite]
    pols = sorted({r["policy"] for r in sr})
    if not sr:
        return {}
    cmax = max(r["cap_candidates"] for r in sr)
    tmax = max(r["cap_seconds"] for r in sr)
    cgrid = np.unique(np.geomspace(1, cmax, n_points).astype(int))
    tgrid = np.geomspace(1e-3, tmax, n_points)
    out = {"candidate_budgets": cgrid.tolist(), "time_budgets": tgrid.tolist(), "by_policy": {}}
    for p in pols:
        pr = [r for r in sr if r["policy"] == p]
        sc = np.asarray([r["candidates"] if r["solved_examples"] else np.inf for r in pr])
        st = np.asarray([r["seconds"] if r["solved_examples"] else np.inf for r in pr])
        out["by_policy"][p] = {
            "solve_rate_vs_candidates": [float((sc <= b).mean()) for b in cgrid],
            "solve_rate_vs_time": [float((st <= b).mean()) for b in tgrid],
        }
    return out
