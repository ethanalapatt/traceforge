"""Stage orchestration: generate -> train -> select -> evaluate -> analysis -> demo -> report.

Every stage is callable on its own (``traceforge generate|train|select|evaluate|analysis``) and
``reproduce`` chains them.  A run directory holds ``manifest.json`` (profile, environment, stage
status, completeness).  Resuming a directory whose profile differs raises ``ResumeError``.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import features as F
from .data.authored import authored_tasks
from .data.builtin import NORMALIZE_NAME_TASK
from .data.generator import GenConfig, dataset_fingerprint, generate_dataset, load_suite, save_suites
from .evaluation import (
    EvalPlan, ResultStore, ResumeError, RunSpec, RssSampler, budget_curves, run_plan, summarize, tune_lambda,
)
from .models import (
    choose_inference_device, load_neural, load_tree, model_path, resolve_device, save_neural, save_tree,
    update_neural_meta, versions,
)
from .policies import PolicyBook
from .profiles import load_profile
from .runtime import hardware_info
from .synth import synthesize
from .tasks import Budgets, EvalTask, TaskSpec, fingerprint, spec_from_dict
from .training import LabelConfig, RowSet, TrainConfig, build_rows, train_neural, train_tree

Log = Callable[[str], None]


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Run:
    """A run directory plus its manifest."""

    def __init__(self, out_dir: str | Path, profile: dict, device: str, resume: bool, log: Log | None = None) -> None:
        self.dir = Path(out_dir)
        self.profile = profile
        self.device_requested = device
        self.fp = fingerprint({"profile": profile, "device": device})
        self._log = log or (lambda m: None)
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            if not resume:
                raise ResumeError(
                    f"{self.dir} already contains a TraceForge run. Pass --resume to continue it, or choose a new --output."
                )
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get("fingerprint") != self.fp:
                raise ResumeError(
                    f"{self.manifest_path} was created with a different profile/device configuration "
                    f"({self.manifest.get('fingerprint')} != {self.fp}); refusing to mix results. Use a new --output."
                )
        else:
            if self.dir.exists() and any(self.dir.iterdir()):
                raise ResumeError(f"{self.dir} exists and is not an empty TraceForge run directory; choose a new --output.")
            self.dir.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "fingerprint": self.fp,
                "profile": profile,
                "device_requested": device,
                "created": _now(),
                "stages": {},
                "environment": {"versions": versions(), "hardware": hardware_info()},
            }
            self.save()

    def save(self) -> None:
        self.manifest["updated"] = _now()
        _atomic_write(self.manifest_path, json.dumps(self.manifest, indent=2, sort_keys=True, default=str))

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        self._log(line)
        with (self.dir / "log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def stage_done(self, name: str) -> bool:
        return bool(self.manifest["stages"].get(name, {}).get("done"))

    def mark(self, name: str, **info) -> None:
        self.manifest["stages"][name] = {"updated": _now(), **info}
        self.save()

    # paths
    @property
    def data(self) -> Path:
        return self.dir / "data"

    @property
    def models(self) -> Path:
        return self.dir / "models"

    @property
    def training(self) -> Path:
        return self.dir / "training"

    @property
    def results(self) -> Path:
        return self.dir / "results"

    def budgets(self, **over) -> Budgets:
        p = self.profile
        kw = dict(max_nodes=p["max_nodes"], max_candidates=p["max_candidates"], seconds=p["seconds"])
        kw.update(over)
        return Budgets(**kw)


# ----------------------------------------------------------------------------- stages


def stage_generate(run: Run, force: bool = False) -> dict:
    if run.stage_done("generate") and not force:
        run.log("generate: already done")
        return run.manifest["stages"]["generate"]
    p = run.profile
    cfg = GenConfig(
        seed=p["seed"],
        max_nodes=p["max_nodes"],
        train_max_size=p["train_max_size"],
        size_weights={int(k): v for k, v in p["size_weights"].items()},
        shift_enabled=p["shift_enabled"],
        quotas=dict(p["quotas"]),
    )
    t0 = time.perf_counter()
    suites, report = generate_dataset(cfg)
    if not report["complete"]:
        raise RuntimeError(f"generator could not fill the requested quotas: {report['counts']} vs {p['quotas']}")
    suites["authored"] = authored_tasks()
    report["counts"]["authored"] = len(suites["authored"])
    manifest = save_suites(run.data, suites, report)
    if not manifest["disjointness"]["ok"]:
        raise RuntimeError(f"split disjointness audit failed: {manifest['disjointness']}")
    fp = dataset_fingerprint(run.data, [s for s in suites])
    run.log(f"generate: {report['counts']} in {time.perf_counter() - t0:.1f}s, rejection rate {report['rejection_rate_among_built']:.2f}, fingerprint {fp}")
    run.mark("generate", done=True, counts=report["counts"], dataset_fingerprint=fp, seconds=time.perf_counter() - t0)
    return run.manifest["stages"]["generate"]


def _label_cfg(run: Run) -> LabelConfig:
    p = run.profile
    return LabelConfig(max_nodes=p["max_nodes"], seed=p["seed"], **p["label"])


def _load_rows(directory: Path, name: str) -> RowSet:
    X = np.load(directory / f"{name}_X.npy")
    y = np.load(directory / f"{name}_y.npy")
    t = np.load(directory / f"{name}_task.npy")
    s = np.load(directory / f"{name}_size.npy")
    g = json.loads((directory / f"{name}_groups.json").read_text())
    return RowSet(X, y, t, s, g)


def stage_train(run: Run, force: bool = False) -> dict:
    if run.stage_done("train") and not force:
        run.log("train: already done")
        return run.manifest["stages"]["train"]
    p = run.profile
    t_all = time.perf_counter()
    ds_fp = run.manifest["stages"]["generate"]["dataset_fingerprint"]
    lc = _label_cfg(run)
    report: dict = {"label_config": lc.__dict__, "dataset_fingerprint": ds_fp, "models": {}}
    rows: dict[str, RowSet] = {}
    for name in ("train", "val"):
        if (run.training / f"{name}_X.npy").exists() and not force:
            rows[name] = _load_rows(run.training, name)
            run.log(f"train: reusing cached {name} rows ({len(rows[name].y)})")
        else:
            t0 = time.perf_counter()
            tasks = load_suite(run.data, name)
            rows[name] = build_rows(tasks, lc, progress=run.log)
            rows[name].save(run.training, name)
            run.log(f"train: built {name} rows {rows[name].stats} in {time.perf_counter() - t0:.1f}s")
        report[f"{name}_rows"] = {
            "rows": int(len(rows[name].y)), "positives": int(rows[name].y.sum()),
            "tasks": len(rows[name].group), "groups": len(set(rows[name].group)),
        }
        if rows[name].stats:
            report[f"{name}_label_stats"] = rows[name].stats
    # group-level disjointness of the supervised rows
    inter = set(rows["train"].group) & set(rows["val"].group)
    if inter:
        raise RuntimeError(f"train/val row groups overlap: {sorted(inter)[:5]}")
    dev = resolve_device(run.device_requested)
    report["training_device"] = dev.to_dict()
    run.log(f"train: device {dev.device} {dev.notes}")
    base_meta = {"dataset_fingerprint": ds_fp, "profile": p["name"], "label_config": lc.__dict__}
    run.models.mkdir(parents=True, exist_ok=True)
    for seed in range(p["neural_seeds"]):
        for kind, drop in (("neural", []), ("neural_nosize", ["ast_size"])):
            if kind == "neural_nosize" and seed != 0:
                continue
            path = model_path(run.models, kind, seed)
            if path.exists() and not force:
                run.log(f"train: {path.name} exists, skipping")
                continue
            cfg = TrainConfig(epochs=p["epochs"], seed=seed, device=dev.device)
            run.log(f"train: {kind} seed {seed}")
            bundle, metrics = train_neural(rows["train"], rows["val"], cfg, drop_features=drop, meta=dict(base_meta, kind=kind), log=run.log)
            tmp = path.with_suffix(".tmp")
            save_neural(tmp, bundle, cfg.hidden)
            os.replace(tmp, path)
            report["models"][path.name] = metrics
            run.log(f"train: {path.name} val_pr_auc={metrics['val_pr_auc']:.3f} recall@20%={metrics['val_ranking_recall_top20']:.3f} size-only={metrics['size_only_val_pr_auc']:.3f}")
        tpath = model_path(run.models, "tree", seed)
        if not tpath.exists() or force:
            model, info = train_tree(rows["train"], rows["val"], seed=seed, log=run.log)
            tmp = tpath.with_suffix(".tmp")
            save_tree(tmp, model, dict(base_meta, kind="tree", seed=seed, tree_info=info))
            os.replace(tmp, tpath)
            report["models"][tpath.name] = info
    report["seconds"] = time.perf_counter() - t_all
    _atomic_write(run.training / "training_report.json", json.dumps(report, indent=2, sort_keys=True, default=str))
    run.mark("train", done=True, seconds=report["seconds"])
    return report


def _tune_tasks(run: Run) -> list[EvalTask]:
    return load_suite(run.data, "val")[: run.profile["tune_tasks"]]


def _analysis_tasks(run: Run) -> list[EvalTask]:
    p = run.profile
    t = load_suite(run.data, "val")[p["tune_tasks"] : p["tune_tasks"] + p["analysis_tasks"]]
    for x in t:
        x.suite = "analysis_val"
    return t


def stage_select(run: Run, force: bool = False) -> dict:
    """Choose penalty strengths (validation tasks only) and the inference device."""
    if run.stage_done("select") and not force:
        run.log("select: already done")
        return json.loads((run.models / "selection.json").read_text())
    p = run.profile
    t0 = time.perf_counter()
    val_rows = _load_rows(run.training, "val")
    probe = val_rows.X[:256]
    bundle0 = load_neural(model_path(run.models, "neural", 0), "cpu")
    inf = choose_inference_device(bundle0, probe, "cpu" if run.device_requested == "cpu" else run.device_requested)
    run.log(f"select: inference device {inf.device} {inf.notes}")
    book = PolicyBook(run.models, 0, inf.device)
    tune = _tune_tasks(run)
    budgets = run.budgets()
    tuning = {}
    for pol in ("heuristic", "tree", "neural"):
        tuning[pol] = tune_lambda(tune, pol, book, p["lambda_grid"], budgets, log=run.log)
    lambdas = {k: v["chosen"] for k, v in tuning.items()}
    sel = {"lambdas": lambdas, "inference": inf.to_dict(), "tuning": tuning, "tune_task_ids": [t.id for t in tune]}
    _atomic_write(run.models / "selection.json", json.dumps(sel, indent=2, sort_keys=True))
    for seed in range(p["neural_seeds"]):
        for kind in ("neural", "neural_nosize"):
            path = model_path(run.models, kind, seed)
            if path.exists():
                update_neural_meta(path, selected_lambda=lambdas["neural"], inference_device=inf.to_dict())
        tp = model_path(run.models, "tree", seed)
        if tp.exists():
            model, meta = load_tree(tp)
            meta["selected_lambda"] = lambdas["tree"]
            save_tree(tp, model, meta)
    run.log(f"select: lambdas {lambdas}")
    run.mark("select", done=True, lambdas=lambdas, inference=inf.to_dict(), seconds=time.perf_counter() - t0)
    return sel


def _selection(run: Run) -> dict:
    return json.loads((run.models / "selection.json").read_text())


def eval_suites(run: Run) -> dict[str, list[EvalTask]]:
    out: dict[str, list[EvalTask]] = {}
    authored = None
    for name, n in run.profile["eval_suites"].items():
        if name in ("authored", "regression"):
            if authored is None:
                authored = load_suite(run.data, "authored")
            pool = [t for t in authored if name == "authored" or t.meta.get("regression")]
            tasks = copy.deepcopy(pool[:n])
        else:
            tasks = load_suite(run.data, name)[:n]
        for t in tasks:
            t.suite = name
        out[name] = tasks
    return out


def _books(run: Run, device: str) -> dict[int, PolicyBook]:
    return {s: PolicyBook(run.models, s, device) for s in range(max(run.profile["neural_seeds"], 1))}


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(deadline - time.perf_counter(), 0.0)


def stage_evaluate(run: Run, deadline: float | None = None) -> dict:
    p = run.profile
    sel = _selection(run)
    lam = sel["lambdas"]
    seeds = range(p["neural_seeds"])
    runs = [RunSpec("uniform", 0, 0), RunSpec("heuristic", 0, lam["heuristic"])]
    runs += [RunSpec("tree", s, lam["tree"]) for s in seeds]
    runs += [RunSpec("neural", s, lam["neural"]) for s in seeds]
    suites = eval_suites(run)
    budgets = run.budgets()
    cfg = {
        "profile": p["name"], "budgets": budgets.to_dict(), "runs": [r.__dict__ for r in runs],
        "suites": {k: [t.id for t in v] for k, v in suites.items()}, "lambdas": lam,
        "inference": sel["inference"]["device"],
        "dataset_fingerprint": run.manifest["stages"]["generate"]["dataset_fingerprint"],
    }
    store = ResultStore(run.results, "main", cfg)
    sampler = RssSampler()
    plan = EvalPlan(suites, runs, budgets, 1, _remaining(deadline) if deadline is not None else p.get("max_total_seconds"))
    t0 = time.perf_counter()
    prog = run_plan(plan, store, _books(run, sel["inference"]["device"]), sampler, sel["inference"]["device"], log=run.log)
    sampler.close()
    store.write_csv()
    rows = store.all_rows()
    summary = summarize(rows)
    summary["completeness"] = prog
    summary["config"] = cfg
    _atomic_write(run.results / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
    _atomic_write(run.results / "curves.json", json.dumps({s: budget_curves(rows, s) for s in list(suites) + ["ALL"]}, sort_keys=True))
    run.mark("evaluate", done=not prog["partial"], partial=prog["partial"], completed=prog["completed"], expected=prog["expected"],
             seconds=time.perf_counter() - t0)
    run.log(f"evaluate: {prog}")
    return summary


def stage_analysis(run: Run, deadline: float | None = None) -> dict:
    """Controls on the validation-derived analysis suite + tiny no-memoization comparison."""
    p = run.profile
    sel = _selection(run)
    lam = sel["lambdas"]["neural"]
    tasks = _analysis_tasks(run)
    budgets = run.budgets()
    runs = [
        RunSpec("uniform", 0, 0),
        RunSpec("neural", 0, lam),
        RunSpec("neural_untrained", 0, lam),
        RunSpec("neural_shuffled", 0, lam),
        RunSpec("neural_nosize", 0, lam),
    ]
    cfg = {"profile": p["name"], "budgets": budgets.to_dict(), "runs": [r.__dict__ for r in runs], "tasks": [t.id for t in tasks], "lam": lam}
    store = ResultStore(run.results, "analysis", cfg)
    sampler = RssSampler()
    books = _books(run, sel["inference"]["device"])
    prog = run_plan(EvalPlan({"analysis_val": tasks}, runs, budgets, 1, _remaining(deadline)), store, books, sampler, sel["inference"]["device"], log=run.log)
    store.write_csv()
    summary = summarize(store.all_rows())
    summary["completeness"] = prog
    # tiny no-memoization comparison (uniform policy only, small cap so it stays cheap)
    nm_tasks = tasks[: p["nomemo_tasks"]]
    nm_budgets = run.budgets(max_candidates=p["nomemo_max_candidates"], seconds=min(p["seconds"], 2.0))
    nm_runs = [
        RunSpec("uniform", 0, 0, label="uniform_memo+prune"),
        RunSpec("uniform", 0, 0, prune_dominated=False, label="uniform_memo_only"),
        RunSpec("uniform", 0, 0, memo="none", label="uniform_nomemo"),
    ]
    nm_store = ResultStore(run.results, "nomemo", {"profile": p["name"], "budgets": nm_budgets.to_dict(), "tasks": [t.id for t in nm_tasks]})
    nm_prog = run_plan(EvalPlan({"analysis_val": nm_tasks}, nm_runs, nm_budgets, 1, _remaining(deadline)), nm_store, books, sampler, "cpu")
    sampler.close()
    nm_store.write_csv()
    summary["nomemo"] = summarize(nm_store.all_rows(), baseline="uniform_memo+prune")
    summary["nomemo"]["completeness"] = nm_prog
    _atomic_write(run.results / "analysis_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    run.mark("analysis", done=not (prog["partial"] or nm_prog["partial"]), partial=prog["partial"] or nm_prog["partial"])
    return summary


def stage_demo(run: Run) -> dict:
    """Synthesize the built-in normalize_name task with the trained neural checkpoint."""
    sel = _selection(run)
    spec = spec_from_dict(NORMALIZE_NAME_TASK)
    out = run.dir / "demo"
    res = synthesize(
        spec, "neural", str(model_path(run.models, "neural", 0)), run.budgets(max_nodes=max(4, min(run.profile["max_nodes"], 7))),
        out, lam=sel["lambdas"]["neural"], device=sel["inference"]["device"],
    )
    (out / "task.json").write_text(json.dumps(NORMALIZE_NAME_TASK, indent=2), encoding="utf-8")
    run.mark("demo", done=True, outcome=res["outcome"])
    return res


def reproduce(
    out_dir: str | Path,
    profile_name: str,
    device: str = "cpu",
    resume: bool = False,
    max_total_seconds: float | None = None,
    log: Log | None = None,
    make_report: bool = True,
) -> Run:
    profile = load_profile(profile_name)
    if max_total_seconds is not None:
        profile["max_total_seconds"] = max_total_seconds
    run = Run(out_dir, profile, device, resume, log)
    deadline = None if profile.get("max_total_seconds") is None else time.perf_counter() + profile["max_total_seconds"]
    stage_generate(run)
    stage_train(run)
    stage_select(run)
    stage_evaluate(run, deadline)
    stage_analysis(run, deadline)
    stage_demo(run)
    if make_report:
        from .reporting import build_report

        build_report(run.dir, run.dir / "report.html")
    return run
