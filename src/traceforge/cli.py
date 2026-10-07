"""``traceforge`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print(msg: str) -> None:
    print(msg, flush=True)


def _maybe_offline(args: argparse.Namespace) -> None:
    if getattr(args, "offline", False):
        from .offline import block_network

        block_network()
        _print("offline mode: outbound network connections are blocked for this process")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .runtime import doctor

    report, ok = doctor()
    if args.json:
        _print(json.dumps(report, indent=2))
    else:
        for c in report["checks"]:
            _print(f"[{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        hw = report["hardware"]
        _print(f"hardware: {hw.get('cpu_brand') or hw['machine']}, {hw['cpu_count_logical']} logical CPUs, {hw['memory_gib']} GiB")
        _print("all required checks passed" if ok else "some required checks FAILED")
    return 0 if ok else 1


def _open_run(path: str):
    from .pipeline import Run

    mp = Path(path) / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{path} is not a TraceForge run directory (no manifest.json); run `traceforge generate` or `reproduce` first")
    m = json.loads(mp.read_text())
    return Run(path, m["profile"], m["device_requested"], resume=True, log=_print)


def cmd_generate(args: argparse.Namespace) -> int:
    from .pipeline import Run, stage_generate
    from .profiles import load_profile

    run = Run(args.output, load_profile(args.profile), args.device, args.resume, _print)
    stage_generate(run, force=args.force)
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    from . import pipeline as P

    _maybe_offline(args)
    run = _open_run(args.output)
    {"train": lambda: P.stage_train(run, args.force), "select": lambda: P.stage_select(run, args.force),
     "evaluate": lambda: P.stage_evaluate(run), "analysis": lambda: P.stage_analysis(run), "demo": lambda: P.stage_demo(run)}[args.cmd]()
    return 0


def _print_summary(run) -> None:
    summary = json.loads((run.results / "summary.json").read_text())
    _print("\nResults (mean over tasks; 'fit' = matches all supplied examples, 'hidden' = also matches every hidden row)")
    for suite, entry in summary["suites"].items():
        if suite == "ALL" and len(summary["suites"]) <= 2:
            continue
        _print(f"\n  suite {suite}")
        _print(f"  {'policy':<10} {'tasks':>5} {'fit':>6} {'hidden':>7} {'mean cand':>10} {'mean s':>8} {'capped s':>9}")
        for pol, d in entry["policies"].items():
            m = d["mean_over_seeds"]
            _print(f"  {pol:<10} {d['n_tasks']:>5} {m['solved_examples']:>6.3f} {m['solved_hidden']:>7.3f} {m['mean_candidates']:>10.0f} {m['mean_seconds']:>8.4f} {m['capped_mean_seconds']:>9.4f}")


def cmd_reproduce(args: argparse.Namespace) -> int:
    from .pipeline import reproduce

    _maybe_offline(args)
    run = reproduce(args.output, args.profile, args.device, args.resume, args.max_total_seconds, _print, make_report=True)
    _print_summary(run)
    prog = run.manifest["stages"].get("evaluate", {})
    _print(f"\nevaluation completeness: {prog.get('completed')}/{prog.get('expected')} runs, partial={prog.get('partial')}")
    _print(f"report: {run.dir / 'report.html'}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    from .models import CheckpointError
    from .synth import synthesize
    from .tasks import Budgets, InvalidTask, load_task_file

    _maybe_offline(args)
    try:
        spec = load_task_file(args.task)
    except InvalidTask as e:
        _print(f"invalid task: {e}")
        return 2
    budgets = Budgets(max_nodes=args.max_nodes, max_candidates=args.max_candidates, seconds=args.seconds)
    try:
        res = synthesize(spec, args.policy, args.checkpoint, budgets, args.output, lam=args.lam, device=args.device)
    except CheckpointError as e:
        _print(f"error: {e}")
        return 2
    s = res["search"]
    c = s["counters"]
    _print(f"outcome: {res['outcome']}  policy={s['policy']} lambda={s['lam']}  candidates={c['candidates']}  states={c['states']}  {s['seconds']:.3f}s")
    if res["outcome"] != "solved":
        _print("no program matched every supplied example within the budgets (this does not mean none exists: it only means the finite grammar/budget was not enough)")
        return 1
    prog = res["program"]
    _print(f"program ({prog['program_size']} AST nodes):\n{prog['program_text']}")
    _print(f"note: {prog['note']}")
    _print(f"wrote {Path(args.output) / 'program.json'}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from .synth import ApplyError, apply_program

    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    try:
        r = apply_program(args.program, args.input, cols, args.output, args.output_column, args.error_column, args.force)
    except ApplyError as e:
        _print(f"error: {e}")
        return 2
    _print(f"wrote {r['output']}: {r['rows']} rows, {r['errors']} row-level error(s)")
    for e in r["error_rows"][:10]:
        _print(f"  line {e['line']}: {e['error']}")
    return 2 if (args.strict and r["errors"]) else 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting import build_report

    out = build_report(args.run_dir, args.output)
    _print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, standalone; open it directly from disk)")
    return 0


def cmd_validate_splits(args: argparse.Namespace) -> int:
    from .data.generator import SUITES, check_split_disjointness, load_suite

    suites = {}
    for n in SUITES:
        if (Path(args.data) / f"{n}.jsonl").exists():
            suites[n] = load_suite(args.data, n)
    audit = check_split_disjointness(suites)
    _print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in audit.items()}, indent=2))
    return 0 if audit["ok"] else 1


def cmd_profiles(args: argparse.Namespace) -> int:
    from .profiles import dump_profiles

    dump_profiles(args.dump)
    _print(f"wrote profile JSON files to {args.dump}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="traceforge", description="Local neural-guided program synthesis for string transformations.")
    p.add_argument("--version", action="version", version=f"traceforge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check the environment")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_doctor)

    g = sub.add_parser("generate", help="stage: generate grouped datasets into a run directory")
    g.add_argument("--profile", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    g.add_argument("--resume", action="store_true")
    g.add_argument("--force", action="store_true")
    g.set_defaults(fn=cmd_generate)

    for name, hlp in (("train", "stage: build labels, train neural/tree/ablation models"), ("select", "stage: choose penalties on validation tasks"),
                      ("evaluate", "stage: run the four-policy benchmark (resumable)"), ("analysis", "stage: controls and no-memoization comparison"),
                      ("demo", "stage: synthesize the built-in demo with the trained checkpoint")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--output", required=True, help="existing run directory")
        s.add_argument("--force", action="store_true")
        s.add_argument("--offline", action="store_true")
        s.set_defaults(fn=cmd_stage)

    r = sub.add_parser("reproduce", help="generate -> train -> select -> evaluate -> analysis -> demo -> report")
    r.add_argument("--profile", required=True, help="smoke | dev | full | path/to/profile.json")
    r.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    r.add_argument("--offline", action="store_true", help="block all outbound network access")
    r.add_argument("--output", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--max-total-seconds", type=float, default=None, help="graceful total time budget; the run is marked partial if exceeded")
    r.set_defaults(fn=cmd_reproduce)

    y = sub.add_parser("synthesize", help="find a program from a task JSON file")
    y.add_argument("task")
    y.add_argument("--policy", default="uniform", choices=["uniform", "heuristic", "tree", "neural"])
    y.add_argument("--checkpoint", default=None)
    y.add_argument("--lam", type=int, default=None, help="penalty strength (default: the checkpoint's validation-selected value)")
    y.add_argument("--max-nodes", type=int, default=7)
    y.add_argument("--max-candidates", type=int, default=100_000)
    y.add_argument("--seconds", type=float, default=5.0)
    y.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"], help="inference device for the neural policy")
    y.add_argument("--output", required=True)
    y.add_argument("--offline", action="store_true")
    y.set_defaults(fn=cmd_synthesize)

    a = sub.add_parser("apply", help="apply a synthesized program to CSV rows (writes a new file)")
    a.add_argument("program")
    a.add_argument("--input", required=True)
    a.add_argument("--columns", required=True, help="comma-separated CSV column(s) mapped to Input(0), Input(1)")
    a.add_argument("--output", required=True)
    a.add_argument("--output-column", default="traceforge_output")
    a.add_argument("--error-column", default="traceforge_error")
    a.add_argument("--force", action="store_true", help="replace an existing output file")
    a.add_argument("--strict", action="store_true", help="exit non-zero if any row failed")
    a.set_defaults(fn=cmd_apply)

    rp = sub.add_parser("report", help="build the standalone offline HTML report")
    rp.add_argument("run_dir")
    rp.add_argument("--output", required=True)
    rp.set_defaults(fn=cmd_report)

    v = sub.add_parser("validate-splits", help="audit group/exact/behavioural disjointness of a data directory")
    v.add_argument("--data", required=True)
    v.set_defaults(fn=cmd_validate_splits)

    pr = sub.add_parser("profiles", help="write the built-in profile JSON files")
    pr.add_argument("--dump", required=True)
    pr.set_defaults(fn=cmd_profiles)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as e:  # noqa: BLE001 - user-facing error boundary
        from .evaluation import ResumeError
        from .models import CheckpointError
        from .offline import NetworkBlockedError

        if isinstance(e, (ResumeError, CheckpointError, NetworkBlockedError)):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
