# TraceForge

A local, offline, neural-guided **program synthesis engine for string transformations**.
Give it a few input → output examples; it searches a small typed DSL for a program that
reproduces them, using a bottom-up enumerator whose priorities come from a small neural model
(or a tree model, or a hand-set heuristic) trained on your own machine. No LLMs, no cloud, no
network after `pip install`.

> **What this is not.** Learning-guided bottom-up synthesis is established work
> (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): BUSTLE, CrossBeam, FlashFill lineage).
> TraceForge is a from-scratch, laptop-scale implementation plus an honest evaluation of *when
> guidance pays for its own overhead*. It does not claim a new algorithm. A returned program is
> **example-consistent**, not proven correct: several programs can fit the same examples.

## Quick start

Requires Python ≥ 3.11 (developed and tested on **3.13 only**), macOS or Linux.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'        # numpy, torch, scikit-learn, matplotlib, psutil (+ pytest, hypothesis)

traceforge doctor              # environment check (reports MPS availability honestly)
pytest                         # 76 tests, ~20 s on an M3

# 90-second demo: no model needed
traceforge synthesize examples/normalize_name.json --policy uniform --output out/demo
traceforge apply out/demo/program.json --input examples/people.csv --columns full_name --output out/people_clean.csv

# train + evaluate everything on this machine, offline (about 10 s on an M3)
traceforge reproduce --profile smoke --device cpu --offline --output runs/smoke
open runs/smoke/report.html    # standalone offline report
```

Use a trained checkpoint (the neural policy never silently falls back to an untrained model):

```bash
traceforge synthesize examples/email_handle.json --policy neural \
    --checkpoint runs/dev/models/neural.pt --output out/email   # λ comes from the checkpoint
```

## Commands

| command | purpose |
|---|---|
| `traceforge doctor [--json]` | versions, CPU/RAM, torch CPU op, optional MPS check |
| `traceforge reproduce --profile smoke\|dev\|full --device cpu\|mps\|auto [--offline] --output DIR [--resume] [--max-total-seconds N]` | generate → train → select → evaluate → controls → demo → report |
| `traceforge generate / train / select / evaluate / analysis / demo --output DIR` | the same stages, individually (run dir must exist) |
| `traceforge synthesize TASK.json --policy uniform\|heuristic\|tree\|neural [--checkpoint F] [--lam N] [--max-nodes N] [--seconds S] --output DIR` | find a program; writes `program.json`, `program.txt`, `result.json`, `trace.json`, `vocabulary.json` |
| `traceforge apply PROGRAM.json --input F.csv --columns COL[,COL] --output OUT.csv [--force] [--strict]` | apply to a CSV; never overwrites the input; per-row errors in `traceforge_error` |
| `traceforge report RUN_OR_SYNTH_DIR --output report.html` | standalone offline HTML (embedded PNG plots, no CDN, no fetch) |
| `traceforge validate-splits --data DIR`, `traceforge profiles --dump configs/` | audits / profile export |

Task file format (`examples/*.json`): `{"schema_version":1,"id":..., "input_columns":[...], "examples":[{"inputs":[...],"output":"..."}]}`
with 3–6 examples and one or two input columns.

## Profiles

| | smoke | dev | full |
|---|---|---|---|
| max AST nodes / train max | 5 / 4 | 7 / 5 | 9 / 7 |
| train / val / dev / test_id / comp_dev / comp_test tasks | 96 / 36 / 6 / – / – / – | 1500 / 300 / 30 / – / 15 / – | 6000 / 800 / 60 / 100 / 30 / 60 |
| eval suites | dev 6 + regression 6 | dev 30 + comp_dev 15 + authored 15 | test_id 100 + comp_test 60 + authored 40 |
| per-task budget | 1 s, 5k candidates | 3 s, 25k | 5 s, 100k |
| neural seeds | 1 | 1 | 3 |

These are starting configurations, not promised runtimes. Full command (not executed in the
build log; see [docs/experiments.md](docs/experiments.md) for what was and was not run):

```bash
traceforge reproduce --profile full --device auto --offline --output runs/full --max-total-seconds 21600
```

## Layout

```
src/traceforge/
  dsl.py            nodes, interpreter, JSON schema, pretty printer, AST sizes
  vocabulary.py     finite per-task parameter vocabulary built only from supplied examples
  tasks.py          TaskSpec / SearchTask / EvalTask (hidden labels live ONLY in EvalTask)
  search/engine.py  bottom-up enumerator: integer priority costs, semantic memo, budgets, traces
  features.py       execution-derived feature schema (v1)
  models.py         MLP / tree checkpoints, device policy (cpu/mps/auto), schema checks
  policies.py       uniform | heuristic | tree | neural + controls
  training.py       weak labels, grouped validation, PR-AUC, training loops
  data/             13 synthetic input families, grouped generator, 40 hand-authored tasks
  evaluation.py     evaluator-side scoring, resumable result store, bootstrap, curves
  pipeline.py       run directories, stages, manifest/fingerprint resume
  reporting.py      standalone HTML report      synth.py   synthesize + apply
tests/              76 tests (pytest + hypothesis)
docs/               architecture, experiments, interview guide, resume evidence, checklist, demo script
configs/            profile JSON exported from code
examples/           task files and a sample CSV
```

## Docs

* [docs/architecture.md](docs/architecture.md) — DSL semantics, memoization, the scheduling-invariant proof, labels, features, leakage boundary, limits
* [docs/experiments.md](docs/experiments.md) — what was run, results tables, caveats
* [docs/interview-guide.md](docs/interview-guide.md) — ~12 questions tied to code and measured results
* [docs/resume-evidence.md](docs/resume-evidence.md) — every number a resume bullet may use, with its source file
* [docs/implementation-checklist.md](docs/implementation-checklist.md), [docs/demo-script.md](docs/demo-script.md)

## Known limitations (short list)

* Python 3.13 only was tested. MPS is optional and only claimed where a run actually used it (see experiments).
* The benchmark is synthetic plus 40 hand-authored tasks; results do not transfer automatically to other string-transformation suites.
* Example-consistency ≠ correctness; hidden-row accuracy is reported separately and is much lower on ambiguous tasks.
* Dominance pruning is an engineering addition on top of the (outputs, size) memo; it is applied to every policy and ablated.
* The tree checkpoint is a joblib pickle: load only files you trained. Neural checkpoints load with `weights_only=True`.
