# Experiments

All numbers below are copied from files under `runs/dev/` and `runs/smoke/` produced by the
commands in §1 on one machine (Apple M3, 8 logical CPUs, 16 GiB, macOS, Python 3.13.7,
torch 2.14.1, numpy 2.5.3, scikit-learn 1.9.1 — from `manifest.json`). Re-running regenerates the
same dataset fingerprint and the same candidate counts/programs (checked: a second dev run
reproduced all 240 result rows' candidates, outcomes and programs exactly); wall-clock seconds vary.

## 1. What was and was not executed

| item | status |
|---|---|
| `pytest` (76 tests, incl. an offline CLI smoke reproduction) | **executed**, all pass (~18 s) |
| `traceforge reproduce --profile smoke --device cpu --offline --output runs/smoke` | **executed**, 48/48 runs |
| `traceforge reproduce --profile dev --device auto --offline --output runs/dev` | **executed**, 240/240 evaluation runs, no partial stage |
| `traceforge reproduce --profile full --device auto --offline --output runs/full --max-total-seconds 21600` | **NOT executed.** No full-profile number appears anywhere in this repository. |
| `synthesize` + `apply` on `examples/normalize_name.json` / `examples/people.csv` | **executed** (and in `test_cli.py`) |
| MPS | **Used for training** in the dev run on this M3 (`training_report.json`: device `mps`). Inference device was chosen by micro-benchmark: **CPU** (0.18 ms vs 1.09 ms per 256-row batch). No other MPS claim is made. |
| Python versions ≠ 3.13, Linux, Intel Mac | not tested |
| Optional SyGuS importer | not implemented |

A first dev run on this machine was slowed to a crawl by the OS suspending the process (CPU
time ≈ 2 % of wall time). It was discarded and replaced by a rerun under `caffeinate`; the
discarded run's rows were identical in candidates/outcomes, and contained no timeouts, so the
stall did not alter results — but only the rerun is reported.

## 2. Dev profile setup

* Tasks: train 1500 (261 program groups), val 300 (100 groups), dev 30 (10 groups), comp_dev 15
  (13 groups, longer programs and held-out operator bigrams), authored 15 of the 40
  hand-written tasks. Generator rejection rate among built candidates: 0.81
  (top reasons: dead node, no-op concat, invalid on supplied, constant output).
* Budgets per task: ≤ 7 AST nodes, 25 000 candidate compositions, 3 s wall clock; identical for all policies and vocabulary-identical.
* Models: MLP 256-128, **62 977 parameters**, trained on weak labels (12 238 train rows / 2 158 positives; val 3 278 / 579) on MPS; ExtraTrees baseline; one neural seed.
* Penalty λ chosen on 60 validation tasks only: heuristic 6, tree 6, neural 3.
* Validation ranking quality (PR-AUC; positive rate ≈ 0.18): **neural 0.670, tree 0.653, neural without AST-size 0.649, AST-size alone 0.181** (≈ the positive rate, i.e. size alone is at chance level). Recall of useful states in the top 20 % of scores: 0.79 / 0.80 / 0.79.

## 3. Main results (single neural seed, single repeat)

`fit` = fraction of tasks whose returned program matches every supplied example;
`hidden` = also matches every hidden row. `capped s` charges unsolved tasks their full time cap.
Source: `runs/dev/results/summary.json`.

**dev (30 tasks, 10 groups)**

| policy | fit | hidden | mean candidates | mean s | capped mean s |
|---|---|---|---|---|---|
| uniform | 0.800 | 0.533 | 11 495 | 0.021 | 0.612 |
| heuristic | 1.000 | 0.700 | 2 472 | 0.019 | 0.019 |
| tree | 1.000 | 0.700 | 1 646 | 0.037 | 0.037 |
| neural | 0.967 | 0.667 | 3 032 | 0.023 | 0.116 |

**comp_dev (15 tasks, 13 groups; compositional shift)**

| policy | fit | hidden | mean candidates | mean s | capped mean s |
|---|---|---|---|---|---|
| uniform | 0.333 | 0.133 | 19 711 | 0.033 | 2.005 |
| heuristic | 0.733 | 0.333 | 13 356 | 0.088 | 0.841 |
| tree | 0.600 | 0.333 | 12 771 | 0.227 | 1.252 |
| neural | 0.600 | 0.333 | 13 218 | 0.098 | 1.221 |

**authored (15 hand-written tasks)**

| policy | fit | hidden | mean candidates | mean s | capped mean s |
|---|---|---|---|---|---|
| uniform | 0.933 | 0.867 | 2 606 | 0.0044 | 0.202 |
| heuristic | 1.000 | 0.933 | 1 258 | 0.0100 | 0.010 |
| tree | 0.933 | 0.867 | 1 775 | 0.0331 | 0.204 |
| neural | 0.933 | 0.867 | 1 828 | 0.0193 | 0.202 |

**All 60 tasks pooled** (not an independent suite): fit uniform 0.717, heuristic 0.933, tree 0.883, neural 0.867; mean candidates 11 327 / 4 890 / 4 460 / 5 278.

Paired bootstrap over tasks (2000 resamples; policy − uniform; 95 % interval), pooled 60 tasks:

| policy | Δ fit | Δ hidden | Δ mean candidates |
|---|---|---|---|
| heuristic | +0.217 [+0.117, +0.317] | +0.150 [+0.050, +0.250] | −6 437 [−8 703, −4 343] |
| tree | +0.167 [+0.083, +0.267] | +0.133 [+0.050, +0.217] | −6 867 [−9 070, −4 697] |
| neural | +0.150 [+0.067, +0.250] | +0.117 [+0.050, +0.200] | −6 049 [−8 141, −4 060] |

Latency on **jointly solved** tasks only (43 of 60; 17 excluded because uniform failed): mean
seconds uniform 0.0105, heuristic 0.0120, neural 0.0077, tree 0.0187; mean candidates uniform
5 921 vs heuristic 1 498, neural 1 027, tree 680. On the 15 authored tasks guided search was
*slower* than uniform on jointly solved tasks (e.g. heuristic 0.0028 s vs 0.0017 s) because those tasks are
tiny and scoring has a fixed cost. Candidates fell everywhere; wall-clock only paid off when
tasks were hard enough that uniform search failed or ran long.

## 4. Controls

On a 30-task subset of **validation** tasks disjoint from the 60 used for λ selection (`analysis_summary.json`):

| policy | fit | mean candidates | mean s |
|---|---|---|---|
| uniform | 0.633 | 13 925 | 0.024 |
| neural (trained) | 1.000 | 2 808 | 0.022 |
| neural, no AST-size feature (retrained) | 0.967 | 4 387 | 0.035 |
| neural, shuffled scores | 0.833 | 9 741 | 0.071 |
| neural, untrained (random weights) | 0.833 | 10 968 | 0.066 |

Reading: the trained model is far better than random/shuffled scorers with identical scoring
overhead, so most of the gain is learned signal rather than the cost-band mechanism. Untrained and
shuffled scorers still beat uniform on solve rate — a random smooth function of the features is
not neutral (it perturbs ordering in ways that sometimes help) — so "any scoring helps a little"
cannot be excluded. Caveat: these tasks come from the split used for checkpoint early-stopping,
so the trained-model row is optimistic (on held-out `dev` the same model fits 0.967).

**No-memoization comparison** (uniform policy, 12 validation tasks, small candidate cap):

| variant | fit | mean candidates |
|---|---|---|
| memo + dominance pruning | 0.417 | 13 026 |
| memo only | 0.417 | 13 295 |
| no memoization | 0.167 | 16 690 |

Semantic memoization is what matters; dominance pruning adds almost nothing here (2 %) — its main justification is bounded frontier size, not speed.

## 5. Honest reading of the results

1. **Guidance helps substantially versus uniform enumeration** on this benchmark: more tasks solved within the same budget and, on jointly solved tasks pooled, 4–9× fewer candidates (uniform 5 921 vs 1 498 / 1 027 / 680); intervals exclude zero.
2. **The learned models did not beat a transparent hand-set heuristic.** The heuristic matched or exceeded neural and tree on every suite (e.g. comp_dev fit 0.733 vs 0.600). Neural ≈ tree. The honest framing is that execution-derived features carry most of the signal; the neural net is a working demonstration of the training/evaluation pipeline, not evidence that learning is necessary here.
3. **Compositional shift hurts everyone** (comp_dev fit ≤ 0.73), consistent with the models being trained on programs ≤ 5 nodes.
4. **Example consistency ≠ correctness.** Even the best policy matches hidden rows on only 0.70 of dev and 0.33 of comp_dev tasks, versus 1.00 / 0.73 consistency: many examples underdetermine the intended program.
5. Wall-clock benefit is real only where uniform search is slow or fails; scoring overhead (per-candidate featurization in Python) dominates on easy tasks.

## 6. Threats to validity

* **Few independent program groups.** dev = 10 groups, comp_dev = 13. Tasks sharing a skeleton are correlated, so task-level bootstrap intervals are *too narrow*; treat them as indicative only.
* One neural seed in dev (3 only in `full`), one timing repeat; seed variance and timing noise are not measured here.
* Synthetic generator + 15 of the 40 authored tasks; no external benchmark. The generator's families overlap in style with the authored tasks.
* Weak labels (one reference program per task); λ selection on 60 tasks, small grid.
* The controls suite overlaps the checkpoint-selection split (see §4).
* The pooled "ALL" rows mix suites and are given only as a summary.

## 7. Smoke profile (CI-sized; `runs/smoke`)

12 tasks (6 dev + 6 regression), 5-node grammar, 1 s / 5 000 candidates: uniform fit 0.500, heuristic/tree/neural 1.000 (mean candidates 2 864 vs 689 / 1 288 / 1 253). It exists to exercise the pipeline offline in ~10 s, not to support performance claims.
