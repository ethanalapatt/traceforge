# Résumé evidence

Every number that may appear in a résumé bullet, with the file it comes from and the
conditions it holds under. Do not quote anything not listed here. Do not call this a new
algorithm or claim a full-profile result: **the full profile has not been run.**

Conditions for all rows marked *dev*: `traceforge reproduce --profile dev --device auto --offline`,
Apple M3 / 16 GiB, Python 3.13.7, one neural seed, one timing repeat, 3 s and 25 000-candidate
budget per task, 60 pooled tasks (dev 30 + comp_dev 15 + authored 15) from 10 + 13 program groups
(+ hand-written tasks). Intervals are task-level bootstrap and are optimistic (correlated tasks).

| claim (use wording close to this) | value | source |
|---|---|---|
| Built a local offline program-synthesis engine over a 9-operator string DSL with a bottom-up semantic-memoized enumerator | 9 operators; searches ≤ 7 AST nodes (dev) | `src/traceforge/dsl.py`, `search/engine.py`, `profiles.py` |
| Proved the penalized priority schedule exhaustive and tested it against an independent brute-force enumerator | bound `c_max=(1+λ)N−λ`; λ ∈ {0,1,3} | `docs/architecture.md` §3; `tests/test_search.py` |
| Test suite | 76 tests incl. Hypothesis properties and an offline smoke reproduction; all pass in ~18 s | `pytest` (`tests/`) |
| Search throughput (uniform policy) | ≈ 5.8 × 10⁵ candidate compositions/s (≈ 2.5 × 10⁶ per-example operator applications/s) | `runs/dev/results/main.jsonl` (sum of `candidates` / sum of `seconds`, uniform rows) |
| Learned guidance solved more tasks within an identical budget | fit all examples: uniform 71.7 % → neural 86.7 % (heuristic 93.3 %, tree 88.3 %) | `runs/dev/results/summary.json` → `suites.ALL.policies` |
| …with fewer candidates | mean candidates 11 327 → 5 278 (neural), −53 % | same |
| Paired bootstrap | neural − uniform: Δfit +0.150 [0.067, 0.250]; Δcandidates −6 049 [−8 141, −4 060] | same → `paired_vs_baseline` |
| Candidate reduction on jointly solved tasks | uniform 5 921 vs neural 1 027 (43/60 tasks) | same → `joint` |
| Training: 62 977-parameter MLP on weak labels, grouped validation | val PR-AUC 0.670 (AST-size alone 0.181; tree 0.653) | `runs/dev/training/training_report.json` |
| Compositional-generalization suite | comp_dev fit: uniform 0.333, heuristic 0.733, neural 0.600, tree 0.600 | `summary.json` → `suites.comp_dev` |
| Controls | neural 1.000 vs untrained 0.833 vs shuffled 0.833 vs uniform 0.633 (30 validation-derived tasks; optimistic) | `runs/dev/results/analysis_summary.json` |
| Example-consistency vs correctness gap | best policy: hidden-row correct 0.70 (dev) vs 1.00 consistent | `summary.json` → `suites.dev` |
| Device policy | cpu/mps/auto; MPS used for training in dev; CPU chosen for inference (0.18 vs 1.09 ms/batch) | `runs/dev/models/selection.json`, `training_report.json` |
| Reproducibility | identical dataset fingerprint and identical candidates/outcomes/programs across two dev runs | `runs/dev/manifest.json` → `stages.generate.dataset_fingerprint` `7f30ef1ce10b7993`; compared against `runs/dev_first_run_discarded` (kept only for this comparison) |

## Suggested bullets (all supported by the table)

* Built **TraceForge**, an offline neural-guided program-synthesis engine for string transformations (9-operator DSL, bottom-up enumeration with semantic memoization, integer-cost priority bands) with a machine-checked scheduling invariant and a 76-test suite including a brute-force oracle.
* Trained a 63 K-parameter scorer on weak labels with program-grouped validation and leak-proof task interfaces; on a 60-task benchmark under identical budgets, guided search solved 87 % of tasks vs 72 % for uniform enumeration with 53 % fewer candidates (paired-bootstrap CI excludes zero).
* Evaluated honestly: ablations/controls (untrained, shuffled, no-size-feature, no-memoization) showed learned signal matters, but a hand-set heuristic matched the neural model; reported example-consistency separately from hidden-row correctness (best 70 % vs 100 % on dev).

## Do not claim

* Any `full`-profile result, any result on Linux/Intel/other Python versions, MPS speedups at inference, novelty of the algorithm, or that learned models beat the heuristic.
* That a returned program is *correct* (only example-consistent).
