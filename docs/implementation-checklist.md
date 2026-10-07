# Implementation checklist

Status legend: ✅ implemented and exercised by a test or a recorded run · 🟡 implemented, evidence is partial · ⛔ not implemented / not run.

| area | status | evidence |
|---|---|---|
| DSL: 9 operators, invalid ≠ empty, bounded strings, node counts, strict JSON, pretty printer | ✅ | `tests/test_dsl.py` (incl. Hypothesis round-trip/totality) |
| Finite per-task vocabulary + saved manifest | ✅ | `vocabulary.py`; authored tasks representable (`test_data.py`); `vocabulary.json` from `synthesize` |
| Bottom-up search, integer costs, semantic memo, lazy budgets, outcomes | ✅ | `tests/test_search.py` |
| Scheduling-invariant proof + tests | ✅ | `docs/architecture.md` §3; oracle tests for λ ∈ {0,1,3} |
| Exhaustive brute-force oracle | ✅ | `test_unlearned_search_agrees_with_brute_force_oracle` |
| Dominance pruning (extension) documented + ablated | ✅ | `test_dominance_pruning_*`, `results/nomemo.*` |
| Policies: uniform, heuristic, tree, neural | ✅ | `policies.py`, `test_eval.py::test_all_policies_*` |
| Controls: untrained, shuffled, no-AST-size, no-memo | ✅ | `results/analysis_summary.json` |
| Grouped splits, rejection stats, fingerprints, behavioural-duplicate check | ✅ | `test_data.py`, `data/splits.json` |
| Evaluator-only hidden labels; leakage test | ✅ | `test_search_receives_no_hidden_information_and_is_label_independent` |
| Weak labels, grouped validation, PR-AUC, size-only baseline | ✅ | `training.py`, `training/training_report.json` |
| Checkpoint schema/metadata; `weights_only` neural load; clear errors | ✅ | `test_ml.py` |
| Device policy cpu/mps/auto | 🟡 | CPU: all tests. MPS: trained on this M3 in the dev profile (see experiments); no claim beyond that machine/run |
| Evaluation: raw JSONL+CSV, paired bootstrap by task, timeout-aware aggregates, joint-solved latency, curves, RSS | ✅ | `test_eval.py`, `results/` |
| Resume + config-fingerprint rejection | ✅ | `test_eval.py`, `test_cli.py` |
| Profiles smoke/dev/full | 🟡 | smoke + dev executed; **full not executed** |
| CLI: doctor, reproduce, synthesize, apply, report (+ stage commands) | ✅ | `test_cli.py` |
| Standalone offline HTML report (escaping, no CDN) | ✅ | `test_cli.py::test_html_report_*`, `test_report_is_standalone_and_offline` |
| Offline smoke run with sockets blocked | ✅ | `test_offline_smoke_reproduction_end_to_end` (CLI `--offline`) |
| Docs: README, architecture, experiments, interview guide, resume evidence, notices | ✅ | this directory |
| Optional SyGuS-derived importer | ⛔ | not implemented (spec: optional, only after the core) |
| Python versions other than 3.13 | ⛔ | not tested |
| Linux / non-Apple hardware | ⛔ | not tested |
