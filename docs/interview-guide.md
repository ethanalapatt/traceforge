# Interview guide

Twelve questions an interviewer can reasonably ask, with answers tied to code and to measured
results (`docs/experiments.md`; numbers from `runs/dev/`). Where the honest answer is unflattering, it is given as is.

**1. What problem does TraceForge solve, in one breath?**
Programming-by-example for strings: from 3–6 input/output pairs it searches a 9-operator DSL
(`dsl.py`) for a program that reproduces them, locally and offline, and can then apply it to a CSV.
The returned program is example-consistent, not proven correct.

**2. Is this novel?**
No. Learning-guided bottom-up synthesis is BUSTLE / CrossBeam territory and the setting is FlashFill's.
It is a from-scratch implementation and an evaluation of when guidance pays off. See `THIRD_PARTY_NOTICES.md`.

**3. How does the search work, and why bottom-up?**
`search/engine.py` enumerates *behaviours*: each state is the vector of outputs on the supplied
examples plus an AST size. Smaller programs are combined into larger ones; because only outputs on
the examples matter, many ASTs collapse to one state (semantic memoization on `(outputs, size)`).
Bottom-up lets us reuse every sub-result and check candidates against the examples immediately.

**4. How does the neural model influence the search without breaking completeness?**
It never prunes. A score `p∈[0,1]` only shifts a state's integer priority cost to
`base + ⌊λ(1−p)⌋`. Everything with AST size ≤ N is still reachable because the level loop runs to
`c_max = (1+λ)N − λ`; the proof is in `docs/architecture.md` §3 and is checked against an independent
brute-force enumerator for λ ∈ {0,1,3} in `tests/test_search.py`.

**5. Why keep AST size separate from priority cost?**
So the node limit stays a hard guarantee ("≤ N nodes") while the model reorders exploration.
A test shows a returned program whose priority cost exceeds its size.

**6. What exactly did you train on, and what are the label flaws?**
Weak labels: proper non-terminal sub-expressions of a task's reference program are positives;
negatives come from a bounded unlearned search plus single-edit perturbations (`training.py`).
Flaw: "not in this one reference program" does not mean "useless" — other solutions may use it —
so outputs are ranking scores, not probabilities. Validation is split by program skeleton so the
model cannot memorize programs; I also report PR-AUC of AST size alone (0.181 ≈ the 0.177 base rate)
so model gains are not overstated (neural 0.670).

**7. How did you prevent leakage?**
Type-level: `SearchTask` has only `(spec, vocab, budgets)`; hidden rows, the reference program and
family/suite names exist only in `EvalTask`. A test swaps the hidden labels/reference and asserts
identical budget-limited search behaviour. Splits are by skeleton group with a disjointness audit
(group, exact spec, probe behaviour); λ and the inference device are chosen on validation tasks only.
The audit also caught a real bug (held-out `Upper∘Concat` while `Lower∘Concat` shared its group).

**8. What are the results, honestly?**
On the dev profile (60 tasks, same budgets): fit-all-examples uniform 0.717 → neural 0.867
(heuristic 0.933, tree 0.883); mean candidates 11 327 → 5 278; paired-bootstrap intervals exclude 0.
But: **a hand-set heuristic matched or beat the learned models on every suite**, neural ≈ tree, and
only 10–13 independent program groups back each held-out suite, so intervals are optimistic.

**9. Did guidance make it faster in wall-clock?**
Not uniformly. On jointly solved tasks pooled, neural 0.0077 s vs uniform 0.0105 s, but heuristic
0.0120 and tree 0.0187 were slower, and on the tiny authored tasks every guided policy was slower
than uniform. Scoring costs Python featurization per batch. The timeout-aware mean (unsolved charged the cap) favours
guidance because uniform fails more often.

**10. What do the controls show?**
Same overhead, different signal: trained neural 1.000 fit / 2 808 candidates vs untrained 0.833 /
10 968 and shuffled 0.833 / 9 741 (uniform 0.633 / 13 925). So the learned signal matters, though
random scorers are not neutral (they beat uniform on solve rate). Caveat: that suite shares the split used for early stopping, so the trained row is optimistic.

**11. What does memoization buy, and what is "dominance pruning"?**
Tiny comparison (12 tasks, uniform): no memo 0.167 fit / 16 690 candidates vs memo 0.417 / 13 295.
Dominance pruning (drop a state if a smaller AST has the same outputs) is my addition beyond the
`(outputs,size)` memo; it is lossless for solving under the node limit (tested against the
oracle) but changed only ~2 % of candidates here. It is applied to all policies and ablated.

**12. What would you do next / what are the limits?**
Run the `full` profile (3 seeds, 100-task in-distribution test, 60 compositional test) — not run
here; add more independent program groups; try learning from search traces rather than one
reference program; vectorize featurization; test other Python versions/platforms. Limits: finite
grammar per task (`exhausted` ≠ impossible), ≤ 2 columns, synthetic + 40 authored tasks, single
machine timing. MPS helped nothing at inference (CPU 0.18 ms vs MPS 1.09 ms per 256-row batch);
it was used only for training.

## Code reading tour (10 minutes)

`dsl.py` (semantics, `unary_fn`) → `vocabulary.py` → `search/engine.py` (level loop, `_compose`,
bounds) → `features.py` → `policies.py` (`heuristic_p`, `FeatureScorer`) → `training.py`
(`task_rows`) → `evaluation.py` (`run_one`, `summarize`) → `tests/test_search.py` (oracle).
