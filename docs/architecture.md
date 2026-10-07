# Architecture

## 1. DSL and semantics (`dsl.py`)

Programs are trees over string-valued expressions.

| node | meaning (Python semantics) | AST nodes |
|---|---|---|
| `Input(i)` | i-th input column (i ∈ {0,1}) | 1 |
| `Literal(s)` | constant string | 1 |
| `Trim(e)` | `str.strip()` (Unicode whitespace) | 1 + e |
| `Lower(e)`, `Upper(e)` | `str.lower()`, `str.upper()` (may change length, e.g. `ß`→`SS`) | 1 + e |
| `Concat(a,b)` | ordered concatenation | 1 + a + b |
| `Token(e, d, k)` | `e.split(d)[k]`; negative k allowed; out of range → **invalid** | 1 + e |
| `Slice(e, a, b)` | `e[a:b]`, `b` may be `None`; never invalid | 1 + e |
| `Replace(e, old, new)` | literal `str.replace`, `old != ""` | 1 + e |

* **Invalid ≠ empty.** `Token` out of range yields the `INVALID_TOKEN` singleton; exceeding the
  string-length bound yields `INVALID_LENGTH`. Invalid propagates through every operator and is
  never equal to `""`. Candidates that are invalid on *any* supplied example are discarded.
* **Literal/operator parameters add no nodes**: `Lower(Replace(Trim(I0)," ","."))` has 4 nodes;
  the spec's email example `Concat(Slice(Lower(I0),0,1), Concat(Literal("."), Lower(I1)))` has 8.
  Repeated subtrees are counted each time they occur.
* **Bounded strings.** Every operator checks `max_string_len` (default 256) *before* allocating
  (e.g. `Replace` computes `len(s) + count(old)·growth` and returns `INVALID_LENGTH` without building the string), so adversarial inputs cannot
  blow memory; tests cover `Concat`, `Replace`, and inputs.
* The interpreter and the search share one cached per-string implementation (`unary_fn`), so
  search-time behaviour and `apply`-time behaviour cannot drift. The evaluator additionally
  re-runs every returned program through `dsl.evaluate`.
* JSON (`to_json`/`from_json`) is a strict schema (unknown ops/keys rejected, ≤ 64 nodes);
  `apply` therefore never executes code from a program file.

## 2. Finite task vocabulary (`vocabulary.py`)

Parameters are drawn from a finite set built **only from the supplied examples**:
≤ 12 literals (6 fixed `"" . space - _ @` + ≤ 6 recurring output fragments that are not copies of
input substrings, support ≥ ⌈0.75 n⌉), ≤ 4 delimiters, token indices {0, 1, −1}, 12 fixed slice
pairs, ≤ 20 replace pairs. The manifest is saved as `vocabulary.json`.
Consequence: `exhausted` means *this finite grammar* was exhausted, never "no program exists".

## 3. Search (`search/engine.py`)

Bottom-up enumeration over *behaviours*: a state is the vector of outputs on the supplied
examples together with an AST size. Only one representative AST per key is kept (struct-of-arrays
storage; the AST is rebuilt solely for the returned solution via recorded lineage).

* **Memoization** keyed by `(outputs, size)`. Because size is part of the key, a smaller
  program for the same outputs is never shadowed by a larger one that was found first.
* **Dominance pruning** (extension): a candidate is dropped if a *smaller* AST with identical
  outputs is already retained. It is lossless for solving with the node limit (a dominated state
  can always be replaced by the smaller one) and measurably shrinks the frontier; it is a flag
  (`prune_dominated`) and is ablated in `results/nomemo.*`.
* **Lossless Concat no-op skip**: a constant-empty operand makes `Concat` re-derive its other
  operand at a larger size, so such pairs are skipped (`noop_skipped`).
* **Priority cost vs. AST size.** terminals cost 1; `base = 1 + Σ child costs`;
  stored cost = `base + ⌊λ(1−p)⌋` where `p ∈ [0,1]` is the policy's usefulness score
  (`p = 1` for uniform, so uniform is plain size-ordered enumeration). The node limit always
  uses AST size, never cost.
* **Lazy, budgeted generation.** Candidates are generated inside nested loops that check the
  candidate, state, time and (sampled) RSS budgets on every iteration; scoring is batched (256)
  and flushed at level end. Outcomes: `solved, timeout, candidate_limit, state_limit,
  exhausted, invalid_task, resource_limit`.

### Scheduling invariant (why a penalty cannot hide a state)

Claim: processing base costs `k = 2, 3, …, c_max` with `c_max = (1+λ)·N − λ` visits every
composition of AST size ≤ N exactly once, after all of its operands are final, for any
score function whose value depends only on `(outputs, size)`.

1. *Bucket k is final once base cost k has been processed.* A composition with base cost c is
   stored in bucket `c + pen`, `pen ≥ 0`, so only compositions with base ≤ k write to bucket k.
2. *Operands are final.* A composition of base cost c is built from states in buckets ≤ c−1
   (`base = 1 + Σ children`, each child cost ≥ 1), all final by (1).
3. *Cost bound.* By induction, every state of AST size s has stored cost ≤ (1+λ)s − λ:
   terminals: 1 = (1+λ)−λ. Composition with children sizes s_i, Σs_i = s−1, ≤ 2 children:
   base ≤ 1 + Σ((1+λ)s_i − λ) ≤ (1+λ)(s−1) + 1 − λ (for one child; two children are no larger);
   adding `⌊λ(1−p)⌋ ≤ λ` gives ≤ (1+λ)s − λ.
4. *Duplicates never improve a bucket.* The penalty depends only on `(outputs,size)`, so a
   second discovery of a key has the same penalty and base cost ≥ the first; discarding it loses
   nothing. Hence the level loop up to `c_max` enumerates exactly the same `(outputs,size)` key
   set as unpenalized enumeration, only in a different order.

Tests: `tests/test_search.py` compares the engine with an independent brute-force enumerator
on a tiny grammar for λ ∈ {0, 1, 3} with a hash-based pseudo-random scorer (key sets equal);
`check_visited=True` asserts no composition is scheduled twice; a worst-case (`p=0`) solution
is found within the bound.

## 4. Features (`features.py`, schema v1, dim = 116)

A candidate is described only by: its outputs on the supplied examples, the supplied inputs and
targets, and its AST size. For each example we compute 37 scalars in [0,1] (length ratio,
equals/prefix/suffix/containment vs the target, case-insensitive variants, common
prefix/suffix lengths, character coverage, character-class fractions, delimiter counts, relations to each input field), then
aggregate mean/min/max over examples (37×3 = 111) and append 5 globals (AST size, #examples, input masks,
mean target length). A per-(example,string) cache avoids recomputation. Checkpoints store the
schema; a mismatch raises a clear `CheckpointError`.

## 5. Training labels (`training.py`) — weak supervision

* **Positives**: proper non-terminal sub-expressions of the task's reference program.
* **Negatives**: states from a bounded *unlearned* search that does not stop at the solution
  (stratified by the positives' sizes, half "hard" by the heuristic score, half random), plus
  single-edit perturbations of positives. Anything with outputs equal to a positive or to the
  final target is excluded.
* The label is therefore "appears in one known solution", not "is useful to *some* solution".
  The model output is a ranking score, not a calibrated probability.
* Validation is grouped by program skeleton; checkpoint selection uses validation PR-AUC;
  `size_only_pr_auc` reports what AST size alone achieves so model gain is not overstated.

## 6. Datasets and splits (`data/`)

13 synthetic input families (names, e-mails, identifiers, paths, dates, codes, …) and a
random-AST sampler with rejection rules (constant, identity, empty, no-op, dead node, unused
column, not representable, invalid, unstable on hidden rows, duplicates). **Group key** = the
program skeleton with parameters normalised, Lower/Upper merged to `Case`, columns relabelled.
Partitions are assigned by `hash(seed|group)` weighted by quotas, so a group never straddles two
suites. *Compositional shift* = size > `train_max_size` **or** a held-out parent→child bigram
(`Case∘Concat`, `Slice∘Replace`, `Replace∘Token`) — evaluated at group granularity (a bug where
`Upper∘Concat` was held out while `Lower∘Concat` shared its group was found by the split audit
and fixed; `tests/test_data.py` has a regression test). The audit checks group, exact-spec and
probe-behaviour overlap; it cannot prove semantic distinctness beyond the probed rows.

40 additional tasks are hand-written (supplied rows and hidden inputs by hand; hidden outputs
come from the stated reference AST), independent of the generator's families.

## 7. Leakage boundary

`SearchTask = (TaskSpec, Vocabulary, Budgets)`. Hidden rows, reference programs, family names
and suite names exist only in `EvalTask`, which only `evaluation.py` and `training.py` (for
labels, on *training* tasks) see. `TaskFeaturizer` takes a `TaskSpec`. A test changes the hidden
labels and the reference and asserts identical candidate-limited search behaviour.
λ is chosen on validation tasks only (most solved → fewer candidates → smaller λ); inference
device is chosen from a micro-benchmark on validation rows.

## 8. Evaluation (`evaluation.py`)

Identical tasks, vocabulary and budgets per policy. Raw JSONL (one row per
suite × task × policy × seed × repeat) plus CSV. Aggregates are **timeout-aware**: unsolved runs
are charged the full wall-clock cap in `capped_mean_seconds`, and latency on *jointly solved*
tasks is reported separately with the excluded count. Paired bootstrap is over **tasks**
(seeds averaged per task first). Candidate counts are deterministic; seconds are not.
Resume: per-store config fingerprint; incompatible configs raise `ResumeError`; rows are keyed
and idempotent; a torn last line is ignored.

## 9. Known limits

Finite grammar; ≤ 2 input columns; no regexes, no conditionals, no loops; tasks sharing a program
group are correlated, so task-level bootstrap intervals understate uncertainty (documented in
experiments); the heuristic and learned scorers see the *supplied target outputs* (legitimately —
they are part of the specification) but nothing else about the task.
