"""Grouped synthetic task generation.

Pipeline: sample a reference AST + input family -> sample supplied rows and hidden query rows
-> execute the reference -> apply documented rejection rules -> check the task is representable
in the vocabulary built *from its supplied examples only* -> route to a partition by the hash of
the program group (skeleton) -> drop cross-partition behavioural duplicates.

Reference programs and hidden rows live only in ``EvalTask``; the search never sees them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .. import dsl
from ..dsl import Node
from ..tasks import EvalTask, Example, TaskSpec, fingerprint, spec_fingerprint
from ..vocabulary import FIXED_LITERALS, SLICE_PAIRS, build_vocabulary
from .families import FAMILIES, Family

SUITES = ("train", "val", "dev", "test_id", "comp_dev", "comp_test")
HELD_OUT_BIGRAMS = (("Upper", "Concat"), ("Slice", "Replace"), ("Replace", "Token"))
FRAGMENT_POOL = ("@example.com", ", ", "user_", ".txt", " - ", "(", ")", "id:", "_v2", "https://", ".com", "!")
REPLACE_NEW = ("", ".", "_", "-", " ")
OP_WEIGHTS = (("Trim", 0.8), ("Lower", 1.2), ("Upper", 0.7), ("Token", 1.6), ("Slice", 1.2), ("Replace", 1.2), ("Concat", 1.8))
MAX_GEN_STRING = 64


@dataclass
class GenConfig:
    seed: int = 0
    max_nodes: int = 7
    train_max_size: int = 5  # programs larger than this are "compositional"
    min_size: int = 2
    size_weights: dict[int, float] = field(default_factory=lambda: {2: 2, 3: 4, 4: 5, 5: 4, 6: 3, 7: 2})
    shift_enabled: bool = True
    held_out_bigrams: tuple[tuple[str, str], ...] = HELD_OUT_BIGRAMS
    quotas: dict[str, int] = field(default_factory=lambda: {"train": 1500, "val": 200, "dev": 30, "comp_dev": 15})
    n_hidden: int = 24
    reuse_skeleton_prob: float = 0.35
    max_per_group: int = 10  # cap on tasks sharing one skeleton in train (keeps splits diverse)
    max_per_group_eval: int = 3  # cap for val/dev/test partitions
    max_attempt_factor: int = 400

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["size_weights"] = {str(k): v for k, v in self.size_weights.items()}
        d["held_out_bigrams"] = [list(b) for b in self.held_out_bigrams]
        return d


# ------------------------------------------------------------------ AST sampling


def _lit_pool_choice(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return rng.choice([l for l in FIXED_LITERALS if l])
    return rng.choice(FRAGMENT_POOL)


def _sample_unary(rng: random.Random, op: str, child: Node, fam: Family) -> Node:
    if op == "Trim":
        return dsl.Trim(child)
    if op == "Lower":
        return dsl.Lower(child)
    if op == "Upper":
        return dsl.Upper(child)
    if op == "Token":
        return dsl.Token(child, rng.choice(fam.delims), rng.choices((0, 1, -1), (0.35, 0.35, 0.3))[0])
    if op == "Slice":
        a, b = rng.choice(SLICE_PAIRS)
        return dsl.Slice(child, a, b)
    old = rng.choice(fam.delims) if fam.delims else " "
    return dsl.Replace(child, old, rng.choice([n for n in REPLACE_NEW if n != old]))


def sample_ast(rng: random.Random, fam: Family, size: int) -> Node:
    n_in = len(fam.columns)
    ops = [(o, w) for o, w in OP_WEIGHTS if not (o == "Token" and not fam.delims)]

    def term(in_concat: bool) -> Node:
        if rng.random() < (0.3 if in_concat else 0.06):
            return dsl.Literal(_lit_pool_choice(rng))
        return dsl.Input(rng.randrange(n_in))

    def gen(sz: int, in_concat: bool = False) -> Node:
        if sz == 1:
            return term(in_concat)
        cand = [(o, w) for o, w in ops if not (o == "Concat" and sz < 3)]
        op = rng.choices([o for o, _ in cand], [w for _, w in cand])[0]
        if op == "Concat":
            a = rng.randint(1, sz - 2)
            return dsl.Concat(gen(a, True), gen(sz - 1 - a, True))
        return _sample_unary(rng, op, gen(sz - 1), fam)

    return gen(size)


def reparam(rng: random.Random, node: Node, fam: Family) -> Node:
    """Same structure, freshly sampled parameters (used for skeleton reuse)."""
    n_in = len(fam.columns)

    def rec(n: Node) -> Node:
        t = type(n)
        if t is dsl.Input:
            return dsl.Input(n.index % n_in)
        if t is dsl.Literal:
            return dsl.Literal(_lit_pool_choice(rng))
        if t is dsl.Concat:
            return dsl.Concat(rec(n.left), rec(n.right))
        return _sample_unary(rng, n.op, rec(n.arg), fam)  # type: ignore[attr-defined]

    return rec(node)


def _group_op(op: str) -> str:
    """Operator name at skeleton-group granularity (Lower/Upper merge into Case)."""
    return "Case" if op in ("Lower", "Upper") else op


def bigrams(node: Node) -> set[tuple[str, str]]:
    """Parent->child operator pairs at group granularity, so that every program in one
    skeleton group has the same shift status and groups never straddle partitions."""
    return {(_group_op(n.op), _group_op(c.op)) for n in dsl.iter_nodes(node) for c in n.children()}


def is_compositional(node: Node, cfg: GenConfig) -> bool:
    """Compositional shift: program longer than ``train_max_size`` OR containing a held-out
    parent->child operator pair.  Defined on the reference program, never on a result."""
    if not cfg.shift_enabled:
        return False
    return dsl.ast_size(node) > cfg.train_max_size or bool(bigrams(node) & {(_group_op(a), _group_op(b)) for a, b in cfg.held_out_bigrams})


# --------------------------------------------------------------------- task building


def _u01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32


def _probe_rows(fam: Family) -> list[tuple[str, ...]]:
    r = random.Random(f"probe|{fam.name}")
    return [fam.sample(r) for _ in range(12)]


_PROBES: dict[str, list[tuple[str, ...]]] = {}


def behavior_fingerprint(ref: Node, fam: Family) -> str:
    """Approximate behavioural identity: outputs of the reference on 12 fixed probe rows of the
    family.  Catches same-behaviour programs with different syntax *on those probes*; cannot
    prove behavioural distinctness elsewhere."""
    rows = _PROBES.setdefault(fam.name, _probe_rows(fam))
    outs = [v if isinstance(v, str) else None for v in dsl.evaluate_rows(ref, rows)]
    return fingerprint([fam.name, outs])


def _noop_count(ref: Node, rows: list[tuple[str, ...]]) -> int:
    k = 0
    for n in dsl.iter_nodes(ref):
        if n.op in dsl.UNARY_OPS:
            c = n.children()[0]
            if dsl.evaluate_rows(n, rows) == dsl.evaluate_rows(c, rows):
                k += 1
    return k


def _without_one_node(n: Node) -> list[Node]:
    """All variants of ``n`` with exactly one unary node replaced by its child."""
    t = type(n)
    if t is dsl.Input or t is dsl.Literal:
        return []
    if t is dsl.Concat:
        return [dsl.Concat(v, n.right) for v in _without_one_node(n.left)] + [
            dsl.Concat(n.left, v) for v in _without_one_node(n.right)
        ]
    child = n.arg  # type: ignore[attr-defined]
    return [child] + [dsl.make_unary(n.op, dsl.unary_params(n), v) for v in _without_one_node(child)]


def dead_node_count(ref: Node, rows: list[tuple[str, ...]]) -> int:
    """Number of unary nodes whose removal changes no output on any of ``rows``."""
    base = dsl.evaluate_rows(ref, rows)
    return sum(dsl.evaluate_rows(v, rows) == base for v in _without_one_node(ref))


def build_task(
    rng: random.Random, ref: Node, fam: Family, cfg: GenConfig
) -> tuple[EvalTask | None, str]:
    """Create one task or return (None, rejection reason)."""
    n_sup = rng.choice((3, 4, 4, 5, 5, 6))
    sup_rows: list[tuple[str, ...]] = []
    tries = 0
    while len(sup_rows) < n_sup and tries < 40:
        tries += 1
        r = fam.sample(rng)
        if r not in sup_rows:
            sup_rows.append(r)
    if len(sup_rows) < 3:
        return None, "too_few_distinct_rows"
    sup_out = dsl.evaluate_rows(ref, sup_rows)
    if any(not isinstance(v, str) for v in sup_out):
        return None, "invalid_on_supplied"
    outs = [str(v) for v in sup_out]
    if len(set(outs)) == 1:
        return None, "constant_output"
    if all(o == "" for o in outs):
        return None, "empty_output"
    if any(len(o) > MAX_GEN_STRING for o in outs):
        return None, "output_too_long"
    for j in range(len(fam.columns)):
        if all(o == r[j] for o, r in zip(outs, sup_rows)):
            return None, "identity_only"
    if len(fam.columns) == 2 and dsl.max_input_index(ref) == 0 and not any(
        isinstance(n, dsl.Input) and n.index == 1 for n in dsl.iter_nodes(ref)
    ) and rng.random() < 0.7:
        return None, "unused_column"
    noops = _noop_count(ref, sup_rows)
    if noops >= 2 or (noops >= 1 and dsl.ast_size(ref) <= 3):
        return None, "excess_noop"
    spec = TaskSpec("tmp", fam.columns, tuple(Example(r, o) for r, o in zip(sup_rows, outs)))
    why = build_vocabulary(spec).contains_program_params(ref)
    if why:
        return None, "not_representable:" + why
    hidden: list[Example] = []
    seen = set(sup_rows)
    drawn = invalid = 0
    while len(hidden) < cfg.n_hidden and drawn < cfg.n_hidden * 4:
        drawn += 1
        r = fam.sample(rng)
        if r in seen:
            continue
        v = dsl.evaluate(ref, r)
        if not isinstance(v, str) or len(v) > MAX_GEN_STRING:
            invalid += 1
            continue
        seen.add(r)
        hidden.append(Example(r, v))
    if len(hidden) < cfg.n_hidden or invalid > 0.3 * drawn:
        return None, "unstable_on_hidden"
    if dead_node_count(ref, sup_rows + [h.inputs for h in hidden]) > 0:
        return None, "dead_node"
    task = EvalTask(spec, tuple(hidden), ref, fam.name, dsl.skeleton(ref), "", {"ref_size": dsl.ast_size(ref)})
    return task, ""


# --------------------------------------------------------------------- the generator


def _partition_for(group: str, shift: bool, cfg: GenConfig) -> str:
    names = ("comp_dev", "comp_test") if shift else ("train", "val", "dev", "test_id")
    q = [max(cfg.quotas.get(n, 0), 0) for n in names]
    tot = sum(q)
    if tot == 0:
        return ""
    u = _u01(f"{cfg.seed}|{group}") * tot
    acc = 0.0
    for n, w in zip(names, q):
        acc += w
        if u < acc:
            return n if w > 0 else ""
    return names[-1]


def generate_dataset(cfg: GenConfig) -> tuple[dict[str, list[EvalTask]], dict]:
    rng = random.Random(cfg.seed)
    suites: dict[str, list[EvalTask]] = {s: [] for s in SUITES if cfg.quotas.get(s, 0) > 0}
    rejections: Counter[str] = Counter()
    fam_names = [f.name for f in FAMILIES]
    programs_by_part: dict[str, list[Node]] = {p: [] for p in suites}
    group_count: Counter[str] = Counter()
    exact_seen: set[str] = set()
    behavior_owner: dict[str, str] = {}
    stats = Counter()
    sizes = sorted(k for k in cfg.size_weights if cfg.min_size <= k <= cfg.max_nodes)
    weights = [cfg.size_weights[k] for k in sizes]
    total_quota = sum(cfg.quotas.get(s, 0) for s in suites)
    max_attempts = total_quota * cfg.max_attempt_factor

    def full(part: str) -> bool:
        return len(suites[part]) >= cfg.quotas[part]

    while not all(full(s) for s in suites) and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        open_parts = [p for p in suites if programs_by_part[p] and not full(p)]
        if open_parts and rng.random() < cfg.reuse_skeleton_prob:
            part_pick = rng.choices(open_parts, [len(programs_by_part[p]) for p in open_parts])[0]
            base = rng.choice(programs_by_part[part_pick])
            need = dsl.max_input_index(base) + 1
            has_token = any(isinstance(n, dsl.Token) for n in dsl.iter_nodes(base))
            fams = [f for f in FAMILIES if len(f.columns) >= max(need, 1) and (f.delims or not has_token)]
            fam = rng.choice(fams)
            ref = reparam(rng, base, fam)
        else:
            fam = rng.choice(FAMILIES)
            ref = sample_ast(rng, fam, rng.choices(sizes, weights)[0])
        skel = dsl.skeleton(ref)
        shift = is_compositional(ref, cfg)
        part = _partition_for(skel, shift, cfg)
        if not part or part not in suites:
            stats["skipped_unused_partition"] += 1
            continue
        if full(part):
            stats["skipped_partition_full"] += 1
            continue
        if group_count[skel] >= (cfg.max_per_group if part == "train" else cfg.max_per_group_eval):
            stats["skipped_group_cap"] += 1
            continue
        task, why = build_task(rng, ref, fam, cfg)
        if task is None:
            rejections[why] += 1
            continue
        fp = spec_fingerprint(task.spec)
        if fp in exact_seen:
            rejections["exact_duplicate_task"] += 1
            continue
        bfp = behavior_fingerprint(ref, fam)
        owner = behavior_owner.get(bfp)
        if owner is not None and owner != part:
            rejections["behavioral_duplicate_cross_split"] += 1
            continue
        exact_seen.add(fp)
        behavior_owner[bfp] = part
        programs_by_part[part].append(ref)
        group_count[skel] += 1
        suites[part].append(task)
        stats["accepted"] += 1
    for part, tasks in suites.items():
        for i, t in enumerate(tasks):
            t.suite = part
            t.spec = TaskSpec(f"{part}-{i:05d}", t.spec.input_columns, t.spec.examples)
    report = {
        "config": cfg.to_dict(),
        "attempts": stats["attempts"],
        "accepted": stats["accepted"],
        "skipped_partition_full": stats["skipped_partition_full"],
        "skipped_unused_partition": stats["skipped_unused_partition"],
        "skipped_group_cap": stats["skipped_group_cap"],
        "rejections": dict(sorted(rejections.items())),
        "rejection_rate_among_built": (
            sum(rejections.values()) / max(1, sum(rejections.values()) + stats["accepted"])
        ),
        "complete": all(full(s) for s in suites),
        "sizes": {p: dict(sorted(Counter(t.meta["ref_size"] for t in ts).items())) for p, ts in suites.items()},
        "groups": {p: len({t.skeleton for t in ts}) for p, ts in suites.items()},
        "counts": {p: len(ts) for p, ts in suites.items()},
        "families": {p: dict(sorted(Counter(t.family for t in ts).items())) for p, ts in suites.items()},
    }
    return suites, report


# ------------------------------------------------------------------------ validation


def check_split_disjointness(suites: dict[str, list[EvalTask]]) -> dict:
    """Group-level disjointness, exact-duplicate and behavioural-duplicate audit.

    What this establishes: no program group (skeleton) and no exact task spec appears in two
    suites; no two tasks in different suites share the reference's behaviour on the family probe
    rows.  What it cannot establish: that no *semantically equivalent* program exists across
    suites outside the probed behaviour, or that input rows do not overlap by coincidence."""
    group_owner: dict[str, str] = {}
    exact_owner: dict[str, str] = {}
    behavior_owner: dict[str, str] = {}
    out = {"group_overlaps": [], "exact_overlaps": [], "behavior_overlaps": []}
    fam = {f.name: f for f in FAMILIES}
    for part, tasks in suites.items():
        for t in tasks:
            g = t.skeleton
            if group_owner.setdefault(g, part) != part:
                out["group_overlaps"].append([g, group_owner[g], part])
            e = spec_fingerprint(t.spec)
            if exact_owner.setdefault(e, part) != part:
                out["exact_overlaps"].append([t.id, exact_owner[e], part])
            if t.reference is not None and t.family in fam:
                b = behavior_fingerprint(t.reference, fam[t.family])
                if behavior_owner.setdefault(b, part) != part:
                    out["behavior_overlaps"].append([t.id, behavior_owner[b], part])
    out["ok"] = not (out["group_overlaps"] or out["exact_overlaps"] or out["behavior_overlaps"])
    return out


# ----------------------------------------------------------------------- persistence


def save_suites(directory: str | Path, suites: dict[str, list[EvalTask]], report: dict) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, tasks in suites.items():
        p = d / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for t in tasks:
                fh.write(json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        files[name] = {"file": p.name, "n": len(tasks), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    audit = check_split_disjointness({k: v for k, v in suites.items() if k != "authored"})
    manifest = {"report": report, "files": files, "disjointness": audit}
    (d / "splits.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def load_suite(directory: str | Path, name: str) -> list[EvalTask]:
    p = Path(directory) / f"{name}.jsonl"
    return [EvalTask.from_dict(json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dataset_fingerprint(directory: str | Path, names: Iterable[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode())
        h.update((Path(directory) / f"{n}.jsonl").read_bytes())
    return h.hexdigest()[:16]
