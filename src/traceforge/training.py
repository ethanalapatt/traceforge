"""Supervised usefulness labels, MLP training, tree baseline.

Labels (weak supervision - see docs/architecture.md):

* positives: proper, non-terminal sub-expressions of the task's *reference* AST (behaviour on
  the supplied examples only);
* negatives: valid, unsolved states from a bounded *unlearned* search, plus valid single-edit
  perturbations of reference sub-expressions; any state whose output vector equals a positive's
  (at any size) or the final target is excluded; negatives are stratified to match the positives'
  AST-size distribution so size alone cannot separate the classes.

Absence from the sampled reference does **not** prove a state cannot contribute to another
correct program.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import dsl
from . import features as F
from .dsl import Node
from .models import NeuralBundle, Normalizer, build_mlp
from .policies import heuristic_p
from .search import search
from .tasks import Budgets, EvalTask, make_search_task
from .vocabulary import Vocabulary


@dataclass
class LabelConfig:
    max_nodes: int = 7
    neg_search_candidates: int = 6000
    neg_search_seconds: float = 2.0
    neg_ratio: int = 4  # stratified negatives per positive
    perturb_per_positive: int = 1
    max_rows_per_task: int = 80
    seed: int = 0


@dataclass
class RowSet:
    X: np.ndarray
    y: np.ndarray
    task: np.ndarray  # task index within this RowSet
    size: np.ndarray
    group: list[str]  # skeleton per task index
    stats: dict = field(default_factory=dict)

    def save(self, directory: Path, name: str) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}_X.npy", self.X)
        np.save(directory / f"{name}_y.npy", self.y)
        np.save(directory / f"{name}_task.npy", self.task)
        np.save(directory / f"{name}_size.npy", self.size)
        (directory / f"{name}_groups.json").write_text(json.dumps(self.group))
        return {"rows": int(len(self.y)), "positives": int(self.y.sum())}


# ------------------------------------------------------------------ perturbations


def _replace_at(n: Node, k: list[int], new: Node) -> Node:
    """Return ``n`` with the k[0]-th node (pre-order) replaced by ``new``."""
    if k[0] == 0:
        k[0] = -1
        return new
    k[0] -= 1
    t = type(n)
    if t is dsl.Input or t is dsl.Literal:
        return n
    if t is dsl.Concat:
        left = _replace_at(n.left, k, new)
        right = _replace_at(n.right, k, new) if k[0] >= 0 else n.right
        return dsl.Concat(left, right)
    arg = _replace_at(n.arg, k, new)  # type: ignore[attr-defined]
    return dsl.make_unary(n.op, dsl.unary_params(n), arg)


def _variant(rng: random.Random, m: Node, vocab: Vocabulary) -> Node | None:
    t = type(m)
    if t is dsl.Input:
        if vocab.n_inputs > 1 and rng.random() < 0.5:
            return dsl.Input(1 - m.index)
        return dsl.Literal(rng.choice(vocab.literals))
    if t is dsl.Literal:
        others = [l for l in vocab.literals if l != m.text]
        return dsl.Literal(rng.choice(others)) if others else None
    if t is dsl.Concat:
        return dsl.Concat(m.right, m.left)
    child = m.arg  # type: ignore[attr-defined]
    ops = vocab.unary_ops()
    same = [o for o in ops if o[0] == m.op and o[1] != dsl.unary_params(m)]
    if same and rng.random() < 0.5:
        op, par = rng.choice(same)
    else:
        op, par = rng.choice([o for o in ops if o[0] != m.op])
    return dsl.make_unary(op, par, child)


def perturbations(rng: random.Random, sub: Node, vocab: Vocabulary, k: int) -> list[Node]:
    out: list[Node] = []
    n_nodes = dsl.ast_size(sub)
    for _ in range(k * 3):
        idx = rng.randrange(n_nodes)
        target = list(dsl.iter_nodes(sub))[idx]
        v = _variant(rng, target, vocab)
        if v is None:
            continue
        out.append(_replace_at(sub, [idx], v))
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------------ labelling


def _proper_nonterminal_subexprs(ref: Node) -> list[Node]:
    return [n for n in dsl.iter_nodes(ref) if n is not ref and n.op not in ("Input", "Literal")]


def task_rows(task: EvalTask, cfg: LabelConfig, rng: random.Random) -> tuple[list[tuple[tuple[str, ...], int]], list[int]] | None:
    """Return ((outs, size) rows, labels) for one training task, or None if unusable."""
    ref = task.reference
    if ref is None:
        return None
    spec = task.spec
    rows_in = [e.inputs for e in spec.examples]
    targets = tuple(e.output for e in spec.examples)
    st = make_search_task(
        spec,
        Budgets(max_nodes=cfg.max_nodes, max_candidates=cfg.neg_search_candidates, seconds=cfg.neg_search_seconds, max_states=200_000),
    )
    pos: dict[tuple[tuple[str, ...], int], None] = {}
    for sub in _proper_nonterminal_subexprs(ref):
        vals = dsl.evaluate_rows(sub, rows_in)
        if any(not isinstance(v, str) for v in vals):
            continue
        outs = tuple(vals)  # type: ignore[arg-type]
        if outs != targets:
            pos[(outs, dsl.ast_size(sub))] = None
    if not pos:
        return None
    pos_outs = {k[0] for k in pos}
    res = search(st, stop_on_solution=False, collect_states=True)
    pool = [s for s in (res.states or []) if s[0] not in pos_outs and s[0] != targets]
    by_size: dict[int, list[tuple[tuple[str, ...], int]]] = {}
    for s in pool:
        by_size.setdefault(s[1], []).append(s)
    negs: dict[tuple[tuple[str, ...], int], None] = {}
    sizes_avail = sorted(by_size)
    fz = F.TaskFeaturizer(spec, cfg.max_nodes)
    hard_order: dict[int, list[tuple[tuple[str, ...], int]]] = {}

    def hardest(sz: int) -> list[tuple[tuple[str, ...], int]]:
        if sz not in hard_order:
            c = by_size[sz][:2000]
            p = heuristic_p(fz.featurize([x[0] for x in c], [x[1] for x in c]))
            hard_order[sz] = [c[int(i)] for i in np.argsort(-p, kind="stable")]
        return hard_order[sz]

    for _o, s in pos:
        if s not in by_size and sizes_avail:
            s = min(sizes_avail, key=lambda z: (abs(z - s), z))
        if s not in by_size:
            continue
        hard_n = cfg.neg_ratio // 2  # half hard (highest heuristic usefulness, same size), half random
        for c in hardest(s)[:hard_n]:
            negs[c] = None
        cands = by_size[s]
        for c in rng.sample(cands, min(len(cands), cfg.neg_ratio - hard_n)):
            negs[c] = None
    for sub in _proper_nonterminal_subexprs(ref):
        for v in perturbations(rng, sub, st.vocab, cfg.perturb_per_positive):
            vals = dsl.evaluate_rows(v, rows_in)
            if any(not isinstance(x, str) for x in vals) or v.op in ("Input", "Literal"):
                continue
            outs = tuple(vals)  # type: ignore[arg-type]
            if outs in pos_outs or outs == targets:
                continue
            negs[(outs, dsl.ast_size(v))] = None
    rows = list(pos) + [n for n in negs if n not in pos]
    labels = [1] * len(pos) + [0] * (len(rows) - len(pos))
    if len(rows) > cfg.max_rows_per_task:
        keep = rng.sample(range(len(rows)), cfg.max_rows_per_task)
        rows = [rows[i] for i in keep]
        labels = [labels[i] for i in keep]
        if sum(labels) == 0:
            return None
    return rows, labels


def build_rows(tasks: Sequence[EvalTask], cfg: LabelConfig, progress=None) -> RowSet:
    rng = random.Random(cfg.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    groups: list[str] = []
    skipped = 0
    for i, t in enumerate(tasks):
        out = task_rows(t, cfg, rng)
        if out is None:
            skipped += 1
            continue
        rows, labels = out
        x = F.TaskFeaturizer(t.spec, cfg.max_nodes).featurize([r[0] for r in rows], [r[1] for r in rows])
        ti = len(groups)
        groups.append(t.skeleton)
        xs.append(x)
        ys.append(np.asarray(labels, dtype=np.float32))
        ts.append(np.full(len(rows), ti, dtype=np.int32))
        ss.append(np.asarray([r[1] for r in rows], dtype=np.int16))
        if progress and (i + 1) % 500 == 0:
            progress(f"labelled {i + 1}/{len(tasks)} tasks")
    if not xs:
        raise RuntimeError("no usable training rows were produced")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    size = np.concatenate(ss)
    stats = {
        "tasks_used": len(groups),
        "tasks_skipped": skipped,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "size_hist_pos": {int(k): int(v) for k, v in zip(*np.unique(size[y == 1], return_counts=True))},
        "size_hist_neg": {int(k): int(v) for k, v in zip(*np.unique(size[y == 0], return_counts=True))},
    }
    return RowSet(X, y, np.concatenate(ts), size, groups, stats)


# ---------------------------------------------------------------------------- metrics


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def ranking_recall(rs: RowSet, scores: np.ndarray, frac: float = 0.2) -> float:
    """Mean per-task recall of known-useful rows within the top ``frac`` of that task's rows."""
    recalls = []
    order = np.argsort(rs.task, kind="stable")
    t_sorted = rs.task[order]
    bounds = np.flatnonzero(np.diff(t_sorted)) + 1
    for seg in np.split(order, bounds):
        yy = rs.y[seg]
        if yy.sum() == 0:
            continue
        k = max(1, math.ceil(frac * len(seg)))
        top = seg[np.argsort(-scores[seg], kind="stable")[:k]]
        recalls.append(rs.y[top].sum() / yy.sum())
    return float(np.mean(recalls)) if recalls else float("nan")


def size_only_pr_auc(train: RowSet, val: RowSet) -> float:
    """PR-AUC of a size-only logistic model: a floor showing size is not the whole signal."""
    from sklearn.linear_model import LogisticRegression

    def f(rs: RowSet) -> np.ndarray:
        s = rs.size.astype(np.float64)[:, None]
        return np.hstack([s, s**2])

    lr = LogisticRegression(max_iter=200).fit(f(train), train.y)
    return pr_auc(val.y, lr.predict_proba(f(val))[:, 1])


# ----------------------------------------------------------------------------- training


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (256, 128)
    epochs: int = 30
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 6
    seed: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["hidden"] = list(self.hidden)
        return d


def train_neural(
    train: RowSet,
    val: RowSet,
    cfg: TrainConfig,
    *,
    drop_features: Sequence[str] = (),
    meta: dict | None = None,
    log=None,
) -> tuple[NeuralBundle, dict]:
    import torch
    import torch.nn.functional as Fn

    if train.y.sum() == 0 or val.y.sum() == 0:
        raise ValueError(
            "training and validation rows must each contain positive labels (checkpoint selection uses "
            "validation PR-AUC); generate more tasks or a larger profile"
        )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    norm = Normalizer.fit(train.X)  # fit on training rows only
    bundle = NeuralBundle(build_mlp(F.FEATURE_DIM, cfg.hidden), norm, list(drop_features), dict(meta or {}))
    dev = cfg.device
    bundle.model.to(dev)
    xtr = torch.from_numpy(bundle.prepare(train.X)).to(dev)
    ytr = torch.from_numpy(train.y).to(dev)
    xva = torch.from_numpy(bundle.prepare(val.X)).to(dev)
    yva = torch.from_numpy(val.y).to(dev)
    n_pos = float(train.y.sum())
    pos_weight = torch.tensor((len(train.y) - n_pos) / max(n_pos, 1.0), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(bundle.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    init_sig = float(sum(p.detach().abs().sum() for p in bundle.model.parameters()))
    best = {"ap": -1.0, "epoch": -1, "state": None}
    history: list[dict] = []
    n = len(ytr)
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        bundle.model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size].to(dev)
            opt.zero_grad()
            logit = bundle.model(xtr[idx]).squeeze(-1)
            loss = Fn.binary_cross_entropy_with_logits(logit, ytr[idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        bundle.model.eval()
        with torch.no_grad():
            lv = bundle.model(xva).squeeze(-1)
            vloss = float(Fn.binary_cross_entropy_with_logits(lv, yva, pos_weight=pos_weight))
            vs = torch.sigmoid(lv).cpu().numpy()
        ap = pr_auc(val.y, vs)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": vloss, "val_pr_auc": ap})
        if log:
            log(f"  epoch {epoch:2d} train_loss={tot / n:.4f} val_loss={vloss:.4f} val_pr_auc={ap:.4f}")
        if ap > best["ap"]:
            best = {"ap": ap, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in bundle.model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg.patience:
            break
    bundle.model.load_state_dict(best["state"])
    bundle.model.to("cpu").eval()  # portable CPU weights
    bundle.device = "cpu"
    final_sig = float(sum(p.detach().abs().sum() for p in bundle.model.parameters()))
    scores = bundle.predict_p(val.X)
    metrics = {
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "val_pr_auc": pr_auc(val.y, scores),
        "val_ranking_recall_top20": ranking_recall(val, scores),
        "val_loss_best": min(h["val_loss"] for h in history),
        "val_positive_rate": float(val.y.mean()),
        "size_only_val_pr_auc": size_only_pr_auc(train, val),
        "train_seconds": time.perf_counter() - t0,
        "param_abs_sum_before": init_sig,
        "param_abs_sum_after": final_sig,
        "n_params": bundle.n_params(),
        "device": dev,
    }
    bundle.meta.update(
        {"seed": cfg.seed, "train_config": cfg.to_dict(), "val_metrics": metrics, "history": history}
    )
    return bundle, metrics


def train_tree(train: RowSet, val: RowSet, seed: int = 0, max_rows: int = 200_000, log=None):
    """ExtraTrees baseline: two fixed configurations, the better one on validation PR-AUC wins."""
    from sklearn.ensemble import ExtraTreesClassifier

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train.y))
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    configs = [
        dict(n_estimators=100, max_depth=12, min_samples_leaf=5),
        dict(n_estimators=100, max_depth=20, min_samples_leaf=3),
    ]
    best = None
    results = []
    for c in configs:
        m = ExtraTreesClassifier(**c, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed)
        m.fit(train.X[idx], train.y[idx])
        s = m.predict_proba(val.X)[:, 1]
        ap = pr_auc(val.y, s)
        results.append({**c, "val_pr_auc": ap, "val_ranking_recall_top20": ranking_recall(val, s)})
        if log:
            log(f"  tree {c} val_pr_auc={ap:.4f}")
        if best is None or ap > best[0]:
            best = (ap, m, c)
    model = best[1]
    model.set_params(n_jobs=1)  # single-threaded inference: avoids thread-pool overhead per call
    return model, {"chosen": best[2], "candidates": results, "val_pr_auc": best[0], "train_rows": int(len(idx)),
                   "val_ranking_recall_top20": [r for r in results if r["val_pr_auc"] == best[0]][0]["val_ranking_recall_top20"]}
