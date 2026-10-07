"""Search policies: uniform, heuristic, tree, neural (+ ablation controls).

All learned/heuristic policies share ``FeatureScorer``: same featurizer, same batching, same
score->penalty formula in the engine.  Only the function mapping a feature matrix to p differs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import features as F
from .models import NeuralBundle, load_neural, load_tree, model_path, random_bundle
from .tasks import SearchTask

POLICIES = ("uniform", "heuristic", "tree", "neural")
CONTROL_POLICIES = ("neural_untrained", "neural_shuffled", "neural_nosize")
ALL_POLICIES = POLICIES + CONTROL_POLICIES


def _col(name: str) -> int:
    return F.FEATURE_INDEX[name]


_H = {n: _col(n) for n in (
    "o_in_t__mean", "ci_o_in_t__mean", "cpl__mean", "csl__mean", "t_cov__mean",
    "ci_eq_t__mean", "is_empty__mean", "len_ratio__mean",
)}


def heuristic_p(x: np.ndarray) -> np.ndarray:
    """Fixed, transparent usefulness score from observed string relations (no learned
    parameters, no task-specific rules; weights are hand-set and untuned):

    p = 0.10 + 0.35 o_in_t + 0.15 ci_o_in_t + 0.10 cpl + 0.10 csl + 0.10 t_cov + 0.10 ci_eq_t
        - 0.40 is_empty - 0.30 max(0, len_ratio - 0.5),  clipped to [0, 1]

    where each term is the mean over examples (see features.py).
    """
    g = lambda n: x[:, _H[n]].astype(np.float64)  # noqa: E731
    p = (
        0.10
        + 0.35 * g("o_in_t__mean")
        + 0.15 * g("ci_o_in_t__mean")
        + 0.10 * g("cpl__mean")
        + 0.10 * g("csl__mean")
        + 0.10 * g("t_cov__mean")
        + 0.10 * g("ci_eq_t__mean")
        - 0.40 * g("is_empty__mean")
        - 0.30 * np.maximum(0.0, g("len_ratio__mean") - 0.5)
    )
    return np.clip(p, 0.0, 1.0)


class FeatureScorer:
    """Featurize -> predict, with per-task feature-row score cache and overhead accounting."""

    def __init__(self, name: str, predict: Callable[[np.ndarray], np.ndarray], device: str = "cpu",
                 shuffle_seed: int | None = None) -> None:
        self.name = name
        self._predict = predict
        self.device = device
        self._shuffle_seed = shuffle_seed
        self._fz: F.TaskFeaturizer | None = None
        self._reset()

    def _reset(self) -> None:
        self._calls = 0
        self._rows = 0
        self._hits = 0
        self._feat_s = 0.0
        self._model_s = 0.0
        self._cache: dict[bytes, float] = {}
        self._rng = np.random.default_rng(self._shuffle_seed) if self._shuffle_seed is not None else None

    def begin(self, task: SearchTask) -> None:
        self._fz = F.TaskFeaturizer(task.spec, task.budgets.max_nodes)
        self._reset()

    def score(self, outs: list[tuple[str, ...]], sizes: list[int]) -> np.ndarray:
        assert self._fz is not None, "begin() must be called first"
        t0 = time.perf_counter()
        x = self._fz.featurize(outs, sizes)
        t1 = time.perf_counter()
        keys = [r.tobytes() for r in x]
        p = np.empty(len(keys), dtype=np.float64)
        todo: dict[bytes, int] = {}
        for i, k in enumerate(keys):
            v = self._cache.get(k)
            if v is not None:
                p[i] = v
                self._hits += 1
            elif k not in todo:
                todo[k] = len(todo)
        if todo:
            rows = np.stack([np.frombuffer(k, dtype=np.float32) for k in todo])
            pred = np.asarray(self._predict(rows), dtype=np.float64)
            for k, j in todo.items():
                self._cache[k] = float(pred[j])
            for i, k in enumerate(keys):
                if k in todo:
                    p[i] = pred[todo[k]]
            self._calls += 1
        if self._rng is not None:
            p = p[self._rng.permutation(len(p))]  # shuffled control: real inference cost, scrambled scores
        t2 = time.perf_counter()
        self._rows += len(keys)
        self._feat_s += t1 - t0
        self._model_s += t2 - t1
        return p

    def stats(self) -> dict:
        return {
            "model_calls": self._calls,
            "rows": self._rows,
            "cache_hits": self._hits,
            "featurize_s": self._feat_s,
            "model_s": self._model_s,
            "device": self.device,
        }


def neural_scorer(bundle: NeuralBundle, name: str = "neural", shuffle_seed: int | None = None) -> FeatureScorer:
    return FeatureScorer(name, bundle.predict_p, bundle.device, shuffle_seed)


def tree_scorer(model) -> FeatureScorer:
    return FeatureScorer("tree", lambda x: model.predict_proba(x)[:, 1])


def heuristic_scorer() -> FeatureScorer:
    return FeatureScorer("heuristic", heuristic_p)


class PolicyBook:
    """Loads and caches every policy for one model seed from a ``models/`` directory."""

    def __init__(self, models_dir: str | Path, seed: int = 0, inference_device: str = "cpu") -> None:
        self.dir = Path(models_dir)
        self.seed = seed
        self.device = inference_device
        self._neural: dict[str, NeuralBundle] = {}
        self._tree = None

    def neural_bundle(self, kind: str = "neural") -> NeuralBundle:
        if kind not in self._neural:
            self._neural[kind] = load_neural(model_path(self.dir, kind, self.seed), self.device)
        return self._neural[kind]

    def get(self, policy: str) -> FeatureScorer | None:
        if policy == "uniform":
            return None
        if policy == "heuristic":
            return heuristic_scorer()
        if policy == "tree":
            if self._tree is None:
                self._tree = load_tree(model_path(self.dir, "tree", self.seed))[0]
            return tree_scorer(self._tree)
        if policy == "neural":
            return neural_scorer(self.neural_bundle("neural"))
        if policy == "neural_nosize":
            return neural_scorer(self.neural_bundle("neural_nosize"), "neural_nosize")
        if policy == "neural_shuffled":
            return neural_scorer(self.neural_bundle("neural"), "neural_shuffled", shuffle_seed=1000 + self.seed)
        if policy == "neural_untrained":
            if "untrained" not in self._neural:
                self._neural["untrained"] = random_bundle(self.seed).to(self.device)
            return neural_scorer(self._neural["untrained"], "neural_untrained")
        raise ValueError(f"unknown policy {policy!r}; choose from {ALL_POLICIES}")
