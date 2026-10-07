"""Execution-derived feature schema (version 1).

A candidate is described *only* by its observed outputs on the supplied examples, its AST size
and the supplied examples themselves.  No task ids, dataset membership, reference programs or
hidden rows reach this module (``TaskFeaturizer`` receives a ``TaskSpec`` and nothing else).

Per example k (candidate output ``o``, desired output ``t``, input fields ``x0, x1``) we compute
``PER_EXAMPLE`` scalars in [0, 1]; the candidate vector is the mean / min / max over examples
(for 0/1 features the mean is the *agreement fraction*) followed by global features, including
explicit input-field masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import TaskSpec

FEATURE_SCHEMA_VERSION = 1
_DELIMS = (" ", ".", "@", "-", "_", ",", "/")
MAX_FEATURE_EXAMPLES = 6
_STR_CLIP = 64

_REL_NAMES = ("eq", "o_in_x", "x_in_o", "ci_o_in_x")
PER_EXAMPLE: tuple[str, ...] = (
    "len_o", "len_ratio", "len_diff", "is_empty", "eq_t", "ci_eq_t", "strip_eq_t",
    "o_prefix_t", "o_suffix_t", "o_in_t", "t_in_o", "ci_o_in_t", "cpl", "csl",
    "ochars_in_t", "t_cov", "frac_letter", "frac_digit", "frac_space", "frac_punct", "frac_upper",
    *(f"count_{i}" for i in range(len(_DELIMS))),
    "ws_tokens",
    *(f"{r}_{j}" for j in range(2) for r in _REL_NAMES),
)
GLOBALS: tuple[str, ...] = ("ast_size", "n_examples", "has_input0", "has_input1", "mean_target_len")
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{n}__{a}" for n in PER_EXAMPLE for a in ("mean", "min", "max")
) + GLOBALS
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_F = len(PER_EXAMPLE)


def schema_dict() -> dict:
    return {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES), "dim": FEATURE_DIM}


class SchemaMismatch(ValueError):
    """Raised when a checkpoint's feature schema differs from the code's."""


def check_schema(saved: dict) -> None:
    cur = schema_dict()
    if saved.get("version") != cur["version"] or list(saved.get("names", [])) != cur["names"]:
        raise SchemaMismatch(
            f"feature schema mismatch: checkpoint v{saved.get('version')} dim={len(saved.get('names', []))} "
            f"vs code v{cur['version']} dim={cur['dim']}; retrain the model with `traceforge train`"
        )


@dataclass
class _ExCtx:
    t: str
    tl: str
    lt: int
    tset: frozenset
    xs: tuple  # ((x, x_lower) | None, ...) length 2


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _example_features(o: str, c: _ExCtx) -> list[float]:
    t, tl, lt = c.t, c.tl, c.lt
    lo = len(o)
    ol = o.lower()
    nonempty = lo > 0
    oset = set(o)
    inter = len(oset & c.tset)
    letters = digits = spaces = punct = upper = 0
    for ch in o:
        if ch.isalpha():
            letters += 1
            if ch.isupper():
                upper += 1
        elif ch.isdigit():
            digits += 1
        elif ch.isspace():
            spaces += 1
        else:
            punct += 1
    inv = 1.0 / lo if lo else 0.0
    lt1 = max(lt, 1)
    f = [
        min(lo, _STR_CLIP) / _STR_CLIP,
        min(lo / lt1, 4.0) / 4.0,
        (max(-32, min(32, lo - lt)) + 32) / 64.0,
        0.0 if nonempty else 1.0,
        1.0 if o == t else 0.0,
        1.0 if ol == tl else 0.0,
        1.0 if o.strip() == t else 0.0,
        1.0 if nonempty and t.startswith(o) else 0.0,
        1.0 if nonempty and t.endswith(o) else 0.0,
        1.0 if nonempty and o in t else 0.0,
        1.0 if lt and t in o else 0.0,
        1.0 if nonempty and ol in tl else 0.0,
        _common_prefix(o, t) / lt1,
        _common_prefix(o[::-1], t[::-1]) / lt1,
        inter / len(oset) if oset else 0.0,
        inter / len(c.tset) if c.tset else 0.0,
        letters * inv, digits * inv, spaces * inv, punct * inv, upper * inv,
    ]
    f += [min(o.count(d), 4) / 4.0 for d in _DELIMS]
    f.append(min(len(o.split()), 8) / 8.0)
    for x in c.xs:
        if x is None:
            f += [0.0, 0.0, 0.0, 0.0]
        else:
            xv, xl = x
            f += [
                1.0 if o == xv else 0.0,
                1.0 if nonempty and o in xv else 0.0,
                1.0 if xv and xv in o else 0.0,
                1.0 if nonempty and ol in xl else 0.0,
            ]
    return f


class TaskFeaturizer:
    """Computes the (B, FEATURE_DIM) float32 matrix for a batch of candidates of one task."""

    def __init__(self, spec: TaskSpec, max_nodes: int, cache_limit: int = 300_000) -> None:
        if len(spec.examples) > MAX_FEATURE_EXAMPLES:
            raise ValueError("too many examples for feature schema")
        self.n = len(spec.examples)
        self.ctx: list[_ExCtx] = []
        for e in spec.examples:
            xs = tuple((x, x.lower()) for x in e.inputs) + (None,) * (2 - len(e.inputs))
            self.ctx.append(_ExCtx(e.output, e.output.lower(), len(e.output), frozenset(e.output), xs))
        self.has0 = 1.0 if spec.n_inputs >= 1 else 0.0
        self.has1 = 1.0 if spec.n_inputs >= 2 else 0.0
        self.mean_tlen = min(float(np.mean([c.lt for c in self.ctx])), _STR_CLIP) / _STR_CLIP
        self.size_norm = 12.0
        self._cache: dict[tuple[int, str], list[float]] = {}
        self._cache_limit = cache_limit

    def featurize(self, outs: list[tuple[str, ...]], sizes: list[int]) -> np.ndarray:
        b, n = len(outs), self.n
        cache = self._cache
        rows: list[list[float]] = []
        for vec in outs:
            for k in range(n):
                key = (k, vec[k])
                r = cache.get(key)
                if r is None:
                    r = _example_features(vec[k], self.ctx[k])
                    if len(cache) < self._cache_limit:
                        cache[key] = r
                rows.append(r)
        arr = np.asarray(rows, dtype=np.float32).reshape(b, n, _F)
        mean = arr.mean(axis=1)
        mn = arr.min(axis=1)
        mx = arr.max(axis=1)
        agg = np.stack([mean, mn, mx], axis=2).reshape(b, _F * 3)
        glob = np.empty((b, len(GLOBALS)), dtype=np.float32)
        glob[:, 0] = np.minimum(np.asarray(sizes, dtype=np.float32) / self.size_norm, 2.0)
        glob[:, 1] = n / float(MAX_FEATURE_EXAMPLES)
        glob[:, 2] = self.has0
        glob[:, 3] = self.has1
        glob[:, 4] = self.mean_tlen
        out = np.concatenate([agg, glob], axis=1)
        return out
