"""Finite, task-specific transformation vocabulary.

Every policy receives *exactly* the vocabulary built here, and it is built **only from the
supplied examples** (inputs and outputs) plus fixed constants - never from hidden rows or a
reference program.

Limits and tie-breaking (also in docs/architecture.md):

* literals   <= 12: fixed ``("", ".", " ", "-", "_", "@")`` first, then up to 6 recurring output
  fragments.  A fragment is a substring (1..12 chars) of the outputs that (a) is not already a
  literal, (b) occurs in the output of >= ceil(0.75 n) examples *without* occurring in that
  example's inputs (so it must be synthesized text, not copied text), and (c) is maximal
  (no longer candidate contains it with equal support).  Rank: support desc, length desc, text.
* delimiters <= 4: candidates are the fixed list ``" @.,-_/:;|"`` plus any other observed
  non-alphanumeric, non-letter character; kept if present in some example input; rank by number
  of examples containing it (desc), then fixed-list position (then code point).
* token indices: 0, 1, -1.
* slice pairs: 12 fixed (start, stop) pairs.
* replace pairs <= 20: old in delimiters, new in single-character/empty literals, old != new,
  old-major order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .tasks import TaskSpec

FIXED_LITERALS: tuple[str, ...] = ("", ".", " ", "-", "_", "@")
FIXED_DELIMITERS: tuple[str, ...] = (" ", "@", ".", ",", "-", "_", "/", ":", ";", "|")
TOKEN_INDICES: tuple[int, ...] = (0, 1, -1)
SLICE_PAIRS: tuple[tuple[int, int | None], ...] = (
    (0, 1), (0, 2), (0, 3), (0, 4), (1, None), (2, None),
    (1, 2), (0, -1), (-1, None), (-2, None), (-3, None), (1, 3),
)
MAX_LITERALS = 12
MAX_DELIMITERS = 4
MAX_REPLACE_PAIRS = 20
MAX_FRAGMENT_LEN = 12


@dataclass(frozen=True)
class Vocabulary:
    n_inputs: int
    literals: tuple[str, ...]
    delimiters: tuple[str, ...]
    token_indices: tuple[int, ...]
    slice_pairs: tuple[tuple[int, int | None], ...]
    replace_pairs: tuple[tuple[str, str], ...]

    def unary_ops(self) -> list[tuple[str, tuple]]:
        """Deterministic list of (op name, parameter tuple) for every unary operator."""
        ops: list[tuple[str, tuple]] = [("Trim", ()), ("Lower", ()), ("Upper", ())]
        ops += [("Token", (d, i)) for d in self.delimiters for i in self.token_indices]
        ops += [("Slice", p) for p in self.slice_pairs]
        ops += [("Replace", p) for p in self.replace_pairs]
        return ops

    def to_dict(self) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "literals": list(self.literals),
            "delimiters": list(self.delimiters),
            "token_indices": list(self.token_indices),
            "slice_pairs": [list(p) for p in self.slice_pairs],
            "replace_pairs": [list(p) for p in self.replace_pairs],
            "limits": {
                "max_literals": MAX_LITERALS,
                "max_delimiters": MAX_DELIMITERS,
                "max_replace_pairs": MAX_REPLACE_PAIRS,
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "Vocabulary":
        return Vocabulary(
            d["n_inputs"], tuple(d["literals"]), tuple(d["delimiters"]), tuple(d["token_indices"]),
            tuple((a, b) for a, b in d["slice_pairs"]), tuple((a, b) for a, b in d["replace_pairs"]),
        )

    def contains_program_params(self, node) -> str | None:
        """Return None if every parameter of ``node`` is representable, else a reason string."""
        from . import dsl

        for n in dsl.iter_nodes(node):
            t = type(n)
            if t is dsl.Input and n.index >= self.n_inputs:
                return "input_column_missing"
            if t is dsl.Literal and n.text not in self.literals:
                return "literal_not_in_vocab"
            if t is dsl.Token and (n.delimiter not in self.delimiters or n.index not in self.token_indices):
                return "token_param_not_in_vocab"
            if t is dsl.Slice and (n.start, n.stop) not in self.slice_pairs:
                return "slice_param_not_in_vocab"
            if t is dsl.Replace and (n.old, n.new) not in self.replace_pairs:
                return "replace_param_not_in_vocab"
        return None


def _recurring_fragments(spec: "TaskSpec", limit: int) -> list[str]:
    n = len(spec.examples)
    need = max(2, ceil(0.75 * n))
    support: dict[str, int] = {}
    for e in spec.examples:
        joined_in = "\x00".join(e.inputs)
        frags = {
            e.output[i:j]
            for i in range(len(e.output))
            for j in range(i + 1, min(len(e.output), i + MAX_FRAGMENT_LEN) + 1)
        }
        for f in frags:
            if f not in joined_in:
                support[f] = support.get(f, 0) + 1
    cands = [f for f, c in support.items() if c >= need and f not in FIXED_LITERALS]
    maximal = [
        f for f in cands
        if not any(g != f and f in g and support[g] == support[f] for g in cands)
    ]
    maximal.sort(key=lambda f: (-support[f], -len(f), f))
    return maximal[:limit]


def _delimiters(spec: "TaskSpec") -> list[str]:
    inputs = [e.inputs for e in spec.examples]
    observed = {ch for row in inputs for s in row for ch in s if not ch.isalnum() and not ch.isspace() and len(ch) == 1}
    candidates = list(FIXED_DELIMITERS) + sorted(c for c in observed if c not in FIXED_DELIMITERS)
    rank = {d: i for i, d in enumerate(candidates)}
    count = {d: sum(any(d in s for s in row) for row in inputs) for d in candidates}
    present = [d for d in candidates if count[d] > 0]
    present.sort(key=lambda d: (-count[d], rank[d], d))
    return present[:MAX_DELIMITERS]


def build_vocabulary(spec: "TaskSpec", max_literals: int = MAX_LITERALS) -> Vocabulary:
    literals = list(FIXED_LITERALS)
    literals += _recurring_fragments(spec, max(0, max_literals - len(literals)))
    delims = _delimiters(spec)
    news = [l for l in literals if len(l) <= 1]
    replace = [(o, nw) for o in delims for nw in news if o != nw][:MAX_REPLACE_PAIRS]
    return Vocabulary(
        n_inputs=spec.n_inputs,
        literals=tuple(literals[:max_literals]),
        delimiters=tuple(delims),
        token_indices=TOKEN_INDICES,
        slice_pairs=SLICE_PAIRS,
        replace_pairs=tuple(replace),
    )


def manifest(vocab: Vocabulary, task_id: str) -> str:
    """JSON text of the saved vocabulary manifest."""
    return json.dumps({"task_id": task_id, **vocab.to_dict()}, ensure_ascii=False, indent=2, sort_keys=True)
