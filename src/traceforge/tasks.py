"""Task schemas.

* :class:`TaskSpec`   - what a user supplies (schema, 3-6 example rows).
* :class:`Budgets`    - resource limits shared by *every* policy.
* :class:`SearchTask` - the only object the search engine ever sees.
* :class:`EvalTask`   - evaluator-only record (hidden queries, reference program, metadata).

The separation is the leakage boundary: nothing in ``search/`` imports ``EvalTask``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import dsl
from .vocabulary import Vocabulary, build_vocabulary

SCHEMA_VERSION = 1
MIN_EXAMPLES = 3
MAX_EXAMPLES = 6


class InvalidTask(ValueError):
    """Task file / examples failed validation (search outcome ``invalid_task``)."""


@dataclass(frozen=True)
class Example:
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class TaskSpec:
    id: str
    input_columns: tuple[str, ...]
    examples: tuple[Example, ...]

    @property
    def n_inputs(self) -> int:
        return len(self.input_columns)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "input_columns": list(self.input_columns),
            "examples": [{"inputs": list(e.inputs), "output": e.output} for e in self.examples],
        }


@dataclass(frozen=True)
class Budgets:
    max_nodes: int = 7
    max_candidates: int = 25_000
    max_states: int = 600_000
    seconds: float = 3.0
    max_string_len: int = dsl.DEFAULT_MAX_STRING_LEN
    max_trace_events: int = 400
    max_rss_mb: float = 8192.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchTask:
    spec: TaskSpec
    vocab: Vocabulary
    budgets: Budgets


@dataclass
class EvalTask:
    spec: TaskSpec
    hidden: tuple[Example, ...]
    reference: dsl.Node | None = None
    family: str = ""
    skeleton: str = ""
    suite: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.spec.id

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "hidden": [{"inputs": list(e.inputs), "output": e.output} for e in self.hidden],
            "reference": dsl.to_json(self.reference) if self.reference is not None else None,
            "family": self.family,
            "skeleton": self.skeleton,
            "suite": self.suite,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "EvalTask":
        spec = spec_from_dict(d["spec"], min_examples=1)
        hidden = tuple(Example(tuple(e["inputs"]), e["output"]) for e in d["hidden"])
        ref = dsl.from_json(d["reference"]) if d.get("reference") is not None else None
        return EvalTask(spec, hidden, ref, d.get("family", ""), d.get("skeleton", ""), d.get("suite", ""), d.get("meta", {}))


def spec_from_dict(
    d: Any,
    *,
    min_examples: int = MIN_EXAMPLES,
    max_examples: int = MAX_EXAMPLES,
    max_string_len: int = dsl.DEFAULT_MAX_STRING_LEN,
) -> TaskSpec:
    """Validate a task dictionary and build a :class:`TaskSpec`.  Raises :class:`InvalidTask`."""
    if not isinstance(d, dict):
        raise InvalidTask("task must be a JSON object")
    if d.get("schema_version") != SCHEMA_VERSION:
        raise InvalidTask(f"unsupported schema_version {d.get('schema_version')!r} (expected {SCHEMA_VERSION})")
    tid = d.get("id")
    if not isinstance(tid, str) or not tid:
        raise InvalidTask("task 'id' must be a nonempty string")
    cols = d.get("input_columns")
    if not isinstance(cols, list) or not 1 <= len(cols) <= 2 or not all(isinstance(c, str) and c for c in cols):
        raise InvalidTask("'input_columns' must be a list of 1 or 2 nonempty strings")
    if len(set(cols)) != len(cols):
        raise InvalidTask("'input_columns' must be distinct")
    exs = d.get("examples")
    if not isinstance(exs, list) or not min_examples <= len(exs) <= max_examples:
        raise InvalidTask(f"'examples' must be a list of {min_examples}-{max_examples} rows")
    out: list[Example] = []
    seen: dict[tuple[str, ...], str] = {}
    for k, e in enumerate(exs):
        if not isinstance(e, dict) or set(e) != {"inputs", "output"}:
            raise InvalidTask(f"example {k}: expected keys 'inputs' and 'output'")
        ins, o = e["inputs"], e["output"]
        if not isinstance(ins, list) or len(ins) != len(cols) or not all(isinstance(x, str) for x in ins):
            raise InvalidTask(f"example {k}: 'inputs' must be {len(cols)} string(s)")
        if not isinstance(o, str):
            raise InvalidTask(f"example {k}: 'output' must be a string")
        if any(len(x) > max_string_len for x in ins) or len(o) > max_string_len:
            raise InvalidTask(f"example {k}: string longer than max_string_len={max_string_len}")
        key = tuple(ins)
        if key in seen and seen[key] != o:
            raise InvalidTask(f"example {k}: contradicts an earlier example with identical inputs")
        seen[key] = o
        out.append(Example(key, o))
    return TaskSpec(tid, tuple(cols), tuple(out))


def load_task_file(path: str | Path, **kw: Any) -> TaskSpec:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InvalidTask(f"task file not found: {p}") from None
    except json.JSONDecodeError as e:
        raise InvalidTask(f"{p}: invalid JSON ({e})") from None
    return spec_from_dict(data, **kw)


def make_search_task(spec: TaskSpec, budgets: Budgets, **vocab_kw: Any) -> SearchTask:
    return SearchTask(spec, build_vocabulary(spec, **vocab_kw), budgets)


def fingerprint(obj: Any) -> str:
    """Stable SHA-256 (first 16 hex chars) of a JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def spec_fingerprint(spec: TaskSpec) -> str:
    """Exact-duplicate fingerprint: column count + ordered examples (the id is excluded)."""
    return fingerprint({"n": spec.n_inputs, "ex": [[list(e.inputs), e.output] for e in spec.examples]})
