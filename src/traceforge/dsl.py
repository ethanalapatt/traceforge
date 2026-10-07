"""TraceForge DSL: immutable AST, canonical printing, JSON round trips, interpreter.

Semantics are *Python string semantics* (code points, ``str.strip/lower/upper/split``,
slicing).  They deliberately do not claim to match any external benchmark's string
semantics (see docs/architecture.md, "DSL semantics").

Invalid results are represented by :class:`Invalid` singletons and are always distinct from
the valid empty string ``""``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterator, Sequence, Union

DEFAULT_MAX_STRING_LEN = 256
MAX_SLICE_BOUND = 64
MAX_TOKEN_INDEX = 8
MAX_PARSE_NODES = 64
SCHEMA_VERSION = 1

UNARY_OPS = ("Trim", "Lower", "Upper", "Token", "Slice", "Replace")
ALL_OPS = ("Input", "Literal") + UNARY_OPS + ("Concat",)


class ProgramError(ValueError):
    """Raised for malformed programs (bad schema, bad parameters, missing columns)."""


class Invalid:
    """Result of an invalid evaluation.  Never equal to a string, including ``""``."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"Invalid({self.reason!r})"


INVALID_TOKEN = Invalid("token_out_of_range")
INVALID_LENGTH = Invalid("max_length_exceeded")

Value = Union[str, Invalid]


def is_invalid(v: object) -> bool:
    return v.__class__ is Invalid


# --------------------------------------------------------------------------- AST nodes


class Node:
    __slots__ = ()

    @property
    def op(self) -> str:
        return type(self).__name__

    def children(self) -> tuple["Node", ...]:
        return ()


def _check_int(name: str, v: object, bound: int) -> None:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProgramError(f"{name} must be an int, got {v!r}")
    if abs(v) > bound:
        raise ProgramError(f"{name}={v} outside bound ±{bound}")


@dataclass(frozen=True, slots=True)
class Input(Node):
    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index not in (0, 1):
            raise ProgramError(f"Input index must be 0 or 1, got {self.index!r}")


@dataclass(frozen=True, slots=True)
class Literal(Node):
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ProgramError("Literal text must be a string")


@dataclass(frozen=True, slots=True)
class Trim(Node):
    arg: Node

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Lower(Node):
    arg: Node

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Upper(Node):
    arg: Node

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Concat(Node):
    left: Node
    right: Node

    def children(self) -> tuple[Node, ...]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class Token(Node):
    arg: Node
    delimiter: str
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.delimiter, str) or self.delimiter == "":
            raise ProgramError("Token delimiter must be a nonempty string")
        _check_int("Token index", self.index, MAX_TOKEN_INDEX)

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Slice(Node):
    arg: Node
    start: int
    stop: int | None

    def __post_init__(self) -> None:
        _check_int("Slice start", self.start, MAX_SLICE_BOUND)
        if self.stop is not None:
            _check_int("Slice stop", self.stop, MAX_SLICE_BOUND)

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Replace(Node):
    arg: Node
    old: str
    new: str

    def __post_init__(self) -> None:
        if not isinstance(self.old, str) or self.old == "":
            raise ProgramError("Replace old must be a nonempty string")
        if not isinstance(self.new, str):
            raise ProgramError("Replace new must be a string")

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


# ------------------------------------------------------------------ structural helpers


def ast_size(node: Node) -> int:
    """AST node count.  Parameters inside Token/Slice/Replace add no nodes; shared subtrees
    are counted every time they occur."""
    t = type(node)
    if t is Input or t is Literal:
        return 1
    if t is Concat:
        return 1 + ast_size(node.left) + ast_size(node.right)
    return 1 + ast_size(node.arg)  # type: ignore[attr-defined]


def iter_nodes(node: Node) -> Iterator[Node]:
    """Pre-order traversal (repeated subtrees are yielded repeatedly)."""
    yield node
    for c in node.children():
        yield from iter_nodes(c)


def unary_params(node: Node) -> tuple:
    t = type(node)
    if t is Token:
        return (node.delimiter, node.index)  # type: ignore[attr-defined]
    if t is Slice:
        return (node.start, node.stop)  # type: ignore[attr-defined]
    if t is Replace:
        return (node.old, node.new)  # type: ignore[attr-defined]
    return ()


def make_unary(op: str, params: tuple, child: Node) -> Node:
    if op == "Trim":
        return Trim(child)
    if op == "Lower":
        return Lower(child)
    if op == "Upper":
        return Upper(child)
    if op == "Token":
        return Token(child, params[0], params[1])
    if op == "Slice":
        return Slice(child, params[0], params[1])
    if op == "Replace":
        return Replace(child, params[0], params[1])
    raise ProgramError(f"unknown unary op {op!r}")


def max_input_index(node: Node) -> int:
    idx = [n.index for n in iter_nodes(node) if isinstance(n, Input)]
    return max(idx) if idx else -1


def validate_program(node: Node, n_inputs: int) -> None:
    if max_input_index(node) >= n_inputs:
        raise ProgramError(f"program references Input({max_input_index(node)}) but task has {n_inputs} column(s)")


# ------------------------------------------------------------------------ interpreter


@lru_cache(maxsize=4096)
def unary_fn(op: str, params: tuple, max_len: int) -> Callable[[str], Value]:
    """Return the per-string implementation of a unary operator.

    Every operator here is the single source of truth used by both the tree interpreter
    and the bottom-up search, so search and evaluation cannot disagree.
    """
    if op == "Trim":
        return str.strip  # type: ignore[return-value]  # never lengthens a string
    if op == "Lower":

        def lower(s: str) -> Value:
            r = s.lower()  # may lengthen for some code points (e.g. 'İ')
            return r if len(r) <= max_len else INVALID_LENGTH

        return lower
    if op == "Upper":

        def upper(s: str) -> Value:
            r = s.upper()  # may lengthen (e.g. 'ß' -> 'SS')
            return r if len(r) <= max_len else INVALID_LENGTH

        return upper
    if op == "Token":
        d, i = params
        if i == 0:
            return lambda s: s.split(d, 1)[0]
        if i == -1:
            return lambda s: s.rsplit(d, 1)[-1]
        if i > 0:

            def tok_pos(s: str) -> Value:
                p = s.split(d, i + 1)
                return p[i] if len(p) > i else INVALID_TOKEN

            return tok_pos

        def tok_neg(s: str) -> Value:
            p = s.split(d)
            return p[i] if len(p) >= -i else INVALID_TOKEN

        return tok_neg
    if op == "Slice":
        a, b = params
        return lambda s: s[a:b]
    if op == "Replace":
        old, new = params
        grow = len(new) - len(old)

        def replace(s: str) -> Value:
            if grow > 0 and len(s) + s.count(old) * grow > max_len:
                return INVALID_LENGTH  # refuse before allocating
            return s.replace(old, new)

        return replace
    raise ProgramError(f"unknown unary op {op!r}")


def concat_values(a: str, b: str, max_len: int) -> Value:
    if len(a) + len(b) > max_len:
        return INVALID_LENGTH
    return a + b


def evaluate(node: Node, inputs: Sequence[str], max_len: int = DEFAULT_MAX_STRING_LEN) -> Value:
    """Evaluate ``node`` on one row.  Never uses eval/exec.  Missing input columns raise
    :class:`ProgramError` (a program/schema mismatch), data-dependent failures return
    :class:`Invalid`."""
    t = type(node)
    if t is Input:
        if node.index >= len(inputs):
            raise ProgramError(f"Input({node.index}) but row has {len(inputs)} column(s)")
        v = inputs[node.index]
        return v if len(v) <= max_len else INVALID_LENGTH
    if t is Literal:
        return node.text if len(node.text) <= max_len else INVALID_LENGTH
    if t is Concat:
        a = evaluate(node.left, inputs, max_len)
        if a.__class__ is Invalid:
            return a
        b = evaluate(node.right, inputs, max_len)
        if b.__class__ is Invalid:
            return b
        return concat_values(a, b, max_len)  # type: ignore[arg-type]
    c = evaluate(node.arg, inputs, max_len)  # type: ignore[attr-defined]
    if c.__class__ is Invalid:
        return c
    return unary_fn(node.op, unary_params(node), max_len)(c)  # type: ignore[arg-type]


def evaluate_rows(node: Node, rows: Sequence[Sequence[str]], max_len: int = DEFAULT_MAX_STRING_LEN) -> list[Value]:
    return [evaluate(node, r, max_len) for r in rows]


# -------------------------------------------------------------------- serialization


def to_json(node: Node) -> dict:
    t = type(node)
    if t is Input:
        return {"op": "Input", "index": node.index}
    if t is Literal:
        return {"op": "Literal", "text": node.text}
    if t is Concat:
        return {"op": "Concat", "left": to_json(node.left), "right": to_json(node.right)}
    d: dict = {"op": node.op, "arg": to_json(node.arg)}  # type: ignore[attr-defined]
    if t is Token:
        d.update(delimiter=node.delimiter, index=node.index)
    elif t is Slice:
        d.update(start=node.start, stop=node.stop)
    elif t is Replace:
        d.update(old=node.old, new=node.new)
    return d


_FIELDS = {
    "Input": {"index"},
    "Literal": {"text"},
    "Trim": {"arg"},
    "Lower": {"arg"},
    "Upper": {"arg"},
    "Concat": {"left", "right"},
    "Token": {"arg", "delimiter", "index"},
    "Slice": {"arg", "start", "stop"},
    "Replace": {"arg", "old", "new"},
}


def from_json(obj: object, max_nodes: int = MAX_PARSE_NODES) -> Node:
    """Strict, bounded parser for the JSON AST format."""
    budget = [max_nodes]

    def rec(o: object) -> Node:
        if not isinstance(o, dict):
            raise ProgramError("AST node must be an object")
        budget[0] -= 1
        if budget[0] < 0:
            raise ProgramError(f"AST exceeds {max_nodes} nodes")
        op = o.get("op")
        if op not in _FIELDS:
            raise ProgramError(f"unknown op {op!r}")
        extra = set(o) - _FIELDS[op] - {"op"}
        missing = _FIELDS[op] - set(o)
        if extra or missing:
            raise ProgramError(f"{op}: unexpected fields {sorted(extra)} / missing fields {sorted(missing)}")
        if op == "Input":
            return Input(o["index"])
        if op == "Literal":
            return Literal(o["text"])
        if op == "Concat":
            return Concat(rec(o["left"]), rec(o["right"]))
        arg = rec(o["arg"])
        if op in ("Trim", "Lower", "Upper"):
            return {"Trim": Trim, "Lower": Lower, "Upper": Upper}[op](arg)
        if op == "Token":
            return Token(arg, o["delimiter"], o["index"])
        if op == "Slice":
            return Slice(arg, o["start"], o["stop"])
        return Replace(arg, o["old"], o["new"])

    return rec(obj)


def canonical_json(node: Node) -> str:
    return json.dumps(to_json(node), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------- pretty printing


def _lit(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def to_str(node: Node) -> str:
    """Canonical single-line form."""
    t = type(node)
    if t is Input:
        return f"Input({node.index})"
    if t is Literal:
        return f"Literal({_lit(node.text)})"
    if t is Concat:
        return f"Concat({to_str(node.left)}, {to_str(node.right)})"
    inner = to_str(node.arg)  # type: ignore[attr-defined]
    if t is Token:
        return f"Token({inner}, {_lit(node.delimiter)}, {node.index})"
    if t is Slice:
        return f"Slice({inner}, {node.start}, {node.stop})"
    if t is Replace:
        return f"Replace({inner}, {_lit(node.old)}, {_lit(node.new)})"
    return f"{node.op}({inner})"


def pretty(node: Node, width: int = 64, _indent: int = 0) -> str:
    """Canonical multi-line form: a node is printed inline when it fits in ``width``
    columns, otherwise its arguments go on separate lines indented by two spaces."""
    one = to_str(node)
    if _indent + len(one) <= width or not node.children():
        return one
    pad = "  " * (_indent // 2 + 1)
    t = type(node)
    parts = [pretty(c, width, _indent + 2) for c in node.children()]
    extra: list[str] = []
    if t is Token:
        extra = [_lit(node.delimiter), str(node.index)]
    elif t is Slice:
        extra = [str(node.start), str(node.stop)]
    elif t is Replace:
        extra = [_lit(node.old), _lit(node.new)]
    lines = [pad + p for p in parts] + ([pad + ", ".join(extra)] if extra else [])
    return node.op + "(\n" + ",\n".join(lines) + "\n" + "  " * (_indent // 2) + ")"


def skeleton(node: Node) -> str:
    """Group key for dataset splits: literal values, delimiters, indices, slice/replace
    parameters are normalized away, Lower/Upper are merged into ``Case`` and input columns
    are relabelled by order of first use."""
    relabel: dict[int, int] = {}

    def rec(n: Node) -> str:
        t = type(n)
        if t is Input:
            relabel.setdefault(n.index, len(relabel))
            return f"I{relabel[n.index]}"
        if t is Literal:
            return "L"
        if t is Concat:
            return f"Concat({rec(n.left)},{rec(n.right)})"
        name = "Case" if t in (Lower, Upper) else n.op
        return f"{name}({rec(n.arg)})"  # type: ignore[attr-defined]

    return rec(node)
