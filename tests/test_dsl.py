import json

import pytest
from hypothesis import given, settings, strategies as st

from traceforge import dsl
from traceforge.dsl import (
    Concat, INVALID_LENGTH, INVALID_TOKEN, Input, Invalid, Literal, Lower, ProgramError, Replace, Slice, Token, Trim, Upper,
    ast_size, evaluate, from_json, is_invalid, pretty, to_json,
)

I0, I1 = Input(0), Input(1)


def test_invalid_is_distinct_from_empty_string():
    out = evaluate(Token(I0, ",", 3), ["a,b"])
    assert is_invalid(out) and out is INVALID_TOKEN
    assert out != "" and "" != out
    assert evaluate(Slice(I0, 5, 9), [""]) == ""  # valid empty string
    assert not is_invalid(evaluate(I0, [""]))
    assert evaluate(Concat(Token(I0, ",", 3), I0), ["a"]) is INVALID_TOKEN  # invalid propagates


def test_trim_uses_python_whitespace_including_unicode():
    assert evaluate(Trim(I0), ["  \tx \n"]) == "x"
    assert evaluate(Trim(I0), ["   "]) == ""
    assert evaluate(Trim(I0), [""]) == ""


def test_case_operations_follow_python_semantics():
    assert evaluate(Upper(I0), ["Straße"]) == "STRASSE"  # length grows
    assert len(evaluate(Lower(I0), ["İ"])) == 2  # code-point count changes
    assert evaluate(Lower(I0), ["ÀÉ"]) == "àé"


def test_token_semantics_repeated_negative_and_out_of_range():
    assert evaluate(Token(I0, ",", 1), ["a,,b"]) == ""  # repeated delimiter -> empty token
    assert evaluate(Token(I0, ",", 2), ["a,,b"]) == "b"
    assert evaluate(Token(I0, ",", -1), ["a,,b"]) == "b"
    assert evaluate(Token(I0, ",", -3), ["a,,b"]) == "a"
    assert evaluate(Token(I0, ",", -4), ["a,,b"]) is INVALID_TOKEN
    assert evaluate(Token(I0, ",", 0), ["nodelim"]) == "nodelim"
    assert evaluate(Token(I0, ",", 1), ["nodelim"]) is INVALID_TOKEN
    assert evaluate(Token(I0, ",", 0), [""]) == ""
    assert evaluate(Token(I0, "ab", 1), ["xabyabz"]) == "y"  # multi-character delimiter
    assert evaluate(Token(I0, ",", -1), [""]) == ""


def test_slice_python_semantics():
    assert evaluate(Slice(I0, 0, 1), ["héllo"]) == "h"
    assert evaluate(Slice(I0, -3, None), ["abcdef"]) == "def"
    assert evaluate(Slice(I0, 1, -1), ["abcdef"]) == "bcde"
    assert evaluate(Slice(I0, 2, 1), ["abc"]) == ""
    assert evaluate(Slice(I0, 0, 1), [""]) == ""


def test_replace_replaces_all_literally():
    assert evaluate(Replace(I0, ".", ""), ["a.b.c"]) == "abc"
    assert evaluate(Replace(I0, "a", "aa"), ["aaa"]) == "aaaaaa"  # no regex, no recursion
    assert evaluate(Replace(I0, "*", "x"), ["a*b"]) == "axb"
    with pytest.raises(ProgramError):
        Replace(I0, "", "x")


def test_concat_operand_order_matters():
    assert evaluate(Concat(I0, I1), ["ab", "cd"]) == "abcd"
    assert evaluate(Concat(I1, I0), ["ab", "cd"]) == "cdab"


def test_input_column_must_exist():
    with pytest.raises(ProgramError):
        evaluate(I1, ["only one"])
    with pytest.raises(ProgramError):
        dsl.validate_program(Concat(I0, I1), 1)
    with pytest.raises(ProgramError):
        Input(2)


def test_length_bound_returns_resource_invalid_not_allocation():
    big = "a" * 100
    assert evaluate(Concat(I0, I0), [big], max_len=150) is INVALID_LENGTH
    assert evaluate(Replace(I0, "a", "bbbb"), [big], max_len=150) is INVALID_LENGTH
    assert evaluate(I0, [big], max_len=50) is INVALID_LENGTH
    assert evaluate(Concat(I0, I0), [big], max_len=200) == big + big


def test_ast_size_counting_rules():
    email = Concat(Slice(Lower(I0), 0, 1), Concat(Literal("."), Lower(I1)))
    assert ast_size(email) == 8  # spec example
    name = Lower(Replace(Trim(I0), " ", "."))
    assert ast_size(name) == 4  # literal parameters add no nodes
    assert ast_size(Token(I0, " ", 0)) == 2 and ast_size(Slice(I0, 0, None)) == 2
    sub = Slice(Lower(I0), 0, 1)  # size 3
    assert ast_size(Concat(sub, sub)) == 7  # repeated subtrees counted repeatedly
    assert ast_size(Concat(I0, I0)) == 3


def test_pretty_printer_matches_spec_example_and_is_canonical():
    email = Concat(Slice(Lower(I0), 0, 1), Concat(Literal("."), Lower(I1)))
    assert pretty(email) == (
        'Concat(\n  Slice(Lower(Input(0)), 0, 1),\n  Concat(Literal("."), Lower(Input(1)))\n)'
    )
    assert pretty(Lower(Replace(Trim(I0), " ", "."))) == 'Lower(Replace(Trim(Input(0)), " ", "."))'
    assert dsl.canonical_json(email) == dsl.canonical_json(from_json(to_json(email)))


def test_from_json_is_strict():
    with pytest.raises(ProgramError):
        from_json({"op": "Eval", "code": "1+1"})
    with pytest.raises(ProgramError):
        from_json({"op": "Input", "index": 0, "extra": 1})
    with pytest.raises(ProgramError):
        from_json({"op": "Token", "arg": {"op": "Input", "index": 0}, "delimiter": "", "index": 0})
    with pytest.raises(ProgramError):
        from_json({"op": "Slice", "arg": {"op": "Input", "index": 0}, "start": True, "stop": None})
    deep = {"op": "Input", "index": 0}
    for _ in range(100):
        deep = {"op": "Trim", "arg": deep}
    with pytest.raises(ProgramError):
        from_json(deep)


params = st.tuples(st.sampled_from([",", " ", "-", "ab"]), st.integers(-3, 3))
slices = st.tuples(st.integers(-5, 5), st.one_of(st.none(), st.integers(-5, 5)))


@st.composite
def asts(draw, depth=3):
    if depth == 0 or draw(st.booleans()):
        return draw(st.one_of(st.builds(Input, st.sampled_from([0, 1])), st.builds(Literal, st.sampled_from(["", ".", "ab"]))))
    kind = draw(st.sampled_from(["Trim", "Lower", "Upper", "Token", "Slice", "Replace", "Concat"]))
    if kind == "Concat":
        return Concat(draw(asts(depth - 1)), draw(asts(depth - 1)))
    child = draw(asts(depth - 1))
    if kind == "Token":
        d, i = draw(params)
        return Token(child, d, i)
    if kind == "Slice":
        a, b = draw(slices)
        return Slice(child, a, b)
    if kind == "Replace":
        return Replace(child, draw(st.sampled_from([",", " ", "a"])), draw(st.sampled_from(["", "x", "yy"])))
    return {"Trim": Trim, "Lower": Lower, "Upper": Upper}[kind](child)


@settings(max_examples=200, deadline=None)
@given(asts())
def test_json_roundtrip_and_deterministic_serialization(node):
    again = from_json(json.loads(json.dumps(to_json(node))))
    assert again == node
    assert dsl.canonical_json(again) == dsl.canonical_json(node)
    assert ast_size(again) == ast_size(node)


@settings(max_examples=200, deadline=None)
@given(asts(), st.text(max_size=12), st.text(max_size=12))
def test_evaluate_total_on_valid_programs(node, a, b):
    out = evaluate(node, [a, b], max_len=64)
    assert isinstance(out, str) or isinstance(out, Invalid)
    if isinstance(out, str):
        assert len(out) <= 64
