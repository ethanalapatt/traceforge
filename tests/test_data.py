import dataclasses

import pytest
from hypothesis import given, settings, strategies as st

from traceforge import dsl
from traceforge.data.authored import authored_tasks, regression_tasks
from traceforge.data.generator import (
    GenConfig, bigrams, check_split_disjointness, dataset_fingerprint, generate_dataset, is_compositional, load_suite, save_suites,
)
from traceforge.dsl import Concat, Input, Lower, Replace, Upper
from traceforge.profiles import load_profile
from traceforge.tasks import Budgets, fingerprint, make_search_task, spec_fingerprint


def small_cfg(seed=0, **kw):
    p = load_profile("smoke")
    base = dict(seed=seed, max_nodes=p["max_nodes"], train_max_size=p["train_max_size"], size_weights={int(k): v for k, v in p["size_weights"].items()},
                shift_enabled=p["shift_enabled"], quotas=dict(p["quotas"]))
    base.update(kw)
    return GenConfig(**base)


@pytest.fixture(scope="module")
def smoke_data():
    return generate_dataset(small_cfg())


def test_generator_fills_quotas_and_reports_rejections(smoke_data):
    suites, report = smoke_data
    assert report["complete"]
    assert {k: len(v) for k, v in suites.items()} == small_cfg().quotas
    assert sum(report["rejections"].values()) > 0 and 0 < report["rejection_rate_among_built"] < 1


def test_splits_are_group_exact_and_behaviour_disjoint(smoke_data):
    suites, _ = smoke_data
    audit = check_split_disjointness(suites)
    assert audit["ok"], audit
    owner = {}
    for part, tasks in suites.items():
        for t in tasks:
            assert owner.setdefault(t.skeleton, part) == part


def test_disjointness_audit_detects_a_planted_overlap(smoke_data):
    suites, _ = smoke_data
    leaked = {k: list(v) for k, v in suites.items()}
    leaked["dev"] = leaked["dev"] + [dataclasses.replace(leaked["train"][0], suite="dev")]
    audit = check_split_disjointness(leaked)
    assert not audit["ok"] and audit["group_overlaps"] and audit["exact_overlaps"]


def test_generation_is_deterministic_and_seed_sensitive(tmp_path, smoke_data):
    suites, report = smoke_data
    again, _ = generate_dataset(small_cfg())
    other, _ = generate_dataset(small_cfg(seed=1))
    for name, d in (("a", suites), ("b", again), ("c", other)):
        save_suites(tmp_path / name, d, report)
    names = list(suites)
    fa, fb, fc = (dataset_fingerprint(tmp_path / n, names) for n in "abc")
    assert fa == fb and fa != fc


def test_saved_tasks_roundtrip_and_hide_labels_from_spec(tmp_path, smoke_data):
    suites, report = smoke_data
    save_suites(tmp_path, suites, report)
    back = load_suite(tmp_path, "dev")
    assert [spec_fingerprint(t.spec) for t in back] == [spec_fingerprint(t.spec) for t in suites["dev"]]
    t = back[0]
    assert t.hidden and t.reference is not None and not hasattr(t.spec, "hidden")
    # supplied examples never leak hidden rows
    assert not ({e.inputs for e in t.spec.examples} & {h.inputs for h in t.hidden})


def test_every_generated_reference_reproduces_supplied_and_hidden_rows(smoke_data):
    suites, _ = smoke_data
    for tasks in suites.values():
        for t in tasks:
            for e in list(t.spec.examples) + list(t.hidden):
                assert dsl.evaluate(t.reference, e.inputs) == e.output
            assert t.vocab_ok if hasattr(t, "vocab_ok") else make_search_task(t.spec, Budgets()).vocab.contains_program_params(t.reference) is None


def test_group_key_merges_case_and_is_stable():
    a = dsl.skeleton(Lower(Concat(Input(0), Input(0))))
    b = dsl.skeleton(Upper(Concat(Input(0), Input(0))))
    assert a == b
    assert dsl.skeleton(Replace(Input(0), "a", "b")) == dsl.skeleton(Replace(Input(0), "x", "y"))


@settings(max_examples=60, deadline=None)
@given(st.sampled_from([Lower, Upper]), st.sampled_from([Lower, Upper]))
def test_shift_status_is_constant_within_a_skeleton_group(c1, c2):
    """Regression: held-out Upper∘Concat once split a Case(Concat(..)) group across train/comp_dev."""
    cfg = small_cfg(shift_enabled=True, train_max_size=9)
    p1, p2 = c1(Concat(Input(0), Input(0))), c2(Concat(Input(0), Input(0)))
    assert dsl.skeleton(p1) == dsl.skeleton(p2)
    assert is_compositional(p1, cfg) == is_compositional(p2, cfg)
    assert ("Case", "Concat") in bigrams(p1)


def test_authored_tasks_are_independent_and_consistent():
    tasks = authored_tasks()
    assert len(tasks) == 40 and len({t.id for t in tasks}) == 40
    assert len(regression_tasks()) == 6
    for t in tasks:
        assert 3 <= len(t.spec.examples) <= 6 and t.hidden
        for e in list(t.spec.examples) + list(t.hidden):
            assert dsl.evaluate(t.reference, e.inputs) == e.output, t.id
        v = make_search_task(t.spec, Budgets(max_nodes=9)).vocab
        assert v.contains_program_params(t.reference) is None, (t.id, v.contains_program_params(t.reference))


def test_fingerprints_are_stable_across_key_order():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
