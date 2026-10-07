import json
import math

import numpy as np
import pytest

import traceforge.training as T
from conftest import task_of
from traceforge import features as F
from traceforge.data.authored import regression_tasks
from traceforge.models import (
    CheckpointError, load_neural, model_path, random_bundle, save_neural, save_tree, load_tree, update_neural_meta,
)
from traceforge.policies import PolicyBook, heuristic_p
from traceforge.tasks import Budgets, make_search_task
from traceforge.training import LabelConfig, TrainConfig, build_rows, pr_auc, train_neural, train_tree


@pytest.fixture(scope="module")
def rows():
    cfg = LabelConfig(max_nodes=5, seed=0)
    return build_rows(regression_tasks(), cfg), cfg


def subset(rs, task_ids):
    """RowSet restricted to the given task indices (task ids remapped to 0..k-1)."""
    from traceforge.training import RowSet

    keep = sorted(task_ids)
    remap = {t: i for i, t in enumerate(keep)}
    m = np.isin(rs.task, keep)
    return RowSet(rs.X[m], rs.y[m], np.array([remap[t] for t in rs.task[m]], dtype=np.int32), rs.size[m], [rs.group[t] for t in keep])


def test_features_are_finite_and_schema_sized():
    t = make_search_task(task_of([(("Ada Lovelace",), "ada"), (("Grace Hopper",), "grace"), (("Alan Turing",), "alan")]).spec, Budgets(max_nodes=5))
    fz = F.TaskFeaturizer(t.spec, 5)
    outs = [("ada", "grace", "alan"), ("", "", ""), ("ADA LOVELACE", "GRACE HOPPER", "ALAN TURING")]
    X = fz.featurize(outs, [2, 1, 3])
    assert X.shape == (3, F.FEATURE_DIM) and np.isfinite(X).all()
    assert len(F.FEATURE_NAMES) == F.FEATURE_DIM == len(set(F.FEATURE_NAMES))


def test_schema_mismatch_is_a_clear_error():
    bad = F.schema_dict()
    bad["version"] += 1
    with pytest.raises(F.SchemaMismatch, match="retrain"):
        F.check_schema(bad)
    bad2 = F.schema_dict()
    bad2["names"] = bad2["names"][:-1]
    with pytest.raises(F.SchemaMismatch):
        F.check_schema(bad2)


def test_labels_have_both_classes_and_exclude_target(rows):
    rs, _ = rows
    assert 0 < rs.y.sum() < len(rs.y)
    assert np.isfinite(rs.X).all()


def test_training_changes_parameters_and_checkpoint_roundtrips(rows, tmp_path, monkeypatch):
    rs, _ = rows
    # use disjoint halves by task group for train/val
    ids = sorted(t for t in set(rs.task.tolist()) if rs.y[rs.task == t].sum() > 0)  # size-2 programs have no positives
    assert len(ids) >= 2
    train, val = subset(rs, ids[0::2]), subset(rs, ids[1::2])  # group disjointness is tested in test_data
    captured = {}
    orig = T.build_mlp

    def spy(*a, **k):
        m = orig(*a, **k)
        captured["init"] = [p.detach().clone() for p in m.parameters()]
        return m

    monkeypatch.setattr(T, "build_mlp", spy)
    bundle, metrics = train_neural(train, val, TrainConfig(epochs=3, seed=1, device="cpu"), meta={"seed": 1})
    moved = [not np.allclose(a.numpy(), b.detach().numpy()) for a, b in zip(captured["init"], bundle.model.parameters())]
    assert all(moved)
    assert bundle.n_params() < 1_000_000
    p = tmp_path / "neural.pt"
    save_neural(p, bundle)
    again = load_neural(p)
    x = val.X[:50]
    assert np.allclose(bundle.predict_p(x), again.predict_p(x), atol=1e-6)
    assert again.meta["feature_schema"]["dim"] == F.FEATURE_DIM
    update_neural_meta(p, selected_lambda=4)
    assert load_neural(p).meta["selected_lambda"] == 4
    assert np.all((bundle.predict_p(x) >= 0) & (bundle.predict_p(x) <= 1))


def test_loading_garbage_or_incompatible_checkpoint_raises_checkpoint_error(tmp_path):
    p = tmp_path / "neural.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_neural(p)
    with pytest.raises(CheckpointError):
        load_neural(tmp_path / "missing.pt")
    b = random_bundle(0)
    save_neural(tmp_path / "ok.pt", b)
    import torch

    blob = torch.load(tmp_path / "ok.pt", weights_only=True)
    meta = json.loads(blob["meta_json"])
    meta["feature_schema"]["version"] = 999
    blob["meta_json"] = json.dumps(meta)
    torch.save(blob, tmp_path / "bad.pt")
    with pytest.raises(CheckpointError, match="schema"):
        load_neural(tmp_path / "bad.pt")


def test_tree_checkpoint_roundtrip(rows, tmp_path):
    rs, _ = rows
    model, info = train_tree(rs, rs, seed=0)
    save_tree(tmp_path / "tree.joblib", model, {"kind": "tree", "x": 1})
    m2, meta = load_tree(tmp_path / "tree.joblib")
    assert meta["x"] == 1
    assert np.allclose(model.predict_proba(rs.X[:20]), m2.predict_proba(rs.X[:20]))


def test_pr_auc_is_correct_on_known_cases():
    y = np.array([1, 0, 1, 0])
    assert pr_auc(y, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(1.0)
    assert pr_auc(y, np.array([0.1, 0.9, 0.2, 0.8])) < 0.6


def test_heuristic_is_bounded_and_deterministic():
    X = np.random.default_rng(0).random((20, F.FEATURE_DIM)).astype(np.float32)
    p = heuristic_p(X)
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1)) and np.array_equal(p, heuristic_p(X))


def test_model_paths_are_stable_by_seed(tmp_path):
    assert model_path(tmp_path, "neural", 0).name == "neural.pt"
    assert model_path(tmp_path, "neural", 2).name == "neural_s2.pt"
    assert model_path(tmp_path, "tree", 0).suffix == ".joblib"
