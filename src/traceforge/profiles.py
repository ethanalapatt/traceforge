"""Resource profiles.  These are adjustable *starting configurations*, not measured runtimes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

PROFILES: dict[str, dict] = {
    "smoke": {
        "name": "smoke",
        "seed": 11,
        "max_nodes": 5,  # deliberately restricted grammar so timing limits cannot flake functional checks
        "train_max_size": 4,
        "size_weights": {"2": 2, "3": 4, "4": 5},
        "shift_enabled": False,
        "quotas": {"train": 96, "val": 36, "dev": 6},
        "eval_suites": {"dev": 6, "regression": 6},
        "seconds": 1.0,
        "max_candidates": 5000,
        "neural_seeds": 1,
        "epochs": 15,
        "lambda_grid": [1, 2, 3, 5],
        "tune_tasks": 12,
        "analysis_tasks": 6,
        "nomemo_tasks": 6,
        "nomemo_max_candidates": 3000,
        "label": {"neg_search_candidates": 3000, "neg_search_seconds": 1.0, "max_rows_per_task": 60},
        "max_total_seconds": None,
    },
    "dev": {
        "name": "dev",
        "seed": 21,
        "max_nodes": 7,
        "train_max_size": 5,
        "size_weights": {"2": 1, "3": 3, "4": 4, "5": 4, "6": 3, "7": 2},
        "shift_enabled": True,
        "quotas": {"train": 1500, "val": 300, "dev": 30, "comp_dev": 15},
        "eval_suites": {"dev": 30, "comp_dev": 15, "authored": 15},
        "seconds": 3.0,
        "max_candidates": 25_000,
        "neural_seeds": 1,
        "epochs": 30,
        "lambda_grid": [1, 2, 3, 4, 6],
        "tune_tasks": 60,
        "analysis_tasks": 30,
        "nomemo_tasks": 12,
        "nomemo_max_candidates": 20_000,
        "label": {"neg_search_candidates": 6000, "neg_search_seconds": 2.0, "max_rows_per_task": 80},
        "max_total_seconds": None,
    },
    "full": {
        "name": "full",
        "seed": 31,
        "max_nodes": 9,
        "train_max_size": 7,
        "size_weights": {"2": 1, "3": 2, "4": 3, "5": 4, "6": 4, "7": 3, "8": 2, "9": 1},
        "shift_enabled": True,
        "quotas": {"train": 6000, "val": 800, "dev": 60, "test_id": 100, "comp_dev": 30, "comp_test": 60},
        "eval_suites": {"test_id": 100, "comp_test": 60, "authored": 40},
        "seconds": 5.0,
        "max_candidates": 100_000,
        "neural_seeds": 3,
        "epochs": 40,
        "lambda_grid": [1, 2, 3, 4, 6],
        "tune_tasks": 100,
        "analysis_tasks": 40,
        "nomemo_tasks": 12,
        "nomemo_max_candidates": 50_000,
        "label": {"neg_search_candidates": 10_000, "neg_search_seconds": 3.0, "max_rows_per_task": 100},
        "max_total_seconds": None,
    },
}


def load_profile(name_or_path: str) -> dict:
    """Load a built-in profile by name, or a JSON file that overrides a built-in one (its
    ``name`` field selects the base)."""
    if name_or_path in PROFILES:
        return copy.deepcopy(PROFILES[name_or_path])
    p = Path(name_or_path)
    if not p.exists():
        raise ValueError(f"unknown profile {name_or_path!r}; built-ins: {sorted(PROFILES)} (or pass a JSON file)")
    data = json.loads(p.read_text())
    base = copy.deepcopy(PROFILES.get(data.get("name", ""), PROFILES["dev"]))
    base.update(data)
    return base


def dump_profiles(directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in PROFILES.items():
        (d / f"{k}.json").write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
