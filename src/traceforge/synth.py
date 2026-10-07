"""User-facing synthesis (``traceforge synthesize``) and CSV application (``traceforge apply``)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import dsl
from . import features as F
from .models import CheckpointError, choose_inference_device, load_neural, load_tree
from .policies import FeatureScorer, heuristic_scorer, neural_scorer, tree_scorer
from .search import search
from .tasks import Budgets, TaskSpec, make_search_task
from .vocabulary import manifest as vocab_manifest

PROGRAM_KIND = "traceforge.program"
DEFAULT_HEURISTIC_LAMBDA = 3
CONSISTENCY_NOTE = (
    "This program matches every supplied example (example consistency). That is not proof that it "
    "is correct on other inputs: several programs can fit the same examples."
)


class ApplyError(ValueError):
    """Bad program file / CSV / arguments for ``apply``."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scorer(policy: str, checkpoint: str | None, device: str, lam: int | None) -> tuple[FeatureScorer | None, int, dict]:
    """Build the scorer for a policy.  Never substitutes an untrained model for a missing
    checkpoint: that raises :class:`CheckpointError` with an actionable message."""
    info: dict = {"policy": policy}
    if policy == "uniform":
        return None, 0, info
    if policy == "heuristic":
        return heuristic_scorer(), DEFAULT_HEURISTIC_LAMBDA if lam is None else lam, info
    if checkpoint is None:
        raise CheckpointError(
            f"--policy {policy} requires --checkpoint (e.g. runs/dev/models/{policy if policy == 'neural' else 'tree'}"
            f"{'.pt' if policy == 'neural' else '.joblib'}). Train one with `traceforge reproduce --profile smoke "
            "--output runs/smoke`, or use --policy uniform to run without a model."
        )
    path = Path(checkpoint)
    if policy == "tree":
        model, meta = load_tree(path)
        info.update(checkpoint=str(path), sha256=_sha256(path), selected_lambda=meta.get("selected_lambda"))
        use = lam if lam is not None else meta.get("selected_lambda")
        if use is None:
            raise CheckpointError("checkpoint has no selected penalty; pass --lam")
        return tree_scorer(model), int(use), info
    if policy == "neural":
        bundle = load_neural(path, "cpu")
        probe = np.random.default_rng(0).random((256, F.FEATURE_DIM)).astype(np.float32)
        choice = choose_inference_device(bundle, probe, device)
        bundle.to(choice.device)
        info.update(
            checkpoint=str(path), sha256=_sha256(path), inference_device=choice.to_dict(),
            selected_lambda=bundle.meta.get("selected_lambda"), val_metrics=bundle.meta.get("val_metrics"),
            seed=bundle.meta.get("seed"), dataset_fingerprint=bundle.meta.get("dataset_fingerprint"),
        )
        use = lam if lam is not None else bundle.meta.get("selected_lambda")
        if use is None:
            raise CheckpointError("checkpoint has no selected penalty; pass --lam")
        return neural_scorer(bundle), int(use), info
    raise ValueError(f"unknown policy {policy!r}")


def synthesize(
    spec: TaskSpec,
    policy: str,
    checkpoint: str | None,
    budgets: Budgets,
    out_dir: str | Path,
    lam: int | None = None,
    device: str = "cpu",
) -> dict:
    scorer, use_lam, info = load_scorer(policy, checkpoint, device, lam)
    task = make_search_task(spec, budgets)
    res = search(task, scorer, use_lam, trace_detail=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocabulary.json").write_text(vocab_manifest(task.vocab, spec.id), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "task_id": spec.id,
        "input_columns": list(spec.input_columns),
        "outcome": res.outcome,
        "search": res.to_dict(),
        "model": info,
        "budgets": budgets.to_dict(),
        "examples": spec.to_dict()["examples"],
    }
    if res.program is not None:
        rows = [e.inputs for e in spec.examples]
        outs = dsl.evaluate_rows(res.program, rows, budgets.max_string_len)
        program_doc = {
            "schema_version": 1,
            "kind": PROGRAM_KIND,
            "task_id": spec.id,
            "input_columns": list(spec.input_columns),
            "program": dsl.to_json(res.program),
            "program_text": dsl.pretty(res.program),
            "program_size": dsl.ast_size(res.program),
            "example_consistent": all(o == e.output for o, e in zip(outs, spec.examples)),
            "note": CONSISTENCY_NOTE,
            "policy": policy,
            "lam": use_lam,
            "model": info,
        }
        (out / "program.json").write_text(json.dumps(program_doc, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        (out / "program.txt").write_text(dsl.pretty(res.program) + "\n", encoding="utf-8")
        summary["program"] = program_doc
    (out / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    (out / "trace.json").write_text(json.dumps(res.trace, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------ apply


def load_program_file(path: str | Path) -> tuple[dsl.Node, dict]:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ApplyError(f"program file not found: {p}") from None
    except json.JSONDecodeError as e:
        raise ApplyError(f"{p}: invalid JSON ({e})") from None
    if not isinstance(doc, dict) or doc.get("kind") != PROGRAM_KIND:
        raise ApplyError(f"{p}: not a TraceForge program file (kind != {PROGRAM_KIND!r})")
    if doc.get("schema_version") != 1:
        raise ApplyError(f"{p}: unsupported schema_version {doc.get('schema_version')!r}")
    try:
        node = dsl.from_json(doc.get("program"))
    except dsl.ProgramError as e:
        raise ApplyError(f"{p}: invalid program: {e}") from None
    cols = doc.get("input_columns")
    if not isinstance(cols, list) or not 1 <= len(cols) <= 2:
        raise ApplyError(f"{p}: 'input_columns' must list 1 or 2 columns")
    try:
        dsl.validate_program(node, len(cols))
    except dsl.ProgramError as e:
        raise ApplyError(f"{p}: {e}") from None
    return node, doc


def apply_program(
    program_path: str | Path,
    input_csv: str | Path,
    columns: list[str],
    output_csv: str | Path,
    output_column: str = "traceforge_output",
    error_column: str = "traceforge_error",
    force: bool = False,
    max_len: int = dsl.DEFAULT_MAX_STRING_LEN * 4,
) -> dict:
    """Apply a program to CSV rows.  Writes a *new* file (never overwrites the input),
    preserves every input row and column, and reports row-level errors in ``error_column``."""
    node, doc = load_program_file(program_path)
    need = len(doc["input_columns"])
    if len(columns) != need:
        raise ApplyError(f"program expects {need} input column(s) ({doc['input_columns']}), got --columns {columns}")
    src, dst = Path(input_csv), Path(output_csv)
    if not src.exists():
        raise ApplyError(f"input CSV not found: {src}")
    if src.resolve() == dst.resolve():
        raise ApplyError("refusing to overwrite the input CSV; choose a different --output")
    if dst.exists() and not force:
        raise ApplyError(f"{dst} already exists; pass --force to replace it")
    with src.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ApplyError("input CSV has no header row")
        if len(set(fields)) != len(fields):
            raise ApplyError("input CSV has duplicate column names")
        missing = [c for c in columns if c not in fields]
        if missing:
            raise ApplyError(f"column(s) {missing} not in CSV header {fields}")
        for extra in (output_column, error_column):
            if extra in fields:
                raise ApplyError(f"CSV already has a column named {extra!r}; choose --output-column")
        rows = list(reader)
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + [output_column, error_column], extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows, start=2):  # line 1 is the header
            vals = [row.get(c) for c in columns]
            out = dict(row)
            if any(v is None for v in vals):
                out[output_column], out[error_column] = "", "missing value in input column"
            else:
                v = dsl.evaluate(node, vals, max_len)
                if isinstance(v, str):
                    out[output_column], out[error_column] = v, ""
                else:
                    out[output_column], out[error_column] = "", f"invalid: {v.reason}"
            if out[error_column]:
                errors.append({"line": i, "error": out[error_column]})
            w.writerow({k: ("" if val is None else val) for k, val in out.items() if k in fields + [output_column, error_column]})
    return {"rows": len(rows), "errors": len(errors), "error_rows": errors[:50], "output": str(dst)}
