"""Neural scorer, tree baseline, checkpoint I/O, and device policy."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import features as F

NORM_CLIP = 6.0
DEFAULT_HIDDEN = (256, 128)


def versions() -> dict:
    import sklearn
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


# ----------------------------------------------------------------------------- devices


def mps_capability_check() -> tuple[bool, str]:
    """Run a tiny op on MPS.  Returns (usable, message)."""
    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return False, "MPS backend not available"
    try:
        x = torch.randn(16, 16, device="mps", dtype=torch.float32)
        y = (x @ x).sum().item()
        torch.mps.synchronize()
        if not np.isfinite(y):
            return False, "MPS capability check produced non-finite output"
        return True, "MPS capability check passed"
    except Exception as e:  # pragma: no cover - platform dependent
        return False, f"MPS capability check failed: {e}"


@dataclass
class DeviceChoice:
    requested: str
    device: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"requested": self.requested, "device": self.device, "notes": self.notes}


def resolve_device(requested: str) -> DeviceChoice:
    """Resolve a *training* device.  Fallbacks are explicit in ``notes``."""
    if requested not in ("cpu", "mps", "auto"):
        raise ValueError("device must be cpu, mps or auto")
    ch = DeviceChoice(requested, "cpu")
    if requested == "cpu":
        return ch
    ok, msg = mps_capability_check()
    ch.notes.append(msg)
    if ok:
        ch.device = "mps"
    elif requested == "mps":
        ch.notes.append("requested --device mps but falling back to cpu")
    return ch


# --------------------------------------------------------------------------- normalizer


class Normalizer:
    """Standardization fit on training rows only; output clamped to ±NORM_CLIP."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @staticmethod
    def fit(x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return Normalizer(mean, np.maximum(std, 1e-3))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -NORM_CLIP, NORM_CLIP).astype(np.float32)


# ------------------------------------------------------------------------------- model


def build_mlp(dim: int = F.FEATURE_DIM, hidden: tuple[int, ...] = DEFAULT_HIDDEN):
    import torch.nn as nn

    layers: list[Any] = []
    d = dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def drop_indices(drop_features: list[str]) -> list[int]:
    return [F.FEATURE_INDEX[n] for n in drop_features]


@dataclass
class NeuralBundle:
    model: Any
    normalizer: Normalizer
    drop_features: list[str]
    meta: dict
    device: str = "cpu"

    def prepare(self, x: np.ndarray) -> np.ndarray:
        z = self.normalizer.transform(x)
        if self.drop_features:
            z[:, drop_indices(self.drop_features)] = 0.0  # masked after normalization
        return z

    def to(self, device: str) -> "NeuralBundle":
        self.model.to(device)
        self.device = device
        return self

    def predict_p(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid usefulness scores in [0, 1] (not a probability of final correctness)."""
        import torch

        z = torch.from_numpy(self.prepare(x))
        with torch.no_grad():
            logit = self.model(z.to(self.device)).squeeze(-1)
            p = torch.sigmoid(logit)
        return p.float().cpu().numpy().astype(np.float64)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))


def model_path(models_dir: str | Path, kind: str, seed: int = 0) -> Path:
    ext = "joblib" if kind.startswith("tree") else "pt"
    name = kind if seed == 0 else f"{kind}_s{seed}"
    return Path(models_dir) / f"{name}.{ext}"


def save_neural(path: str | Path, bundle: NeuralBundle, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in bundle.model.state_dict().items()}
    meta = dict(bundle.meta)
    meta.setdefault("feature_schema", F.schema_dict())
    meta["drop_features"] = list(bundle.drop_features)
    meta["versions"] = versions()
    meta["hidden"] = list(hidden)
    torch.save(
        {
            "state_dict": sd,
            "normalizer_mean": torch.from_numpy(bundle.normalizer.mean),
            "normalizer_std": torch.from_numpy(bundle.normalizer.std),
            "meta_json": json.dumps(meta, sort_keys=True, default=_json_default),
        },
        path,
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


class CheckpointError(RuntimeError):
    pass


def load_neural(path: str | Path, device: str = "cpu") -> NeuralBundle:
    """Load a portable CPU checkpoint (``weights_only`` load; no pickled code is executed)."""
    import torch

    path = Path(path)
    if not path.exists():
        raise CheckpointError(
            f"checkpoint not found: {path}\n"
            "Train one with `traceforge reproduce --profile smoke --output runs/smoke` (or "
            "`traceforge train ...`), or run with `--policy uniform` which needs no model."
        )
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        meta = json.loads(blob["meta_json"])
        F.check_schema(meta["feature_schema"])
        hidden = tuple(meta.get("hidden", DEFAULT_HIDDEN))
        model = build_mlp(F.FEATURE_DIM, hidden)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        norm = Normalizer(blob["normalizer_mean"].numpy(), blob["normalizer_std"].numpy())
    except F.SchemaMismatch as e:
        raise CheckpointError(f"{path}: {e}") from e
    except Exception as e:  # corrupt/foreign file, missing keys, shape mismatch
        raise CheckpointError(f"{path} is not a loadable TraceForge neural checkpoint ({type(e).__name__}: {str(e)[:200]})") from e
    return NeuralBundle(model, norm, list(meta.get("drop_features", [])), meta).to(device)


def update_neural_meta(path: str | Path, **updates: Any) -> None:
    import torch

    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.loads(blob["meta_json"])
    meta.update(updates)
    blob["meta_json"] = json.dumps(meta, sort_keys=True, default=_json_default)
    torch.save(blob, path)


def save_tree(path: str | Path, model: Any, meta: dict) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("feature_schema", F.schema_dict())
    meta["versions"] = versions()
    joblib.dump({"model": model, "meta": meta}, path, compress=3)


def load_tree(path: str | Path) -> tuple[Any, dict]:
    """Load a tree baseline.  NOTE: joblib/pickle executes code; only load files you trained."""
    import joblib

    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"tree model not found: {path}; run `traceforge train` first")
    try:
        blob = joblib.load(path)
        F.check_schema(blob["meta"]["feature_schema"])
        return blob["model"], blob["meta"]
    except F.SchemaMismatch as e:
        raise CheckpointError(f"{path}: {e}") from e
    except Exception as e:
        raise CheckpointError(f"{path} is not a loadable TraceForge tree checkpoint ({type(e).__name__}: {str(e)[:200]})") from e


def random_bundle(seed: int, drop_features: list[str] | None = None) -> NeuralBundle:
    """Randomly initialized (untrained) network - used only as an ablation control."""
    import torch

    torch.manual_seed(seed)
    model = build_mlp()
    model.eval()
    norm = Normalizer(np.zeros(F.FEATURE_DIM, np.float32), np.ones(F.FEATURE_DIM, np.float32))
    return NeuralBundle(model, norm, drop_features or [], {"kind": "untrained", "seed": seed})


def benchmark_inference(bundle: NeuralBundle, x: np.ndarray, device: str, reps: int = 20) -> float:
    """Median seconds per ``predict_p`` call of the given batch on ``device``."""
    prev = bundle.device
    bundle.to(device)
    bundle.predict_p(x)  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        bundle.predict_p(x)
        ts.append(time.perf_counter() - t)
    bundle.to(prev)
    return float(np.median(ts))


def choose_inference_device(bundle: NeuralBundle, x: np.ndarray, requested: str) -> DeviceChoice:
    """CPU is the reference.  ``auto`` measures CPU and (if usable) MPS on a representative
    batch and picks the faster one; both timings are recorded in ``notes``."""
    ch = DeviceChoice(requested, "cpu")
    if requested == "cpu":
        return ch
    ok, msg = mps_capability_check()
    ch.notes.append(msg)
    if not ok:
        if requested == "mps":
            ch.notes.append("requested mps for inference but falling back to cpu")
        return ch
    t_cpu = benchmark_inference(bundle, x, "cpu")
    t_mps = benchmark_inference(bundle, x, "mps")
    ch.notes.append(f"inference batch={len(x)}: cpu {t_cpu * 1e3:.3f} ms/call, mps {t_mps * 1e3:.3f} ms/call")
    if requested == "mps" or t_mps < t_cpu:
        ch.device = "mps"
    return ch
