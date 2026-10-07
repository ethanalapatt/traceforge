"""Environment / hardware description and ``traceforge doctor``."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def hardware_info() -> dict:
    import psutil

    info = {
        "machine": platform.machine(),
        "system": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "cpu_brand": None,
    }
    if sys.platform == "darwin":
        try:
            info["cpu_brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip() or None
        except Exception:
            pass
    return info


def doctor() -> tuple[dict, bool]:
    """Returns (report, ok)."""
    from .models import mps_capability_check, versions

    report: dict = {"python": sys.version.split()[0], "ok": True, "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        report["ok"] = report["ok"] and ok

    check("python>=3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    for mod in ("numpy", "torch", "sklearn", "matplotlib", "psutil"):
        try:
            m = __import__(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # pragma: no cover
            check(f"import {mod}", False, str(e))
    for mod in ("pytest", "hypothesis"):
        try:
            m = __import__(mod)
            check(f"import {mod} (dev)", True, getattr(m, "__version__", ""))
        except Exception:
            report["checks"].append({"name": f"import {mod} (dev)", "ok": True, "detail": "not installed (only needed for tests)"})
    try:
        import torch

        x = torch.randn(4, 4)
        check("torch cpu op", bool((x @ x).isfinite().all()))
    except Exception as e:  # pragma: no cover
        check("torch cpu op", False, str(e))
    ok, msg = mps_capability_check()
    report["checks"].append({"name": "mps (optional)", "ok": True, "detail": msg + ("" if ok else " - CPU is the reference path")})
    report["mps_usable"] = ok
    report["hardware"] = hardware_info()
    report["versions"] = versions()
    return report, report["ok"]
