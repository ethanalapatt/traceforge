import pytest

from traceforge.tasks import Budgets, Example, TaskSpec, make_search_task
from traceforge.vocabulary import Vocabulary


def spec_of(rows, cols=("x",), tid="t"):
    return TaskSpec(tid, tuple(cols), tuple(Example(tuple(i), o) for i, o in rows))


def task_of(rows, cols=("x",), **budget):
    kw = dict(max_nodes=5, max_candidates=50_000, seconds=30.0)
    kw.update(budget)
    return make_search_task(spec_of(rows, cols), Budgets(**kw))


def tiny_vocab(n_inputs=1):
    """A deliberately tiny grammar so exhaustive enumeration is cheap."""
    return Vocabulary(
        n_inputs=n_inputs,
        literals=(".",),
        delimiters=("-",),
        token_indices=(0, 1),
        slice_pairs=((0, 1), (1, None)),
        replace_pairs=(("-", ""),),
    )


@pytest.fixture
def rows_dash():
    return [(("ab-cd",), "zzzz"), (("Ef-gh",), "zzzz"), (("ij-KL",), "zzzz")]


import subprocess
import sys
from pathlib import Path


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """One real end-to-end smoke reproduction through the CLI with network access blocked."""
    out = tmp_path_factory.mktemp("runs") / "smoke"
    proc = subprocess.run(
        [sys.executable, "-m", "traceforge.cli", "reproduce", "--profile", "smoke", "--device", "cpu", "--offline", "--output", str(out)],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return out, proc.stdout
