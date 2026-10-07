import csv
import json
import re
import socket
import subprocess
import sys

import pytest

from traceforge import dsl
from traceforge.cli import main
from traceforge.offline import NetworkBlockedError, block_network, unblock_network
from traceforge.pipeline import Run
from traceforge.profiles import load_profile
from traceforge.reporting import build_report, safe_json
from traceforge.evaluation import ResumeError

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_offline_guard_blocks_sockets_and_restores():
    block_network()
    try:
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(NetworkBlockedError):
            socket.getaddrinfo("example.com", 80)
        s = socket.socket(socket.AF_INET)
        with pytest.raises(NetworkBlockedError):
            s.connect(("93.184.216.34", 80))
        s.close()
    finally:
        unblock_network()
    assert socket.create_connection.__name__ != "_blocked"


def test_offline_smoke_reproduction_end_to_end(smoke_run):
    out, stdout = smoke_run
    assert "outbound network connections are blocked" in stdout
    m = json.loads((out / "manifest.json").read_text())
    assert all(m["stages"][s]["done"] for s in ("generate", "train", "select", "evaluate", "analysis", "demo"))
    assert m["stages"]["evaluate"]["completed"] == m["stages"]["evaluate"]["expected"] > 0
    for f in ("results/main.jsonl", "results/main.csv", "results/summary.json", "results/curves.json", "models/neural.pt", "models/tree.joblib", "report.html", "demo/program.json"):
        assert (out / f).exists(), f
    rows = [json.loads(l) for l in (out / "results/main.jsonl").read_text().splitlines()]
    assert {r["policy"] for r in rows} == {"uniform", "heuristic", "tree", "neural"}
    for r in rows:  # every claimed solution is re-verifiable from the saved row alone
        if r["outcome"] == "solved":
            assert r["solved_examples"] == 1


def test_reproduce_refuses_existing_dir_without_resume_and_incompatible_resume(smoke_run):
    out, _ = smoke_run
    with pytest.raises(ResumeError, match="--resume"):
        Run(out, load_profile("smoke"), "cpu", resume=False)
    with pytest.raises(ResumeError, match="different profile"):
        Run(out, load_profile("dev"), "cpu", resume=True)
    Run(out, load_profile("smoke"), "cpu", resume=True)  # compatible resume is allowed


def test_resume_of_a_finished_run_does_nothing_new(smoke_run):
    out, _ = smoke_run
    before = (out / "results/main.jsonl").read_text()
    proc = subprocess.run([sys.executable, "-m", "traceforge.cli", "reproduce", "--profile", "smoke", "--device", "cpu", "--offline", "--output", str(out), "--resume"],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert (out / "results/main.jsonl").read_text() == before


def test_report_is_standalone_and_offline(smoke_run, tmp_path):
    out, _ = smoke_run
    html = (out / "report.html").read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*["\']https?:', html) and "<link" not in html
    assert not re.search(r"\b(fetch|XMLHttpRequest|import\()\b", html)
    assert 'data:image/png;base64' in html and "Limitations" in html and "example consistency" in html.lower()
    again = build_report(out, tmp_path / "r.html")
    assert again.read_text(encoding="utf-8") == html or abs(len(again.read_text()) - len(html)) < 5000


def test_synthesize_and_apply_roundtrip_with_csv_escaping(tmp_path):
    ex = ROOT / "examples"
    rc = main(["synthesize", str(ex / "normalize_name.json"), "--policy", "uniform", "--output", str(tmp_path / "s")])
    assert rc == 0
    prog = tmp_path / "s/program.json"
    src = tmp_path / "in.csv"
    src.write_text('id,full_name\n1,"  Ada ""the"" Lovelace, Countess  "\n2,Grace  Hopper\n3,"multi\nline"\n', encoding="utf-8")
    dst = tmp_path / "out.csv"
    assert main(["apply", str(prog), "--input", str(src), "--columns", "full_name", "--output", str(dst)]) == 0
    with open(dst, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and [r["id"] for r in rows] == ["1", "2", "3"]
    assert rows[1]["traceforge_output"] == "grace..hopper"
    assert rows[0]["full_name"] == '  Ada "the" Lovelace, Countess  ' and rows[2]["full_name"] == "multi\nline"
    assert src.read_text(encoding="utf-8").startswith("id,full_name")  # input untouched
    assert main(["apply", str(prog), "--input", str(src), "--columns", "full_name", "--output", str(dst)]) == 2  # no overwrite
    assert main(["apply", str(prog), "--input", str(src), "--columns", "full_name", "--output", str(src), "--force"]) == 2  # never the input
    assert main(["apply", str(prog), "--input", str(src), "--columns", "nope", "--output", str(tmp_path / "x.csv")]) == 2


def test_apply_reports_row_level_errors_without_aborting(tmp_path):
    prog = {"schema_version": 1, "kind": "traceforge.program", "task_id": "t", "input_columns": ["a"],
            "program": dsl.to_json(dsl.Token(dsl.Input(0), ",", 1))}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(prog))
    src = tmp_path / "i.csv"
    src.write_text("a\n\"x,y\"\nnodelim\n", encoding="utf-8")
    dst = tmp_path / "o.csv"
    assert main(["apply", str(p), "--input", str(src), "--columns", "a", "--output", str(dst)]) == 0
    rows = list(csv.DictReader(open(dst, newline="")))
    assert rows[0]["traceforge_output"] == "y" and rows[0]["traceforge_error"] == ""
    assert rows[1]["traceforge_output"] == "" and rows[1]["traceforge_error"].startswith("invalid")
    assert main(["apply", str(p), "--input", str(src), "--columns", "a", "--output", str(tmp_path / "o2.csv"), "--strict"]) == 2


def test_apply_rejects_foreign_or_malicious_program_files(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"kind": "traceforge.program", "schema_version": 1, "input_columns": ["a"], "program": {"op": "Eval", "code": "__import__('os')"}}))
    src = tmp_path / "i.csv"
    src.write_text("a\nx\n")
    assert main(["apply", str(p), "--input", str(src), "--columns", "a", "--output", str(tmp_path / "o.csv")]) == 2


def test_neural_without_checkpoint_is_an_error_not_a_silent_fallback(tmp_path, capsys):
    rc = main(["synthesize", str(ROOT / "examples/normalize_name.json"), "--policy", "neural", "--output", str(tmp_path / "s")])
    assert rc == 2 and "--checkpoint" in capsys.readouterr().out
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    assert main(["synthesize", str(ROOT / "examples/normalize_name.json"), "--policy", "neural", "--checkpoint", str(bad), "--output", str(tmp_path / "s2")]) == 2


def test_invalid_task_files_are_rejected_clearly(tmp_path, capsys):
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"schema_version": 1, "id": "x", "input_columns": ["a"], "examples": [{"inputs": ["a"], "output": "1"}, {"inputs": ["a"], "output": "2"}, {"inputs": ["b"], "output": "3"}]}))
    assert main(["synthesize", str(bad), "--output", str(tmp_path / "o")]) == 2
    assert "invalid task" in capsys.readouterr().out


def test_html_report_escapes_hostile_strings(tmp_path):
    evil = "</script><img src=x onerror=alert(1)>"
    task = {"schema_version": 1, "id": "<b>evil</b>", "input_columns": ["a"],
            "examples": [{"inputs": [evil + "a"], "output": evil + "a<i>"}, {"inputs": [evil + "b"], "output": evil + "b<i>"},
                         {"inputs": [evil + "c"], "output": evil + "c<i>"}]}
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps(task))
    assert main(["synthesize", str(tf), "--policy", "uniform", "--max-nodes", "5", "--output", str(tmp_path / "s")]) == 0
    out = build_report(tmp_path / "s", tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "<img src=x" not in html and "&lt;img src=x" in html
    assert "<b>evil</b>" not in html
    assert html.count("</script>") == html.count("<script")  # the payload cannot close a script block early
    assert "\\u003ci\\u003e" in html and 'Literal(\\"<i>' not in html  # the program's literal is neutralized inside the JSON blob


def test_safe_json_neutralizes_script_breakouts():
    s = safe_json({"a": "</script><!--", "b": " &"})
    assert "<" not in s and ">" not in s and " " not in s
    assert json.loads(s) == {"a": "</script><!--", "b": " &"}


def test_doctor_runs(capsys):
    assert main(["doctor"]) == 0
    assert "all required checks passed" in capsys.readouterr().out
