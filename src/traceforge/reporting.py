"""Standalone offline HTML report (no CDN, fonts or fetch calls).

The report is a *view of saved results*: every number, trace and curve comes from files in the
run directory.  Plots are embedded PNGs; the only JavaScript is a small local step-through.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from . import dsl

COLORS = {
    "uniform": "#6b7280", "heuristic": "#E69F00", "tree": "#009E73", "neural": "#0072B2",
    "neural_untrained": "#CC79A7", "neural_shuffled": "#D55E00", "neural_nosize": "#56B4E9",
}
FRESH_ROWS = [["  Katherine Johnson  "], ["Dorothy  Vaughan"], ["mary JACKSON"], [" Annie Easley "], ["Ünal ÇELIK"]]


def esc(x: object) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def safe_json(obj: object) -> str:
    """JSON for embedding inside <script>: neutralizes </script>, <!-- and U+2028/9."""
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fig_to_img(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def plot_curves(curves: dict, title: str) -> str:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for pol, d in curves["by_policy"].items():
        c = COLORS.get(pol, "#333")
        ax[0].plot(curves["candidate_budgets"], d["solve_rate_vs_candidates"], label=pol, color=c, lw=1.8)
        ax[1].plot(curves["time_budgets"], d["solve_rate_vs_time"], label=pol, color=c, lw=1.8)
    for a, xl in zip(ax, ("candidate-evaluation budget (log)", "wall-clock budget, seconds (log; includes scoring)")):
        a.set_xscale("log")
        a.set_xlabel(xl)
        a.set_ylim(-0.02, 1.02)
    ax[0].set_ylabel("tasks solved (example-consistent)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    return _fig_to_img(fig, f"Solve rate versus budget: {title}")


def plot_outcomes(rows: list[dict]) -> str:
    plt = _mpl()
    pols = sorted({r["policy"] for r in rows})
    outs = ["solved", "candidate_limit", "timeout", "state_limit", "exhausted", "resource_limit", "invalid_task"]
    palette = ["#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#999999", "#000000", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(pols) + 1.2))
    left = [0.0] * len(pols)
    for o, col in zip(outs, palette):
        vals = [sum(r["outcome"] == o for r in rows if r["policy"] == p) / max(1, sum(r["policy"] == p for r in rows)) for p in pols]
        if any(vals):
            ax.barh(pols, vals, left=left, label=o, color=col)
            left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("fraction of runs")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.invert_yaxis()
    return _fig_to_img(fig, "Search termination reasons per policy")


def plot_solved_vs_hidden(suite_entry: dict) -> str:
    plt = _mpl()
    pols = list(suite_entry["policies"])
    a = [suite_entry["policies"][p]["mean_over_seeds"]["solved_examples"] for p in pols]
    b = [suite_entry["policies"][p]["mean_over_seeds"]["solved_hidden"] for p in pols]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = range(len(pols))
    ax.bar([i - 0.2 for i in x], a, 0.4, label="fits every supplied example", color="#0072B2")
    ax.bar([i + 0.2 for i in x], b, 0.4, label="also matches every hidden row", color="#E69F00")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pols, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    return _fig_to_img(fig, "Example consistency versus hidden-row correctness")


def plot_levels(trace: dict) -> str:
    plt = _mpl()
    ev = trace.get("events", [])
    if not ev:
        return ""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar([e["level"] for e in ev], [e["attempted"] for e in ev], color="#0072B2", label="candidates attempted")
    ax.bar([e["level"] for e in ev], [e["new_states"] for e in ev], color="#E69F00", label="new semantic states kept")
    ax.set_yscale("symlog")
    ax.set_xlabel("priority-cost level")
    ax.legend(frameon=False, fontsize=8)
    return _fig_to_img(fig, "Candidates explored per priority-cost level")


def plot_training(history: list[dict]) -> str:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    e = [h["epoch"] for h in history]
    ax[0].plot(e, [h["train_loss"] for h in history], label="train", color="#0072B2")
    ax[0].plot(e, [h["val_loss"] for h in history], label="validation", color="#D55E00")
    ax[0].set_title("weighted BCE loss")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(e, [h["val_pr_auc"] for h in history], color="#009E73")
    ax[1].set_title("validation PR-AUC")
    for a in ax:
        a.set_xlabel("epoch")
    return _fig_to_img(fig, "Neural scorer training history")


def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


def summary_table(entry: dict) -> str:
    head = ["policy", "tasks", "fit examples", "match hidden", "mean accuracy", "mean candidates", "mean s", "capped mean s", "scoring s", "model calls", "peak RSS MiB"]
    rows = []
    for pol, d in entry["policies"].items():
        m, sd = d["mean_over_seeds"], d["sd_over_seeds"]

        def cell(k, nd=3):
            s = _fmt(m[k], nd)
            return s + (f" ± {sd[k]:.{nd}f}" if sd else "")

        rows.append(
            [pol, d["n_tasks"], cell("solved_examples"), cell("solved_hidden"), cell("mean_accuracy"), cell("mean_candidates", 0),
             cell("mean_seconds", 4), cell("capped_mean_seconds", 4), _fmt(d["mean_scoring_s"], 4), _fmt(d["mean_model_calls"], 1), _fmt(d["peak_rss_mb"], 0)]
        )
    return _table(head, rows, raw_cols=set(range(2, 11)))


def _table(head: list[str], rows: list[list], raw_cols: set[int] | None = None) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    body = ""
    for r in rows:
        cells = "".join(f"<td>{c if (raw_cols and i in raw_cols) else esc(c)}</td>" for i, c in enumerate(r))
        body += f"<tr>{cells}</tr>"
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def paired_table(entry: dict) -> str:
    rows = []
    for pol, d in entry["paired_vs_baseline"].items():
        def ci(k, nd=3):
            x = d[k]
            if not x.get("n"):
                return "–"
            return f"{x['mean_diff']:+.{nd}f} [{x['ci95'][0]:+.{nd}f}, {x['ci95'][1]:+.{nd}f}]"

        rows.append([pol, ci("solved_examples"), ci("solved_hidden"), ci("candidates", 0), ci("capped_seconds", 4)])
    if not rows:
        return "<p class='muted'>No paired comparison available.</p>"
    return _table(["policy − uniform", "Δ fit examples", "Δ match hidden", "Δ mean candidates", "Δ capped mean s"], rows)


def joint_table(entry: dict) -> str:
    rows = []
    for pol, d in entry["joint"].items():
        rows.append([pol, f"{d['n_joint']}/{d['n_tasks']}", d["excluded"], _fmt(d["mean_seconds"], 4), _fmt(d["baseline_mean_seconds"], 4),
                     _fmt(d["mean_candidates"], 0), _fmt(d["baseline_mean_candidates"], 0)])
    if not rows:
        return "<p class='muted'>No jointly solved tasks.</p>"
    return _table(["policy", "jointly solved", "excluded tasks", "mean s (policy)", "mean s (uniform)", "mean cand (policy)", "mean cand (uniform)"], rows)


# --------------------------------------------------------------------- program / trace


def _trace_rows(program: dsl.Node, rows: list[list[str]]) -> list[dict]:
    """For each row: the value of every sub-expression in post-order (real interpreter output)."""
    exprs: list[dsl.Node] = []

    def rec(n: dsl.Node) -> None:
        for c in n.children():
            rec(c)
        exprs.append(n)

    rec(program)
    out = []
    for r in rows:
        steps = []
        for n in exprs:
            v = dsl.evaluate(n, r)
            steps.append({"expr": dsl.to_str(n), "value": v if isinstance(v, str) else f"<invalid: {v.reason}>"})
        out.append({"inputs": r, "steps": steps})
    return out


def demo_section(demo_dir: Path) -> str:
    res = _load(demo_dir / "result.json")
    if not res:
        return "<p class='muted'>No demo synthesis found in this run.</p>"
    prog_doc = res.get("program")
    srch = res["search"]
    c = srch["counters"]
    exrows = "".join(
        f"<tr><td>{esc(' | '.join(e['inputs']))}</td><td>{esc(e['output'])}</td></tr>" for e in res["examples"]
    )
    parts = [f"<p>Task <code>{esc(res['task_id'])}</code> · policy <b>{esc(srch['policy'])}</b> (penalty λ={esc(srch['lam'])}) · outcome <b>{esc(res['outcome'])}</b></p>",
             f'<div class="tablewrap"><table><thead><tr><th>supplied input</th><th>expected output</th></tr></thead><tbody>{exrows}</tbody></table></div>']
    if not prog_doc:
        parts.append("<p>No program was found within the budgets.</p>")
        return "\n".join(parts)
    node = dsl.from_json(prog_doc["program"])
    fresh = _trace_rows(node, FRESH_ROWS)
    before_after = "".join(
        f"<tr><td><code>{esc(t['inputs'][0])}</code></td><td><code>{esc(t['steps'][-1]['value'])}</code></td></tr>" for t in fresh
    )
    parts += [
        f"<h3>Discovered program ({prog_doc['program_size']} AST nodes)</h3><pre>{esc(prog_doc['program_text'])}</pre>",
        f"<p class='note'>{esc(prog_doc['note'])}</p>",
        "<h3>Before / after on fresh rows</h3>",
        f'<div class="tablewrap"><table><thead><tr><th>before (new input)</th><th>after (program output)</th></tr></thead><tbody>{before_after}</tbody></table></div>',
        "<h3>Step through the program</h3>",
        '<div id="stepper"><label>Row <select id="rowsel"></select></label> <button id="prev">◀ prev</button> <button id="next">next ▶</button>'
        '<pre id="stepout"></pre></div>',
        f'<script type="application/json" id="trace-data">{safe_json(fresh)}</script>',
        "<h3>Search effort for this synthesis</h3>",
        _table(["candidates attempted", "invalid", "memo duplicates", "dominated", "states kept", "interpreter calls", "seconds", "model calls", "scoring s"],
               [[c["candidates"], c["invalid"], c["duplicates"], c["dominated"], c["states"], c["interpreter_calls"], _fmt(srch["seconds"], 4),
                 srch["scoring"].get("model_calls", 0), _fmt(srch["scoring"].get("featurize_s", 0) + srch["scoring"].get("model_s", 0), 4)]]),
        plot_levels(srch.get("trace", {})),
    ]
    lineage = srch.get("trace", {}).get("lineage")
    if lineage:
        rows = "".join(f"<tr><td><code>{esc(s['expr'])}</code></td><td>{s['size']}</td><td>{esc(' | '.join(s['values']))}</td></tr>" for s in lineage)
        parts.append(f'<h3>Lineage on the supplied examples</h3><div class="tablewrap"><table><thead><tr><th>sub-expression</th><th>nodes</th><th>value per supplied example</th></tr></thead><tbody>{rows}</tbody></table></div>')
    return "\n".join(parts)


STEPPER_JS = """
(function(){
  var el=document.getElementById('trace-data'); if(!el) return;
  var data=JSON.parse(el.textContent), sel=document.getElementById('rowsel'), out=document.getElementById('stepout'), k=0;
  data.forEach(function(d,i){var o=document.createElement('option'); o.value=i; o.textContent=d.inputs.join(' | '); sel.appendChild(o);});
  function show(){var d=data[+sel.value], s=['input: '+JSON.stringify(d.inputs)];
    for(var i=0;i<=k&&i<d.steps.length;i++){s.push((i+1)+'. '+d.steps[i].expr+'  ->  '+JSON.stringify(d.steps[i].value));}
    out.textContent=s.join('\\n');}
  sel.onchange=function(){k=0;show();};
  document.getElementById('next').onclick=function(){var d=data[+sel.value]; if(k<d.steps.length-1)k++; show();};
  document.getElementById('prev').onclick=function(){if(k>0)k--; show();};
  show();
})();
"""

CSS = """
:root{--bg:#fff;--fg:#1c1f23;--muted:#5b6470;--card:#f6f7f9;--line:#d8dce2;--accent:#0072B2;--warn:#b45309}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171a;--fg:#e6e8eb;--muted:#9aa3ad;--card:#1d2126;--line:#343a42;--accent:#56B4E9;--warn:#f0a85c}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}h3{margin-top:1.4em}
pre,code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.88em}
pre{background:var(--card);padding:12px;border-radius:8px;overflow-x:auto;border:1px solid var(--line)}
.tablewrap{overflow-x:auto;margin:.6em 0}table{border-collapse:collapse;width:100%;font-size:.9em}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}th{background:var(--card)}
td:first-child{white-space:normal}
.muted{color:var(--muted)}.note{background:var(--card);border-left:4px solid var(--accent);padding:8px 12px;border-radius:4px}
.warn{border-left-color:var(--warn)}img{max-width:100%;height:auto;background:#fff;border-radius:6px;border:1px solid var(--line)}
button,select{font:inherit;padding:3px 8px}.badge{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--card);border:1px solid var(--line);font-size:.8em}
"""

LIMITATIONS = """
<ul>
<li><b>Example consistency is not correctness.</b> A returned program matches the supplied examples; hidden-row accuracy is reported separately and is lower whenever the examples are ambiguous.</li>
<li><b>Bounded language.</b> 9 string operators, a finite per-task vocabulary and an AST-size limit. <code>exhausted</code> means that finite grammar was exhausted, not that the task is impossible.</li>
<li><b>Weak labels.</b> Training positives are sub-expressions of one sampled reference program; other programs could also use "negative" states. The model's sigmoid is a usefulness score, not a probability of correctness.</li>
<li><b>Synthetic benchmark.</b> Tasks come from this repository's generator (plus 40 hand-authored tasks); results do not transfer automatically to other string-transformation benchmarks. Held-out evaluation is by program group, not by input distribution.</li>
<li><b>Timing.</b> Wall-clock figures come from one process on one machine, single repeat unless stated; RSS is sampled (50 ms) and includes any MPS allocations (unified memory). Candidate counts are deterministic; times are not.</li>
<li><b>Dominance pruning</b> (dropping a state when a smaller AST with identical outputs exists) is an engineering optimization on top of the (outputs, size) memoization; it is applied identically to every policy and ablated in the no-memoization table.</li>
</ul>
"""


def build_report(run_dir: str | Path, output: str | Path) -> Path:
    d = Path(run_dir)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load(d / "manifest.json")
    parts: list[str] = []
    title = "TraceForge report"
    parts.append(f"<h1>{esc(title)}</h1><p class='muted'>Local neural-guided program synthesis for string transformations. Everything on this page is rendered from saved result files in <code>{esc(d.name)}</code>; nothing is loaded from the network.</p>")
    parts.append("<p>Give TraceForge a few examples of a text transformation; it searches for an executable program, using a neural model trained locally to prioritize useful intermediate computations. "
                 "Learning-guided bottom-up synthesis is not new (see BUSTLE, CrossBeam in <code>THIRD_PARTY_NOTICES.md</code>); this is a from-scratch laptop-scale implementation and an honest evaluation of when guidance pays for its overhead.</p>")
    if manifest is None:
        parts.append("<h2>Synthesis result</h2>" + demo_section(d))
    else:
        stages = manifest["stages"]
        partial = [k for k, v in stages.items() if v.get("partial")]
        missing = [k for k in ("generate", "train", "select", "evaluate", "analysis", "demo") if not stages.get(k, {}).get("done") and k not in partial]
        status = "complete" if not partial and not missing else "PARTIAL / INCOMPLETE"
        cls = "note" if status == "complete" else "note warn"
        ev = stages.get("evaluate", {})
        parts.append(f"<p class='{cls}'><b>Run status: {esc(status)}.</b> Profile <b>{esc(manifest['profile']['name'])}</b>; evaluation completed {esc(ev.get('completed', '?'))}/{esc(ev.get('expected', '?'))} runs"
                     + (f"; partial stages: {esc(', '.join(partial))}" if partial else "") + (f"; not finished: {esc(', '.join(missing))}" if missing else "")
                     + ". Partial results must not be cited as full-run evidence.</p>")
        parts.append("<h2>1. Demo: a program found from examples</h2>" + demo_section(d / "demo"))
        summary = _load(d / "results/summary.json")
        rows = [json.loads(l) for l in (d / "results/main.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (d / "results/main.jsonl").exists() else []
        if summary:
            parts.append("<h2>2. Benchmark: four policies, identical tasks, vocabulary and budgets</h2>")
            parts.append(f"<p>Budgets per task: {esc(json.dumps(summary['config']['budgets']))}. Penalties λ (chosen on validation tasks only): {esc(json.dumps(summary['config']['lambdas']))}. "
                         f"Inference device: {esc(summary['config']['inference'])}.</p>")
            curves = _load(d / "results/curves.json") or {}
            for suite, entry in summary["suites"].items():
                parts.append(f"<h3>Suite: {esc(suite)}</h3>")
                parts.append(summary_table(entry))
                parts.append(plot_solved_vs_hidden(entry))
                parts.append("<p class='muted'>Paired bootstrap over tasks (2000 resamples) of policy − uniform; seeds are averaged per task first, so seeds do not masquerade as extra tasks.</p>")
                parts.append(paired_table(entry))
                parts.append("<p class='muted'>Latency on jointly solved tasks only (excluded tasks listed); the capped mean in the table above charges every unsolved run its full time cap.</p>")
                parts.append(joint_table(entry))
                if suite in curves and curves[suite]:
                    parts.append(plot_curves(curves[suite], f"suite {suite}"))
            if rows:
                parts.append("<h3>Termination reasons</h3>" + plot_outcomes(rows))
        asum = _load(d / "results/analysis_summary.json")
        if asum:
            parts.append("<h2>3. Controls (validation-derived analysis suite)</h2>")
            parts.append("<p>Untrained and shuffled scores keep the neural-inference overhead but remove (or scramble) the learned signal; <code>neural_nosize</code> is retrained without the AST-size feature.</p>")
            e = asum["suites"].get("analysis_val")
            if e:
                parts.append(summary_table(e))
            nm = asum.get("nomemo", {}).get("suites", {}).get("analysis_val")
            if nm:
                parts.append("<h3>Tiny no-memoization comparison (uniform policy, small candidate cap)</h3>" + summary_table(nm))
        tr = _load(d / "training/training_report.json")
        sel = _load(d / "models/selection.json")
        if tr:
            parts.append("<h2>4. Training</h2>")
            parts.append(f"<p>Training device: {esc(json.dumps(tr['training_device']))}. Train rows {esc(json.dumps(tr.get('train_rows')))}; validation rows {esc(json.dumps(tr.get('val_rows')))} (split by program group).</p>")
            mrows = []
            for name, m in tr["models"].items():
                if "val_pr_auc" in m:
                    mrows.append([name, _fmt(m.get("val_pr_auc")), _fmt(m.get("val_ranking_recall_top20")), _fmt(m.get("size_only_val_pr_auc")), _fmt(m.get("val_positive_rate")), m.get("best_epoch", "–"), m.get("n_params", "–")])
            if mrows:
                parts.append(_table(["model", "val PR-AUC", "recall of useful @ top 20%", "size-only PR-AUC", "positive rate", "best epoch", "params"], mrows))
            ckpt_hist = None
            try:
                from .models import load_neural

                b = load_neural(d / "models/neural.pt")
                ckpt_hist = b.meta.get("history")
                parts.append(f"<p>Checkpoint <code>neural.pt</code>: seed {esc(b.meta.get('seed'))}, {b.n_params()} parameters, dataset fingerprint <code>{esc(b.meta.get('dataset_fingerprint'))}</code>, feature schema v{esc(b.meta['feature_schema']['version'])} (dim {esc(b.meta['feature_schema']['dim'])}).</p>")
            except Exception as ex:  # pragma: no cover
                parts.append(f"<p class='muted'>Checkpoint metadata unavailable: {esc(ex)}</p>")
            if ckpt_hist:
                parts.append(plot_training(ckpt_hist))
        if sel:
            trows = []
            for pol, t in sel["tuning"].items():
                for g in t["grid"]:
                    trows.append([pol, g["lam"], f"{g['solved']}/{t['n_tasks']}", _fmt(g["mean_candidates"], 0), _fmt(g["mean_seconds"], 4), "✔" if g["lam"] == t["chosen"] else ""])
            parts.append("<h3>Penalty selection (validation tasks only)</h3>" + _table(["policy", "λ", "solved", "mean candidates", "mean s", "chosen"], trows))
            parts.append(f"<p>Inference device choice: {esc(json.dumps(sel['inference']))}</p>")
        splits = _load(d / "data/splits.json")
        if splits:
            r = splits["report"]
            parts.append("<h2>5. Data and splits</h2>")
            parts.append(f"<p>Counts {esc(json.dumps(r['counts']))}; program groups per suite {esc(json.dumps(r['groups']))}. Generator rejection rate among built candidates: {r['rejection_rate_among_built']:.2f}; reasons: <code>{esc(json.dumps(r['rejections']))}</code>.</p>")
            parts.append(f"<p>Disjointness audit: <b>{'passed' if splits['disjointness']['ok'] else 'FAILED'}</b> (group, exact-task and probe-behaviour overlap; it cannot prove semantic distinctness beyond the probed rows). File checksums: <code>{esc(json.dumps({k: v['sha256'][:12] for k, v in splits['files'].items()}))}</code></p>")
        if rows:
            parts.append("<h2>6. Failure examples</h2>" + failure_section(d, rows))
        parts.append("<h2>7. Provenance</h2>")
        parts.append(f"<pre>{esc(json.dumps({'hardware': manifest['environment']['hardware'], 'versions': manifest['environment']['versions'], 'profile': manifest['profile'], 'stages': stages}, indent=2, sort_keys=True))}</pre>")
    parts.append("<h2>Limitations</h2>" + LIMITATIONS)
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>" + "\n".join(parts) + f"</main><script>{STEPPER_JS}</script></body></html>"
    )
    out.write_text(doc, encoding="utf-8")
    return out


def failure_section(d: Path, rows: list[dict]) -> str:
    from .data.generator import load_suite

    by_id = {}
    for suite in {r["suite"] for r in rows}:
        name = "authored" if suite in ("authored", "regression") else suite
        try:
            for t in load_suite(d / "data", name):
                by_id[t.id] = t
        except FileNotFoundError:
            pass
    unsolved = [r for r in rows if r["policy"] == "neural" and not r["solved_examples"]][:5]
    wrong = [r for r in rows if r["policy"] == "neural" and r["solved_examples"] and not r["solved_hidden"]][:5]
    out = []
    for title, rs in (
        ("Neural policy: no consistent program within budget", unsolved),
        ("Neural policy: fits the supplied examples but fails hidden rows", wrong),
    ):
        out.append(f"<h3>{esc(title)}</h3>")
        if not rs:
            out.append("<p class='muted'>None in this run.</p>")
            continue
        for r in rs:
            t = by_id.get(r["task_id"])
            ex = "; ".join(f"{list(e.inputs)} → {e.output!r}" for e in (t.spec.examples[:3] if t else []))
            line = f"<p><code>{esc(r['task_id'])}</code> ({esc(r['suite'])}) outcome <b>{esc(r['outcome'])}</b>, {r['candidates']} candidates. Supplied: {esc(ex)}"
            if t is not None and r.get("program_json"):
                prog = dsl.from_json(json.loads(r["program_json"]))
                line += f"<br>Selected program: <code>{esc(r['program'])}</code> · hidden rows correct {r['hidden_correct']}/{r['hidden_total']}"
                for h in t.hidden:
                    got = dsl.evaluate(prog, h.inputs)
                    if got != h.output:
                        gv = got if isinstance(got, str) else f"<invalid: {got.reason}>"
                        line += f"<br>First mismatching hidden row: input {esc(list(h.inputs))} expected {esc(repr(h.output))}, program produced {esc(repr(gv))}"
                        break
                if t.reference is not None:
                    line += f"<br>Reference (evaluator-only): <code>{esc(dsl.to_str(t.reference))}</code>"
            out.append(line + "</p>")
    return "\n".join(out)
