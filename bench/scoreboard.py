"""Scoreboard utilities: spdf-vs-liteparse comparison, regression gate, publishing.

  python bench/scoreboard.py show     [run.json]
  python bench/scoreboard.py gate     <baseline.json> <candidate.json>   # exit 0 ok / 2 regression
  python bench/scoreboard.py publish  <candidate.json> [--label TEXT]   # -> results/latest.json,
                                                                        #    results/history.jsonl, RESULTS.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"

# (metric path, direction, tie tolerance, regression tolerance)
#   direction: +1 higher is better, -1 lower is better
#   tolerances on "rel:" are relative (fraction), otherwise absolute
HEADLINE = [
    ("odl.overall", +1, 0.0005, 0.002),
    ("odl.nid", +1, 0.0005, 0.002),
    ("odl.nid_s", +1, 0.0005, 0.002),
    ("odl.teds", +1, 0.0005, 0.003),
    ("odl.mhs", +1, 0.0005, 0.003),
    ("odl.failures", -1, 0, 0),
    ("odl.sec_per_doc", -1, "rel:0.03", "rel:0.20"),
    ("odl.peak_rss_mb", -1, "rel:0.03", "rel:0.20"),
    ("edge.nid_mean", +1, 0.0005, 0.002),
    ("edge.f1_mean", +1, 0.0005, 0.002),
    ("edge.robust_pass", +1, 0, 0),
    ("edge.huge_sec", -1, "rel:0.03", "rel:0.25"),
    ("edge.huge_rss_mb", -1, "rel:0.03", "rel:0.25"),
]
CASE_TIE, CASE_REG = 0.005, 0.01
HEADLINE_MUST_WIN = ["odl.overall", "odl.nid", "odl.teds", "odl.mhs", "edge.nid_mean", "edge.f1_mean"]


def get(d: dict, path: str):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _tol(t, ref):
    if isinstance(t, str) and t.startswith("rel:"):
        return abs(ref or 0) * float(t[4:])
    return t


def metric_rows(run: dict):
    """Yield (name, spdf_value, lite_value, direction, tie_tol, reg_tol)."""
    s, l = run["engines"].get("spdf", {}), run["engines"].get("liteparse", {})
    for name, direction, tie, reg in HEADLINE:
        yield name, get(s, name), get(l, name), direction, tie, reg
    cases = sorted(set((get(s, "edge.cases") or {}).keys()) | set((get(l, "edge.cases") or {}).keys()))
    for c in cases:
        p = f"edge.cases.{c}.score"
        yield f"edge.case.{c}", get(s, p), get(l, p), +1, CASE_TIE, CASE_REG


def compare_engines(run: dict) -> dict:
    rows, wins, losses, ties = [], 0, 0, 0
    for name, sv, lv, d, tie, _ in metric_rows(run):
        if sv is None or lv is None:
            status = "n/a"
        else:
            t = _tol(tie, lv)
            diff = (sv - lv) * d
            status = "win" if diff > t else ("loss" if diff < -t else "tie")
        wins += status == "win"; losses += status == "loss"; ties += status == "tie"
        rows.append({"metric": name, "spdf": sv, "liteparse": lv, "better": "higher" if d > 0 else "lower", "status": status})
    head = {r["metric"]: r["status"] for r in rows}
    headline_wins = all(head.get(m) == "win" for m in HEADLINE_MUST_WIN)
    # "beats" = never worse on any tracked metric (ties allowed, e.g. both 1.0 on a
    # robustness case) AND strictly better on the headline accuracy metrics.
    return {"rows": rows, "wins": wins, "losses": losses, "ties": ties,
            "beats_liteparse": losses == 0 and headline_wins,
            "no_losses": losses == 0}


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def render_console(cmp: dict) -> str:
    out = [f"{'metric':34s} {'spdf':>10s} {'liteparse':>10s}  status"]
    for r in cmp["rows"]:
        out.append(f"{r['metric']:34s} {fmt(r['spdf']):>10s} {fmt(r['liteparse']):>10s}  {r['status']}")
    out.append(f"wins={cmp['wins']} ties={cmp['ties']} losses={cmp['losses']} beats_liteparse={cmp['beats_liteparse']}")
    return "\n".join(out)


def gate(baseline: dict, cand: dict) -> tuple[bool, list[str], list[str]]:
    """Return (ok, regressions, improvements) for spdf between two runs."""
    regs, imps = [], []
    b_rows = {n: (sv, d, tie, reg) for n, sv, _, d, tie, reg in metric_rows(baseline)}
    cand_status = {r["metric"]: r["status"] for r in compare_engines(cand)["rows"]}
    for n, cv, _, d, tie, reg in metric_rows(cand):
        if n not in b_rows:
            continue
        bv = b_rows[n][0]
        if bv is None or cv is None:
            if bv is not None and cv is None:
                regs.append(f"{n}: missing in candidate")
            continue
        delta = (cv - bv) * d
        perf = n.endswith(("sec_per_doc", "huge_sec", "rss_mb"))
        if delta < -_tol(reg, bv):
            if perf and cand_status.get(n) == "win":
                # speed/memory headroom may be spent on accuracy as long as spdf still
                # beats LiteParse on that metric; it is reported, not blocking
                imps.append(f"(perf cost, still ahead of LiteParse) {n}: {fmt(bv)} -> {fmt(cv)}")
                continue
            regs.append(f"{n}: {fmt(bv)} -> {fmt(cv)}")
        elif delta > _tol(tie, bv) and not n.endswith(("sec_per_doc", "huge_sec", "rss_mb")):
            imps.append(f"{n}: {fmt(bv)} -> {fmt(cv)}")
        elif delta > _tol(reg, bv):  # speed/memory only count when clearly outside noise
            imps.append(f"{n}: {fmt(bv)} -> {fmt(cv)}")
    real = [i for i in imps if not i.startswith("(perf cost")]
    notes = [i for i in imps if i.startswith("(perf cost")]
    # perf-cost notes ride along with real improvements but never count as one on their own
    return (not regs), regs, (real + notes) if real else []


def render_md(run: dict, cmp: dict, history: list[dict]) -> str:
    v = run["versions"]
    lines = [
        "# spdf vs LiteParse — benchmark scoreboard",
        "",
        "<!-- generated by bench/scoreboard.py publish; do not edit by hand -->",
        "",
        f"- Run: {run['timestamp']} · commit `{run['commit']}`",
        f"- Engines: spdf `{v.get('spdf')}` vs LiteParse `{v.get('liteparse')}` (CLI, one process per document, no OCR on odl; default OCR on edge)",
        f"- odl = [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench) "
        f"({get(run, 'engines.spdf.odl.docs') or '–'} docs, harness scorer). edge = bench/edge generated corpus.",
        f"- spdf output format on odl: `{get(run, 'engines.spdf.odl.format')}`; LiteParse: `markdown`.",
        "",
        f"**Wins {cmp['wins']} · ties {cmp['ties']} · losses {cmp['losses']}** — "
        + ("spdf beats LiteParse: no losses on any tracked metric and wins on every headline accuracy metric."
           if cmp["beats_liteparse"] else "spdf does not yet beat LiteParse (needs zero losses + wins on odl.overall/nid/teds/mhs and edge nid/f1)."),
        "",
        "| metric | better | spdf | LiteParse | status |",
        "|---|---|---:|---:|---|",
    ]
    for r in cmp["rows"]:
        st = {"win": "✅ win", "loss": "❌ loss", "tie": "➖ tie"}.get(r["status"], r["status"])
        lines.append(f"| {r['metric']} | {r['better']} | {fmt(r['spdf'])} | {fmt(r['liteparse'])} | {st} |")
    if history:
        lines += ["", "## Progress (spdf)", "", "| date (UTC) | commit | odl overall | edge NID | wins/ties/losses | note |", "|---|---|---:|---:|---|---|"]
        for h in history[-30:]:
            lines.append(f"| {h['timestamp'][:16]} | `{h['commit']}` | {fmt(h.get('odl_overall'))} | {fmt(h.get('edge_nid'))} | "
                         f"{h['wins']}/{h['ties']}/{h['losses']} | {h.get('label', '')} |")
    lines += ["", "Reproduce: see bench/README.md (`bench/setup.ps1`, then `python bench/run_bench.py`)."]
    return "\n".join(lines) + "\n"


def publish(cand_path: Path, label: str = "") -> None:
    run = json.loads(cand_path.read_text(encoding="utf-8"))
    cmp = compare_engines(run)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    entry = {
        "timestamp": run["timestamp"], "commit": run["commit"], "label": label,
        "odl_overall": get(run, "engines.spdf.odl.overall"), "edge_nid": get(run, "engines.spdf.edge.nid_mean"),
        "wins": cmp["wins"], "ties": cmp["ties"], "losses": cmp["losses"],
        "spdf": {r["metric"]: r["spdf"] for r in cmp["rows"]},
        "liteparse": {r["metric"]: r["liteparse"] for r in cmp["rows"]},
    }
    hist_p = RESULTS / "history.jsonl"
    with hist_p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    history = [json.loads(l) for l in hist_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    (BENCH / "RESULTS.md").write_text(render_md(run, cmp, history), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 1
    cmd = argv[1]
    if cmd == "show":
        p = Path(argv[2]) if len(argv) > 2 else RESULTS / "latest.json"
        print(render_console(compare_engines(json.loads(p.read_text(encoding="utf-8")))))
        return 0
    if cmd == "gate":
        b = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        c = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
        ok, regs, imps = gate(b, c)
        print(json.dumps({"ok": ok, "regressions": regs, "improvements": imps,
                          "vs_liteparse": {k: v for k, v in compare_engines(c).items() if k != "rows"}}, indent=1))
        return 0 if ok else 2
    if cmd == "publish":
        label = argv[argv.index("--label") + 1] if "--label" in argv else ""
        publish(Path(argv[2]), label)
        return 0
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
