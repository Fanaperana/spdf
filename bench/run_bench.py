"""spdf vs LiteParse benchmark harness.

Suites
------
odl   opendataloader-bench (200 real-world PDFs, markdown ground truth).
      Metrics: NID / NID-S (reading order), TEDS (tables), MHS (headings),
      overall = mean(NID, TEDS, MHS) -- the harness's own scorer, imported
      from the pinned checkout.  LiteParse publishes its own number on this
      suite (0.886 for v2, no OCR), so it is the headline comparison.
edge  bench/edge/corpus: generated edge cases (multi-column, tables, rotation,
      CJK/RTL, ligatures, encryption, corrupt files, scans, huge docs, ...).
      Text cases are scored with NID (normalised Indel similarity, whitespace
      collapsed) and token F1 against the generator's ground truth; robustness
      cases score 1 when the engine exits in time without crashing.

Both engines are driven through their CLIs, one process per document, with
wall time and peak RSS sampled via psutil, so speed/memory are like-for-like.

Usage
-----
  python bench/run_bench.py                       # all suites, both engines
  python bench/run_bench.py --suites edge --engines spdf
  python bench/run_bench.py --odl-limit 40        # quick subset (same 40 docs every time)
  python bench/run_bench.py --out bench/.cache/candidate.json

LiteParse results are cached per LiteParse version under bench/.cache and only
recomputed with --refresh-liteparse, so iterating on spdf is cheap.
Environment overrides: SPDF_BIN, LIT_BIN, SPDF_BENCH_ODL (bench checkout).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psutil
from rapidfuzz.distance import Indel

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
CACHE = BENCH / ".cache"
EXE = ".exe" if os.name == "nt" else ""
ODL_DIR = Path(os.environ.get("SPDF_BENCH_ODL", CACHE / "opendataloader-bench"))
EDGE_DIR = BENCH / "edge" / "corpus"
TIMEOUT = 180


# --------------------------------------------------------------------------- engines

def spdf_bin() -> Path:
    return Path(os.environ.get("SPDF_BIN", REPO / "target" / "release" / f"spdf{EXE}"))


def lit_bin() -> str:
    if os.environ.get("LIT_BIN"):
        return os.environ["LIT_BIN"]
    cand = Path(sys.executable).parent / f"lit{EXE}"
    return str(cand) if cand.exists() else "lit"


def version_of(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def spdf_formats() -> set[str]:
    try:
        h = subprocess.run([str(spdf_bin()), "parse", "--help"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {"text"}
    m = re.search(r"possible values: ([^\]]+)\]", h)
    return {v.strip() for v in m.group(1).split(",")} if m else {"text"}


def engine_cmd(engine: str, pdf: Path, out: Path, fmt: str, ocr: bool, password: str | None) -> list[str]:
    if engine == "spdf":
        cmd = [str(spdf_bin()), "parse", str(pdf), "-q", "--format", fmt, "-o", str(out)]
    elif engine == "liteparse":
        cmd = [lit_bin(), "parse", str(pdf), "-q", "--format", fmt, "-o", str(out)]
    else:
        raise ValueError(engine)
    if not ocr:
        cmd.append("--no-ocr")
    if password:
        cmd += ["--password", password]
    return cmd


def run_measured(cmd: list[str], timeout: float = TIMEOUT) -> dict:
    """Run ``cmd``; return rc, wall seconds, peak RSS (MB, incl. children), stderr tail."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak = 0
    timed_out = False
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    while proc.poll() is None:
        if ps is not None:
            try:
                rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
        if time.perf_counter() - t0 > timeout:
            timed_out = True
            for p in ([ps] + ps.children(recursive=True)) if ps else []:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            proc.kill()
            break
        time.sleep(0.005)
    _, err = proc.communicate()
    wall = time.perf_counter() - t0
    err_s = err.decode("utf-8", "replace")
    return {
        "rc": proc.returncode,
        "wall": wall,
        "peak_rss_mb": peak / 1e6,
        "timed_out": timed_out,
        "panic": ("panicked" in err_s) or (proc.returncode is not None and (proc.returncode < 0 or proc.returncode >= 0xC0000000)),
        "stderr_tail": err_s[-400:],
    }


# --------------------------------------------------------------------------- scoring helpers

PAGE_MARKER = re.compile(r"^[ \t]*-{2,}[ \t]*Page[ \t]+\d+[ \t]*-{2,}[ \t]*$", re.M | re.I)


def norm_text(s: str) -> str:
    # page-separator banners (LiteParse text mode prints "--- Page N ---") are
    # formatting, not content: drop them for both engines.
    s = PAGE_MARKER.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    return norm_text(s).lower().split()


def token_f1(gt: str, pred: str) -> float:
    g, p = Counter(tokens(gt)), Counter(tokens(pred))
    if not g and not p:
        return 1.0
    inter = sum((g & p).values())
    if inter == 0:
        return 0.0
    prec, rec = inter / sum(p.values()), inter / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def nid(gt: str, pred: str) -> float:
    return Indel.normalized_similarity(norm_text(gt), norm_text(pred))


# --------------------------------------------------------------------------- suites

def odl_docs(limit: int | None) -> list[Path]:
    pdfs = sorted((ODL_DIR / "pdfs").glob("*.pdf"))
    if limit:
        step = max(1, len(pdfs) // limit)
        pdfs = pdfs[::step][:limit]  # deterministic spread over the corpus
    return pdfs


def run_parallel(jobs, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: j(), jobs))


def suite_odl(engine: str, limit: int | None, refresh: bool, workers: int, ver: str) -> dict:
    sys.path.insert(0, str(ODL_DIR / "src"))
    from evaluator import _aggregate_document_scores, _evaluate_single_document  # type: ignore

    fmt = "markdown" if engine == "liteparse" or "markdown" in spdf_formats() else "text"
    key = engine if engine == "spdf" else f"liteparse-{re.sub(r'[^0-9A-Za-z.]+', '_', ver)}"
    pred_dir = CACHE / "odl-pred" / key
    md_dir = pred_dir / "markdown"
    timing_file = pred_dir / "timings.json"
    docs = odl_docs(limit)
    timings = {}
    if engine == "liteparse" and timing_file.exists() and not refresh:
        timings = json.loads(timing_file.read_text())
    todo = [d for d in docs if d.stem not in timings]
    if engine == "spdf" and md_dir.exists():
        shutil.rmtree(md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    def job(pdf: Path):
        out = md_dir / f"{pdf.stem}.md"
        r = run_measured(engine_cmd(engine, pdf, out, fmt, ocr=False, password=None))
        return pdf.stem, r

    for stem, r in run_parallel([lambda p=p: job(p) for p in todo], workers):
        timings[stem] = r
    if engine == "liteparse":
        timing_file.write_text(json.dumps(timings))

    gt_dir = ODL_DIR / "ground-truth" / "markdown"
    docs_scores = []
    per_doc = {}
    for d in docs:
        s = _evaluate_single_document(d.stem, gt_dir / f"{d.stem}.md", md_dir / f"{d.stem}.md")
        docs_scores.append(s)
        per_doc[d.stem] = {"overall": s.overall, "nid": s.nid, "teds": s.teds, "mhs": s.mhs}
    agg = _aggregate_document_scores(docs_scores)["score"]
    t = [timings[d.stem] for d in docs]
    walls = [x["wall"] for x in t]
    return {
        "format": fmt,
        "docs": len(docs),
        "overall": agg["overall_mean"],
        "nid": agg["nid_mean"],
        "nid_s": agg["nid_s_mean"],
        "teds": agg["teds_mean"],
        "mhs": agg["mhs_mean"],
        "sec_per_doc": statistics.fmean(walls),
        "sec_median": statistics.median(walls),
        "peak_rss_mb": max(x["peak_rss_mb"] for x in t),
        "failures": sum(1 for x in t if x["rc"] != 0 or x["timed_out"]),
        "per_doc": per_doc,
    }


def suite_edge(engine: str, workers: int) -> dict:
    cases = sorted(EDGE_DIR.glob("*.json"))
    if not cases:
        raise SystemExit("edge corpus missing: run python bench/edge/make_corpus.py")
    out_dir = CACHE / "edge-out" / engine
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    def job(meta_path: Path):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        name = meta_path.stem
        pdf = meta_path.with_suffix(".pdf")
        out = out_dir / f"{name}.txt"
        r = run_measured(engine_cmd(engine, pdf, out, "text", ocr=True, password=meta.get("password")), timeout=TIMEOUT)
        pred = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
        res = {"wall": r["wall"], "peak_rss_mb": r["peak_rss_mb"], "rc": r["rc"],
               "timed_out": r["timed_out"], "panic": r["panic"], "kind": meta["kind"],
               "tags": meta.get("tags", [])}
        if meta["kind"] == "robust":
            res["score"] = 0.0 if (r["timed_out"] or r["panic"]) else 1.0
        else:
            res["nid"] = nid(meta["gt"], pred)
            res["f1"] = token_f1(meta["gt"], pred)
            res["score"] = res["nid"]
        if r["rc"] != 0:
            res["stderr_tail"] = r["stderr_tail"][-200:]
        return name, res

    # the huge doc runs alone so its timing/memory is not polluted
    huge = [c for c in cases if "huge" in c.stem]
    rest = [c for c in cases if c not in huge]
    results = dict(run_parallel([lambda c=c: job(c) for c in rest], workers))
    for c in huge:
        n, r = job(c)
        results[n] = r
    text_cases = {k: v for k, v in results.items() if v["kind"] == "text"}
    robust = {k: v for k, v in results.items() if v["kind"] == "robust"}
    huge_r = next((v for k, v in results.items() if "huge" in k), None)
    return {
        "cases": results,
        "nid_mean": statistics.fmean(v["nid"] for v in text_cases.values()),
        "f1_mean": statistics.fmean(v["f1"] for v in text_cases.values()),
        "ocr_nid_mean": (statistics.fmean(v["nid"] for v in text_cases.values() if "ocr" in v.get("tags", []))
                         if any("ocr" in v.get("tags", []) for v in text_cases.values()) else None),
        "robust_pass": statistics.fmean(v["score"] for v in robust.values()) if robust else 1.0,
        "huge_sec": huge_r["wall"] if huge_r else None,
        "huge_rss_mb": huge_r["peak_rss_mb"] if huge_r else None,
        "edge_sec_total": sum(v["wall"] for v in results.values()),
    }


# --------------------------------------------------------------------------- main

def git_head() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="odl,edge")
    ap.add_argument("--engines", default="spdf,liteparse")
    ap.add_argument("--odl-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="parallel docs; >1 is faster but makes timings noisier")
    ap.add_argument("--refresh-liteparse", action="store_true")
    ap.add_argument("--out", default=str(CACHE / "candidate.json"))
    ap.add_argument("--reuse-liteparse-from", default=str(BENCH / "results" / "latest.json"),
                    help="when liteparse is not in --engines, copy its numbers from this file")
    args = ap.parse_args()
    suites = [s for s in args.suites.split(",") if s]
    engines = [e for e in args.engines.split(",") if e]
    CACHE.mkdir(exist_ok=True)

    if not spdf_bin().exists():
        raise SystemExit(f"spdf binary not found at {spdf_bin()} (cargo build --release -p spdf-cli)")
    versions = {"spdf": version_of([str(spdf_bin())]), "liteparse": version_of([lit_bin()])}
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_head(),
        "odl_limit": args.odl_limit,
        "versions": versions,
        "engines": {},
    }
    for eng in engines:
        res = {}
        for s in suites:
            t0 = time.time()
            if s == "odl":
                res["odl"] = suite_odl(eng, args.odl_limit, args.refresh_liteparse, args.workers, versions[eng])
            elif s == "edge":
                res["edge"] = suite_edge(eng, args.workers)
            print(f"[{eng}] {s} done in {time.time() - t0:.0f}s", file=sys.stderr)
        result["engines"][eng] = res
    if "liteparse" not in engines and Path(args.reuse_liteparse_from).exists():
        prev = json.loads(Path(args.reuse_liteparse_from).read_text(encoding="utf-8"))
        if prev.get("odl_limit") == args.odl_limit and "liteparse" in prev.get("engines", {}):
            result["engines"]["liteparse"] = prev["engines"]["liteparse"]
            result["liteparse_reused_from"] = prev.get("commit")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(BENCH))
    from scoreboard import compare_engines, render_console  # noqa: E402
    print(render_console(compare_engines(result)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
