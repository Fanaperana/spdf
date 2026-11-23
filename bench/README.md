# bench/ — spdf vs LiteParse

Reproducible head-to-head between spdf and [LiteParse](https://github.com/run-llama/liteparse)
(pinned in `requirements.txt`). The scoreboard lives in [RESULTS.md](RESULTS.md); raw numbers in
`results/latest.json`, one line per accepted run in `results/history.jsonl`, and every
improvement-loop attempt (accepted or rejected) in `progress.jsonl`.

## Suites

| suite | corpus | metrics |
|---|---|---|
| `odl` | [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench) @ `7af1d8f`, 200 real PDFs with markdown ground truth. LiteParse publishes its own score on this suite. | NID / NID-S (reading order), TEDS (tables), MHS (heading hierarchy), overall = mean(NID, TEDS, MHS), using the harness's own scorer; s/doc, peak RSS, failures |
| `edge` | `edge/make_corpus.py` → 46 generated PDFs: 2/3-column, bordered/borderless/sparse tables, page rotation 90/180/270, rotated text, CJK, Korean, Arabic/Hebrew RTL, diacritics, ligatures, tiny/huge text, headers/footers, faux-bold, justified, letter-spaced, invisible OCR layer, empty pages, mixed page sizes, AcroForm values, AES-256 user/owner passwords, stripped ToUnicode, truncated / missing / bogus xref, image-only scans (straight, skewed, 100 dpi, noisy JPEG, table, Chinese, accented Latin, rotated 90°, and a born-digital + scanned mixed document), a 400-page doc, and garbage / zero-byte / HTML-named-.pdf robustness inputs | per case NID + token F1 vs generator ground truth (robustness cases: 1 = exits in time without a crash); means, 400-page wall time + peak RSS |

Both engines run through their CLIs, one process per document (`--no-ocr` on `odl`,
default OCR on `edge`), so every number includes process start-up, model/engine load and
teardown — the cost a user pays per file. Each document is run **3 times (median reported)**
after one untimed warm-up run per suite for each engine (OS file cache, on-disk shader/model
caches). Peak memory is the larger of polled RSS (incl. child processes) and the process's
exact `PeakWorkingSetSize`, read from its handle after exit, so short runs are not under-counted.
LiteParse emits `--format markdown` on `odl`; spdf emits markdown when its CLI offers it, else text.

OCR is not the same engine on both sides: LiteParse 2.19.0's default build uses bundled
Tesseract (`tesseract-rs`, CPU); spdf uses PP-OCRv5 on Burn (GPU via wgpu, CPU fallback).
The comparison is like-for-like at the level that matters to a user — same input, same CLI
invocation, same default settings, same machine — not engine-for-engine.

**Beating LiteParse** = zero losses on every tracked metric (ties allowed within the band, e.g.
both 1.0 on a robustness case) and strict wins on every headline metric:

- accuracy: `odl.overall/nid/teds/mhs`, `edge.nid_mean`, `edge.f1_mean`, `edge.ocr_nid_mean`
- latency: `odl.sec_per_doc` (mean), `odl.sec_median`, `odl.sec_p95`, `edge.total_sec`,
  `edge.ocr_sec_mean`, `edge.ocr_sec_max`, `edge.huge_sec`
- memory: `odl.peak_rss_mb`, `odl.rss_mean_mb`, `edge.max_rss_mb`, `edge.rss_mean_mb`,
  `edge.ocr_rss_mean_mb`, `edge.ocr_rss_max_mb`, `edge.huge_rss_mb`

Every edge case is also tracked individually for score, wall time (`edge.case.<name>.sec`) and
peak RSS (`edge.case.<name>.rss_mb`); none may lose. Tie bands vs LiteParse: 0.0005 on accuracy
aggregates, 0.005 per case score, 3% on time/memory (floors 5 ms / 1 MB; per case 10 ms / 2 MB).

**Regression gate** (used by the improvement loop before every commit): no spdf metric may get
worse than its band — 0.002 on accuracy aggregates (0.003 TEDS/MHS), 0.01 on a case score, 7% on
latency/memory aggregates (floors 10 ms / 3 MB) and 15% per case (floors 30 ms / 5 MB). Accuracy
can't be bought with latency or memory, and vice versa.

## Run

```powershell
pwsh bench/setup.ps1                                  # once
bench/.venv/Scripts/python bench/run_bench.py          # full run -> bench/.cache/candidate.json
bench/.venv/Scripts/python bench/run_bench.py --engines spdf --odl-limit 40   # quick check
bench/.venv/Scripts/python bench/scoreboard.py show bench/.cache/candidate.json
bench/.venv/Scripts/python bench/scoreboard.py gate bench/results/latest.json bench/.cache/candidate.json
bench/.venv/Scripts/python bench/scoreboard.py publish bench/.cache/candidate.json --label "note"
```

LiteParse predictions are cached per LiteParse version under `bench/.cache`; pass
`--refresh-liteparse` after bumping it. The older `benchmark/` directory (token F1 vs raw
tesseract on `example/`) is kept for history; `bench/` is the tracked scoreboard.

## Quick mode (for iterating, not for gating)

```powershell
bench\.venv\Scripts\python.exe bench\run_bench.py --suites edge --engines spdf --cases "scanned*,simple,huge_400_pages,invisible_ocr_layer,table_bordered,cjk_chinese"
bench\.venv\Scripts\python.exe bench\scoreboard.py quick bench\.cache\quick.json
```

`--cases` takes comma-separated globs over edge case names. The run is still one process per document with
a warm-up and a median of `--repeats` (default 3), but only over the chosen cases, so it takes about a
minute instead of the full run. It writes `bench/.cache/quick.json` (outputs in `bench/.cache/edge-out-quick`)
and is marked `partial`. `scoreboard.py quick` prints each case as new / accepted baseline / LiteParse and
flags regressions against the accepted baseline. `gate` and `publish` refuse partial runs: acceptance always
uses the full 200-document odl suite plus all edge cases.
