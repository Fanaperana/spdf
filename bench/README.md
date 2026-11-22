# bench/ — spdf vs LiteParse

Reproducible head-to-head between spdf and [LiteParse](https://github.com/run-llama/liteparse)
(pinned in `requirements.txt`). The scoreboard lives in [RESULTS.md](RESULTS.md); raw numbers in
`results/latest.json`, one line per accepted run in `results/history.jsonl`, and every
improvement-loop attempt (accepted or rejected) in `progress.jsonl`.

## Suites

| suite | corpus | metrics |
|---|---|---|
| `odl` | [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench) @ `7af1d8f`, 200 real PDFs with markdown ground truth. LiteParse publishes its own score on this suite. | NID / NID-S (reading order), TEDS (tables), MHS (heading hierarchy), overall = mean(NID, TEDS, MHS), using the harness's own scorer; s/doc, peak RSS, failures |
| `edge` | `edge/make_corpus.py` → 40 generated PDFs: 2/3-column, bordered/borderless/sparse tables, page rotation 90/180/270, rotated text, CJK, Korean, Arabic/Hebrew RTL, diacritics, ligatures, tiny/huge text, headers/footers, faux-bold, justified, letter-spaced, invisible OCR layer, empty pages, mixed page sizes, AcroForm values, AES-256 user/owner passwords, stripped ToUnicode, truncated / missing / bogus xref, image-only scans (straight + skewed), a 400-page doc, and garbage / zero-byte / HTML-named-.pdf robustness inputs | per case NID + token F1 vs generator ground truth (robustness cases: 1 = exits in time without a crash); means, 400-page wall time + peak RSS |

Both engines run through their CLIs, one process per document (`--no-ocr` on `odl`,
default OCR on `edge`), timed with peak RSS sampled by psutil. LiteParse emits
`--format markdown` on `odl`; spdf emits markdown when its CLI offers it, else text.

**Beating LiteParse** = zero losses on every tracked metric (ties allowed, e.g. both 1.0 on
a robustness case) and strict wins on `odl.overall`, `odl.nid`, `odl.teds`, `odl.mhs`,
`edge.nid_mean`, `edge.f1_mean`. Tie bands: 0.0005 on aggregates, 0.005 per edge case, 3% on
time/memory.

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
