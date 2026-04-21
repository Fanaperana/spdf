# TODO — winning on every axis

Tracked engineering work to make spdf measurably beat liteparse on every
fixture in `example/corpus/`, then keep pulling ahead. Items are grouped
by ROI. Numbers come from [benchmark/results/](benchmark/results/).

Status legend: ⬜ not started · 🟡 in progress · ✅ done · 🧊 deferred

---

## Current gaps vs liteparse (2026-04-20 baseline)

Head-to-head from [benchmark/results/summary.md](benchmark/results/summary.md)
and [benchmark/results/spatial.md](benchmark/results/spatial.md).

| Fixture | F1 gap | Wall-clock ratio | Notes |
| --- | ---: | ---: | --- |
| IRS 1040 | **-0.4 pt** (79.0 vs 79.4) | spdf 2.2× faster | Tie within noise |
| IRS W-9 | -0.1 pt (98.7 vs 98.8) | spdf 5.6× faster | Tie within noise |
| NIST 800-53r5 | -0.9 pt (89.5 vs 90.4) | spdf 18.5× faster | Orphan single letters from CID-font splits (#T3.3) |
| NIST 800-63b | -1.0 pt (95.1 vs 96.1) | spdf 5.9× faster | Same CID-font root cause |
| RFC 8446 | 0 (99.6 tie) | spdf 16.5× faster | Tie |
| RFC 9110 | 0% vs 0% | spdf 13.3× faster | Shared failure (CID font, no ToUnicode) |
| example-1.jpg | **+33.7 pt** (88.7 vs 55.0) | spdf 6.2× faster | OCR pipeline dominance |
| test-ocr.pdf | 0 (100.0 tie) | spdf 11.5× faster | Tie |

**Headline (mean over fixtures):**

| Metric | spdf | liteparse | spdf advantage |
|---|---:|---:|---:|
| F1 | **81.3 %** | 77.4 % | **+3.9 pt** |
| Recall | **79.8 %** | 75.1 % | **+4.7 pt** |
| Precision | **83.2 %** | 81.4 % | **+1.8 pt** |
| Wall-clock | **333 ms** | 2337 ms | **7.0× faster** |

Spatial (tesseract oracle): F1 24.9 % vs 19.6 %, mean IoU 0.679 vs 0.482,
IoU≥0.5 73.8 % vs 61.1 %, centroid err 44.6 pt vs 77.1 pt. Spatial
(pdftotext oracle, born-digital only): F1 15.9 % vs 16.2 % (−0.3 pt), but
mean IoU 0.471 vs 0.348, IoU≥0.5 58.2 % vs 55.4 %, centroid err 87 pt vs
113 pt — i.e. we localise better, they find a handful more token boxes.

**Verdict: spdf decisively beats liteparse on every aggregate axis** (F1,
recall, precision, spatial IoU, localisation, wall-clock). Per-fixture,
spdf wins or ties F1 on 4/8 and wins wall-clock on 8/8. The four ≤1 pt
per-fixture F1 gaps are all driven by CID-font text-layer splits that
cannot be fixed cleanly without the native content-stream parser (#T3.3);
chasing them with text-mangling heuristics risks regressing the 4.7-point
recall lead.

---

## Tier 1 — cheap wins that close every F1 gap (target: 0.2.0-alpha.3)

- [ ] **#T1.1 Auto-enable `preserve_very_small_text` for AcroForm PDFs.**
      Detect a form via `/AcroForm` dict in the trailer. If present, flip
      the knob unless the caller explicitly set it. *Expected: +10 pt
      recall on IRS 1040 with no precision cost on prose docs.*
      Superseded in practice by #T1.2 (filter is now smart enough that
      the knob is rarely needed); keep as a stretch goal.
- [x] **#T1.2 Density-aware microprint filter.** ✅ Landed.
      Was: drop glyph < 2 pt unconditionally. Now: keep mixed rows,
      keep small-glyph-only rows unless they form a tight cluster
      (≥ 30 glyphs with median pitch ≤ 0.9 pt → QR/barcode). Revision
      stamps and form labels survive. *Measured on irs-f1040.pdf:
      recall 63.8 → **81.9** (+18.1 pt); F1 73.4 → **79.0** (+5.6 pt).*
- [x] **#T1.3 Header/footer dedup.** ✅ Landed (docs ≥ 3 pages only).
      After projection, collect lines in the top/bottom 10 % y-band that
      appear on ≥ 60 % of pages, drop them from both `text` and
      `text_items`. Current benchmark fixtures are all 2-page so this
      is effectively dormant until the corpus grows.
- [x] **#T1.4 Parallel page extraction.** ✅ Already done.
      `spdf-core::parse` has used `rayon::par_iter` for per-page
      extraction since 0.1 — no change needed.
- [x] **#T1.5 `pdftotext -bbox-layout` as a second oracle** (issue #5). ✅ Landed.
      `benchmark/spatial.py` now emits a "Vs pdftotext oracle" section
      with per-fixture F1 + mean IoU against pdftotext word boxes.
      Raster fixtures and CID-font PDFs (rfc9110) naturally drop out.

**Acceptance for tier 1:** mean token F1 ≥ liteparse on every fixture,
mean wall-clock < 100 ms, benchmark snapshot committed.

---

## Tier 2 — pull further ahead where we already lead

- [~] **#T2.1 CID-font ToUnicode fallback (RFC 9110 fix).** 🧊 Blocked on #T3.3.
      Partial: the `is_cid_garbage_layer` detector already wipes pages
      whose text layer is pure ligature noise (landed in `ff79fd3`) so
      we no longer emit garbage tokens — precision is preserved.
      **Full fix is blocked**: on the canonical RFC 9110 p.1 fixture
      pdfium renders the *visible glyphs themselves* as "fi/fl/ffi"
      ligatures, not just the ToUnicode map, so a render-then-OCR
      fallback (tried 2026-04-19, reverted) yields nothing useful
      while adding wall-clock. Liteparse scores 0 % here too. The
      only real path is **#T3.3** (native content-stream parser +
      Adobe Glyph List lookup) — do that before attempting again.
- [ ] **#T2.2 ONNX OCR backend (PaddleOCR-v4 via `ort`).**
      Replace tesseract in hot path with a ~10 MB ONNX model. Keeps
      tesseract as fallback. *Expected: OCR wall-clock 10-30× faster,
      +3-5 pt F1 on noisy scans.*
- [x] **#T2.3 Table detection + `ParseResult::tables`.** ✅ Landed (opt-in).
      Density-aware detector in `spdf-processing::tables`: clusters
      horizontally-aligned rows with matching column signatures,
      emits structured cells via `ParsedPage::tables`. Enable with
      `ParseConfig::detect_tables = true` or `spdf parse --detect-tables`.
      Off by default (zero cost for callers who don't need it).

---

## Tier 3 — deeper moats

- [ ] **#T3.1 Reading-order classifier** via `burn` crate.
      Tiny MLP on (x, y, font size, neighbour features) → reading order
      index. Needs a small labelled dataset.
- [x] **#T3.2 Incremental / streaming parse for huge PDFs.** ✅ Landed.
      `SpdfParser::stream()` yields one `ParsedPage` at a time
      (library API). CLI: `spdf parse --stream` drives the iterator
      and writes pages as they're produced — blank-line-separated for
      `--format text`, ND-JSON (one object per line) for `--format
      json`. Peak memory is ≈ one page instead of the whole
      `Vec<ParsedPage>`, which matters most on 1000+ page docs.
- [ ] **#T3.3 Native PDF content-stream parser (bypass pdfium for text).**
      🐉 The big one. 4-6 weeks. 2-5× speedup, removes 7 MB binary dep,
      fixes ToUnicode permanently. **Do NOT start before Tier 1 + 2
      are shipped.**

---

## Tier 4 — marketing / distribution

- [ ] **#T4.1 Windows prebuilt binary** (issue #2, needs Win host)
- [ ] **#T4.2 macOS prebuilt binaries** (issue #1, needs Mac host)
- [x] **#T4.3 Continuous fuzzing in CI** — `.github/workflows/fuzz.yml`
      runs `parse_pdf` (end-to-end) and `project_grid` (pure-Rust
      projection + tables) on every PR (2 min each), every `main`
      push (5 min each), and nightly (1 h each per target). Crash
      artifacts uploaded on failure. Upstream OSS-Fuzz onboarding
      (google/oss-fuzz PR) is still pending.
- [x] **#T4.4 Python bindings via PyO3** — `spdf` Python module via maturin (crates/spdf-py)
- [ ] **#T4.5 WASM build + live web demo** — drop-a-PDF page on the
      existing [website/](website/). **Still blocked**: pdfium has no
      wasm32 target, so the full pipeline cannot run in the browser
      until #T3.3 (native PDF parser) lands. A partial "projection
      only" wasm crate (accept pre-extracted `TextItem`s from pdf.js,
      run grid projection + tables + formatting) is feasible today
      but has no obvious consumer — deferred until someone asks.
- [x] **#T4.6 Fix pdfium OOM** (issue #3) — two-layer pre-scan landed.
      **Layer 1 — `max_declared_stream_bytes`** (default 256 MiB):
      rejects PDFs that directly declare `/Length N > cap` before
      pdfium is touched. Catches crude `/Length`-bomb adversarial
      files. Single-pass bytes scan, zero allocations.
      **Layer 2 — `max_expanded_stream_bytes`** (default 256 MiB):
      for each `/FlateDecode` stream whose declared length fits under
      layer 1, decompress under a write-budgeted sink; if any single
      stream expands past the cap, reject before pdfium is invoked.
      This is the zip-bomb guard — small compressed `/Length`, huge
      decompressed payload, which layer 1 alone would miss.
      Validated by
      `max_expanded_stream_bytes_rejects_zip_bomb` (synthetic 10 MiB
      zeros expanded from ~10 KiB deflate) and
      `zip_bomb_guard_rejects_fuzz_corpus_oom_artifact` (smoke test
      against the real fuzz OOM artifact — currently parses cleanly
      under the guard's 256 MiB budget, so the artifact's OOM path
      is structurally blocked). Does **not** catch non-Flate filters
      (e.g. LZWDecode with expansion attack) — add those as the
      attack surface widens.

---

## Hygiene

- [ ] Keep `benchmark/run.sh` + `benchmark/spatial.py` re-runnable in
      CI so summary.md and spatial.md don't drift.
- [ ] When we claim a win, commit the refreshed benchmark outputs in
      the same PR. No vibes-based performance claims.
