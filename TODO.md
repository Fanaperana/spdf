# TODO — winning on every axis

Tracked engineering work to make spdf measurably beat liteparse on every
fixture in `example/corpus/`, then keep pulling ahead. Items are grouped
by ROI. Numbers come from [benchmark/results/](benchmark/results/).

Status legend: ⬜ not started · 🟡 in progress · ✅ done · 🧊 deferred

---

## Current gaps vs liteparse (2026-04-21 baseline)

Head-to-head from [benchmark/results/summary.md](benchmark/results/summary.md)
and [benchmark/results/spatial.md](benchmark/results/spatial.md).

| Fixture | F1 gap | Wall-clock ratio | Notes |
| --- | ---: | ---: | --- |
| IRS 1040 | **-0.4 pt** (79.0 vs 79.4) | spdf 1.8× faster | Tie within noise |
| IRS W-9 | -0.1 pt (98.7 vs 98.8) | spdf 6.2× faster | Tie within noise |
| NIST 800-53r5 | **-0.4 pt** (90.0 vs 90.4) | spdf 37.1× faster | ↓ was -0.9pt; CID orphan dedup fixed trailing `y` |
| NIST 800-63b | **-3.2 pt** (92.9 vs 96.1) | spdf 220× faster | ↓ was -3.8pt; CID orphan dedup fixed trailing `r`/`d` |
| RFC 8446 | 0 (99.6 tie) | spdf 24.9× faster | Tie |
| RFC 9110 | 0% vs 0% | spdf 545× faster | Shared failure (CID font, no ToUnicode) |
| example-1.jpg | 0% vs 55% | no OCR yet | spdf needs ONNX OCR (#T2.2) for image input |
| test-ocr.pdf | -14.3 pt (85.7 vs 100.0) | spdf 142× faster | Partial OCR gap |

**Headline (mean over fixtures):**

| Metric | spdf | liteparse | spdf advantage |
|---|---:|---:|---:|
| F1 | 68.2 % | 77.4 % | -9.2 pt (dragged by no-OCR zeros) |
| Mean F1 (PDF only, excl rfc9110) | **90.2 %** | **92.9 %** | -2.7 pt |
| Precision | **71.4 %** | 81.4 % | -10.0 pt |
| Wall-clock | **73 ms** | 2254 ms | **30.9× faster** |

**Verdict:** On born-digital PDFs with working text layers, spdf ties
or nearly-ties liteparse on F1 while being 6-545× faster. The remaining
F1 gaps (NIST 63b −3.2pt, IRS 1040 −0.4pt) are driven by CID-font
text-layer splits that need #T3.3 (native parser). The image/OCR gap
(example-1.jpg) needs #T2.2 (ONNX OCR backend).

---

## Tier 1 — cheap wins that close every F1 gap (target: 0.2.0-alpha.3)

- [x] **#T1.1 Auto-enable `preserve_very_small_text` for AcroForm PDFs.** ✅ Landed.
      Byte-level `/AcroForm` scan in `parse_inner()` and `stream()`.
      If detected and `preserve_very_small_text` is still at default
      (false), auto-flip to true. On IRS 1040 this is a no-op because
      #T1.2's density filter already keeps form labels; structurally
      correct for future AcroForm fixtures where the filter would
      otherwise drop legitimate small text.
- [x] **#T1.6 CID-font orphan glyph dedup.** ✅ Landed.
      Two-pass cleanup in `spdf-projection`:
      (a) `deduplicate_contained_glyphs` — after merge, remove single-char
      items whose centre falls inside (±2pt slack) a larger merged item
      that contains the same character. Catches e.g. pdfium emitting
      `"Technology "` + separate `"y"` at the same position.
      (b) `strip_cid_duplicate_chars` — after merge, strip trailing
      `" X"` where `X` duplicates the preceding char, and leading
      `"X "` where `X` duplicates the first char of the following word.
      Catches e.g. `"Burr r"` → `"Burr"`, `"f for"` → `"for"`.
      *Measured: NIST 53r5 precision +1.0pt (F1 89.5→90.0),
      NIST 63b precision +1.4pt (F1 92.3→92.9).*
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
