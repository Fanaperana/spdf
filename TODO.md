# TODO — winning on every axis

Tracked engineering work to make spdf measurably beat liteparse on every
fixture in `example/corpus/`, then keep pulling ahead. Items are grouped
by ROI. Numbers come from [benchmark/results/](benchmark/results/).

Status legend: ⬜ not started · 🟡 in progress · ✅ done · 🧊 deferred

---

## Current gaps vs liteparse (2026-04-19 baseline)

| Fixture | F1 gap | Root cause (hypothesis) |
| --- | ---: | --- |
| IRS 1040 | **-6.0 pt** (73.4 vs 79.4) | Recall 63.8% vs 81.8% — we drop small form text |
| NIST 800-53r5 | -0.9 pt | Precision 97.9% vs 100% — we emit header/footer crud |
| NIST 800-63b | -1.0 pt | Same pattern — false-positive tokens |
| IRS W-9 | -0.1 pt | Tie within noise |
| RFC 9110 | 0% vs 0% | Shared bug — CID font, no ToUnicode |
| RFC 8446 | 99.6 both | Tie |
| example-1.jpg | **+33.7 pt win** (88.7 vs 55.0) | OCR pipeline strength |

Headline today: spdf mean F1 **80.6 %** in **366 ms**; liteparse 77.4 % in
2594 ms. Mean IoU 0.679 vs 0.482.

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

- [ ] **#T2.1 CID-font ToUnicode fallback (RFC 9110 fix).**
      When a glyph has no ToUnicode map, render the glyph to a small
      bitmap and route that single glyph to tesseract at ≥ 300 DPI.
      Cache by `(font_id, cid)` so each unique glyph hits OCR once per
      document. *Expected: RFC 9110 jumps 0 → ~95 %. Pure lead move;
      liteparse also scores 0 %.*
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
- [ ] **#T3.2 Incremental / streaming parse for huge PDFs.**
      Drop memory 10-50× on 1000+ page docs; enables pipe-through.
- [ ] **#T3.3 Native PDF content-stream parser (bypass pdfium for text).**
      🐉 The big one. 4-6 weeks. 2-5× speedup, removes 7 MB binary dep,
      fixes ToUnicode permanently. **Do NOT start before Tier 1 + 2
      are shipped.**

---

## Tier 4 — marketing / distribution

- [ ] **#T4.1 Windows prebuilt binary** (issue #2, needs Win host)
- [ ] **#T4.2 macOS prebuilt binaries** (issue #1, needs Mac host)
- [ ] **#T4.3 OSS-Fuzz onboarding** (issue #4) — continuous-fuzz badge
- [ ] **#T4.4 Python bindings via PyO3** — opens data-science market
- [ ] **#T4.5 WASM build + live web demo** — drop-a-PDF page on the
      existing [website/](website/)
- [ ] **#T4.6 Fix pdfium OOM** (issue #3) — still open; real fix requires
      process isolation or a pre-scanner

---

## Hygiene

- [ ] Keep `benchmark/run.sh` + `benchmark/spatial.py` re-runnable in
      CI so summary.md and spatial.md don't drift.
- [ ] When we claim a win, commit the refreshed benchmark outputs in
      the same PR. No vibes-based performance claims.
