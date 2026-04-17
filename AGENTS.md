# spdf — Agent Guide

Rust rewrite of `liteparse/` (TypeScript). This doc is the single source of
truth for where things live and how they relate. Keep it short; expand per-crate
READMEs for the gory details.

## Crates

| Crate | Purpose | Mirrors (TS) |
| --- | --- | --- |
| `spdf-types` | Public schema: `TextItem`, `ParsedPage`, `ParseResult`, `ParseConfig`. No heavy deps. | `liteparse/src/core/types.ts` |
| `spdf-processing` | Pure text/geometry helpers: bbox, clean, markup, search, ocr-utils. | `liteparse/src/processing/*` (minus grid) |
| `spdf-projection` | Spatial grid reconstruction (the crown jewel). | `liteparse/src/processing/gridProjection.ts`, `grid.ts` |
| `spdf-pdf` | `PdfEngine` trait + PDFium implementation (text + render + images). | `liteparse/src/engines/pdf/*` |
| `spdf-ocr` | `OcrEngine` trait + HTTP + Tesseract implementations. | `liteparse/src/engines/ocr/*` |
| `spdf-convert` | Shell out to LibreOffice / ImageMagick for non-PDF inputs. | `liteparse/src/conversion/convertToPdf.ts` |
| `spdf-output` | JSON + text formatters. | `liteparse/src/output/*` |
| `spdf-core` | `SpdfParser` orchestrator: load → extract → OCR → project → format. | `liteparse/src/core/parser.ts` |
| `spdf-cli` | `spdf` binary (clap). Installed as `spdf` and `lit`. | `liteparse/cli/parse.ts` |
| `spdf-node` | `@llamaindex/spdf` via napi-rs (excluded from default workspace builds). | `liteparse/src/lib.ts` exports |
| `xtask` | Parity harness, benches, pdfium downloader. | N/A |

## Key Invariants

- Coordinates use PDF points, top-left origin (matches liteparse).
- All numeric work uses `f64` to stay bit-compatible with JS `Number`.
- Iteration order of "JS-object-keyed" structures is preserved via `IndexMap`
  to keep the projection algorithm deterministic against liteparse snapshots.
- Public JSON output: `serde(rename_all = "camelCase")` + `preserve_order`
  serializer + `skip_serializing_if = Option::is_none`.
- Parity target: same visible text + bboxes within ε=0.5pt vs liteparse JSON,
  verified by the `xtask parity` harness.
