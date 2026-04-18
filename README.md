<div align="center">

# spdf

**Fast, spatial PDF parsing in Rust.**

Extract text with preserved columns, tables, and layout — plus optional OCR
for scans, format conversion for Office docs, and a single self-contained
binary.

[![CI](https://github.com/Fanaperana/spdf/actions/workflows/ci.yml/badge.svg)](https://github.com/Fanaperana/spdf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](rust-toolchain.toml)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#install)

</div>

---

## Why spdf

Most PDF-to-text tools collapse whitespace, shuffle columns, and emit one
giant line salad — fine for search indexing, useless for anything that cares
about *where* things appear on the page (invoices, tax bills, property
records, scientific tables, legal forms).

`spdf` keeps the geometry:

- **Column-aware projection** — tables, two-column layouts, sidebars, and
  indented blocks come back in reading order with their spatial structure
  intact.
- **Faux-bold & shadow dedup** — PDFs that "draw text twice" to simulate
  bold no longer produce `TaTax Infofo`; you get `Tax Info`.
- **Word reconstruction** — PDFium-style per-glyph extraction is stitched
  back into words (`1 8 3 6` → `1836`) using a liteparse-compatible merge
  heuristic.
- **QR / barcode / microprint filtering** — the hundreds of tiny numeric
  glyphs that encode a QR code are auto-dropped so they don't destroy the
  surrounding table.
- **Optional OCR** — Tesseract locally, or any HTTP OCR server (PaddleOCR,
  EasyOCR, etc.) for image-only pages. One flag to turn it off when you
  know the PDF is born-digital.
- **Format conversion** — Office docs via LibreOffice shell-out, images via
  ImageMagick, all behind the same CLI.
- **One static binary.** Install PDFium once, ship `spdf` anywhere.

## Comparison

Benchmarked on two real-world U.S. county tax documents (TAX_APPEAL_MOM
set) with `--no-ocr`. Token-level F1 measured against the same documents
parsed through [LiteParse](https://github.com/run-llama/liteparse) using the
provided reference outputs in [tests/parity/](tests/parity/).

| Feature                              | spdf (this project) | LiteParse  | pdftotext  | pypdfium2  |
| ------------------------------------ | :-----------------: | :--------: | :--------: | :--------: |
| Language                             | Rust                | TypeScript | C++        | C++/Python |
| Single static binary                 | ✅                  | ❌ (Node)  | ✅         | ❌         |
| Column-aware text projection         | ✅                  | ✅         | partial    | ❌         |
| Faux-bold shadow dedup               | ✅                  | ✅         | ❌         | ❌         |
| QR / microprint filter               | ✅                  | ✅         | ❌         | ❌         |
| OCR fallback (Tesseract + HTTP)      | ✅                  | ✅         | ❌         | ❌         |
| Office-format conversion             | ✅                  | ✅         | ❌         | ❌         |
| Batch mode                           | ✅                  | ✅         | ❌         | ❌         |
| JSON output with per-item bboxes     | ✅                  | ✅         | ❌         | partial    |
| C ABI FFI crate                      | ✅                  | ❌         | ✅         | ✅         |
| **Token F1 vs LiteParse (tax bill)** | **0.990**           | 1.000      | ~0.82      | ~0.80      |
| **Token F1 vs LiteParse (PRC)**      | **0.922**           | 1.000      | ~0.75      | ~0.78      |
| **Startup time (cold)**              | ~25 ms              | ~450 ms    | ~10 ms     | ~120 ms    |

Parity harness and golden outputs live in [tests/parity/](tests/parity/);
run `python3 tests/parity/compare.py` to reproduce.

## Install

```sh
# from source (requires Rust 1.85+)
cargo install --path crates/spdf-cli

# or build locally
cargo build --release -p spdf-cli
./target/release/spdf --help
```

### Runtime dependency: PDFium

`spdf` dynamically loads a PDFium shared library. On macOS:

```sh
brew install pdfium
```

Or download a prebuilt binary from
[bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/releases)
and point `PDFIUM_LIB_PATH` at it.

## Quick start

```sh
# Plain text with preserved layout
spdf parse invoice.pdf --no-ocr --format text

# Structured JSON with per-glyph bounding boxes
spdf parse invoice.pdf --no-ocr --format json > out.json

# OCR-only mode for scanned PDFs
spdf parse scan.pdf --ocr-language eng

# Use an external OCR server (PaddleOCR, EasyOCR, etc.)
spdf parse scan.pdf --ocr-server-url http://localhost:8000

# Render specific pages
spdf parse book.pdf --target-pages 1-3,7,12-15

# Dump pages as PNGs
spdf screenshot report.pdf -o ./pages --dpi 200

# Batch-convert a directory of PDFs
spdf batch-parse ./inputs ./outputs --format text
```

## Library usage

```rust
use spdf_core::LiteParse;
use spdf_types::ParseConfig;

let parser = LiteParse::new(ParseConfig {
    ocr_enabled: false,
    ..Default::default()
});
let result = parser.parse_path("invoice.pdf")?;
for page in &result.pages {
    println!("--- page {} ---\n{}", page.page_num, page.text);
}
```

## Architecture

```
crates/
  spdf-types/        public schema
  spdf-processing/   text / geometry / markup helpers
  spdf-projection/   spatial reconstruction (the crown jewel)
  spdf-pdf/          PdfEngine trait + PDFium impl
  spdf-ocr/          OcrEngine trait + Tesseract + HTTP impls
  spdf-convert/      LibreOffice / ImageMagick shell-outs
  spdf-output/       JSON + text formatters
  spdf-core/         orchestrator
  spdf-cli/          spdf binary
  spdf-ffi/          C ABI cdylib
xtask/               parity harness, benches, pdfium fetcher
```

See [AGENTS.md](AGENTS.md) for the full crate map and
[CONTRIBUTING.md](CONTRIBUTING.md) for development workflow.

## Roadmap

- Node bindings (`@spdf/node`) on top of `spdf-ffi`
- Python bindings via PyO3
- `spdf serve` — a local HTTP service compatible with the liteparse API
- Optional ML-based reading-order classifier (opt-in, `burn` feature flag)

## Acknowledgements

The spatial projection algorithm is inspired by and benchmarked against
[LiteParse](https://github.com/run-llama/liteparse) by the LlamaIndex team.
Rendering is powered by [PDFium](https://pdfium.googlesource.com/pdfium/).
OCR uses [Tesseract](https://github.com/tesseract-ocr/tesseract).

## License

[MIT](LICENSE) © 2026 spdf contributors.
