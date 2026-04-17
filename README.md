# spdf

Fast Rust port of [LiteParse](../liteparse/README.md) — spatial PDF parsing with
pluggable OCR and format conversion. Organised as a Cargo workspace; the main
user-facing artifact is the `spdf` CLI, with a `spdf-core` library crate and
`@llamaindex/spdf` Node bindings on the roadmap.

See [AGENTS.md](AGENTS.md) for the crate map and implementation plan.

## Status

Early scaffolding. The workspace builds and the test suite is green, but the
full feature set (spatial grid projection, OCR merge, format conversion, batch
mode, screenshots-to-disk, Node bindings) is still being ported from liteparse.

## Build

```sh
cargo check --workspace
cargo test --workspace
```

## Runtime dependency: PDFium

`spdf-pdf` dynamically loads the PDFium shared library at runtime. Install it
with your system package manager or download a prebuilt bundle:

- macOS (Homebrew): `brew install pdfium` or drop `libpdfium.dylib` somewhere
  on `DYLD_LIBRARY_PATH`.
- Debian / Ubuntu: `apt install libpdfium-dev`, or download a prebuilt release
  from <https://github.com/bblanchon/pdfium-binaries/releases> and set
  `LD_LIBRARY_PATH`.
- Windows: unpack a prebuilt `pdfium.dll` into the same directory as
  `spdf.exe` or anywhere on `PATH`.

An `xtask pdfium-download` helper is planned.

## Layout

```
crates/
  spdf-types/        public schema
  spdf-processing/   text/geometry helpers
  spdf-projection/   spatial reconstruction (the crown jewel)
  spdf-pdf/          PdfEngine trait + PDFium impl
  spdf-ocr/          OcrEngine trait + HTTP + Tesseract impls
  spdf-convert/      LibreOffice / ImageMagick shell-outs
  spdf-output/       JSON + text formatters
  spdf-core/         orchestrator (LiteParse equivalent)
  spdf-cli/          spdf binary
xtask/               parity harness, benches, pdfium fetcher
```
