# Changelog

All notable changes to `spdf` land here. We follow
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Benchmark corpus** — three public-domain PDFs under
  [example/corpus/](example/corpus/) (IRS Form 1040, NIST SP 800-63B
  excerpt, RFC 8446 excerpt) so the benchmark exercises real-world
  form/prose/RFC layouts instead of just the 2 original fixtures.
- **Spatial-precision benchmark** (`benchmark/spatial.py`) that scores
  IoU and centroid error against raw-tesseract ground truth.
- **Property tests** for `spdf-projection::project_page`: panic-freedom,
  input-alphabet preservation, and stability under input shuffle.
- **Fuzz harness** (`fuzz/`) with a `parse_pdf` cargo-fuzz target that
  exercises `SpdfParser::parse(bytes)` with adversarial blobs.
- **Windows CI** — `test` matrix now covers Ubuntu, macOS, and
  Windows. A separate `msrv` job pins the minimum supported Rust
  version (1.85) and a `rustdoc` job gates on doc warnings.
- `CHANGELOG.md`, `SECURITY.md` update, and a documented stability
  policy for the pre-1.0 API.

### Fixed
- Rust tesseract binding defaults to `PSM_SINGLE_BLOCK`; we now force
  `PSM_AUTO` to match the Tesseract CLI and recover ~2× word yield on
  mixed-layout images (example-1.jpg: 92 → 230 words).
- OCR words within 2 pt of each other no longer suppress each other
  through the overlap filter (snapshot `existing_len` pre-OCR).
- pdfium-rendered PNGs have no `pHYs` chunk; libtesseract fell back to
  70 DPI. We now set `user_defined_dpi` before `set_image_from_mem`.

### Changed
- OCR confidence threshold bumped from 0.1 to 0.3.
- `run_ocr` now runs per-page OCR in parallel via rayon.
- Tesseract instances are cached `thread_local!` keyed on
  `(datapath, languages)`.

## [0.1.0] — initial alpha

Initial public alpha. Column-aware grid projection, bundled pdfium,
optional Tesseract OCR, CLI + FFI crates, and a parity harness.

---

## Stability policy

spdf is pre-1.0. That means:

- The **JSON wire format** (`ParseResult`, `TextItem`, `ParsedPage`) is
  considered stable — it mirrors LiteParse's output and is covered by
  the parity harness. Breaking changes here will bump the minor
  version and be called out under **Changed** above.
- The **Rust library API** (`spdf-core`, `spdf-types`) may change in
  any minor release until we cut 1.0. We still note every break in the
  **Changed** section.
- The **CLI flags** are best-effort stable; removals or renames go
  through one release of deprecation warnings.
- The **C ABI** in `spdf-ffi` is not yet stable; do not assume symbol
  stability across versions.

Once the test corpus and fuzz harness have a few months of real usage
without undiscovered soundness issues, we'll cut 1.0 and commit to
semver for the library surface too.
