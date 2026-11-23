<div align="center">

# spdf

**spdf** — *speed-f*: a **s**patial **PDF** parser.
Column-aware text extraction at a lightning pace, in Rust.

<!-- typewriter.gif placeholder — drop a recording of the website hero
     here when we cut one. The phrases cycle through:
       "fast, spatial PDF parsing in Rust."
       "speedf. a lightning-fast PDF parser."
       "spacial PDF parser. column-aware by design."
       "single static binary. PDFium embedded."
       "OCR built in. PP-OCRv5, no Tesseract needed."
-->

Extract text with preserved columns, tables, and layout — plus built-in
neural OCR for scans, format conversion for Office docs, and a single
self-contained binary.

[![CI](https://github.com/Fanaperana/spdf/actions/workflows/ci.yml/badge.svg)](https://github.com/Fanaperana/spdf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](rust-toolchain.toml)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#install)

</div>

---

## Get started in 10 minutes

Assumes Rust 1.85+ is already installed ([rustup.rs](https://rustup.rs);
installing Rust itself is extra). On a machine with a warm cargo cache the
install below took 7 m 21 s; a cold crate cache or the first OCR model
download (~110 MB) can take longer.

**1. Install** (builds current `main` from GitHub):

```sh
cargo install --git https://github.com/Fanaperana/spdf --locked spdf-cli
spdf --version
# -> spdf 0.2.0-alpha.2
```

The build bundles PDFium and fetches the PP-OCRv5 models, then embeds the
OCR weights in the binary. Nothing else needs to be installed to run it.

**2. Parse a born-digital PDF** (OCR off — faster when you know the PDF has
a text layer):

```sh
spdf parse invoice.pdf --no-ocr                    # plain text, layout preserved
spdf parse invoice.pdf --no-ocr --format markdown  # Markdown
spdf parse invoice.pdf --no-ocr --format json      # JSON with per-item bboxes
```

**3. Parse a scan.** OCR is **on by default**; just run `parse`:

```sh
spdf parse scan.pdf
spdf parse scan.pdf --format markdown
```

Each of these commands takes well under a second on a typical document
(60–90 ms for a born-digital page, ~0.26 s for the first OCR parse of
[example/test-ocr.pdf](example/test-ocr.pdf) on the machine above). To try
them on real files, use the fixtures in the repo:
[tests/corpus/hello.pdf](tests/corpus/hello.pdf),
[example/corpus/irs-fw9-p1-2.pdf](example/corpus/irs-fw9-p1-2.pdf) (a
born-digital form) and [example/test-ocr.pdf](example/test-ocr.pdf) (an
image-only page).

`--format` accepts `text` (default), `json`, and `markdown`. See
[OCR](#ocr) for devices, languages, and where the models live.

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
- **Built-in OCR** — PP-OCRv5 compiled into the binary for image-only
  pages; no Python, ONNX runtime, or system Tesseract. An HTTP OCR server
  or local Tesseract are optional alternatives. `--no-ocr` turns it off
  when you know the PDF is born-digital.
- **Format conversion** — Office docs via LibreOffice shell-out, images via
  ImageMagick, all behind the same CLI.
- **One self-contained binary** — PDFium and the OCR weights are bundled
  at build time.

## spdf vs LiteParse 2.19.0

Current head-to-head against [LiteParse](https://github.com/run-llama/liteparse)
2.19.0 (`lit 2.19.0`) and spdf `0.2.0-alpha.2`, from
[bench/results/latest.json](bench/results/latest.json) (run 2026-11-23).
Full scoreboard, including every per-case row: [bench/RESULTS.md](bench/RESULTS.md).

**125 wins · 36 ties · 1 loss** across all tracked metrics. Because of that
one loss, spdf does **not** yet beat LiteParse on every metric.

`odl` = 200 real PDFs from opendataloader-bench (no OCR). `edge` = 48
generated edge cases (default OCR on). Higher is better for accuracy; lower
is better for time and memory.

### Accuracy

| metric | spdf | LiteParse | |
| --- | ---: | ---: | --- |
| odl overall | **0.9015** | 0.8862 | win |
| odl NID (reading order) | **0.9265** | 0.9114 | win |
| odl TEDS (tables) | **0.8351** | 0.8277 | win |
| odl MHS (headings) | **0.8515** | 0.8320 | win |
| edge NID, mean | **0.9747** | 0.8686 | win |
| edge token F1, mean | **0.9664** | 0.8850 | win |
| edge OCR NID, mean (scanned cases) | **0.9901** | 0.7490 | win |

### Speed (seconds, per-process wall time)

| metric | spdf | LiteParse | |
| --- | ---: | ---: | --- |
| odl s/doc, mean | **0.0426** | 0.0943 | win |
| odl s/doc, median | **0.0419** | 0.0945 | win |
| odl s/doc, p95 | **0.0473** | 0.1107 | win |
| edge total (48 cases) | **4.6501** | 9.6424 | win |
| edge OCR s/doc, mean | **0.3668** | 0.4744 | win |
| edge OCR s/doc, max | **0.6079** | 0.9237 | win |
| edge 400-page doc | **0.1679** | 0.3328 | win |

### Memory (peak RSS, MB)

| metric | spdf | LiteParse | |
| --- | ---: | ---: | --- |
| odl peak | **30.8** | 53.1 | win |
| odl mean | **15.0** | 30.9 | win |
| edge max | **67.0** | 158.8 | win |
| edge mean | **22.2** | 73.9 | win |
| edge OCR mean | **57.3** | 90.3 | win |
| edge OCR max | **67.0** | 122.1 | win |
| edge 400-page doc | **17.8** | 40.9 | win |

### The one loss

| case | metric | spdf | LiteParse | |
| --- | --- | ---: | ---: | --- |
| `scanned_chinese` | wall time (s) | 0.3375 | **0.2969** | **loss** |
| `scanned_chinese` | score | **1.0000** | 0.0392 | win |
| `scanned_chinese` | peak RSS (MB) | **67.0** | 85.5 | win |

spdf is about 41 ms slower than LiteParse on the scanned Chinese page. On
the same page it recovers the text (score 1.0000 vs 0.0392) and uses less
memory.

### Methodology

- **Suites:** [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
  at commit `7af1d8f` (200 PDFs with markdown ground truth, scored with
  the harness's own scorer: NID, TEDS, MHS, overall = mean of the three),
  plus 48 generated edge cases from
  [bench/edge/make_corpus.py](bench/edge/make_corpus.py) (multi-column,
  tables, rotation, CJK / RTL, encryption, broken xrefs, image-only scans,
  a 400-page doc, garbage inputs, …), scored by NID and token F1 against
  the generator's ground truth.
- **CLI, one process per document** for both engines (`--no-ocr` on `odl`,
  default OCR on `edge`), so every number includes process start-up,
  engine/model load, and teardown — the cost a user pays per file.
- Each document runs **3 times; the median is reported**, after one
  untimed warm-up run per suite for each engine.
- **Peak memory** is the larger of polled RSS (including child processes)
  and the process's exact `PeakWorkingSetSize`.
- **OCR engines differ:** LiteParse 2.19.0's default build uses bundled
  Tesseract; spdf uses PP-OCRv5 on Burn. The comparison is like-for-like at
  the user/CLI level (same input, same invocation, default settings, same
  machine), not engine-for-engine.
- Tie bands, the win/loss rules, and the regression gate are documented in
  [bench/README.md](bench/README.md).

### Reproduce

```powershell
pwsh bench/setup.ps1                                  # once
bench/.venv/Scripts/python bench/run_bench.py          # full run -> bench/.cache/candidate.json
bench/.venv/Scripts/python bench/scoreboard.py show bench/.cache/candidate.json
bench/.venv/Scripts/python bench/scoreboard.py gate bench/results/latest.json bench/.cache/candidate.json
```

`scoreboard.py show bench/results/latest.json` prints the accepted board.
`gate` and `publish` refuse partial `--cases` runs: acceptance always uses
the full 200-document `odl` suite plus all edge cases. The older
`benchmark/` directory (token F1 vs raw Tesseract on `example/`) is kept
for history only; `bench/` is the tracked scoreboard.

### Feature comparison

| Feature                              | spdf (this project)       | LiteParse          | pdftotext  | pypdfium2  |
| ------------------------------------ | :-----------------------: | :----------------: | :--------: | :--------: |
| Language                             | Rust                      | TypeScript         | C++        | C++/Python |
| Single self-contained binary         | ✅                        | ❌ (Node)          | ✅         | ❌         |
| Column-aware text projection         | ✅                        | ✅                 | partial    | ❌         |
| Faux-bold shadow dedup               | ✅                        | ✅                 | ❌         | ❌         |
| QR / microprint filter               | ✅                        | ✅                 | ❌         | ❌         |
| Default OCR                          | PP-OCRv5 (built in)       | Tesseract (bundled)| ❌         | ❌         |
| HTTP OCR server option               | ✅                        | ✅                 | ❌         | ❌         |
| Office-format conversion             | ✅                        | ✅                 | ❌         | ❌         |
| Batch mode                           | ✅                        | ✅                 | ❌         | ❌         |
| JSON output with per-item bboxes     | ✅                        | ✅                 | ❌         | partial    |
| Markdown output                      | ✅                        | ✅                 | ❌         | ❌         |
| C ABI FFI crate                      | ✅                        | ❌                 | ✅         | ✅         |

## OCR

OCR is **enabled by default** in the CLI. Pages that need it are rendered
and recognised; `--no-ocr` turns it off.

- **Engine:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  PP-OCRv5 — mobile text detection (DBNet) and recognition (SVTR-LCNet,
  CTC), plus the PP-OCRv5 **server** recogniser for CJK lines and a Latin
  recogniser for Latin-script lines.
- **Native code, no runtime deps:** the models are compiled to native
  [Burn](https://burn.dev) 0.22 code at build time (`burn-onnx`). No
  Python, no ONNX runtime, and no system Tesseract on the default path.
- **Devices:** wgpu GPU (Vulkan / DX12 / Metal) with CPU fallback, and
  CUDA when built with `--features neural-ocr-cuda`. Override the choice
  with `SPDF_OCR_DEVICE=cpu|wgpu|cuda`.
- **Models:** fetched and SHA-256-verified at **build** time into
  - `%LOCALAPPDATA%\spdf\models` on Windows,
  - `$XDG_CACHE_HOME/spdf/models` or `~/.cache/spdf/models` on Unix,
  - or `$SPDF_OCR_MODEL_DIR` if set,

  then **embedded in the final binary**. Files already cached with the
  right hash are not downloaded again, and users of an already-built
  binary need no separate model download. Offline (and docs.rs) builds
  still compile; the neural engine then reports its models as unavailable.
- **Alternatives:** `--ocr-server-url http://host:port` sends pages to an
  HTTP OCR server; building with `--features tesseract` adds local
  Tesseract (`--tessdata-path` applies only to Tesseract).
  `--ocr-language` (default `en`) selects the language.

```sh
# CUDA build
cargo install --git https://github.com/Fanaperana/spdf --locked spdf-cli --features neural-ocr-cuda
SPDF_OCR_DEVICE=cuda spdf parse scan.pdf

# Force CPU inference
SPDF_OCR_DEVICE=cpu spdf parse scan.pdf
```

## Production-readiness

spdf is pre-1.0. The table below tracks what we've hardened so you can
decide whether it fits your threat model; see [CHANGELOG.md](CHANGELOG.md)
for per-release detail.

| Area | Status |
| --- | --- |
| JSON output schema (byte-compatible with LiteParse) | stable (covered by parity harness) |
| Typed error enum at the public API (`SpdfError`) | stable |
| Benchmark corpus | [bench/](bench/): opendataloader-bench (200 PDFs) + 48 generated edge cases; public-domain fixtures in [example/](example/) |
| Property tests (`cargo test -p spdf-projection proptests`) | panic-freedom + shuffle-stability |
| Fuzz harness (`cargo +nightly fuzz run parse_pdf`) | [fuzz/](fuzz/) — run before exposing to untrusted input |
| Cross-platform CI | Linux + macOS + Windows; MSRV 1.85; rustdoc warnings gated |
| Resource guards (`timeout_secs`, `max_input_bytes`, `max_pages`) | available via [`SpdfParser::builder`](crates/spdf-core/src/lib.rs) |
| Security policy | see [SECURITY.md](SECURITY.md) |
| CLI / Rust library API | best-effort stable, breaks noted in [CHANGELOG.md](CHANGELOG.md) |
| `spdf-ffi` C ABI | **unstable** — symbols may change across 0.x releases |
| crates.io publication | [`spdf-core`](https://crates.io/crates/spdf-core), [`spdf-cli`](https://crates.io/crates/spdf-cli), and all sub-crates are on crates.io from 0.2.0-alpha.1 (the published 0.2.0-alpha.2 predates built-in OCR; see [Install](#install)) |

**Recommended posture** when parsing untrusted PDFs today:

```rust
let parser = SpdfParser::builder()
    .timeout_secs(30)           // defensive deadline
    .max_input_bytes(50 << 20)  // 50 MiB input cap
    .max_pages(500)             // reject page-tree bombs
    .build();
```

Then wrap the process in a resource-capped sandbox (`systemd-run
--property=MemoryMax=1G`, Firejail, Docker `--memory=`). Follow the
full hardening checklist in [SECURITY.md](SECURITY.md).

## Install

### From GitHub (recommended)

```sh
# Requires Rust 1.85+ (https://rustup.rs). Then:
cargo install --git https://github.com/Fanaperana/spdf --locked spdf-cli
```

This builds current `main` with the default `neural-ocr` feature.

> **Note on crates.io and prebuilt binaries.** `spdf-cli 0.2.0-alpha.2` on
> crates.io was published on 2026-04-19 and is behind current `main`
> despite the same version string: its default features are empty and OCR
> there is Tesseract-only, opt-in. The GitHub release for that tag only
> attaches a Linux x86_64 tarball of the same older build. Use the command
> above until a new release is cut.

### From source

```sh
git clone https://github.com/Fanaperana/spdf
cd spdf
cargo install --path crates/spdf-cli

# or build without installing
cargo build --release -p spdf-cli
./target/release/spdf --help
```

### What the build does

- **PDFium** is downloaded and bundled at build time, then extracted to a
  user cache on first run. Set `PDFIUM_LIB_PATH` to use a system PDFium
  library instead.
- **OCR models** (PP-OCRv5) are fetched and SHA-256-verified on the first
  build, then embedded in the binary — see [OCR](#ocr) for the cache
  location.
- **Features:** `neural-ocr` (default), `neural-ocr-cuda` (adds CUDA;
  select it at runtime with `SPDF_OCR_DEVICE=cuda`), `tesseract` (optional
  local Tesseract, needs a system Tesseract/Leptonica and a C toolchain).

### Prebuilt binaries

| Target | Archive | Status |
| --- | --- | :---: |
| `x86_64-unknown-linux-gnu` | `spdf-<version>-x86_64-unknown-linux-gnu.tar.gz` | attached to v0.2.0-alpha.2 (older build, no built-in OCR) |
| `aarch64-unknown-linux-gnu` | `spdf-<version>-aarch64-unknown-linux-gnu.tar.gz` | ⬜ TODO |
| `x86_64-apple-darwin` | `spdf-<version>-x86_64-apple-darwin.tar.gz` | ⬜ TODO — build on macOS Intel |
| `aarch64-apple-darwin` | `spdf-<version>-aarch64-apple-darwin.tar.gz` | ⬜ TODO — build on Apple Silicon |
| `x86_64-pc-windows-msvc` | `spdf-<version>-x86_64-pc-windows-msvc.zip` | ⬜ TODO — build on Windows |

To produce a release tarball on a new host:

```sh
cargo build --release -p spdf-cli
VER=0.2.0-alpha.2
TARGET=$(rustc -vV | awk '/^host:/ {print $2}')
DIR="spdf-${VER}-${TARGET}"
mkdir -p "dist/${DIR}"
cp target/release/spdf "dist/${DIR}/"          # use spdf.exe on Windows
cp LICENSE README.md CHANGELOG.md "dist/${DIR}/"
tar czf "dist/${DIR}.tar.gz" -C dist "${DIR}"   # or zip on Windows
sha256sum "dist/${DIR}.tar.gz" > "dist/${DIR}.tar.gz.sha256"
gh release upload v${VER} "dist/${DIR}.tar.gz" "dist/${DIR}.tar.gz.sha256"
```

### Platform support matrix

| Platform | Core parsing | Built-in OCR (default) | Notes |
| --- | :---: | :---: | --- |
| Linux x86_64 | ✅ | ✅ | |
| macOS (Intel + Apple Silicon) | ✅ | ✅ | GPU via Metal |
| Windows x86_64 | ✅ | ✅ | benchmark machine; GPU via DX12 / Vulkan |

Built-in OCR needs no system packages on any platform. The optional
`tesseract` feature links `libtesseract` + `libleptonica` via `bindgen`
and needs those libraries plus clang; on Windows that means `vcpkg
install tesseract leptonica --triplet x64-windows` and `LIBCLANG_PATH`.
Most users don't need it.

## CLI reference

```sh
# Confirm the install
spdf --version

# Plain text with preserved layout
spdf parse invoice.pdf --no-ocr --format text

# Markdown
spdf parse invoice.pdf --no-ocr --format markdown

# Read from stdin (handy for piping through curl / aws s3 cp)
cat invoice.pdf | spdf parse - --no-ocr

# Structured JSON with per-glyph bounding boxes
spdf parse invoice.pdf --no-ocr --format json > out.json

# Password-protected PDFs
spdf parse confidential.pdf --password 'hunter2'

# Keep very-small glyphs (<2 pt) — useful for IRS-style form field labels
spdf parse irs-fw9.pdf --preserve-small-text

# Scanned PDFs: built-in OCR runs by default
spdf parse scan.pdf

# Use an external OCR server (PaddleOCR, EasyOCR, etc.)
spdf parse scan.pdf --ocr-server-url http://localhost:8000

# Render specific pages
spdf parse book.pdf --target-pages 1-3,7,12-15

# Dump pages as PNGs
spdf screenshot report.pdf -o ./pages --dpi 200

# Batch-convert a directory of PDFs
spdf batch-parse ./inputs ./outputs --format text
```

Exit codes: `0` success, `1` parse/OCR error (with `Error: …` on stderr),
`2` invalid CLI arguments (standard clap).

## Library usage

```rust
use spdf_core::SpdfParser;
use spdf_types::ParseConfig;

let parser = SpdfParser::new(ParseConfig {
    ocr_enabled: false,
    ..Default::default()
});
let result = parser.parse(std::path::Path::new("invoice.pdf"))?;
for page in &result.pages {
    println!("--- page {} ---\n{}", page.page_num, page.text);
}
# Ok::<(), spdf_types::SpdfError>(())
```

Or use the ergonomic builder if you only want to tweak a few knobs:

```rust
use spdf_core::SpdfParser;

let parser = SpdfParser::builder()
    .timeout_secs(30)
    .max_input_bytes(50 << 20)
    .ocr_enabled(false)
    .build();
let result = parser.parse(std::path::Path::new("invoice.pdf"))?;
# Ok::<(), spdf_types::SpdfError>(())
```

## Architecture

```
crates/
  spdf-types/        public schema
  spdf-processing/   text / geometry / markup helpers
  spdf-projection/   spatial reconstruction (the crown jewel)
  spdf-pdf/          PdfEngine trait + PDFium impl
  spdf-ocr/          OcrEngine trait + built-in PP-OCRv5 (Burn) + HTTP + optional Tesseract
  spdf-convert/      LibreOffice / ImageMagick shell-outs
  spdf-output/       JSON + text + markdown formatters
  spdf-core/         orchestrator
  spdf-cli/          spdf binary
  spdf-ffi/          C ABI cdylib
bench/               spdf vs LiteParse scoreboard
xtask/               parity harness, benches, pdfium fetcher
```

See [AGENTS.md](AGENTS.md) for the full crate map and
[CONTRIBUTING.md](CONTRIBUTING.md) for development workflow.

## Roadmap

- Node bindings (`@spdf/node`) on top of `spdf-ffi`
- Python bindings via PyO3
- `spdf serve` — a local HTTP parsing service
- Prebuilt Windows and macOS binaries, and a crates.io release with built-in OCR

## Release process (for maintainers)

Releases are fully automated by [`.github/workflows/release.yml`](.github/workflows/release.yml):

1. Bump `version` in the workspace `Cargo.toml` and update [CHANGELOG.md](CHANGELOG.md).
2. `git commit -am "release 0.x.y" && git push`
3. `git tag -a v0.x.y -m "spdf 0.x.y" && git push origin v0.x.y`

The workflow then:
- builds the `x86_64-unknown-linux-gnu` tarball and attaches it to the
  GitHub release (creating one if missing),
- runs `cargo publish` for all 9 crates in topological order using the
  `CRATES_IO_TOKEN` repository secret, with retry-on-429 for the
  crates.io new-crate rate limit.

To smoke-test the pipeline without consuming a real version number, run
the workflow manually from the **Actions** tab with `dry_run = true` —
it skips uploads and runs `cargo publish --dry-run` for every crate.

## Acknowledgements

spdf is an independent Rust project authored by
[Fanaperana](https://github.com/Fanaperana). The spatial projection
algorithm was inspired by (and is benchmarked against)
[LiteParse](https://github.com/run-llama/liteparse), but spdf is not a
port or rewrite — it's its own implementation, with its own engine
choices (PDFium + PP-OCRv5 on Burn), its own data model, and its own
hardening work. Rendering is powered by
[PDFium](https://pdfium.googlesource.com/pdfium/); OCR uses
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv5 models
run on [Burn](https://burn.dev), with optional
[Tesseract](https://github.com/tesseract-ocr/tesseract) support.

## License

[MIT](LICENSE) © 2026 spdf contributors.
