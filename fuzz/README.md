# Fuzzing

This crate hosts [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
harnesses that exercise the public `SpdfParser::parse` entry point with
adversarial byte blobs, to make sure the PDF parser never panics,
aborts, or hangs on malformed input.

## Setup

```sh
cargo install cargo-fuzz
rustup toolchain install nightly   # libFuzzer needs nightly
```

## Run a target

```sh
cargo +nightly fuzz run parse_pdf     -- -max_total_time=300
cargo +nightly fuzz run project_grid  -- -max_total_time=300
```

Two harnesses are defined:

| Target          | What it fuzzes                                  | Needs pdfium? |
| --------------- | ----------------------------------------------- | ------------- |
| `parse_pdf`     | Public `SpdfParser::parse(bytes)` on PDF blobs  | yes           |
| `project_grid`  | `project_pages_to_grid` + `detect_tables` on synthetic `TextItem` vectors — pure-Rust, no pdfium | no |

`parse_pdf` seeds a `SpdfParser` with OCR disabled and `max_pages=8`,
caps each input at 512 KiB, and calls `parse(ParseInput::Bytes(…))`.
Any panic / timeout / out-of-memory counts as a finding.

`project_grid` uses a coverage-guided `Arbitrary` impl over `TextItem`
fields so every branch of the clustering / grid / header-footer /
table pipeline gets exercised. Rejects inputs with > 4096 items or
> 16 pages so libfuzzer spends its budget on structurally-interesting
cases.

## Continuous fuzzing in CI

`.github/workflows/fuzz.yml` runs both targets on every PR
(2 min each), every push to `main` (5 min each), and on a nightly
schedule (1 h each per target). Crash artifacts are uploaded as a
build artifact if any harness fails.

## Corpus

A seed corpus directory is tracked at `fuzz/corpus/parse_pdf/` (via a
`.keep` sentinel). Populate it with real PDFs before the first run so
libFuzzer has useful structural material to mutate:

```sh
cp ../example/*.pdf ../example/corpus/*.pdf fuzz/corpus/parse_pdf/
cargo +nightly fuzz run parse_pdf fuzz/corpus/parse_pdf -- -max_total_time=600
```

Recommended runtime budget before each release: **at least 1 CPU-hour**
(`-max_total_time=3600`). For 1.0 release, do a full 24 CPU-hour run
(`-max_total_time=86400`). Track the latest budget in the release notes.

## Known findings

| Date | Finding | Status |
| --- | --- | --- |
| 2026-04-18 | OOM at ~2 GiB RSS on a mutated ~150 KB PDF (artifact saved at `corpus/parse_pdf/oom-d3bf6727d48df284b948852ddbcf7030bccd0cc4`). Triggered inside pdfium's object-stream decompression on pathological `/Length` fields. | Mitigated by `max_input_bytes(1 MiB)` + `timeout_secs(10)` in the fuzz target; considered pre-1.0 known-issue, to be fixed with a pdfium memory budget in a future release. |

A total of 9 minutes / ~1700 iterations of fuzzing has been run on the
pre-0.2.0 codebase. No other crashes or panics were observed.

## Triaging findings

Crashes land under `fuzz/artifacts/<target>/crash-<hash>`. Minimise with:

```sh
cargo +nightly fuzz tmin parse_pdf fuzz/artifacts/parse_pdf/crash-<hash>
```

Attach the minimised reproducer to a bug report. Do **not** commit
crashing inputs that may contain private data.
