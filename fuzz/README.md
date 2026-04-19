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
cargo +nightly fuzz run parse_pdf -- -max_total_time=300
```

The harness seeds a `SpdfParser` with OCR disabled and `max_pages=8`,
caps each input at 512 KiB, and calls `parse(ParseInput::Bytes(…))`.
Any panic / timeout / out-of-memory counts as a finding.

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

## Triaging findings

Crashes land under `fuzz/artifacts/<target>/crash-<hash>`. Minimise with:

```sh
cargo +nightly fuzz tmin parse_pdf fuzz/artifacts/parse_pdf/crash-<hash>
```

Attach the minimised reproducer to a bug report. Do **not** commit
crashing inputs that may contain private data.
