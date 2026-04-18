# Contributing to spdf

Thanks for your interest in making `spdf` better! This project welcomes
issues, pull requests, and design discussions.

## Getting set up

```sh
git clone https://github.com/Fanaperana/spdf.git
cd spdf
cargo build --workspace
cargo test  --workspace
```

You'll need:

- Rust **1.85+** (see [rust-toolchain.toml](rust-toolchain.toml))
- A PDFium shared library available at runtime (see [README](README.md#install))
- Optional: Tesseract for the OCR tests, LibreOffice for the conversion tests

## Development workflow

- Run `cargo fmt` and `cargo clippy --all-targets --all-features` before
  opening a PR. CI enforces both.
- Keep commits focused. One logical change per commit.
- New behaviour needs a test. Regression fixes need a test.
- Parity with reference outputs lives in [tests/parity/](tests/parity/).
  If you change projection or rendering, rerun:
  ```sh
  python3 tests/parity/compare.py
  ```
  and commit the updated golden files when the change is intentional.

## Project layout

See the [architecture section](README.md#architecture) in the README for the
crate map. In short:

- `crates/spdf-projection/` — the spatial reconstruction heuristics. Most
  quality fixes happen here.
- `crates/spdf-pdf/` — PDFium integration.
- `crates/spdf-core/` — orchestrator that wires everything together.
- `crates/spdf-cli/` — command-line binary.
- `xtask/` — parity harness, benches, pdfium fetcher.

## Pull requests

1. Fork and create a feature branch.
2. Make your change with tests.
3. Ensure `cargo test --workspace` is green.
4. Open a PR describing *what* changed and *why*. Link any related issues.
5. Be patient and kind during review.

## Reporting issues

Please include:

- spdf version (`spdf --version`) and platform
- A minimal reproducer (ideally a tiny PDF we can redistribute)
- Expected vs actual output
- Redact any sensitive content first

## Code of Conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md).
