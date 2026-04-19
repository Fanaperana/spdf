# Security Policy

## Supported versions

`spdf` is pre-1.0; only the latest release on `main` receives security
updates.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| < main  | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security reports.**

Use GitHub's [private vulnerability reporting](https://github.com/Fanaperana/spdf/security/advisories/new)
to disclose privately. We aim to acknowledge reports within 72 hours and
publish a fix within 14 days for critical issues.

When reporting, please include:

- A description of the issue and its potential impact
- Reproduction steps or a proof-of-concept
- Your preferred credit name for the advisory (optional)

## Scope

- Memory-safety bugs in `spdf` Rust code (including `unsafe` blocks)
- Input-validation failures that could lead to crashes, denials of service,
  or arbitrary code execution when processing untrusted PDFs
- Dependency vulnerabilities that affect a released version

Issues in upstream PDFium, Tesseract, or other third-party binaries should
be reported to their respective projects.

## Supply chain

- Dependencies are pinned via `Cargo.lock`.
- CI runs `cargo audit` on every push.
- Release artifacts (when published) will be signed.

## Hardening against adversarial PDFs

PDF is a notoriously adversarial format. If you are parsing untrusted
documents, follow these practices:

- **Cap input size** at the ingress layer *and* via
  `ParseConfig::max_input_bytes` (not set by default).
- **Cap page count** via `ParseConfig::max_pages` (default 1000). A
  malicious PDF with a million-entry page tree will still slow down
  pdfium's tree walk even if you cap pages.
- **Set a timeout** via `ParseConfig::timeout_secs`. Fuzzing has found
  at least one ~150 KB mutated PDF that exhausts ~2 GiB of RSS inside
  pdfium's object-stream decoder; a per-parse timeout is the simplest
  backstop. See [`fuzz/README.md`](fuzz/README.md) for the current
  known-findings table.
- **Run under a resource budget.** On Linux, wrap the process in
  `systemd-run --scope --property=MemoryMax=1G --property=CPUQuota=200%`
  or a seccomp-jailed sandbox. On macOS, use `launchd` limits.
- **Treat file paths as untrusted.** When reading user-supplied paths,
  validate them against an allowlist; spdf does not perform its own
  path sanitisation.
- **Keep pdfium up to date.** spdf bundles pdfium at build time from
  [bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries);
  rebuild against a fresh release after any upstream PDFium CVE.
- **Run the fuzz harness** (`fuzz/README.md`) before exposing spdf to
  untrusted input paths in production.

We plan to land additional hardening (pdfium memory ceiling,
deeper property-based tests for the projection algorithm, OSS-Fuzz
integration) before 1.0. Track progress under
[issue label `hardening`](https://github.com/Fanaperana/spdf/issues?q=label%3Ahardening).
