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
