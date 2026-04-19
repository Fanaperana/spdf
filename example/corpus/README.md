# Public-domain corpus

Small selection of publicly redistributable PDFs used by `benchmark/run.sh`
and the unit tests. Everything here is either a U.S. Government work
(public domain under 17 USC §105) or IETF material (RFC series,
unrestricted redistribution).

| File | Source | License |
| --- | --- | --- |
| `irs-f1040.pdf` | [IRS Form 1040](https://www.irs.gov/pub/irs-pdf/f1040.pdf) (2-page tax form with a dense field/table layout) | Public domain (U.S. government work) |
| `irs-fw9-p1-2.pdf` | [IRS Form W-9](https://www.irs.gov/pub/irs-pdf/fw9.pdf), first 2 pages (short form + instructions prose) | Public domain (U.S. government work) |
| `nist-sp-800-63b-p1-2.pdf` | [NIST SP 800-63B](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf), first 2 pages (prose + section headings) | Public domain (U.S. government work) |
| `nist-sp-800-53r5-p1-2.pdf` | [NIST SP 800-53r5](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf), first 2 pages (security controls, heavy tables) | Public domain (U.S. government work) |
| `rfc8446-p1-2.pdf` | [RFC 8446 — TLS 1.3](https://www.rfc-editor.org/rfc/pdfrfc/rfc8446.txt.pdf), first 2 pages (plain-text RFC typesetting) | IETF Trust Legal Provisions; unrestricted redistribution |
| `rfc9110-p1-2.pdf` | [RFC 9110 — HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.pdf), first 2 pages (modern IETF layout) | IETF Trust Legal Provisions; unrestricted redistribution |
| `cjk-unicode-p1-2.pdf` | [Unicode 17.0 CJK Unified Ideographs chart](https://unicode.org/charts/PDF/U4E00.pdf), first 2 pages (exercises non-ASCII character handling) | © Unicode, Inc. — Unicode code charts are freely redistributable |
| `encrypted.pdf` | `irs-f1040.pdf` re-encrypted with Ghostscript (`-sUserPassword=secret -dEncryptionR=3 -dKeyLength=128`) | Derived from public-domain work; exercises the password-required code path |
| `malformed.pdf` | First 200 bytes of `rfc8446-p1-2.pdf` (truncated) | Derived; exercises the malformed-input code path |

Large PDFs were trimmed to their first two pages with Ghostscript so the
committed corpus stays under ~500 KB total:

```sh
gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dFirstPage=1 -dLastPage=2 \
   -sOutputFile=trimmed.pdf input.pdf
```

## Adding new fixtures

1. Confirm the file is genuinely redistributable (U.S. gov / CC0 / public
   domain). Do not commit private tax documents, invoices, medical records,
   or anything under an unknown license.
2. Prefer trimming multi-page PDFs to 1–3 representative pages.
3. Keep the total corpus under ~2 MB so the benchmark runs quickly in CI.
4. Re-run `make benchmark-update` and commit the regenerated results.
