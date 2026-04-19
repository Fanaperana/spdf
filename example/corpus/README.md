# Public-domain corpus

Small selection of publicly redistributable PDFs used by `benchmark/run.sh`
and the unit tests. Everything here is either a U.S. Government work
(public domain under 17 USC §105) or IETF material (RFC series,
unrestricted redistribution).

| File | Source | License |
| --- | --- | --- |
| `irs-f1040.pdf` | [IRS Form 1040](https://www.irs.gov/pub/irs-pdf/f1040.pdf) (2-page tax form with a dense field/table layout) | Public domain (U.S. government work) |
| `nist-sp-800-63b-p1-2.pdf` | [NIST SP 800-63B](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf), first 2 pages (prose + section headings) | Public domain (U.S. government work) |
| `rfc8446-p1-2.pdf` | [RFC 8446 — TLS 1.3](https://www.rfc-editor.org/rfc/pdfrfc/rfc8446.txt.pdf), first 2 pages (plain-text RFC typesetting) | IETF Trust Legal Provisions; unrestricted redistribution |

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
