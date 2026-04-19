# Spatial benchmark — spdf vs liteparse

Ground truth: raw tesseract (PSM_AUTO, `user_defined_dpi=150`) run directly on the rendered image. PDFs are rasterised at 150 DPI via `pdftoppm`. Pixel bboxes converted to PDF points (× 72/150). Tokens are lowercased and pure-punctuation noise dropped before matching.


Matching: per ground-truth word, pick the engine word with the same text and maximum IoU (ties by centroid distance). `iou≥0.5` gives the localisation-precision rate.


## Per-fixture

| fixture | engine | tokens | matched | F1 | mean IoU | median IoU | IoU≥0.5 | centroid err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| example-1.jpg | spdf | 232 | 212 | 90.6% | 0.976 | 1.000 | 97.6% | 4.50 pt |
| test-ocr.pdf | spdf | 6 | 5 | 62.5% | 0.952 | 0.953 | 100.0% | 0.64 pt |
| irs-f1040.pdf | spdf | 773 | 115 | 15.5% | 0.476 | 0.794 | 55.7% | 97.90 pt |
| irs-fw9-p1-2.pdf | spdf | 133 | 29 | 5.6% | 0.517 | 0.817 | 58.6% | 169.21 pt |
| nist-sp-800-53r5-p1-2.pdf | spdf | 14 | 3 | 11.5% | 0.964 | 0.963 | 100.0% | 0.35 pt |
| nist-sp-800-63b-p1-2.pdf | spdf | 72 | 14 | 12.4% | 0.678 | 0.851 | 78.6% | 84.12 pt |
| rfc8446-p1-2.pdf | spdf | 26 | 1 | 1.0% | 0.869 | 0.869 | 100.0% | 0.44 pt |
| rfc9110-p1-2.pdf | spdf | 4 | 0 | 0.0% | 0.000 | 0.000 | 0.0% | 0.00 pt |

## Mean over fixtures

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 24.9% | 0.679 | 73.8% | 44.64 pt |

## Vs pdftotext oracle (born-digital PDFs)

Ground truth is `pdftotext -bbox-layout` word boxes. Excludes raster fixtures and PDFs whose text layer pdftotext cannot read (e.g. CID fonts with no ToUnicode). Higher is better; this isolates spatial accuracy on the cases where the PDF actually has a ground truth.

| fixture | engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---|---:|---:|---:|---:|
| test-ocr.pdf | spdf | 54.5% | 0.000 | 0.0% | 337.26 pt |
| irs-f1040.pdf | spdf | 21.9% | 0.397 | 59.9% | 80.49 pt |
| irs-fw9-p1-2.pdf | spdf | 5.5% | 0.557 | 69.0% | 101.50 pt |
| nist-sp-800-53r5-p1-2.pdf | spdf | 14.6% | 0.834 | 100.0% | 1.63 pt |
| nist-sp-800-63b-p1-2.pdf | spdf | 13.0% | 0.661 | 78.6% | 83.82 pt |
| rfc8446-p1-2.pdf | spdf | 1.9% | 0.846 | 100.0% | 5.48 pt |
| rfc9110-p1-2.pdf | spdf | 0.0% | 0.000 | 0.0% | 0.00 pt |

### Mean (pdftotext oracle)

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 15.9% | 0.471 | 58.2% | 87.17 pt |
