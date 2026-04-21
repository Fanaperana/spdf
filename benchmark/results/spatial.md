# Spatial benchmark — spdf vs liteparse

Ground truth: raw tesseract (PSM_AUTO, `user_defined_dpi=150`) run directly on the rendered image. PDFs are rasterised at 150 DPI via `pdftoppm`. Pixel bboxes converted to PDF points (× 72/150). Tokens are lowercased and pure-punctuation noise dropped before matching.


Matching: per ground-truth word, pick the engine word with the same text and maximum IoU (ties by centroid distance). `iou≥0.5` gives the localisation-precision rate.


## Per-fixture

| fixture | engine | tokens | matched | F1 | mean IoU | median IoU | IoU≥0.5 | centroid err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| example-1.jpg | spdf | 232 | 212 | 90.6% | 0.976 | 1.000 | 97.6% | 4.50 pt |
| example-1.jpg | liteparse | 147 | 109 | 56.9% | 0.667 | 0.844 | 67.9% | 28.03 pt |
| test-ocr.pdf | spdf | 6 | 5 | 62.5% | 0.952 | 0.953 | 100.0% | 0.64 pt |
| test-ocr.pdf | liteparse | 6 | 4 | 50.0% | 0.957 | 0.957 | 100.0% | 0.54 pt |
| irs-f1040.pdf | spdf | 773 | 115 | 15.5% | 0.476 | 0.794 | 55.7% | 97.90 pt |
| irs-f1040.pdf | liteparse | 281 | 84 | 17.0% | 0.351 | 0.552 | 52.4% | 135.73 pt |
| irs-fw9-p1-2.pdf | spdf | 133 | 29 | 5.6% | 0.517 | 0.817 | 58.6% | 169.21 pt |
| irs-fw9-p1-2.pdf | liteparse | 133 | 28 | 5.4% | 0.348 | 0.547 | 53.6% | 175.61 pt |
| nist-sp-800-53r5-p1-2.pdf | spdf | 14 | 3 | 11.5% | 0.964 | 0.963 | 100.0% | 0.35 pt |
| nist-sp-800-53r5-p1-2.pdf | liteparse | 7 | 1 | 4.4% | 0.634 | 0.634 | 100.0% | 2.01 pt |
| nist-sp-800-63b-p1-2.pdf | spdf | 72 | 14 | 12.4% | 0.678 | 0.851 | 78.6% | 84.12 pt |
| nist-sp-800-63b-p1-2.pdf | liteparse | 56 | 20 | 19.0% | 0.471 | 0.622 | 65.0% | 103.50 pt |
| rfc8446-p1-2.pdf | spdf | 26 | 1 | 1.0% | 0.869 | 0.869 | 100.0% | 0.44 pt |
| rfc8446-p1-2.pdf | liteparse | 36 | 4 | 3.7% | 0.427 | 0.548 | 50.0% | 171.02 pt |
| rfc9110-p1-2.pdf | spdf | 0 | 0 | 0.0% | 0.000 | 0.000 | 0.0% | 0.00 pt |
| rfc9110-p1-2.pdf | liteparse | 4 | 0 | 0.0% | 0.000 | 0.000 | 0.0% | 0.00 pt |

## Mean over fixtures

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 24.9% | 0.679 | 73.8% | 44.64 pt |
| liteparse | 19.6% | 0.482 | 61.1% | 77.05 pt |

## Vs pdftotext oracle (born-digital PDFs)

Ground truth is `pdftotext -bbox-layout` word boxes. Excludes raster fixtures and PDFs whose text layer pdftotext cannot read (e.g. CID fonts with no ToUnicode). Higher is better; this isolates spatial accuracy on the cases where the PDF actually has a ground truth.

| fixture | engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---|---:|---:|---:|---:|
| test-ocr.pdf | spdf | 54.5% | 0.000 | 0.0% | 337.26 pt |
| test-ocr.pdf | liteparse | 54.5% | 0.000 | 0.0% | 337.26 pt |
| irs-f1040.pdf | spdf | 21.9% | 0.397 | 59.9% | 80.49 pt |
| irs-f1040.pdf | liteparse | 25.9% | 0.521 | 65.8% | 87.44 pt |
| irs-fw9-p1-2.pdf | spdf | 5.5% | 0.557 | 69.0% | 101.50 pt |
| irs-fw9-p1-2.pdf | liteparse | 5.3% | 0.438 | 71.4% | 105.83 pt |
| nist-sp-800-53r5-p1-2.pdf | spdf | 14.6% | 0.834 | 100.0% | 1.63 pt |
| nist-sp-800-53r5-p1-2.pdf | liteparse | 5.9% | 0.609 | 100.0% | 2.31 pt |
| nist-sp-800-63b-p1-2.pdf | spdf | 13.0% | 0.661 | 78.6% | 83.82 pt |
| nist-sp-800-63b-p1-2.pdf | liteparse | 17.0% | 0.433 | 70.6% | 121.85 pt |
| rfc8446-p1-2.pdf | spdf | 1.9% | 0.846 | 100.0% | 5.48 pt |
| rfc8446-p1-2.pdf | liteparse | 4.6% | 0.435 | 80.0% | 137.64 pt |
| rfc9110-p1-2.pdf | spdf | 0.0% | 0.000 | 0.0% | 0.00 pt |
| rfc9110-p1-2.pdf | liteparse | 0.0% | 0.000 | 0.0% | 0.00 pt |

### Mean (pdftotext oracle)

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 15.9% | 0.471 | 58.2% | 87.17 pt |
| liteparse | 16.2% | 0.348 | 55.4% | 113.19 pt |
