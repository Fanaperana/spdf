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
| nist-sp-800-63b-p1-2.pdf | spdf | 72 | 14 | 12.4% | 0.678 | 0.851 | 78.6% | 84.12 pt |
| nist-sp-800-63b-p1-2.pdf | liteparse | 56 | 20 | 19.0% | 0.471 | 0.622 | 65.0% | 103.50 pt |
| rfc8446-p1-2.pdf | spdf | 26 | 1 | 1.0% | 0.869 | 0.869 | 100.0% | 0.44 pt |
| rfc8446-p1-2.pdf | liteparse | 36 | 4 | 3.7% | 0.427 | 0.548 | 50.0% | 171.02 pt |

## Mean over fixtures

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 36.4% | 0.790 | 86.4% | 37.52 pt |
| liteparse | 29.3% | 0.575 | 67.1% | 87.76 pt |
