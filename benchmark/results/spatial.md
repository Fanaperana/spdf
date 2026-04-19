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

## Mean over fixtures

| engine | F1 | mean IoU | IoU≥0.5 | centroid err |
|---|---:|---:|---:|---:|
| spdf      | 76.5% | 0.964 | 98.8% | 2.57 pt |
| liteparse | 53.5% | 0.812 | 83.9% | 14.28 pt |
