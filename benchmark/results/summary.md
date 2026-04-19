# Benchmark — spdf vs liteparse

Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).

Token regex: `[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*`, case-insensitive multiset precision/recall.


## Per-fixture

| fixture | engine | wall-clock | tokens | recall | precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| irs-f1040.pdf | spdf | 270 ms | 1591 | 81.9% | 76.3% | 79.0% |
| irs-fw9-p1-2.pdf | spdf | 91 ms | 2253 | 99.1% | 98.4% | 98.7% |
| nist-sp-800-53r5-p1-2.pdf | spdf | 16 ms | 96 | 82.5% | 97.9% | 89.5% |
| nist-sp-800-63b-p1-2.pdf | spdf | 842 ms | 222 | 93.5% | 96.8% | 95.1% |
| rfc8446-p1-2.pdf | spdf | 122 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc9110-p1-2.pdf | spdf | 289 ms | 0 | 0.0% | 0.0% | 0.0% |
| example-1.jpg | spdf | 1204 ms | 231 | 82.0% | 96.5% | 88.7% |
| test-ocr.pdf | spdf | 400 ms | 20 | 100.0% | 100.0% | 100.0% |

## Mean over fixtures

| engine | mean recall | mean precision | mean F1 | mean wall-clock |
|---|---:|---:|---:|---:|
| spdf      | 79.8% | 83.2% | 81.3% | 404 ms |
