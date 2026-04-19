# Benchmark — spdf vs liteparse

Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).

Token regex: `[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*`, case-insensitive multiset precision/recall.


## Per-fixture

| fixture | engine | wall-clock | tokens | recall | precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| irs-f1040.pdf | spdf | 268 ms | 1094 | 63.8% | 86.5% | 73.4% |
| irs-f1040.pdf | liteparse | 541 ms | 1575 | 81.8% | 77.0% | 79.4% |
| irs-fw9-p1-2.pdf | spdf | 76 ms | 2253 | 99.1% | 98.4% | 98.7% |
| irs-fw9-p1-2.pdf | liteparse | 465 ms | 2253 | 99.1% | 98.4% | 98.8% |
| nist-sp-800-53r5-p1-2.pdf | spdf | 17 ms | 96 | 82.5% | 97.9% | 89.5% |
| nist-sp-800-53r5-p1-2.pdf | liteparse | 461 ms | 94 | 82.5% | 100.0% | 90.4% |
| nist-sp-800-63b-p1-2.pdf | spdf | 969 ms | 222 | 93.5% | 96.8% | 95.1% |
| nist-sp-800-63b-p1-2.pdf | liteparse | 5530 ms | 226 | 95.2% | 96.9% | 96.1% |
| rfc8446-p1-2.pdf | spdf | 20 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc8446-p1-2.pdf | liteparse | 375 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc9110-p1-2.pdf | spdf | 235 ms | 8 | 0.0% | 0.0% | 0.0% |
| rfc9110-p1-2.pdf | liteparse | 3101 ms | 8 | 0.0% | 0.0% | 0.0% |
| example-1.jpg | spdf | 1067 ms | 231 | 82.0% | 96.5% | 88.7% |
| example-1.jpg | liteparse | 7070 ms | 146 | 42.3% | 78.8% | 55.0% |
| test-ocr.pdf | spdf | 274 ms | 20 | 100.0% | 100.0% | 100.0% |
| test-ocr.pdf | liteparse | 3212 ms | 20 | 100.0% | 100.0% | 100.0% |

## Mean over fixtures

| engine | mean recall | mean precision | mean F1 | mean wall-clock |
|---|---:|---:|---:|---:|
| spdf      | 77.5% | 84.5% | 80.6% | 366 ms |
| liteparse | 75.1% | 81.4% | 77.4% | 2594 ms |
