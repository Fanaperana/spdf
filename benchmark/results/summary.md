# Benchmark — spdf vs liteparse

Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).

Token regex: `[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*`, case-insensitive multiset precision/recall.


## Per-fixture

| fixture | engine | wall-clock | tokens | recall | precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| irs-f1040.pdf | spdf | 252 ms | 1591 | 81.9% | 76.3% | 79.0% |
| irs-f1040.pdf | liteparse | 460 ms | 1575 | 81.8% | 77.0% | 79.4% |
| irs-fw9-p1-2.pdf | spdf | 66 ms | 2252 | 99.1% | 98.4% | 98.7% |
| irs-fw9-p1-2.pdf | liteparse | 408 ms | 2253 | 99.1% | 98.4% | 98.8% |
| nist-sp-800-53r5-p1-2.pdf | spdf | 10 ms | 95 | 82.5% | 98.9% | 90.0% |
| nist-sp-800-53r5-p1-2.pdf | liteparse | 371 ms | 94 | 82.5% | 100.0% | 90.4% |
| nist-sp-800-63b-p1-2.pdf | spdf | 21 ms | 207 | 88.3% | 98.1% | 92.9% |
| nist-sp-800-63b-p1-2.pdf | liteparse | 4628 ms | 226 | 95.2% | 96.9% | 96.1% |
| rfc8446-p1-2.pdf | spdf | 14 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc8446-p1-2.pdf | liteparse | 349 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc9110-p1-2.pdf | spdf | 5 ms | 0 | 0.0% | 0.0% | 0.0% |
| rfc9110-p1-2.pdf | liteparse | 2728 ms | 8 | 0.0% | 0.0% | 0.0% |
| example-1.jpg | spdf | 196 ms | 0 | 0.0% | 0.0% | 0.0% |
| example-1.jpg | liteparse | 6232 ms | 146 | 42.3% | 78.8% | 55.0% |
| test-ocr.pdf | spdf | 20 ms | 15 | 75.0% | 100.0% | 85.7% |
| test-ocr.pdf | liteparse | 2855 ms | 20 | 100.0% | 100.0% | 100.0% |

## Mean over fixtures

| engine | mean recall | mean precision | mean F1 | mean wall-clock |
|---|---:|---:|---:|---:|
| spdf      | 65.8% | 71.4% | 68.2% | 73 ms |
| liteparse | 75.1% | 81.4% | 77.4% | 2254 ms |
