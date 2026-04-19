# Benchmark — spdf vs liteparse

Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).

Token regex: `[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*`, case-insensitive multiset precision/recall.


## Per-fixture

| fixture | engine | wall-clock | tokens | recall | precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| irs-f1040.pdf | spdf | 271 ms | 1094 | 63.8% | 86.5% | 73.4% |
| irs-f1040.pdf | liteparse | 466 ms | 1575 | 81.8% | 77.0% | 79.4% |
| nist-sp-800-63b-p1-2.pdf | spdf | 788 ms | 222 | 93.5% | 96.8% | 95.1% |
| nist-sp-800-63b-p1-2.pdf | liteparse | 4679 ms | 226 | 95.2% | 96.9% | 96.1% |
| rfc8446-p1-2.pdf | spdf | 29 ms | 399 | 99.5% | 99.7% | 99.6% |
| rfc8446-p1-2.pdf | liteparse | 358 ms | 399 | 99.5% | 99.7% | 99.6% |
| example-1.jpg | spdf | 1041 ms | 231 | 82.0% | 96.5% | 88.7% |
| example-1.jpg | liteparse | 6531 ms | 146 | 42.3% | 78.8% | 55.0% |
| test-ocr.pdf | spdf | 266 ms | 20 | 100.0% | 100.0% | 100.0% |
| test-ocr.pdf | liteparse | 2965 ms | 20 | 100.0% | 100.0% | 100.0% |

## Mean over fixtures

| engine | mean recall | mean precision | mean F1 | mean wall-clock |
|---|---:|---:|---:|---:|
| spdf      | 87.8% | 95.9% | 91.4% | 479 ms |
| liteparse | 83.8% | 90.5% | 86.0% | 3000 ms |
