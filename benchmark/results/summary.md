# Benchmark — spdf vs liteparse

Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).

Token regex: `[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*`, case-insensitive multiset precision/recall.


## Per-fixture

| fixture | engine | wall-clock | tokens | recall | precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| example-1.jpg | spdf | 1018 ms | 231 | 82.0% | 96.5% | 88.7% |
| example-1.jpg | liteparse | 6346 ms | 146 | 42.3% | 78.8% | 55.0% |
| test-ocr.pdf | spdf | 252 ms | 20 | 100.0% | 100.0% | 100.0% |
| test-ocr.pdf | liteparse | 2860 ms | 20 | 100.0% | 100.0% | 100.0% |

## Mean over fixtures

| engine | mean recall | mean precision | mean F1 | mean wall-clock |
|---|---:|---:|---:|---:|
| spdf      | 91.0% | 98.3% | 94.3% | 635 ms |
| liteparse | 71.1% | 89.4% | 77.5% | 4603 ms |
